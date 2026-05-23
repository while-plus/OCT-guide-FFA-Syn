import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import torch
import pandas as pd
# from torchvision.transforms._transforms_video import NormalizeVideo
import math
from torch.utils.data import DistributedSampler, Sampler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from skimage.draw import line



class CSVDataset(Dataset):
    def __init__(self, csv_path, array_dir, transform=None, sample_oct_num=4):
        super(CSVDataset, self).__init__()

        self.csv_path = csv_path
        self.array_dir = array_dir
        
        self.sample_oct_num = sample_oct_num
        data_df = pd.read_csv(self.csv_path)
        data_df.drop_duplicates(inplace=True)
        data_df.reset_index(drop=True, inplace=True)

        # data_df = data_df.iloc[:5000]
        self.data_df = [row.to_dict() for _, row in tqdm(data_df.iterrows(), total=len(data_df))]
        

        self.dataset_len = len(data_df)
        # self.use_augment = use_augment
        # self.image_embed_dir = image_embed_dir
        # self.split = split
        self.line_pad_num = 100 #int(np.ceil(100 * 2**0.5))
        self.transform = transform

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index
        row = self.data_df[sample_index]

        ffa_file = row['FFA']
        ffa_image = self.load_image_file(self.array_dir, ffa_file)

        cfp_file = row['CFP']
        cfp_image = self.load_image_file(self.array_dir, cfp_file)


        oct_file_lst = [elem for elem in eval(row['OCT']) if len(elem[-1]) != 0]
        oct_image_lst, line_cords_lst = [], []

        for oct_file, cords in oct_file_lst:
            if len(oct_image_lst) >= self.sample_oct_num:
                break
            line = self.fitting_line(cords)
            if line is None:
                continue
            oct_image = self.load_image_file(self.array_dir, oct_file)
            oct_image_lst.append(oct_image)
            line_cords_lst.append(line)        
            


        for _ in range(max(self.sample_oct_num-len(oct_image_lst), 0)):
            oct_image_lst.append(np.zeros((512, 512, 3), dtype=np.uint8))
            line_cords_lst.append(-np.ones([self.line_pad_num, 2], dtype=np.int64))
        line_cords_lst = np.stack(line_cords_lst, 0)

        aug = self.transform is not None and random.random() < 0.5
        if aug:
            transformed = self.transform(image=cfp_image, masks=[ffa_image]+oct_image_lst)
            cfp_image = transformed['image']
            ffa_image = transformed['masks'][0]
            oct_image_lst = transformed['masks'][1:]

            
        cfp_image = ToTensor()(cfp_image) #* 2 - 1
        ffa_image = ToTensor()(ffa_image)[0:1, ...] #* 2 - 1
        oct_image_lst = torch.stack([ToTensor()(oct_image) for oct_image in oct_image_lst])

        return {'A': cfp_image, 'OCT': oct_image_lst,
                'B': ffa_image, 'Line Cords': line_cords_lst, 
                'image file': ffa_file} #\
                # {'A': aug_cfp_image, 'OCT': aug_oct_image_lst,
                # 'B': aug_ffa_image, 'Line Cords': line_cords_lst, }
    

    def load_image_file(self, image_dir, image_file):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def fitting_line(self, cords):
        # return: [self.line_pad_num, 2]
        x1, y1, x2, y2 = [int(x / 512 * 32) for x in cords]
        row, col = line(y1, x1, y2, x2)
        if len(row) > self.line_pad_num:
            return None
        row = np.pad(row, [0, self.line_pad_num-len(row)], 'constant', constant_values=-1)
        col = np.pad(col, [0, self.line_pad_num-len(col)], 'constant', constant_values=-1)
        return np.stack([row, col], -1)     


class GroupedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        # import torch.distributed as dist
        # if num_replicas is None:
        #     num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        # if rank is None:
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        # super().__init__(dataset, num_replicas, rank, shuffle, seed=42)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = 42
        self.shuffle = shuffle
        self.epoch = 0
        # 预计算每个样本的 T
        self.Ts = dataset.Ts
        self.groups = {}
        for idx, t in enumerate(self.Ts):
            self.groups.setdefault(t, []).append(idx)

    def __iter__(self):
        rank_indices = {}
        # 对每个 T 分组独立打乱并分配
        for t, group in self.groups.items():
            group = np.array(group)
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                shuffle_idx = torch.randperm(len(group), generator=g).numpy()
                group = group[shuffle_idx]

            # 平均切分给各 rank
            total = len(group)
            per_rank = math.floor(total / self.num_replicas)
            start = self.rank * per_rank
            end = start + per_rank
            for rank in range(self.num_replicas):
                local_indices = group[start:end]
                rank_indices.setdefault(rank, []).extend(local_indices.tolist())

        # indices = np.array(indices)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shuffle_idx = torch.randperm(self.__len__(), generator=g).numpy()
            for rank in range(self.num_replicas):
                rank_indices[rank] = np.array(rank_indices[rank])[shuffle_idx]


        indices = [rank_indices[rank][i] for i in range(self.__len__()) for rank in range(self.num_replicas) ]

        # print(self.rank, indices)
        # for idx in indices:
        #     yield idx
        return iter(indices)

    def __len__(self):
        return sum([math.floor(len(v) / self.num_replicas) for v in self.groups.values()])
    
    def set_epoch(self, epoch):
        self.epoch = epoch

class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level =noise_level
        
    def __getitem__(self, index):
        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        else:
            # if noise !=0, A and B make different transform
            item_A = self.transform1(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
            item_B = self.transform1(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
            
            
            
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        item_A = self.transform(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        if self.unaligned:
            item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

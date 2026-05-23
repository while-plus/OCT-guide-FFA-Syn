import os
import cv2
from tqdm import tqdm
from cal_ssim import ssim
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import argparse

import lpips
from torchmetrics.image.fid import FrechetInceptionDistance


lpips_metric = lpips.LPIPS()
fid_metric = FrechetInceptionDistance(normalize=True)

def PSNR(target, prediction, max_value=255):
    mse = F.mse_loss(prediction, target)
    psnr_value = 10 * torch.log10(max_value ** 2 / mse)
    return psnr_value.item()


def read_image_lst(ref_image_dir, fake_image_dir):
    ref_image_file_lst = sorted(os.listdir(ref_image_dir))
    fake_image_file_lst = sorted(os.listdir(fake_image_dir))
    
    # assert ref_image_file_lst == fake_image_file_lst
    if ref_image_file_lst!= fake_image_file_lst:
        print("Warning: ref_image_file_lst!= fake_image_file_lst")
    ref_images, fake_images = [], []

    for image_file in fake_image_file_lst:
        ref_image_path = os.path.join(ref_image_dir, image_file)
        fake_image_path = os.path.join(fake_image_dir, image_file)
        ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
        fake_image = cv2.imread(fake_image_path, cv2.IMREAD_GRAYSCALE)

        # ref_image = cv2.resize(ref_image, (256, 256))
        # fake_image = cv2.resize(fake_image, (512, 512))
        
        ref_images.append(ref_image)
        fake_images.append(fake_image)

    return np.array(ref_images), np.array(fake_images)

def cal_metrics(ref_image_dir, fake_image_dir, device, batch_size=1):
    ref_images, fake_images = read_image_lst(ref_image_dir, fake_image_dir)

    ref_tensor = torch.Tensor(np.array(ref_images)).to(device)
    fake_tensor = torch.Tensor(np.array(fake_images)).to(device)
    psnr_score = PSNR(ref_tensor, fake_tensor, max_value=255)


    ref_tensor = ref_tensor.unsqueeze(1) / 255
    fake_tensor = fake_tensor.unsqueeze(1) / 255
    ssim_score = ssim(ref_tensor, fake_tensor).detach().cpu().item() * 100

    torch.cuda.empty_cache()
    ref_tensor = ref_tensor.repeat(1,3,1,1)
    fake_tensor = fake_tensor.repeat(1,3,1,1)

    
    num_split = ref_tensor.shape[0] // batch_size


    lpips_metric.to(device)
    fid_metric.to(device)
    fid_metric.reset()
    lpips_score_lst = []
    for batch_real, batch_fake in zip(ref_tensor.chunk(num_split, dim=0), fake_tensor.chunk(num_split, dim=0)):
        with torch.no_grad():
            score = lpips_metric(batch_real, batch_fake, normalize=True)
            lpips_score_lst.append(score)
            ## resize 到 299
            batch_real = TF.resize(batch_real, size=[299, 299], interpolation=InterpolationMode.BILINEAR)
            batch_fake = TF.resize(batch_fake, size=[299, 299], interpolation=InterpolationMode.BILINEAR)
            fid_metric.update(batch_real, real=True)
            fid_metric.update(batch_fake, real=False)

    lpips_score = torch.concat(lpips_score_lst, dim=0)   
    lpips_score = lpips_score.mean().detach().cpu().numpy().item() * 100
    fid_score = fid_metric.compute().detach().cpu().numpy().item()
    return psnr_score, ssim_score, fid_score, lpips_score
    # return lpips_score, fid_score


def format_metrics(ref_image_dir, fake_image_dir, device, psnr_score, ssim_score, fid_score, lpips_score):
    return "\n".join(
        [
            "Evaluation Results",
            f"  Reference Dir : {ref_image_dir}",
            f"  Generated Dir : {fake_image_dir}",
            f"  Device        : {device}",
            f"  PSNR          : {psnr_score:.4f}",
            f"  SSIM x100     : {ssim_score:.4f}",
            f"  FID           : {fid_score:.4f}",
            f"  LPIPS x100    : {lpips_score:.4f}",
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate generated images against references.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--ref-image-dir", default="./data/FFA", help="Directory of reference images.")
    parser.add_argument("--fake-image-dir", default="./outputs/test/FFA", help="Directory of generated images.")
    args = parser.parse_args()

    psnr_score, ssim_score, fid_score, lpips_score = cal_metrics(
        args.ref_image_dir,
        args.fake_image_dir,
        device=args.device,
    )
    print(
        format_metrics(
            args.ref_image_dir,
            args.fake_image_dir,
            args.device,
            psnr_score,
            ssim_score,
            fid_score,
            lpips_score,
        )
    )

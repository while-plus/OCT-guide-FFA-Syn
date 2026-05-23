#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config
from .datasets import ImageDataset, ValDataset, CSVDataset
from Model.CycleGan import *
from Model.vitunet import *
from .utils import Resize,ToTensor,smooothing_loss
from .utils import Logger
from .reg import Reg
from .transformer import Transformer_2D
from skimage import measure
import numpy as np
import cv2
import albumentations as A
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
import json
from tqdm import tqdm
from accelerate.utils import set_seed
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.functional as F





@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     

def tensor_to_array(image_tensor):
    # image_tensor = (image_tensor+1) / 2
    array = image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    return array
    
    


    

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.accelerator = Accelerator(kwargs_handlers=[
                                        DistributedDataParallelKwargs(find_unused_parameters=True)])
        set_seed(42)
        # def networks
        model_args = {
            'image_shape'        : (3, 512, 512),
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 6,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            # 'unet_features_list' : [96, 192, 384],
            'unet_features_list' : [48, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
            'out_ch'             : 1
        }
        self.netG_A2B = ViTUNetGenerator(**model_args)

        # self.netG_A2B = Generator(config['input_nc'], config['output_nc'])
        self.netD_B = Discriminator(config['output_nc'])
        self.optimizer_G = torch.optim.Adam([param for param in self.netG_A2B.parameters() if param.requires_grad_], 
                                            lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        self.optimizer_G = self.accelerator.prepare(self.optimizer_G)
        self.netG_A2B, self.netD_B, self.optimizer_D_B = \
            self.accelerator.prepare(self.netG_A2B, self.netD_B, self.optimizer_D_B)

        # self.clip_loss = self.accelerator.prepare(CLIPLOSS())
        

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['output_nc'],config['output_nc'])
            self.spatial_transform = Transformer_2D()#.cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            self.R_A, self.spatial_transform, self.optimizer_R_A = \
                self.accelerator.prepare(self.R_A, self.spatial_transform, self.optimizer_R_A)


        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()


        # Dataset loader
        train_transform = A.Compose([A.HorizontalFlip(p=1)])
        train_dataset = CSVDataset(config['train_csv'], config['data_dir'], train_transform)
        self.dataloader = DataLoader(train_dataset, batch_size=config['batchSize'], 
                                     shuffle=True, num_workers=config['n_cpu'])

        val_dataset = CSVDataset(config['test_csv'], config['data_dir'])
        self.val_data = DataLoader(val_dataset, batch_size=1, 
                                    shuffle=False, num_workers=config['n_cpu'], drop_last=False)
        self.dataloader, self.val_data = self.accelerator.prepare(self.dataloader, self.val_data)

    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            self.netG_A2B.train()
            for batch in tqdm(self.dataloader, desc=f'Epoch: {epoch+1}', total=len(self.dataloader), disable=not self.is_main):
                # Set model input
                # real_A = Variable(self.input_A.copy_(batch['A']))
                # real_B = Variable(self.input_B.copy_(batch['B']))
                
                real_A = batch['A']
                real_B = batch['B']
                oct_images = batch['OCT']                             # B, K, H, W
                line_cords = batch['Line Cords']                      # B, K, 2
                


                self.optimizer_R_A.zero_grad()
                self.optimizer_G.zero_grad()
                # self.optimizer_oct.zero_grad()
                #### regist sys loss
                
                fake_B, contrastive_loss = self.netG_A2B(real_A, oct_images, line_cords, real_B)
                # print(oct_feats.shape, ffa_feats.shape, ground_truths.shape)
                Trans = self.R_A(fake_B, real_B) 
                SysRegist_A2B = self.spatial_transform(fake_B,Trans)
                SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B)###SR
                pred_fake0 = self.netD_B(fake_B)
                adv_loss = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, 
                                                                    torch.ones_like(pred_fake0).to(pred_fake0))
                #### smooth loss
                SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                
                ### OCT pred loss

                CLIP_loss = [l*self.config['CLIP_lamda'] for l in contrastive_loss]

                total_loss = SM_loss + adv_loss + SR_loss
                for l in CLIP_loss:
                    total_loss = total_loss+l       #+ OCT_MSE_loss#+ CLIP_loss
                if self.is_main:
                    print(SM_loss.item(),adv_loss.item(),SR_loss.item(), \
                          [l.item() for l in CLIP_loss])        #OCT_MSE_loss.item())#,CLIP_loss.item())
                # toal_loss.backward()
                self.accelerator.backward(total_loss)
                self.optimizer_R_A.step()
                self.optimizer_G.step()
                # self.optimizer_oct.step()



                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    # sparse_oct_features = self.oct_encoder(oct_images, line_cords)
                    fake_B, _ = self.netG_A2B(real_A, oct_images, line_cords)
                pred_fake0 = self.netD_B(fake_B)
                pred_real = self.netD_B(real_B)
                loss_D_B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake0, torch.zeros_like(pred_fake0).to(pred_fake0)) + \
                            self.config['Adv_lamda'] * self.MSE_loss(pred_real, torch.ones_like(pred_real).to(pred_real))


                # loss_D_B.backward()
                self.accelerator.backward(loss_D_B)
                self.optimizer_D_B.step()


            if (epoch+1) % 20 == 0:
                with torch.no_grad():
                    self.test(epoch)
                if self.is_main:
                    save_ckpt_dir = os.path.join(self.config['save_root'], 'ckpt')
                    os.makedirs(save_ckpt_dir, exist_ok=True)
                    torch.save(self.netG_A2B.state_dict(), 
                               os.path.join(save_ckpt_dir, f'netG_A2B_{epoch+1}.pth'))


            self.accelerator.wait_for_everyone()

                
    @property
    def is_main(self):
        return self.accelerator.is_main_process        
    

    def test(self, epoch):
        save_image_dir = os.path.join(self.config['save_root'], 'Images', f"Epoch_{epoch+1}")
        self.netG_A2B.eval()
        for batch in tqdm(self.val_data, desc=f'Epoch: {epoch+1}', total=len(self.val_data), disable=not self.is_main):
            real_A = batch['A']
            oct_images = batch['OCT']                             # B, K, H, W
            line_cords = batch['Line Cords']                      # B, K, 2

            with torch.no_grad():
                fake_B, _ = self.netG_A2B(real_A, oct_images, line_cords)

                for image_file, gen_image in zip(batch['image file'], tensor_to_array(fake_B)):
                    image_file = image_file[:image_file.rfind('.')] + '.jpg'
                    image_path = os.path.join(save_image_dir, image_file)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    cv2.imwrite(image_path, gen_image)
                    
    
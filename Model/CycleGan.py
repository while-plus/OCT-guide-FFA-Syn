import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange








# from .dit import DiT

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.Sequential(
#             # nn.MaxPool2d(2),
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.conv = DoubleConv(out_channels, out_channels)
#     def forward(self, x):
#         x = self.up(x)
#         x = self.conv(x)
#         return x
    
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             # nn.MaxPool2d(2),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc):
#         super(Generator, self).__init__()

#         self.inc = DoubleConv(input_nc, 64)

#         self.downs = nn.Sequential(Down(64, 128),
#                         Down(128, 256),
#                         Down(256, 512),
#                         Down(512, 512))     #  3.94784 M
        
#         self.body = DiT(input_size=32, in_channels=512, depth=6, 
#                hidden_size=256, patch_size=2, num_heads=8,
#                learn_sigma=False)       # 8.480256 M
#         self.ups = nn.Sequential(
#                         Up(512, 256),
#                         Up(256, 128),
#                         Up(128, 64),
#                         Up(64, 64),
#                         )
#         self.out_conv = nn.Sequential(
#                             nn.Conv2d(64, output_nc, kernel_size=1),
#                             nn.Tanh())


#     def forward(self, x, cond):
#         x = self.inc(x)
#         x = self.downs(x)       ## [4, 512, 32, 32], 3.94784 M
#         x = self.body(x, cond)   
#         x = self.ups(x)
#         x = self.out_conv(x)
#         return x, None

def load_resnet_encoder():
    from torchvision.models import resnet34
    model = resnet34()#.to(device)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    model.avgpool = nn.Identity() # nn.AdaptiveAvgPool2d((1, 32))
    model.fc = nn.Identity()
    return model



def modulate(x, shift, scale):
    return x * (1 + scale) + shift



class ResidualBlock(nn.Module):
    def __init__(self, in_features, mid_features=None, n_heads=8):
        super(ResidualBlock, self).__init__()
        mid_features = in_features
        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)
        
        self.fc_down = nn.Sequential(nn.Conv2d(in_features, mid_features, 1),
                                nn.InstanceNorm2d(mid_features))
        self.fc_down_c = nn.Sequential(nn.Conv2d(512, in_features, 1),
                                nn.InstanceNorm2d(in_features))
        self.fc_up = nn.Sequential(nn.Conv2d(mid_features, in_features, 1),
                                nn.InstanceNorm2d(in_features))


        self.norm_1 = nn.InstanceNorm2d(mid_features)
        self.norm_2 = nn.InstanceNorm2d(mid_features)
        self.cross_attn = nn.MultiheadAttention(mid_features, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(mid_features, mid_features*3),
            nn.InstanceNorm2d(mid_features*3),
            nn.Linear(mid_features*3, mid_features),
        )

        # self.adaLN_modulation = nn.Sequential(
        #                 nn.SiLU(),
        #                 nn.Linear(512, 3 * in_features, bias=True)
        #             )
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)

    def forward(self, x, c):
        # print(x.shape, c.shape)
        x = x + self.conv_block(x)
        if c is None:
            return x
        # c = self.cord_mlp(c)
        _, _, H, W = x.shape


        x = self.map_to_tokens(x)       # self.fc_down(x)
        c = self.map_to_tokens(self.fc_down_c(c))        # self.fc_down_c(c)
        y, _ = self.cross_attn(x, x, c)
        y = self.norm_1(y) + x

        y = self.norm_2(self.ffn(y)) + y
        y = self.token_to_map(y, H, W)
        # y = self.fc_up(y)
        return y

    def map_to_tokens(self, x):
        x = rearrange(x, 'b c h w -> (h w) b c')
        return x
    
    def token_to_map(self, x, H, W):
        x = rearrange(x, '(h w) b c -> b c h w', h=H, w=W)
        return x

    
        # print(x.shape)
        # msa = self.adaLN_modulation(c)
        # B, C = msa.shape
        # shift_msa, scale_msa, gate_msa = msa.reshape(B, C, 1, 1).chunk(3, dim=1)
        # # print(shift_msa.shape)        # B, C, 1, 1
        # # print(x.shape)                # B, C, H, W

        # x = x + gate_msa * self.conv_block(modulate(self.norm(x), shift_msa, scale_msa))
        # return x

class Body(nn.Module):
    def __init__(self, n_residual_blocks, in_features):
        super().__init__()
                # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]
        self.model_body = nn.ModuleList(model_body)

    def forward(self, x, c):
        for layer in self.model_body:
            x = layer(x, c)
        return x




class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, downsample_times=2, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(downsample_times):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2



        self.model_body = Body(n_residual_blocks, in_features)

        # self.cord_mlp = nn.Sequential(
        #     nn.Linear(4, in_features),
        #     nn.GELU(),
        #     nn.Linear(in_features, in_features)
        # )
        # self.clip_fc = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(512, 512),
        # )

            
        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(downsample_times):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        # self.model_body_1 = nn.Sequential(*model_body[:n_residual_blocks//2])
        # self.model_body_2 = nn.Sequential(*model_body[n_residual_blocks//2:])
        self.model_tail = nn.Sequential(*model_tail)
        
        self.encoder = load_resnet_encoder()
        self.feat_shape = [512, 512 // 2**downsample_times, 512 // 2**downsample_times] 
        self.sparse_oct_map = nn.Parameter(torch.zeros(1, *self.feat_shape), True)#.to(oct_features)

    def forward(self, x, oct_images, line_cords, ffa=None):
        # oct_features = self.get_oct_sparse_feats(oct_images, line_cords)
        oct_features = None
        # print(oct_features.shape)
        x = self.model_head(x)
        # if ref_cord is not None:
        #     ref_cord = self.cord_mlp(ref_cord)
        #     ref_cord = ref_cord.view(ref_cord.size(0), ref_cord.size(1), 1, 1)
        #     x = x + ref_cord
        # x = self.model_body(x)
        x = self.model_body(x, oct_features)

        x = self.model_tail(x)

        return x

    def get_oct_sparse_feats(self, oct, line_cords):
        batch, slice_num = oct.shape[:2]
        oct = rearrange(oct, 'b s c h w -> (b s) c h w')
        
        oct_features = self.encoder(oct).reshape(-1, *self.feat_shape)
        oct_features = oct_features.mean(dim=2, keepdim=True)
        oct_features = rearrange(oct_features, '(b s) c h w -> b s c h w', b=batch)
        
        sparse_oct_maps = self.sparse_oct_map.repeat(batch, 1, 1, 1)
        sparse_oct_cnts = torch.zeros(batch, *self.feat_shape).to(oct_features)
        for b in range(batch):
            # sparse_oct_map = torch.zeros(1, *self.feat_shape).to(oct_features)
            for s in range(slice_num):
                oct_slice_feat = oct_features[b, s, ...]
                line = line_cords[b, s, ...]
                row, col = line[:, 0], line[:, 1]
                ## 剔除 pad 值 -1
                valid_pad_mask = row != -1
                row, col = row[valid_pad_mask], col[valid_pad_mask]
                ## 限制范围
                valid_mask = torch.logical_and(torch.logical_and(row < self.feat_shape[1], row >= 0), 
                                            torch.logical_and(col < self.feat_shape[2], col >= 0))

                if valid_pad_mask.sum() != 0 and valid_mask.sum() != 0:
                    inter_oct_slice = F.interpolate(oct_slice_feat.unsqueeze(0), 
                                                    size=[1, valid_pad_mask.sum()], mode='bilinear')
                    # print(valid_pad_mask.sum() != 0 and valid_mask.sum())
                    sparse_oct_maps[b:b+1, :, row[valid_mask], col[valid_mask]] += inter_oct_slice[..., 0, valid_mask]
                    sparse_oct_cnts[b:b+1, :, row[valid_mask], col[valid_mask]] += 1
        sparse_oct_maps = sparse_oct_maps / sparse_oct_cnts.clip(min=1)
            # sparse_oct_maps.append(sparse_oct_map)
        return sparse_oct_maps


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

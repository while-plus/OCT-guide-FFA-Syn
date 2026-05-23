# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import copy
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)

def get_norm_layer(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name)

def get_norm_layer_fn(norm):
    return lambda features : get_norm_layer(norm, features)

def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)

def select_optimizer(parameters, optimizer):
    name, kwargs = extract_name_kwargs(optimizer)

    if name == 'AdamW':
        return torch.optim.AdamW(parameters, **kwargs)

    if name == 'Adam':
        return torch.optim.Adam(parameters, **kwargs)

    raise ValueError("Unknown optimizer: '%s'" % name)

def select_loss(loss):
    name, kwargs = extract_name_kwargs(loss)

    if name.lower() in [ 'l1', 'mae' ]:
        return nn.L1Loss(**kwargs)

    if name.lower() in [ 'l2', 'mse' ]:
        return nn.MSELoss(**kwargs)

    raise ValueError("Unknown loss: '%s'" % name)




def get_downsample_x2_conv2_layer(features, **kwargs):
    return (
        nn.Conv2d(features, features, kernel_size = 2, stride = 2, **kwargs),
        features
    )

def get_downsample_x2_conv3_layer(features, **kwargs):
    return (
        nn.Conv2d(
            features, features, kernel_size = 3, stride = 2, padding = 1,
            **kwargs
        ),
        features
    )

def get_downsample_x2_pixelshuffle_layer(features, **kwargs):
    out_features = 4 * features
    return (nn.PixelUnshuffle(downscale_factor = 2, **kwargs), out_features)

def get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features * 4

    layer = nn.Sequential(
        nn.PixelUnshuffle(downscale_factor = 2, **kwargs),
        nn.Conv2d(
            out_features, out_features, kernel_size = 3, padding = 1
        ),
    )

    return (layer, out_features)

def get_upsample_x2_deconv2_layer(features, **kwargs):
    return (
        nn.ConvTranspose2d(
            features, features, kernel_size = 2, stride = 2, **kwargs
        ),
        features
    )

def get_upsample_x2_upconv_layer(features, **kwargs):
    layer = nn.Sequential(
        nn.Upsample(scale_factor = 2, **kwargs),
        nn.Conv2d(features, features, kernel_size = 3, padding = 1),
    )

    return (layer, features)

def get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features // 4

    layer = nn.Sequential(
        nn.PixelShuffle(upscale_factor = 2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 1),
    )

    return (layer, out_features)

def get_downsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'conv':
        return get_downsample_x2_conv2_layer(features, **kwargs)

    if name == 'conv3':
        return get_downsample_x2_conv3_layer(features, **kwargs)

    if name == 'avgpool':
        return (nn.AvgPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'maxpool':
        return (nn.MaxPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'pixel-unshuffle':
        return get_downsample_x2_pixelshuffle_layer(features, **kwargs)

    if name == 'pixel-unshuffle-conv':
        return get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)

def get_upsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'deconv':
        return get_upsample_x2_deconv2_layer(features, **kwargs)

    if name == 'upsample':
        return (nn.Upsample(scale_factor = 2, **kwargs), features)

    if name == 'upsample-conv':
        return get_upsample_x2_upconv_layer(features, **kwargs)

    if name == 'pixel-shuffle':
        return (nn.PixelShuffle(upscale_factor = 2, **kwargs), features // 4)

    if name == 'pixel-shuffle-conv':
        return get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Upsample Layer: '%s'" % name)





def calc_tokenized_size(image_shape, token_size):
    # image_shape : (C, H, W)
    # token_size  : (H_t, W_t)
    if image_shape[1] % token_size[0] != 0:
        raise ValueError(
            "Token width %d does not divide image width %d" % (
                token_size[0], image_shape[1]
            )
        )

    if image_shape[2] % token_size[1] != 0:
        raise ValueError(
            "Token height %d does not divide image height %d" % (
                token_size[1], image_shape[2]
            )
        )

    # result : (N_h, N_w)
    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1])

def img_to_tokens(image_batch, token_size):
    # image_batch : (N, C, H, W)
    # token_size  : (H_t, W_t)

    # result : (N, C, N_h, H_t, W)
    result = image_batch.view(
        (*image_batch.shape[:2], -1, token_size[0], image_batch.shape[3])
    )

    # result : (N, C, N_h, H_t, W       )
    #       -> (N, C, N_h, H_t, N_w, W_t)
    result = result.view((*result.shape[:4], -1, token_size[1]))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, N_h, N_w, C, H_t, W_t)
    result = result.permute((0, 2, 4, 1, 3, 5))

    return result

def img_from_tokens(tokens):
    # tokens : (N, N_h, N_w, C, H_t, W_t)
    # result : (N, C, N_h, H_t, N_w, W_t)
    result = tokens.permute((0, 3, 1, 4, 2, 5))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, C, N_h, H_t, N_w * W_t)
    #        = (N, C, N_h, H_t, W)
    result = result.reshape((*result.shape[:4], -1))

    # result : (N, C, N_h, H_t, W)
    #       -> (N, C, N_h * H_t, W)
    #        = (N, C, H, W)
    result = result.reshape((*result.shape[:2], -1, result.shape[4]))

    return result

class PositionWiseFFN(nn.Module):

    def __init__(self, features, ffn_features, activ = 'gelu', **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Linear(features, ffn_features),
            get_activ_layer(activ),
            nn.Linear(ffn_features, features),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(
        self, features, ffn_features, n_heads, activ = 'gelu', norm = None,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = get_norm_layer(norm, features)
        self.atten = nn.MultiheadAttention(features, n_heads)

        self.cross_norm = get_norm_layer(norm, features)
        self.cross_atten = nn.MultiheadAttention(features, n_heads)
        self.cross_conv = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(2 * features, features, bias=True)
                )

        self.norm2 = get_norm_layer(norm, features)
        self.ffn   = PositionWiseFFN(features, ffn_features, activ)

        # self.oct_fc = nn.Sequential(
        #                 nn.SiLU(),
        #                 nn.Linear(features, features, bias=True)
        #             )
        

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(features, 6 * features, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        # self.rezero = rezero

        # if rezero:
        #     self.re_alpha = nn.Parameter(torch.zeros((1, )))
        # else:
        #     self.re_alpha = 1

        self.cfp_fc = nn.Sequential(
                        nn.Linear(features, features),
                        get_norm_layer(norm, features),
                        get_activ_layer(activ),
                    )
        self.oct_fc = nn.Sequential( 
                        nn.Linear(features, features),
                        get_norm_layer(norm, features),
                        get_activ_layer(activ),

                    )
        # self.cfp_norm1 = get_norm_layer(norm, features)
        # self.oct_norm1 = get_norm_layer(norm, features)

        self.join_attn = nn.MultiheadAttention(features, n_heads)

        self.oct_norm1 = get_norm_layer(norm, features)
        self.oct_atten = nn.MultiheadAttention(features, n_heads)

        self.cfp_norm2 = get_norm_layer(norm, features)
        self.oct_norm2 = get_norm_layer(norm, features)

        self.cfp_ffn   = PositionWiseFFN(features, ffn_features, activ)
        self.oct_ffn   = PositionWiseFFN(features, features, activ)


    def forward(self, x, oct_tokens=None):
        # x: (Seq, Batch, channel)
        # oct_tokens: (Seq, Batch, channel)
        if oct_tokens is None:
            ## Self Attn
            # Step 1: Multi-Head Self Attention
            y1 = self.norm1(x)
            y1, _ = self.atten(y1, y1, y1)
            y = x + y1
            # Step 2: PositionWise Feed Forward Network
            y2 = self.norm2(y)
            y2 = self.ffn(y2)
            y  = y + y2
            return y, _
        ## Cross Attention
        # y1 = self.norm1(x)
        # y1, _ = self.atten(y1, y1, y1)
        # y = x + y1


        # oct_tokens = self.oct_fc(oct_tokens)
        # # print(oct_tokens.shape, q.shape)
        # k = torch.concat([q, oct_tokens])
        # v = torch.concat([oct_tokens, q])
        # join_y, _ = self.join_attn(q, k, v)
        # kv = self.cross_norm(y)  
        # oct_tokens = self.oct_fc(oct_tokens)
        # # # print(oct_tokens.shape, q.shape)
        # # k = torch.concat([q, oct_tokens])
        # # v = torch.concat([oct_tokens, q])
        # cross_y, _ = self.cross_atten(oct_tokens, kv, kv)
        # y = cross_y + y

        # # Step 2: PositionWise Feed Forward Network
        # y2 = self.norm2(y)
        # y2 = self.ffn(y2)
        # y  = y + y2
        # return y, None


        
        # Brodacasting & AdaLN
        # oct_tokens = self.oct_fc(oct_tokens)
        # oct_tokens1 = self.oct_norm1(oct_tokens)
        # oct_tokens1, _ = self.oct_atten(oct_tokens1, oct_tokens1, oct_tokens1)
        # oct_tokens = oct_tokens + oct_tokens1

        # oct_tokens2 = self.oct_norm2(oct_tokens)
        # oct_tokens2 = self.oct_ffn(oct_tokens2)
        # oct_tokens = oct_tokens + oct_tokens2

        oct_tokens = self.oct_fc(oct_tokens)
        y_cross = self.cfp_fc(x)
        y_cross, _ = self.cross_atten(y_cross, y_cross, oct_tokens)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(y_cross).chunk(6, dim=-1)


        y1 = self.modulate(self.norm1(x), shift_msa, scale_msa)
        y1, _ = self.atten(y1, y1, y1)
        y = x +  gate_msa * y1

        y2 = self.modulate(self.norm2(y), shift_mlp, scale_mlp)
        y2 = self.ffn(y2)
        y  = y + gate_mlp * y2
        return y, y_cross


        ## CROSS ATTN
        #  x: (Seq, Batch, channel)
        # sparse_oct_tokens: (Seq, Batch, channel)
        # y1 = self.norm1(x)
        # y1, _atten_weights = self.atten(y1, y1, y1)
        # y = x + y1

        # if sparse_oct_tokens is not None:
        #     ## Brodacasting
        #     sparse_oct_tokens = self.oct_fc(sparse_oct_tokens)
        #     y_cross = self.cross_norm(y)
        #     y_cross, _ = self.cross_atten(y_cross, y_cross, sparse_oct_tokens)
        #     ## Add Fusion
        #     y = y + y_cross
        # y2 = self.norm2(y)
        # y2 = self.ffn(y2)
        # y  = y + y2

        # x = self.cfp_fc(x)
        # sparse_oct_tokens = self.oct_fc(sparse_oct_tokens)


        ## JOINT ATTN
        # y1 = self.cfp_fc(x)
        # oct_tokens1 = self.oct_fc(oct_tokens)

        # qk = torch.concat([y1, oct_tokens1])   # 2 * L, N, features
        # v = torch.concat([oct_tokens1, y1])

        # join_y, _ = self.join_attn(qk, qk, v)
        
        # y1, oct_tokens1 = torch.chunk(join_y, 2, dim=0)
        # y = y1 + x
        # oct_tokens_y = oct_tokens1 + oct_tokens
        

        # y2 = self.cfp_norm2(y)
        # y2 = self.cfp_ffn(y2)
        # y  = y + y2
        

        # oct_tokens_2 = self.oct_norm2(oct_tokens_y)
        # oct_tokens_2 = self.oct_ffn(oct_tokens_2)
        # oct_tokens  = oct_tokens_2 + oct_tokens_y
        # return y, oct_tokens
    
    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    # def extra_repr(self):
    #     return 're_alpha = %e' % (self.re_alpha, )

class TransformerEncoder(nn.Module):

    def __init__(
        self, features, ffn_features, n_heads, n_blocks, activ, norm,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = nn.ModuleList([
            TransformerBlock(
                features, ffn_features, n_heads, activ, norm, rezero
            ) for _ in range(n_blocks)
        ])

        self.feat_shape = [384, 32, 32] 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        width = self.feat_shape[0]
        scale = width ** -0.5
        self.proj_1 = nn.Parameter(scale * torch.randn(width, width))
        self.proj_2 = nn.Parameter(scale * torch.randn(width, width))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, sparse_oct_tokens, ffa=None):
        # x : (N, L, features)
        # sparse_oct_tokens: (N, L, features)

        # y : (L, N, features)
        # print(sparse_oct_tokens.shape)
        y = x.permute((1, 0, 2))
        if ffa is not None:
            ffa_y = ffa.permute((1, 0, 2))

        if sparse_oct_tokens is not None:
            sparse_oct_tokens = sparse_oct_tokens.permute((1, 0, 2))

        contrastive_loss = []
        for idx, block in enumerate(self.encoder):
            # if idx % 2 == 0:
            y, y_cross = block(y, sparse_oct_tokens)
            if ffa is not None:
            #    ffa_y, _ = block(ffa_y)
               contrastive_loss.append(self.contrastive(block(ffa_y)[0], y))

            # else:
            #     y, _ = block(y, None)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))
        if sparse_oct_tokens is not None:
            sparse_oct_tokens = sparse_oct_tokens.permute((1, 0, 2))
        return result, sparse_oct_tokens, y_cross, contrastive_loss

    def contrastive(self, feat_1, feat_2):
        # feat_1 : (N, L, C)
        # feat_2 : (N, T, C)
        # ground_truth : (N, L, T)
        
        feat_1 = feat_1 @ self.proj_1
        feat_2 = feat_2 @ self.proj_2
        feat_1 = feat_1 / feat_1.norm(dim=-1, keepdim=True)
        feat_2 = feat_2 / feat_2.norm(dim=-1, keepdim=True)
        
        b, l, _ = feat_1.shape
        ground_truth = torch.arange(l).repeat(b, 1).long().to(feat_1.device)
        # if feat_1.shape[1] == 0:
        #     return torch.zeros([]).to(feat_1)
        logits_per_1 = einsum('nlc,ntc->nlt', feat_1, feat_2) * self.logit_scale.exp()
        logits_per_2 = logits_per_1.transpose(1, 2)
            
        loss = (
            self.criterion(logits_per_1, ground_truth)
            +  self.criterion(logits_per_2, ground_truth)
        ) / 2
    
        return loss
class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, features, height, width, **kwargs):
        super().__init__(**kwargs)
        self.projector = nn.Linear(2, features)
        self._height   = height
        self._width    = width

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        x_norm = 2 * x / (self._width  - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2)
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim = 2)

        return torch.sin(self.projector(z))

class ViTInput(nn.Module):

    def __init__(
        self, input_features, embed_features, features, height, width,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._height   = height
        self._width    = width

        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)

        x, y   = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed  = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # x     : (N, L, input_features)
        # embed : (1, height * width, embed_features)
        #       = (1, L, embed_features)
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1, L, embed_features)
        #      -> (N, L, embed_features)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features)
        result = torch.cat([embed, x], dim = 2)

        # (N, L, features)
        return self.output(result)

class PixelwiseViT(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.trans_input_oct = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, rezero
        )

        self.trans_output = nn.Linear(features, image_shape[0])
        self.trans_output_oct = nn.Linear(features, image_shape[0])
        

        self.oct_fc = nn.Conv2d(512, 384, 1)

    def forward(self, x, sparse_oct_tokens, ffa):
        # x : (N, C, H, W)
        # print(x.shape)
        # itokens : (N, C, H * W)

        # print(sparse_oct_features.shape)
        # print(x.shape)
        # x = sparse_oct_features + x
        itokens = x.view(*x.shape[:2], -1)
        itokens = itokens.permute((0, 2, 1))
        y = self.trans_input(itokens)

        if ffa is not None:
            ffa_tokens = ffa.view(*ffa.shape[:2], -1)
            ffa_tokens = ffa_tokens.permute((0, 2, 1))
            ffa_y = self.trans_input(ffa_tokens)
        else:
            ffa_y = None
        # itokens : (N, C, H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        # y : (N, L, features)
        # print(itokens.shape)

        if sparse_oct_tokens is not None:
            sparse_oct_tokens = self.oct_fc(sparse_oct_tokens)    
            sparse_oct_tokens = rearrange(sparse_oct_tokens, 'b c h w -> b (h w) c')
            # print(sparse_oct_features.shape)
            sparse_oct_tokens = self.trans_input_oct(sparse_oct_tokens)
        
        
        y, sparse_oct_tokens, y_cross, contrastive_loss = self.encoder(y, sparse_oct_tokens, ffa_y)


        otokens = self.trans_output(y).permute((0, 2, 1))
        # otokens : (N, L, C) -> (N, C, L)
        # result : (N, C, H, W)
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])
        if sparse_oct_tokens is not None:
            sparse_oct_tokens = self.trans_output_oct(sparse_oct_tokens).permute((0, 2, 1))
            oct_results = sparse_oct_tokens.view(*sparse_oct_tokens.shape[:2], *self.image_shape[1:])
        else:
            oct_results = None
        
        return result, oct_results, contrastive_loss



class UnetBasicBlock(nn.Module):

    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
            get_activ_layer(activ),

            get_norm_layer(norm, mid_features),
            nn.Conv2d(
                mid_features, out_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x)

class UNetEncBlock(nn.Module):

    def __init__(
        self, features, activ, norm, downsample, input_shape, **kwargs
    ):
        super().__init__(**kwargs)

        self.downsample, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = UnetBasicBlock(C, features, activ, norm)

        self.output_shape = (output_features, H//2, W//2)

    def get_output_shape(self):
        return self.output_shape

    def forward(self, x):
        # print('x: ', x.shape)
        r = self.block(x)
        # print('r: ', r.shape)
        y = self.downsample(r)
        return (y, r)

class UNetDecBlock(nn.Module):

    def __init__(
        self, output_shape, activ, norm, upsample, input_shape,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_shape[0]
        )

        self.block = UnetBasicBlock(
            2 * input_features, output_shape[0], activ, norm,
            mid_features = max(input_features, input_shape[0])
        )

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x, r):
        # x : (N, C, H_in, W_in)
        # r : (N, C, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )



class DecBlock(nn.Module):

    def __init__(
        self, output_shape, activ, norm, upsample, input_shape,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_shape[0]
        )

        self.block = UnetBasicBlock(
            input_features, output_shape[0], activ, norm,
            mid_features = max(input_features, input_shape[0])
        )

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x):
        # x : (N, C, H_in, W_in)
        # r : (N, C, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # result : (N, C_out, H_out, W_out)
        return self.block(x)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )


class UNetBlock(nn.Module):

    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.get_output_shape()
        self.inner_module = None

        self.deconv = UNetDecBlock(
            image_shape, activ, norm, upsample, self.inner_shape, rezero
        )
        self.deconv_oct = DecBlock(
            image_shape, activ, norm, upsample, self.inner_shape, rezero
        )
    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x, sparse_oct_features, ffa=None):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)
        if ffa is not None:
            (ffa_y, _) = self.conv(ffa)
        else:
            ffa_y = None
        y, oct_tokens, contrastive_loss = self.inner_module(y, sparse_oct_features, ffa_y)
        
        # y : (N, C_inner, H_inner, W_inner)

        # y : (N, C, H, W)
        y = self.deconv(y, r)
        if oct_tokens is None:
            return y, None
        oct_results = self.deconv_oct(oct_tokens)
        return y, oct_results, contrastive_loss

class UNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        out_ch, rezero = True, **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self.out_ch = out_ch

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = UNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                rezero
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.unet = unet_layers[0]

    def _construct_input_layer(self, activ):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self):
        self.layer_output = nn.Conv2d(
            self.features_list[0], self.out_ch, kernel_size = 1
        )
        self.layer_output_oct = nn.Conv2d(
            self.features_list[0], self.out_ch, kernel_size = 1
        )
    def get_innermost_block(self):
        result = self.unet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x, sparse_oct_features, ffa=None):
        # x : (N, C, H, W)

        y = self.layer_input(x)     # [4, 48, 512, 512]
        
        ffa_y = self.layer_input(x) if ffa is not None else None
        y, oct_results, contrastive_loss = self.unet(y, sparse_oct_features, ffa_y)
      
        # print(y.shape)
        y = self.layer_output(y)    # [4, 48, 512, 512]
        if oct_results is not None:
            oct_results = self.layer_output_oct(oct_results)
        return y, oct_results, contrastive_loss


import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np



class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(2, 0, 1)    # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # print(x.shape, self.positional_embedding.shape)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[1:, ...]



class OCTEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.feat_shape = [512, 64, 64] 
        self.feat_shape = [512, 32, 32] 
        self.encoder = self.load_oct_encoder()
        self.encoder.requires_grad_(False)
        self.attn_pool = AttentionPool2d(self.feat_shape[2], 512, 8)
        self.attn_pool.requires_grad_(False)

        ckpt_path = './data/OCTEncoder.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        message = self.load_state_dict({k:v for k, v in ckpt.items()},strict=False)
        print(message)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        width = self.feat_shape[0]
        scale = width ** -0.5
        self.proj_oct = nn.Parameter(scale * torch.randn(width, width))
        self.proj_ffa = nn.Parameter(scale * torch.randn(width, width))
        self.criterion = nn.CrossEntropyLoss()

        self.sparse_oct_map = nn.Parameter(torch.zeros(1, *self.feat_shape), True)#.to(oct_features)


    def forward(self, oct, line_cords, ffa=None):
        # sparse_oct_features = self.get_oct_sparse_feats(oct, line_cords)
        # if ffa is not None:
        #     oct_feats, ffa_feats, ground_truths = self.get_pair_feats(ffa, sparse_oct_features, line_cords)
        #     cont_loss = self.contrastive(oct_feats, ffa_feats, ground_truths)
        #     return sparse_oct_features, cont_loss
        # return sparse_oct_features

        sparse_oct_features = self.get_oct_sparse_feats(oct, line_cords)
        if ffa is not None:
            if ffa.shape[1] == 1:
                ffa = ffa.repeat(1, 3, 1, 1)
            ffa_feats = self.encoder(ffa).reshape(-1, *self.feat_shape)
            # b, c, h, w = ffa_feats.shape
            ffa_feats = rearrange(ffa_feats, 'b c h w -> b (h w) c')
            return sparse_oct_features, ffa_feats
        return sparse_oct_features
    def get_oct_sparse_feats(self, oct, line_cords):
        batch, slice_num = oct.shape[:2]
        oct = rearrange(oct, 'b s c h w -> (b s) c h w')
        
        oct_features = self.encoder(oct).reshape(-1, *self.feat_shape)
        oct_features = oct_features.mean(dim=2, keepdim=True)
        # oct_features = rearrange(oct_features, '(b s) c h w -> b s c h w', b=batch)

        oct_features = self.attn_pool(oct_features)
        # print(oct_features.shape)
        oct_features = rearrange(oct_features, 'w (b s) c-> b s c 1 w', b=batch)

        
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
    
    def get_pair_feats(self, ffa, sparse_oct_features, line_cords):
        ## 获得 OCT 和 FFA 配对的特征
        if ffa.shape[1] == 1:
            ffa = ffa.repeat(1, 3, 1, 1)
        ffa_feats = self.encoder(ffa).reshape(-1, *self.feat_shape)
        b, c, h, w = ffa_feats.shape
        ffa_feats = rearrange(ffa_feats, 'b c h w -> b (h w) c')
        
        # sparse_oct_feats = rearrange(sparse_oct_features, 'b c h w -> (b h w) c')
        batch_line_cords = rearrange(line_cords, 'b s t c -> b (s t) c')
        valid_oct_tokens_lst, ground_truth_lst = [], []

        for sparse_oct_feat, line_cords in zip(sparse_oct_features, batch_line_cords):
            row, col = line_cords[..., 0], line_cords[..., 1]
            ## 限制范围
            valid_mask = torch.logical_and(torch.logical_and(row < self.feat_shape[1], row >= 0), 
                                            torch.logical_and(col < self.feat_shape[2], col >= 0))

            row = row[valid_mask]
            col = col[valid_mask]
            token_idxs = row * w + col
            sparse_oct_feat = rearrange(sparse_oct_feat, 'c h w -> (h w) c')
            
            valid_oct_token = sparse_oct_feat[token_idxs, :]
            # print(valid_oct_token.shape)
            num_tokens = len(valid_oct_token)
            ground_truth = torch.zeros(num_tokens, h*w)
            ground_truth[torch.arange(num_tokens), token_idxs] = 1
            valid_oct_tokens_lst.append(valid_oct_token)
            ground_truth_lst.append(ground_truth)

        oct_feats = [torch.concat(valid_oct_tokens_lst[i:i+1] + valid_oct_tokens_lst[:i] + valid_oct_tokens_lst[i+1:])
                    for i in range(len(valid_oct_tokens_lst))]
        oct_feats = torch.stack(oct_feats, dim=0)

        num_valid_tokens = oct_feats.shape[1]

        ground_truth_lst = [torch.concat([gt, torch.zeros(num_valid_tokens-gt.shape[0], h*w)]) for gt in ground_truth_lst]
        ground_truths = torch.stack(ground_truth_lst, dim=0).to(sparse_oct_features)


        oct_feats = oct_feats @ self.proj_oct
        ffa_feats = ffa_feats @ self.proj_ffa
        oct_feats = oct_feats / oct_feats.norm(dim=-1, keepdim=True)
        ffa_feats = ffa_feats / ffa_feats.norm(dim=-1, keepdim=True)

        
        return oct_feats, ffa_feats, ground_truths
        # self.get_pair_feats(sparse_oct_features, ffa_feats, batch_line_cords)

    def contrastive(self, feat_1, feat_2, ground_truth):
        # feat_1 : (N, L, C)
        # feat_2 : (N, T, C)
        # ground_truth : (N, L, T)
        if feat_1.shape[1] == 0:
            return torch.zeros([]).to(feat_1)
        logits_per_1 = einsum('nlc,ntc->nlt', feat_1, feat_2) * self.logit_scale.exp()
        logits_per_2 = logits_per_1.transpose(1, 2)
            
        loss = (
            self.criterion(logits_per_1, ground_truth)
            +  self.criterion(logits_per_2, ground_truth.transpose(1, 2))
        ) / 2
    
        return loss


    def load_oct_encoder(self):
        from torchvision.models import resnet34, resnet18
        model = resnet34()#.to(device)
        # model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # model.layer4 = nn.Identity()
        model.avgpool = nn.Identity() # nn.AdaptiveAvgPool2d((1, 32))
        model.fc = nn.Identity()


        return model
    

class ViTUNetGenerator(nn.Module):

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, unet_features_list, unet_activ, unet_norm,
        unet_downsample = 'conv',
        unet_upsample   = 'upsample-conv',
        unet_rezero     = False,
        rezero          = True,
        activ_output    = None,
        out_ch          = None,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.image_shape = image_shape
        out_ch = image_shape[0] if out_ch is None else out_ch

        self.net = UNet(
            unet_features_list, unet_activ, unet_norm, image_shape,
            unet_downsample, unet_upsample, out_ch, unet_rezero
        )

        bottleneck = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, norm,
            image_shape = self.net.get_inner_shape(),
            rezero      = rezero
        )

        self.net.set_bottleneck(bottleneck)
        self.oct_encoder = OCTEncoder()
        self.output = get_activ_layer(activ_output)


        self.feat_shape = [512, 32, 32] 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        width = self.feat_shape[0]
        scale = width ** -0.5
        self.proj_oct = nn.Parameter(scale * torch.randn(384, width))
        self.proj_ffa = nn.Parameter(scale * torch.randn(width, width))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, oct_images, line_cords, ffa=None):
        # x : (N, C, H, W)
        # if ffa is None:
        #     sparse_oct_features = self.oct_encoder(oct_images, line_cords)
        #     result = self.net(x, sparse_oct_features)
        #     return self.output(result)
        
        # sparse_oct_features, cont_loss = self.oct_encoder(oct_images, line_cords, ffa)
        # result = self.net(x, sparse_oct_features)
        # return self.output(result), cont_loss
        sparse_oct_features = self.oct_encoder(oct_images, line_cords)

            # sparse_oct_features = None
        result, oct_results, contrastive_loss = self.net(x, sparse_oct_features, ffa)
            # print(result.shape, oct_results.shape)
        return self.output(result), contrastive_loss#self.output(oct_results)
 

        # oct_feats, ffa_feats = self.oct_encoder(oct_images, line_cords, ffa)
        # result, oct_feats = self.net(x, oct_feats)

        # oct_feats = oct_feats @ self.proj_oct
        # ffa_feats = ffa_feats @ self.proj_ffa
        # oct_feats = oct_feats / oct_feats.norm(dim=-1, keepdim=True)
        # ffa_feats = ffa_feats / ffa_feats.norm(dim=-1, keepdim=True)

        # # print(oct_feats.shape, ffa_feats.shape)
        # cont_loss = self.contrastive(oct_feats, ffa_feats)
        # return self.output(result), cont_loss



        # sparse_oct_features = None
        # result, _ = self.net(x, sparse_oct_features)
        # # if ffa is None:
        # #     return self.output(result)
        # return self.output(result), None#, torch.zeros([]).to(x)
        
    def contrastive(self, feat_1, feat_2):
        # feat_1 : (N, L, C)
        # feat_2 : (N, T, C)
        # ground_truth : (N, L, T)
        b, l, _ = feat_1.shape
        ground_truth = torch.arange(l).repeat(b, 1).long().to(feat_1.device)
        # if feat_1.shape[1] == 0:
        #     return torch.zeros([]).to(feat_1)
        logits_per_1 = einsum('nlc,ntc->nlt', feat_1, feat_2) * self.logit_scale.exp()
        logits_per_2 = logits_per_1.transpose(1, 2)
            
        loss = (
            self.criterion(logits_per_1, ground_truth)
            +  self.criterion(logits_per_2, ground_truth)
        ) / 2
    
        return loss



num_classes = 200


#@title 2. DEFINISI ARSITEKTUR MODEL (Modifikasi)
#region 2. DEFINISI ARSITEKTUR MODEL (Modifikasi)

""" Dna blocks used for Modification

A PyTorch impl of Dna blocks

Paper: Modification: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()

        channels_per_group = c // self.groups

        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        out = x.view(b, -1, h, w)
        return out

class DyReLU(nn.Module):
    def __init__(self, num_func=2, use_bias=False, scale=2., serelu=False):
        """
        num_func: -1: none
                   0: relu
                   1: SE
                   2: dy-relu
        """
        super(DyReLU, self).__init__()

        assert(num_func>=-1 and num_func<=2)
        self.num_func = num_func
        self.scale = scale

        serelu = serelu and num_func == 1
        self.act = nn.ReLU6(inplace=True) if num_func == 0 or serelu else nn.Sequential()

    def forward(self, x):
        if isinstance(x, tuple):
            out, a = x
        else:
            out = x

        out = self.act(out)


        if self.num_func == 1:    # SE
            a = a * self.scale
            out = out * a
        elif self.num_func == 2:  # DY-ReLU
            _, C, _, _ = a.shape
            a1, a2 = torch.split(a, [C//2, C//2], dim=1)
            a1 = (a1 - 0.5) * self.scale + 1.0 #  0.0 -- 2.0
            a2 = (a2 - 0.5) * self.scale       # -1.0 -- 1.0
            out = torch.max(out*a1, out*a2)
            
        return out

class HyperFunc(nn.Module):
    def __init__(self, token_dim, oup, sel_token_id=0, reduction_ratio=4):
        super(HyperFunc, self).__init__()

        self.sel_token_id = sel_token_id
        squeeze_dim = token_dim // reduction_ratio
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, squeeze_dim),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_dim, oup),
            h_sigmoid()
        )


    def forward(self, x):
        if isinstance(x, tuple):
            x, attn = x

        if self.sel_token_id == -1:
            hp = self.hyper(x).permute(1, 2, 0)         # bs x hyper_dim x T

            bs, T, H, W = attn.shape
            attn = attn.view(bs, T, H*W)
            hp = torch.matmul(hp, attn)                  # bs x hyper_dim x HW
            h = hp.view(bs, -1, H, W)
        else:
            t = x[self.sel_token_id]
            h = self.hyper(t)
            h = torch.unsqueeze(torch.unsqueeze(h, 2), 3)
        return h

class MaxDepthConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(MaxDepthConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, oup, (3,1), stride, (1, 0), bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, oup, (1,3), stride, (0, 1), bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        out = torch.max(y1, y2)
        return out

class Local2GlobalAttn(nn.Module):
    def __init__(
        self,
        inp,
        token_dim=128,
        token_num=6,
        inp_res=0,
        norm_pos='post',
        drop_path_rate=0.
    ):
        super(Local2GlobalAttn, self).__init__()

        num_heads = 2
        self.scale = (inp // num_heads) ** -0.5

        self.q = nn.Linear(token_dim, inp)
        self.proj = nn.Linear(inp, token_dim)
        
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)


    def forward(self, x):
        features, tokens = x
        bs, C, _, _ = features.shape

        t = self.q(tokens).permute(1, 0, 2) # from T x bs x Ct to bs x T x Ct
        k = features.view(bs, C, -1)        # bs x C x HW
        attn = (t @ k) * self.scale

        attn_out = attn.softmax(dim=-1)             # bs x T x HW
        attn_out = (attn_out @ k.permute(0, 2, 1))  # bs x T x C
                                                    # note here: k=v without transform
        t = self.proj(attn_out.permute(1, 0, 2))    #T x bs x C

        tokens = tokens + self.drop_path(t)
        tokens = self.layer_norm(tokens)

        return tokens

class Local2Global(nn.Module):
    def __init__(
            self,
            inp,
            block_type='mlp',
            token_dim=128,
            token_num=6,
            inp_res=0,
            attn_num_heads=2,
            use_dynamic=False,
            norm_pos='post',
            drop_path_rate=0.,
            remove_proj_local=True,
        ):
        super(Local2Global, self).__init__()
        print(f'L2G: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.num_heads = attn_num_heads
        self.token_num = token_num 
        self.norm_pos = norm_pos
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )


        if 'mlp' in block_type:
            self.mlp = nn.Linear(inp_res, token_num)

        if 'attn' in block_type:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.q = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(inp, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)
        
        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.k = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.v = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            

    def forward(self, x):
        features, tokens = x # features: bs x C x H x W
                             #   tokens: T x bs x Ct

        bs, C, H, W = features.shape
        T, _, _ = tokens.shape
        attn = None

        if 'mlp' in self.block:
            t_sum = self.mlp(features.view(bs, C, -1)).permute(2, 0, 1) # T x bs x C            

        if 'attn' in self.block:
            t = self.q(tokens).view(T, bs, self.num_heads, -1).permute(1, 2, 0, 3)  # from T x bs x Ct to bs x N x T x Ct/N
            if self.remove_proj_local:
                k = features.view(bs, self.num_heads, -1, H*W)                          # bs x N x C/N x HW
                attn = (t @ k) * self.scale                                             # bs x N x T x HW
    
                attn_out = attn.softmax(dim=-1)                 # bs x N x T x HW
                attn_out = (attn_out @ k.transpose(-1, -2))     # bs x N x T x C/N (k: bs x N x C/N x HW)
                                                                # note here: k=v without transform
            else:
                k = self.k(features).view(bs, self.num_heads, -1, H*W)                          # bs x N x C/N x HW
                v = self.v(features).view(bs, self.num_heads, -1, H*W)                          # bs x N x C/N x HW 
                attn = (t @ k) * self.scale                                             # bs x N x T x HW
    
                attn_out = attn.softmax(dim=-1)                 # bs x N x T x HW
                attn_out = (attn_out @ v.transpose(-1, -2))     # bs x N x T x C/N (k: bs x N x C/N x HW)
                                                                # note here: k=v without transform
 
            t_a = attn_out.permute(2, 0, 1, 3)              # T x bs x N x C/N
            t_a = t_a.reshape(T, bs, -1)

            if 'mlp' in self.block:
                t_sum = t_sum + t_a
            else:
                t_sum = t_a

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.proj(t_sum)
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if attn is not None:
            bs, Nh, Ca, HW = attn.shape
            attn = attn.view(bs, Nh, Ca, H, W)

        return tokens, attn

#---------------------------------------------------------------------------------------------------------------

class RelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, n_head, d_k, d_v, out_channels, attn_dropout=0.1, ff_dropout=0.1, attn_bias=False):
        super().__init__()
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(in_channels, n_head * d_k, bias=attn_bias)
        self.K = nn.Linear(in_channels, n_head * d_k, bias=attn_bias)
        self.V = nn.Linear(in_channels, n_head * d_v, bias=attn_bias)
        self.ff = nn.Linear(n_head * d_v, out_channels)
        self.attn_dropout = nn.Dropout2d(attn_dropout)
        self.ff_dropout = nn.Dropout(ff_dropout)
        self.relative_bias = nn.Parameter(
            torch.randn(n_head, ((inp_h << 1) - 1) * ((inp_w << 1) - 1)),
            requires_grad=True
        )
        self.register_buffer('relative_indices', self._get_relative_indices(inp_h, inp_w))

    def _get_relative_indices(self, height, width):
        ticks_y, ticks_x = torch.arange(height), torch.arange(width)
        grid_y, grid_x = torch.meshgrid(ticks_y, ticks_x)
        area = height * width
        out = torch.empty(area, area).fill_(float('nan'))
        for idx_y in range(height):
            for idx_x in range(width):
                rel_indices_y = grid_y - idx_y + height
                rel_indices_x = grid_x - idx_x + width
                flatten_indices = (rel_indices_y * width + rel_indices_x).view(-1)
                out[idx_y * width + idx_x] = flatten_indices
        assert not out.isnan().any(), '`relative_indices` have blank indices'
        assert (out >= 0).all(), '`relative_indices` have negative indices'
        return out.long()

    def _interpolate_relative_bias(self, height, width):
        relative_bias = self.relative_bias.view(1, self.n_head, (self.inp_h << 1) - 1, -1)
        relative_bias = F.interpolate(relative_bias, size=((height << 1) - 1, (width << 1) - 1), mode='bilinear', align_corners=True)
        return relative_bias.view(self.n_head, -1)

    def update_relative_bias_and_indices(self, height, width):
        self.relative_indices = self._get_relative_indices(height, width)
        self.relative_bias = self._interpolate_relative_bias(height, width)

    def forward(self, x):
        b, c, H, W, h = *x.shape, self.n_head
    
        len_x = H * W
        x = x.view(b, c, len_x).transpose(-1, -2)
        q = self.Q(x).view(b, len_x, self.n_head, self.d_k).transpose(1, 2)
        k = self.K(x).view(b, len_x, self.n_head, self.d_k).transpose(1, 2)
        v = self.V(x).view(b, len_x, self.n_head, self.d_v).transpose(1, 2)

        if H == self.inp_h and W == self.inp_w:
            relative_indices = self.relative_indices
            relative_bias = self.relative_bias
        else:
            relative_indices = self._get_relative_indices(H, W).to(x.device)
            relative_bias = self._interpolate_relative_bias(H, W)

        relative_indices = relative_indices.view(1, 1, *relative_indices.size()).expand(b, h, -1, -1)
        relative_bias = relative_bias.view(1, relative_bias.size(0), 1, relative_bias.size(1)).expand(b, -1, len_x, -1)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        similarity = torch.matmul(q, k.transpose(-1, -2)) + relative_biases
        similarity = similarity.softmax(dim=-1)
        similarity = self.attn_dropout(similarity)
        
        out = torch.matmul(similarity, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.n_head * self.d_v)
        out = self.ff(out)
        out = self.ff_dropout(out)
        out = out.transpose(-1, -2).view(b, -1, H, W)
        return out


class GlobalBlock(nn.Module):
    def __init__(
        self,
        block_type='mlp',
        token_dim=128,
        token_h=2,
        token_w=3,
        mlp_token_exp=4,
        attn_num_heads=4,
        use_dynamic=False,
        use_ffn=False,
        norm_pos='post',
        drop_path_rate=0.,
        token_num=6  # Add token_num as a parameter
    ):
        super(GlobalBlock, self).__init__()

        print(f'G2G: {attn_num_heads} heads')

        self.block = block_type
        self.num_heads = attn_num_heads
        self.token_h = token_h
        self.token_w = token_w
        self.norm_pos = norm_pos
        self.use_dynamic = use_dynamic
        self.use_ffn = use_ffn
        self.ffn_exp = 2
        self.token_num = token_num  # Assign token_num to self.token_num

        if self.use_ffn:
            print('use ffn')
            self.ffn = nn.Sequential(
                nn.Linear(token_dim, token_dim * self.ffn_exp),
                nn.GELU(),
                nn.Linear(token_dim * self.ffn_exp, token_dim)
            )
            self.ffn_norm = nn.LayerNorm(token_dim)

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, token_dim),
                h_sigmoid(),
            )

        if 'mlp' in self.block:
            self.token_mlp = nn.Sequential(
                nn.Linear(token_h * token_w, token_h * token_w * mlp_token_exp),
                nn.GELU(),
                nn.Linear(token_h * token_w * mlp_token_exp, token_h * token_w),
            )

        if 'attn' in self.block:
            self.relative_attn = RelativeAttention(
                inp_h=token_h,
                inp_w=token_w,
                in_channels=token_dim,
                n_head=attn_num_heads,
                d_k=token_dim // attn_num_heads,
                d_v=token_dim // attn_num_heads,
                out_channels=token_dim,
                attn_dropout=0.1,
                ff_dropout=0.1,
                attn_bias=False
            )

        self.channel_mlp = nn.Linear(token_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        tokens = x
        T, bs, C = tokens.shape

        if 'mlp' in self.block:
            t = self.token_mlp(tokens.permute(1, 2, 0))  # bs x channel x token_num
            t_sum = t.permute(2, 0, 1)                    # token_num x bs x channel

        if 'attn' in self.block:
            tokens_reshaped = tokens.permute(1, 2, 0).view(bs, C, self.token_h, self.token_w)  # bs x C x H x W
            attn_out = self.relative_attn(tokens_reshaped)  # Apply RelativeAttention
            attn_out = attn_out.reshape(T, bs, C)  # Reshape back to (T, bs, C)
            t_sum = attn_out + t_sum if 'mlp' in self.block else attn_out

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            t_sum = t_sum * alp

        t_sum = self.channel_mlp(t_sum)  # token_num x bs x channel
        tokens = tokens + self.drop_path(t_sum)
        tokens = self.layer_norm(tokens)

        if self.use_ffn:
            t_ffn = self.ffn(tokens)
            tokens = tokens + t_ffn
            tokens = self.ffn_norm(tokens)

        return tokens


#---------------------------------------------------------------------------------------------------------------

class Global2Local(nn.Module):
    def __init__(
        self,
        inp,
        inp_res=0,
        block_type='mlp',
        token_dim=128,
        token_num=6,
        attn_num_heads=2,
        use_dynamic=False,
        drop_path_rate=0.,
        remove_proj_local=True, 
    ):
        super(Global2Local, self).__init__()
        print(f'G2L: {attn_num_heads} heads, inp: {inp}, token: {token_dim}')

        self.token_num = token_num
        self.num_heads = attn_num_heads
        self.block = block_type
        self.use_dynamic = use_dynamic

        if self.use_dynamic:
            self.alpha_scale = 2.0
            self.alpha = nn.Sequential(
                nn.Linear(token_dim, inp),
                h_sigmoid(),
            )


        if 'mlp' in self.block:
            self.mlp = nn.Linear(token_num, inp_res)

        if 'attn' in self.block:
            self.scale = (inp // attn_num_heads) ** -0.5
            self.k = nn.Linear(token_dim, inp)

        self.proj = nn.Linear(token_dim, inp)
        self.drop_path = DropPath(drop_path_rate)

        self.remove_proj_local = remove_proj_local
        if self.remove_proj_local == False:
            self.q = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.fuse = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
 
    def forward(self, x):
        out, tokens = x

        if self.use_dynamic:
            alp = self.alpha(tokens) * self.alpha_scale
            v = self.proj(tokens)
            v = (v * alp).permute(1, 2, 0)
        else:
            v = self.proj(tokens).permute(1, 2, 0)  # from T x bs x Ct -> T x bs x C -> bs x C x T 

        bs, C, H, W = out.shape
        if 'mlp' in self.block:
            g_sum = self.mlp(v).view(bs, C, H, W)       # bs x C x T -> bs x C x H x W

        if 'attn' in self.block:
            if self.remove_proj_local:
                q = out.view(bs, self.num_heads, -1, H*W).transpose(-1, -2)                         # bs x N x HW x C/N
            else:
                q = self.q(out).view(bs, self.num_heads, -1, H*W).transpose(-1, -2)                         # bs x N x HW x C/N

            k = self.k(tokens).permute(1, 2, 0).view(bs, self.num_heads, -1, self.token_num)    # from T x bs x Ct -> bs x C x T -> bs x N x C/N x T
            attn = (q @ k) * self.scale                         # bs x N x HW x T

            attn_out = attn.softmax(dim=-1)                     # bs x N x HW x T
            
            vh = v.view(bs, self.num_heads, -1, self.token_num) # bs x N x C/N x T
            attn_out = (attn_out @ vh.transpose(-1, -2))        # bs x N x HW x C/N
                                                                # note here k != v
            g_a = attn_out.transpose(-1, -2).reshape(bs, C, H, W)   # bs x C x HW

            if self.remove_proj_local == False:
                g_a = self.fuse(g_a)            

            g_sum = g_sum + g_a if 'mlp' in self.block else g_a

        out = out + self.drop_path(g_sum)

        return out

##########################################################################################################
# Dna Blocks
##########################################################################################################
class DnaBlock3(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        exp_ratios, #(e1, e2)
        kernel_size=(3,3),
        dw_conv='dw',
        group_num=1,
        se_flag=[2,0,2,0],
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        inp_res=49,
        gbr_type='mlp',
        gbr_dynamic=[False, False, False],
        gbr_ffn=False,
        gbr_before_skip=False,
        mlp_token_exp=4,
        norm_pos='post',
        drop_path_rate=0.,
        cnn_drop_path_rate=0.,
        attn_num_heads=2,
        remove_proj_local=True,
    ):
        super(DnaBlock3, self).__init__()

        print(f'block: {inp_res}, cnn-drop {cnn_drop_path_rate:.4f}, mlp-drop {drop_path_rate:.4f}')
        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id

        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False
        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            if dw_conv == 'dw':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp*e1, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp*e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    # pw-linear
                    nn.Conv2d(inp*e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
            elif dw_conv == 'sepdw':
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp*e1//2, (3,1), (stride,1), (1,0), groups=inp, bias=False),
                    nn.BatchNorm2d(inp*e1//2),
                    nn.Conv2d(inp*e1//2, inp*e1, (1,3), (1, stride), (0,1), groups=inp*e1//2, bias=False),
                    nn.BatchNorm2d(inp*e1),
                    nn.ReLU6(inplace=True),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                    # pw-linear
                    nn.Conv2d(inp*e1, oup, 1, 1, 0, groups=group_num, bias=False),
                    nn.BatchNorm2d(oup),
                )
 
        else:
            # conv (dw->pw->dw->pw)
            self.se_flag = se_flag
            hidden_dim1 = round(inp * e1)
            hidden_dim2 = round(oup * e2)

            if dw_conv == 'dw':
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1, k1, stride, k1//2, groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'maxdw':
                self.conv1 = nn.Sequential(
                    MaxDepthConv(inp, hidden_dim1, stride),
                    ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'sepdw':
                self.conv1 = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim1//2, (3,1), (stride,1), (1,0), groups=inp, bias=False),
                    nn.BatchNorm2d(hidden_dim1//2),
                    nn.Conv2d(hidden_dim1//2, hidden_dim1, (1,3), (1, stride), (0,1), groups=hidden_dim1//2, bias=False),
                    nn.BatchNorm2d(hidden_dim1),
                    ChannelShuffle(inp) if group_num > 1 else nn.Sequential()
                )
 
            num_func = se_flag[0] 
            self.act1 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper1 = HyperFunc(
                token_dim, 
                hidden_dim1 * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[0] > 0 else nn.Sequential()
                

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
            num_func = -1
            #num_func = 1 if se_flag[1] == 1 else -1 
            self.act2 = DyReLU(num_func=num_func, scale=2.)


            if dw_conv == 'dw':
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2, k2, 1, k2//2, groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential()
                )
            elif dw_conv == 'maxdw':
                self.conv3 = nn.Sequential(
                    MaxDepthConv(oup, hidden_dim2, 1),
                )
            elif dw_conv == 'sepdw':
                self.conv3 = nn.Sequential(
                    nn.Conv2d(oup, hidden_dim2//2, (3,1), (1,1), (1,0), groups=oup, bias=False),
                    nn.BatchNorm2d(hidden_dim2//2),
                    nn.Conv2d(hidden_dim2//2, hidden_dim2, (1,3), (1, 1), (0,1), groups=hidden_dim2//2, bias=False),
                    nn.BatchNorm2d(hidden_dim2),
                    ChannelShuffle(oup) if group_num > 1 else nn.Sequential()
                )
           
            num_func = se_flag[2]
            self.act3 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper3 = HyperFunc(
                token_dim, 
                hidden_dim2 * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[2] > 0 else nn.Sequential()
 

            self.conv4 = nn.Sequential(
                nn.Conv2d(hidden_dim2, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup)
            )
            num_func = 1 if se_flag[3] == 1 else -1 
            self.act4 = DyReLU(num_func=num_func, scale=2.)
            self.hyper4 = HyperFunc(
                token_dim, 
                oup * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[3] > 0 else nn.Sequential()
 

            self.drop_path = DropPath(cnn_drop_path_rate)

            # l2g, gb, g2l
            self.local_global = Local2Global(
                inp,
                block_type = gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic = gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic = gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate
            )
 
            oup_res = inp_res // (stride * stride)

            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic = gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
        else:
            # step 1: local to global
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            # step 2: conv1 + conv2
            out = self.conv1(features)

            # process attn: mean, downsample if stride > 1, and softmax
            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1) # bs x T x H x W
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            out = self.act2(out)

            # step 4: conv3 + conv 4
            out_cp = out
            out = self.conv3(out)
            if self.se_flag[2] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            out = self.conv4(out)
            if self.se_flag[3] > 0:
                hp = self.hyper4((tokens, attn))
                out = self.act4((out, hp))
            else:
                out = self.act4(out)

            out = self.drop_path(out) + out_cp

            # step 3: global to local
            out = self.global_local((out, tokens))

        if self.identity:
            out = out + features

        return (out, tokens)


class DnaBlock(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        exp_ratios, #(e1, e2)
        kernel_size=(3,3),
        dw_conv='dw',
        group_num=1,
        se_flag=[2,0,2,0],
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        inp_res=49,
        gbr_type='mlp',
        gbr_dynamic=[False, False, False],
        gbr_ffn=False,
        gbr_before_skip=False,
        mlp_token_exp=4,
        norm_pos='post',
        drop_path_rate=0.,
        cnn_drop_path_rate=0.,
        attn_num_heads=2,
        remove_proj_local=True,
    ):
        super(DnaBlock, self).__init__()

        print(f'block: {inp_res}, cnn-drop {cnn_drop_path_rate:.4f}, mlp-drop {drop_path_rate:.4f}')
        if isinstance(exp_ratios, tuple):
            e1, e2 = exp_ratios
        else:
            e1, e2 = exp_ratios, 4
        k1, k2 = kernel_size

        self.stride = stride
        self.hyper_token_id = hyper_token_id

        self.gbr_before_skip = gbr_before_skip
        self.identity = stride == 1 and inp == oup
        self.use_conv_alone = False
        if e1 == 1 or e2 == 0:
            self.use_conv_alone = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp*e1, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp*e1),
                nn.ReLU6(inplace=True),
                ChannelShuffle(inp) if group_num > 1 else nn.Sequential(),
                # pw-linear
                nn.Conv2d(inp*e1, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # conv (pw->dw->pw)
            self.se_flag = se_flag
            hidden_dim = round(inp * e1)

            self.conv1 = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(hidden_dim),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
            )

            num_func = se_flag[0] 
            self.act1 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper1 = HyperFunc(
                token_dim, 
                hidden_dim * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[0] > 0 else nn.Sequential()
                

            self.conv2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, k1, stride, k1//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            num_func = se_flag[2] # note here we used index 2 to be consistent with block2
            self.act2 = DyReLU(num_func=num_func, scale=2., serelu=True)
            self.hyper2 = HyperFunc(
                token_dim, 
                hidden_dim * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[2] > 0 else nn.Sequential()
 

            self.conv3 = nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=group_num, bias=False),
                nn.BatchNorm2d(oup),
                ChannelShuffle(group_num) if group_num > 1 else nn.Sequential()
            )
            num_func = 1 if se_flag[3] == 1 else -1 
            self.act3 = DyReLU(num_func=num_func, scale=2.)
            self.hyper3 = HyperFunc(
                token_dim, 
                oup * num_func, 
                sel_token_id=hyper_token_id, 
                reduction_ratio=hyper_reduction_ratio
            ) if se_flag[3] > 0 else nn.Sequential()
 

            self.drop_path = DropPath(cnn_drop_path_rate)

            # l2g, gb, g2l
            self.local_global = Local2Global(
                inp,
                block_type = gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                inp_res=inp_res,
                use_dynamic = gbr_dynamic[0],
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

            self.global_block = GlobalBlock(
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                mlp_token_exp=mlp_token_exp,
                use_dynamic = gbr_dynamic[1],
                use_ffn=gbr_ffn,
                norm_pos=norm_pos,
                drop_path_rate=drop_path_rate
            )
 
            oup_res = inp_res // (stride * stride)

            self.global_local = Global2Local(
                oup,
                oup_res,
                block_type=gbr_type,
                token_dim=token_dim,
                token_num=token_num,
                use_dynamic = gbr_dynamic[2],
                drop_path_rate=drop_path_rate,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,
            )

    def forward(self, x):
        features, tokens = x
        if self.use_conv_alone:
            out = self.conv(features)
            if self.identity:
                out = self.drop_path(out) + features

        else:
            # step 1: local to global
            tokens, attn = self.local_global((features, tokens))
            tokens = self.global_block(tokens)

            # step 2: conv1 + conv2 + conv3
            out = self.conv1(features)

            # process attn: mean, downsample if stride > 1, and softmax
            if self.hyper_token_id == -1:
                attn = attn.mean(dim=1) # bs x T x H x W
                if self.stride > 1:
                    _, _, H, W = out.shape
                    attn = F.adaptive_avg_pool2d(attn, (H, W))
                attn = torch.softmax(attn, dim=1)

            if self.se_flag[0] > 0:
                hp = self.hyper1((tokens, attn))
                out = self.act1((out, hp))
            else:
                out = self.act1(out)

            out = self.conv2(out)
            if self.se_flag[2] > 0:
                hp = self.hyper2((tokens, attn))
                out = self.act2((out, hp))
            else:
                out = self.act2(out)

            out = self.conv3(out)
            if self.se_flag[3] > 0:
                hp = self.hyper3((tokens, attn))
                out = self.act3((out, hp))
            else:
                out = self.act3(out)

            # step 3: global to local and skip
            if self.gbr_before_skip == True:
                out = self.global_local((out, tokens))
                if self.identity:
                    out = self.drop_path(out) + features
            else:
                if self.identity:
                    out = self.drop_path(out) + features
                out = self.global_local((out, tokens))

        return (out, tokens)

##########################################################################################################
# classifier
##########################################################################################################
class MergeClassifier(nn.Module):
    def __init__(
        self, inp, 
        oup=1280, 
        ch_exp=6, 
        num_classes=num_classes, 
        drop_rate=0., 
        drop_branch=[0.0, 0.0],
        group_num=1, 
        token_dim=128, 
        cls_token_num=1, 
        last_act='relu',
        hyper_token_id=0,
        hyper_reduction_ratio=4
    ):
        super(MergeClassifier, self).__init__()

        self.drop_branch=drop_branch
        self.cls_token_num = cls_token_num

        hidden_dim = inp * ch_exp
        self.conv = nn.Sequential(
            ChannelShuffle(group_num) if group_num > 1 else nn.Sequential(),
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=group_num, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        self.last_act = last_act
        num_func = 2 if last_act == 'dyrelu' else 0 
        self.act = DyReLU(num_func=num_func, scale=2.)
 
        self.hyper = HyperFunc(
            token_dim, 
            hidden_dim * num_func, 
            sel_token_id=hyper_token_id, 
            reduction_ratio=hyper_reduction_ratio
        ) if last_act == 'dyrelu' else nn.Sequential()
 
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )

        if cls_token_num > 0:
            cat_token_dim = token_dim * cls_token_num 
        elif cls_token_num == 0:
            cat_token_dim = token_dim
        else:
            cat_token_dim = 0

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + cat_token_dim, oup),
            nn.BatchNorm1d(oup),
            h_swish()
        )

        self.classifier = nn.Sequential(
           nn.Dropout(drop_rate),
           nn.Linear(oup, num_classes)
       )

    def forward(self, x):
        features, tokens = x

        x = self.conv(features)

        if self.last_act == 'dyrelu':
            hp = self.hyper(tokens)
            x = self.act((x, hp))
        else:
            x = self.act(x)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        ps = [x]
        
        if self.cls_token_num == 0:
            avg_token = torch.mean(F.relu6(tokens), dim=0)
            ps.append(avg_token)
        elif self.cls_token_num < 0:
            pass
        else:
            for i in range(self.cls_token_num):
                ps.append(tokens[i])

        # drop branch
        if self.training and self.drop_branch[0] + self.drop_branch[1] > 1e-8:
            rd = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
            keep_local = 1 - self.drop_branch[0]
            keep_global = 1 - self.drop_branch[1]
            rd_local = (keep_local + rd).floor_()
            rd_global = -((rd - keep_global).floor_())
            ps[0] = ps[0].div(keep_local) * rd_local
            ps[1] = ps[1].div(keep_global) * rd_global

        x = torch.cat(ps, dim=1)
        x = self.fc(x)

        x = self.classifier(x)
        return x


""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn


from timm.models.features import FeatureListNet, FeatureDictNet, FeatureHookNet
from timm.models.hub import has_hf_hub, download_cached_file, load_state_dict_from_hf
from torch.hub import load_state_dict_from_url
from timm.models.layers import Conv2dSame, Linear


_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_custom_pretrained(model, default_cfg=None, load_fn=None, progress=False, check_hash=False):
    r"""Loads a custom (read non .pth) weight file

    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.

    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        model: The instantiated model to load weights into
        default_cfg (dict): Default pretrained model cfg
        load_fn: An external stand alone fn that loads weights into provided model, otherwise a fn named
            'laod_pretrained' on the model will be called if it exists
        progress (bool, optional): whether or not to display a progress bar to stderr. Default: False
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file. Default: False
    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    if not pretrained_url:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    cached_file = download_cached_file(default_cfg['url'], check_hash=check_hash, progress=progress)

    if load_fn is not None:
        load_fn(model, cached_file)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(cached_file)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def load_pretrained(model, default_cfg=None, num_classes=num_classes, in_chans=3, filter_fn=None, strict=True, progress=False):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        default_cfg (Optional[Dict]): default configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint
        progress (bool): enable progress bar for weight download

    """
    default_cfg = default_cfg or getattr(model, 'default_cfg', None) or {}
    pretrained_url = default_cfg.get('url', None)
    hf_hub_id = default_cfg.get('hf_hub', None)
    if not pretrained_url and not hf_hub_id:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if hf_hub_id and has_hf_hub(necessary=not pretrained_url):
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({hf_hub_id})')
        state_dict = load_state_dict_from_hf(hf_hub_id)
    else:
        _logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        state_dict = load_state_dict_from_url(pretrained_url, progress=progress, map_location='cpu')
    if filter_fn is not None:
        # for backwards compat with filter fn that take one arg, try one first, the two
        try:
            state_dict = filter_fn(state_dict)
        except TypeError:
            state_dict = filter_fn(state_dict, model)

    input_convs = default_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = default_cfg.get('classifier', None)
    label_offset = default_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != default_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                del state_dict[classifier_name + '.weight']
                del state_dict[classifier_name + '.bias']
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            # FIXME extra checks to ensure this is actually the FC classifier layer and not a diff Linear layer?
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def overlay_external_default_cfg(default_cfg, kwargs):
    """ Overlay 'external_default_cfg' in kwargs on top of default_cfg arg.
    """
    external_default_cfg = kwargs.pop('external_default_cfg', None)
    if external_default_cfg:
        default_cfg.pop('url', None)  # url should come from external cfg
        default_cfg.pop('hf_hub', None)  # hf hub id should come from external cfg
        default_cfg.update(external_default_cfg)


def set_default_kwargs(kwargs, names, default_cfg):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # default_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    FIXME this sequence of overlay default_cfg, set default kwargs, filter kwargs
    could/should be replaced by an improved configuration mechanism

    Args:
        default_cfg: input default_cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Overlay default cfg values from `external_default_cfg` if it exists in kwargs
    overlay_external_default_cfg(default_cfg, kwargs)
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if default_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)
    set_default_kwargs(kwargs, names=default_kwarg_names, default_cfg=default_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        pretrained_custom_load: bool = False,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        default_cfg (dict): model's default pretrained/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault('architecture', variant)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg
    
    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', num_classes))
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model)
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict)

    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg
    
    return model


def model_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params based on ordering
        return [p for p in model.parameters()][:-2]
    else:
        return model.parameters()



""" Model Registry
Hacked together by / Copyright 2020 Ross Wightman
"""

import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_model_default_key', 'has_model_default_key', 'get_model_default_value', 'is_model_pretrained']

_module_to_models = defaultdict(set)  # dict of sets to check membership of model in module
_model_to_module = {}  # mapping of model names to module names
_model_entrypoints = {}  # mapping of model names to entrypoint fns
_model_has_pretrained = set()  # set of model names that have pretrained weight url present
_model_default_cfgs = dict()  # central repo for model default_cfgs


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False  # check if model has a pretrained url to allow filtering on this
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        # this will catch all models that have entrypoint matching cfg key, but miss any aliasing
        # entrypoints or non-matching combos
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', module='', pretrained=False, exclude_filters='', name_matches_cfg=False):
    """ Return list of available model names, sorted alphabetically

    Args:
        filter (str) - Wildcard filter string that works with fnmatch
        module (str) - Limit model selection to a specific sub-module (ie 'gen_efficientnet')
        pretrained (bool) - Include only models with pretrained weights if True
        exclude_filters (str or list[str]) - Wildcard filters to exclude models after including them with filter
        name_matches_cfg (bool) - Include only models w/ model_name matching default_cfg name (excludes some aliases)

    Example:
        model_list('gluon_resnet*') -- returns all models starting with 'gluon_resnet'
        model_list('*resnext*, 'resnet') -- returns all models with 'resnext' in 'resnet' module
    """
    if module:
        models = list(_module_to_models[module])
    else:
        models = _model_entrypoints.keys()
    if filter:
        models = fnmatch.filter(models, filter)  # include these models
    if exclude_filters:
        if not isinstance(exclude_filters, (tuple, list)):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf)  # exclude these models
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    if pretrained:
        models = _model_has_pretrained.intersection(models)
    if name_matches_cfg:
        models = set(_model_default_cfgs).intersection(models)
    return list(sorted(models, key=_natural_key))


def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def list_modules():
    """ Return list of module names that contain models / model entrypoints
    """
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    """Check if a model exists within a subset of modules
    Args:
        model_name (str) - name of model to check
        module_names (tuple, list, set) - names of modules to search in
    """
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def has_model_default_key(model_name, cfg_key):
    """ Query model default_cfgs for existence of a specific key.
    """
    if model_name in _model_default_cfgs and cfg_key in _model_default_cfgs[model_name]:
        return True
    return False


def is_model_default_key(model_name, cfg_key):
    """ Return truthy value for specified model default_cfg key, False if does not exist.
    """
    if model_name in _model_default_cfgs and _model_default_cfgs[model_name].get(cfg_key, False):
        return True
    return False


def get_model_default_value(model_name, cfg_key):
    """ Get a specific model default_cfg value by key. None if it doesn't exist.
    """
    if model_name in _model_default_cfgs:
        return _model_default_cfgs[model_name].get(cfg_key, None)
    else:
        return None


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained


"""Modification V1

A PyTorch impl of Modification-V1.
 
Paper: Modification: Bridging MobileNet and Transformer (CVPR 2022)
       https://arxiv.org/abs/2108.05895

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

__all__ = ['Modification']
  
def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': num_classes, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = {
    'default': _cfg(url=''),
}

class Modification(nn.Module):
    def __init__(
        self,
        block_args,
        num_classes=num_classes,
        img_size=224,
        width_mult=1.,
        in_chans=3,
        stem_chs=16,
        num_features=1280,
        dw_conv='dw',
        kernel_size=(3,3),
        cnn_exp=(6,4),
        group_num=1,
        se_flag=[2,0,2,0],
        hyper_token_id=0,
        hyper_reduction_ratio=4,
        token_dim=128,
        token_num=6,
        cls_token_num=1,
        last_act='relu',
        last_exp=6,
        gbr_type='mlp',
        gbr_dynamic=[False, False, False],
        gbr_norm='post',
        gbr_ffn=False,
        gbr_before_skip=False,
        gbr_drop=[0.0, 0.0],
        mlp_token_exp=4,
        drop_rate=0.,
        drop_path_rate=0.,
        cnn_drop_path_rate=0.,
        attn_num_heads = 2,
        remove_proj_local=True,
        ):

        super(Modification, self).__init__()

        cnn_drop_path_rate = drop_path_rate
        mdiv = 8 if width_mult > 1.01 else 4
        self.num_classes = num_classes

        #global tokens
        self.tokens = nn.Embedding(token_num, token_dim) 

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.ReLU6(inplace=True)
        )
        input_channel = stem_chs

        # blocks
        layer_num = len(block_args)
        inp_res = img_size * img_size // 4
        layers = []
        for idx, val in enumerate(block_args):
            b, t, c, n, s, t2 = val # t2 for block2 the second expand
            block = eval(b)

            t = (t, t2)
            output_channel = _make_divisible(c * width_mult, mdiv) if idx > 0 else _make_divisible(c * width_mult, 4) 

            drop_path_prob = drop_path_rate * (idx+1) / layer_num
            cnn_drop_path_prob = cnn_drop_path_rate * (idx+1) / layer_num

            layers.append(block(
                input_channel, 
                output_channel, 
                s, 
                t, 
                dw_conv=dw_conv,
                kernel_size=kernel_size,
                group_num=group_num,
                se_flag=se_flag,
                hyper_token_id=hyper_token_id,
                hyper_reduction_ratio=hyper_reduction_ratio,
                token_dim=token_dim, 
                token_num=token_num,
                inp_res=inp_res,
                gbr_type=gbr_type,
                gbr_dynamic=gbr_dynamic,
                gbr_ffn=gbr_ffn,
                gbr_before_skip=gbr_before_skip,
                mlp_token_exp=mlp_token_exp,
                norm_pos=gbr_norm,
                drop_path_rate=drop_path_prob,
                cnn_drop_path_rate=cnn_drop_path_prob,
                attn_num_heads=attn_num_heads,
                remove_proj_local=remove_proj_local,        
            ))
            input_channel = output_channel

            if s == 2:
                inp_res = inp_res // 4

            for i in range(1, n):
                layers.append(block(
                    input_channel, 
                    output_channel, 
                    1, 
                    t, 
                    dw_conv=dw_conv,
                    kernel_size=kernel_size,
                    group_num=group_num,
                    se_flag=se_flag,
                    hyper_token_id=hyper_token_id,
                    hyper_reduction_ratio=hyper_reduction_ratio,
                    token_dim=token_dim, 
                    token_num=token_num,
                    inp_res=inp_res,
                    gbr_type=gbr_type,
                    gbr_dynamic=gbr_dynamic,
                    gbr_ffn=gbr_ffn,
                    gbr_before_skip=gbr_before_skip,
                    mlp_token_exp=mlp_token_exp,
                    norm_pos=gbr_norm,
                    drop_path_rate=drop_path_prob,
                    cnn_drop_path_rate=cnn_drop_path_prob,
                    attn_num_heads=attn_num_heads,
                    remove_proj_local=remove_proj_local,
                ))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # last layer of local to global
        self.local_global = Local2Global(
            input_channel,
            block_type = gbr_type,
            token_dim=token_dim,
            token_num=token_num,
            inp_res=inp_res,
            use_dynamic = gbr_dynamic[0],
            norm_pos=gbr_norm,
            drop_path_rate=drop_path_rate,
            attn_num_heads=attn_num_heads
        )

        # classifer
        self.classifier = MergeClassifier(
            input_channel, 
            oup=num_features, 
            ch_exp=last_exp,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_branch=gbr_drop,
            group_num=group_num,
            token_dim=token_dim,
            cls_token_num=cls_token_num,
            last_act = last_act,
            hyper_token_id=hyper_token_id,
            hyper_reduction_ratio=hyper_reduction_ratio
        )

        #initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                if m.bias is not None: 
                    m.bias.data.zero_()


    def forward(self, x):
        # setup tokens
        bs, _, _, _ = x.shape
        z = self.tokens.weight
        tokens = z[None].repeat(bs, 1, 1).clone()
        tokens = tokens.permute(1, 0, 2)
 
        # stem -> features -> classifier
        x = self.stem(x)
        x, tokens = self.features((x, tokens))
        tokens, attn = self.local_global((x, tokens))
        y = self.classifier((x, tokens))

        return y

def _create_modification(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        Modification, 
        variant, 
        pretrained,
        default_cfg=default_cfgs['default'],
        **kwargs)
    print(model)

    return model

common_model_kwargs = dict(
    cnn_drop_path_rate = 0.1,
    dw_conv = 'dw',
    kernel_size=(3, 3),
    cnn_exp = (6, 4),
    cls_token_num = 1,
    hyper_token_id = 0,
    hyper_reduction_ratio = 4,
    attn_num_heads = 2,
    gbr_norm = 'post',
    mlp_token_exp = 4,
    gbr_before_skip = False,
    gbr_drop = [0., 0.],
    last_act = 'relu',
    remove_proj_local = True,
)


def modification_508m(pretrained=False, **kwargs):

    #stem = 24
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  24, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  40, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  40, 1, 1, 3], #3
        ['DnaBlock3', 6,  72, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  3,  72, 1, 1, 3], #5
        ['DnaBlock3', 6, 128, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4, 128, 1, 1, 4], #7
        ['DnaBlock',  6, 176, 1, 1, 4], #8
        ['DnaBlock',  6, 176, 1, 1, 4], #9
        ['DnaBlock3', 6, 240, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 240, 1, 1, 4], #11
        ['DnaBlock',  6, 240, 1, 1, 4], #12
    ]
   
    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1920,
        stem_chs = 24,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_508m", pretrained, **model_kwargs)
    return model


def modification_294m(pretrained=False, **kwargs):

    #stem = 16
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  16, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  24, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  4,  24, 1, 1, 4], #3
        ['DnaBlock3', 6,  48, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  4,  48, 1, 1, 4], #5
        ['DnaBlock3', 6,  96, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  96, 1, 1, 4], #7
        ['DnaBlock',  6, 128, 1, 1, 4], #8
        ['DnaBlock',  6, 128, 1, 1, 4], #9
        ['DnaBlock3', 6, 192, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 192, 1, 1, 4], #11
        ['DnaBlock',  6, 192, 1, 1, 4], #12
    ]
  
    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1920,
        stem_chs = 16,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_294m", pretrained, **model_kwargs)
    return model


def modification_214m(pretrained=False, **kwargs):

    #stem = 12
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  20, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  20, 1, 1, 4], #3
        ['DnaBlock3', 6,  40, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  4,  40, 1, 1, 4], #5
        ['DnaBlock3', 6,  80, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  80, 1, 1, 4], #7
        ['DnaBlock',  6, 112, 1, 1, 4], #8
        ['DnaBlock',  6, 112, 1, 1, 4], #9
        ['DnaBlock3', 6, 160, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 160, 1, 1, 4], #11
        ['DnaBlock',  6, 160, 1, 1, 4], #12
    ]


    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1600,
        stem_chs = 12,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_214m", pretrained, **model_kwargs)
    return model


def modification_151m(pretrained=False, **kwargs):

    #stem = 12
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  16, 1, 2, 4], #2 56x56 (2)
        ['DnaBlock',  3,  16, 1, 1, 3], #3
        ['DnaBlock3', 6,  32, 1, 2, 4], #4 28x28 (2)
        ['DnaBlock',  3,  32, 1, 1, 3], #5
        ['DnaBlock3', 6,  64, 1, 2, 4], #6 14x14 (4)
        ['DnaBlock',  4,  64, 1, 1, 4], #7
        ['DnaBlock',  6,  88, 1, 1, 4], #8
        ['DnaBlock',  6,  88, 1, 1, 4], #9
        ['DnaBlock3', 6, 128, 1, 2, 4], #10 7x7 (3)
        ['DnaBlock',  6, 128, 1, 1, 4], #11
        ['DnaBlock',  6, 128, 1, 1, 4], #12
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1280,
        stem_chs = 12,
        token_num = 6,
        token_dim = 192,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_151m", pretrained, **model_kwargs)
    return model


def modification_96m(pretrained=False, **kwargs):

    #stem = 12
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 2,  12, 1, 1, 0], #1 112x112 (1)
        ['DnaBlock3', 6,  16, 1, 2, 4], #2 56x56 (1)
        ['DnaBlock3', 6,  32, 1, 2, 4], #3 28x28 (2)
        ['DnaBlock',  3,  32, 1, 1, 3], #4
        ['DnaBlock3', 6,  64, 1, 2, 4], #5 14x14 (3)
        ['DnaBlock',  4,  64, 1, 1, 4], #6
        ['DnaBlock',  6,  88, 1, 1, 4], #7
        ['DnaBlock3', 6, 128, 1, 2, 4], #8 7x7 (2)
        ['DnaBlock',  6, 128, 1, 1, 4], #9
    ]


    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1280,
        stem_chs = 12,
        token_num = 4,
        token_dim = 128,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_96m", pretrained, **model_kwargs)
    return model


def modification_52m(pretrained=False, **kwargs):

    #stem = 8
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 3,  12, 1, 2, 0], #1 56x56 (2)
        ['DnaBlock',  3,  12, 1, 1, 3], #2
        ['DnaBlock3', 6,  24, 1, 2, 4], #3 28x28 (2)
        ['DnaBlock',  3,  24, 1, 1, 3], #4
        ['DnaBlock3', 6,  48, 1, 2, 4], #5 14x14 (3)
        ['DnaBlock',  4,  48, 1, 1, 4], #6
        ['DnaBlock',  6,  64, 1, 1, 4], #7
        ['DnaBlock3', 6,  96, 1, 2, 4], #8 7x7 (2)
        ['DnaBlock',  6,  96, 1, 1, 4], #9
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 1,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1024,
        stem_chs = 8,
        token_num = 3,
        token_dim = 128,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_52m", pretrained, **model_kwargs)
    return model


def modification_26m(pretrained=False, **kwargs):

    #stem = 8
    dna_blocks = [ 
        #b, e1,  c, n, s, e2
        ['DnaBlock3', 3,  12, 1, 2, 0], #1 56x56 (2)
        ['DnaBlock',  3,  12, 1, 1, 3], #2
        ['DnaBlock3', 6,  24, 1, 2, 4], #3 28x28 (2)
        ['DnaBlock',  3,  24, 1, 1, 3], #4
        ['DnaBlock3', 6,  48, 1, 2, 4], #5 14x14 (3)
        ['DnaBlock',  4,  48, 1, 1, 4], #6
        ['DnaBlock',  6,  64, 1, 1, 4], #7
        ['DnaBlock3', 6,  96, 1, 2, 4], #8 7x7 (2)
        ['DnaBlock',  6,  96, 1, 1, 4], #9
    ]

    model_kwargs = dict(
        block_args = dna_blocks,
        width_mult = 1.0,
        se_flag = [2,0,2,0],
        group_num = 4,
        gbr_type = 'attn',
        gbr_dynamic = [True, False, False],
        gbr_ffn = True,
        num_features = 1024,
        stem_chs = 8,
        token_num = 3,
        token_dim = 128,
        **common_model_kwargs,
        **kwargs,   
    )
    model = _create_modification("modification_26m", pretrained, **model_kwargs)
    return model


#endregion



#region 9. FUNGSI TRAINING
#@title 9. FUNGSI TRAINING

""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

#region Import
import argparse
import copy
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')
#endregion

#region Hyperparameter

class Args:
    ### CUSTOM ###
    # Load this checkpoint as if they were the pretrained weights (with adaptation) (default: None).
    pretrained_path = '/home/tasi2425111/for_hpc/baru/ti_mod/4_new_train/output/train/20250405-152714-modification_294m-224/checkpoint-75.pth.tar'
    # Resume full model and optimizer state from checkpoint (default: '')
    resume = '/home/tasi2425111/for_hpc/baru/ti_mod/4_new_train/output/train/20250405-152714-modification_294m-224/checkpoint-75.pth.tar'
    # path to dataset (root dir)
    data_dir = '/home/tasi2425111/restructured-resized-tiny-imagenet-200'  #Disesuaikan dengan kebutuhan
    # number of label classes (Model default if None)
    num_classes = 200  #Disesuaikan dengan kebutuhan
    # Name of model to train (default: "resnet50")
    model = 'modification_294m'  #Disesuaikan dengan kebutuhan
    # Device (accelerator) to use.
    device = 'cuda:1'
    # Input image center crop percent (for validation only)
    crop_pct = None ## Tidak diikutkan karena sudah diresize
    # Use AutoAugment policy. "v0" or "original". (default: None)
    aa = 'rand-m15-n2' ## Operation = 2 , Magnitude = 15
    # mixup alpha, mixup enabled if > 0. (default: 0.)
    mixup = 0.8
    #loss_type = Softmax #Sudah default di code training
    # Label smoothing (default: 0.1)
    smoothing = 0.1
    # number of epochs to train (default: 300)
    epochs = 300
    # Input batch size for training (default: 128)
    batch_size = 20
    # Validation batch size override (default: None)
    validation_batch_size = 20
    # Optimizer (default: "sgd")
    opt = 'AdamW'
    # learning rate, overrides lr-base if set (default: None)
    lr = 1e-3
    # lower lr bound for cyclic schedulers that hit 0 (default: 0)
    min_lr = 1e-5
    # Learning rate scheduler (default: "cosine")
    sched = 'cosine'
    # epochs to warmup LR, if scheduler supports
    warmup_epochs = 10000
    # weight decay (default: 2e-5)
    weight_decay = 0.05
    # Clip gradient norm (default: None, no clipping)
    clip_grad = 1.0
    # Decay factor for model weights moving average (default: 0.9998)
    model_ema_decay = None
    # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
    input_size = (3, 224, 224)
    # Override mean pixel value of dataset
    mean = (0.485, 0.456, 0.406)
    # Override std deviation of dataset
    std = (0.229, 0.224, 0.225)


    ### DEFAULT ###
    # path to dataset (positional is *deprecated*, use --data-dir)
    data = None
    # dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)
    dataset = ''
    # dataset train split (default: train)
    train_split = 'train'
    # dataset validation split (default: validation)
    val_split = 'validation'
    # Manually specify num samples in train split, for IterableDatasets.
    train_num_samples = None
    # Manually specify num samples in validation split, for IterableDatasets.
    val_num_samples = None
    # Allow download of dataset for torch/ and tfds/ datasets that support it.
    dataset_download = False
    # path to class to idx mapping file (default: "")
    class_map = ''
    # Dataset image conversion mode for input images.
    input_img_mode = None
    # Dataset key for input images.
    input_key = None
    # Dataset key for target labels.
    target_key = None
    # Allow huggingface dataset import to execute code downloaded from the dataset's repo.
    dataset_trust_remote_code = False
    # Start with pretrained version of specified network (if avail)
    pretrained = False
    # Load this checkpoint into model after initialization (default: none)
    initial_checkpoint = ''
    # prevent resume of optimizer state when resuming model
    no_resume_opt = False
    # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.
    gp = None
    # Image size (default: None => model default)
    img_size = None
    # Image input channels (default: None => 3)
    in_chans = None
    # Image resize interpolation type (overrides model)
    interpolation = ''
    # Use channels_last memory layout
    channels_last = False
    # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')
    fuser = ''
    # The number of steps to accumulate gradients (default: 1)
    grad_accum_steps = 1
    # Enable gradient checkpointing through model blocks/stages
    grad_checkpointing = False
    # enable experimental fast-norm
    fast_norm = False
    # Model kwargs
    model_kwargs = {}
    # Head initialization scale
    head_init_scale = None
    # Head initialization bias value
    head_init_bias = None
    # torch.compile mode (default: None).
    torchcompile_mode = None
    # torch.jit.script the full model , action='store_true'
    torchscript = False
    # Enable compilation w/ specified backend (default: inductor). const='inductor'
    torchcompile = None
    # use NVIDIA Apex AMP or Native AMP for mixed precision training
    amp = False
    # lower precision AMP dtype (default: float16)
    amp_dtype = 'float16'
    # AMP impl to use, "native" or "apex" (default: native)
    amp_impl = 'native'
    # Model dtype override (non-AMP) (default: float32)
    model_dtype = None
    # Force broadcast buffers for native DDP to off.
    no_ddp_bb = False
    # torch.cuda.synchronize() end of each step
    synchronize_step = False
    # Local rank for distributed training
    local_rank = 0
    # Python imports for device backend modules.
    device_modules = None
    # Optimizer Epsilon (default: None, use opt default)
    opt_eps = None
    # Optimizer Betas (default: None, use opt default)
    opt_betas = None
    # Optimizer momentum (default: 0.9)
    momentum = 0.9
    # Gradient clipping mode. One of ("norm", "value", "agc")
    clip_mode = 'norm'
    # layer-wise learning rate decay (default: None)
    layer_decay = None
    # action=utils.ParseKwargs
    opt_kwargs = {}
    # Apply LR scheduler step on update instead of epoch end.
    sched_on_updates = False
    # base learning rate: lr = lr_base * global_batch_size / base_size
    lr_base = 0.1
    # base learning rate batch size (divisor, default: 256).
    lr_base_size = 256
    # base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)
    lr_base_scale = ''
    # learning rate noise on/off epoch percentages
    lr_noise = None
    # learning rate noise limit percent (default: 0.67)
    lr_noise_pct = 0.67
    # learning rate noise std-dev (default: 1.0)
    lr_noise_std = 1.0
    # learning rate cycle len multiplier (default: 1.0)
    lr_cycle_mul = 1.0
    # amount to decay each learning rate cycle (default: 0.5)
    lr_cycle_decay = 0.5
    # learning rate cycle limit, cycles enabled if > 1
    lr_cycle_limit = 1
    # learning rate k-decay for cosine/poly (default: 1.0)
    lr_k_decay = 1.0
    # warmup learning rate (default: 1e-5)
    warmup_lr = 1e-5
    # epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
    epoch_repeats = 0.
    # manual epoch number (useful on restarts)
    start_epoch = None
    # list of decay epoch indices for multistep lr. must be increasing
    decay_milestones = [90, 180, 270]
    # epoch interval to decay LR
    decay_epochs = 90
    # Exclude warmup period from decay schedule.
    warmup_prefix = False
    # epochs to cooldown LR at min_lr, after cyclic schedule ends
    cooldown_epochs = 0
    # patience epochs for Plateau LR scheduler (default: 10)
    patience_epochs = 10
    # LR decay rate (default: 0.1)
    decay_rate = 0.1
    # Disable all training augmentation, override other train aug args
    no_aug = False
    # Crop-mode in train
    train_crop_mode = None
    # Random resize scale (default: 0.08 1.0)
    scale = [0.08, 1.0]
    # Random resize aspect ratio (default: 0.75 1.33)
    ratio = [3. / 4., 4. / 3.]
    # Horizontal flip training aug probability
    hflip = 0.5
    # Vertical flip training aug probability
    vflip = 0.
    # Color jitter factor (default: 0.4)
    color_jitter = 0.4
    # Probability of applying any color jitter.
    color_jitter_prob = None
    # Probability of applying random grayscale conversion.
    grayscale_prob = None
    # Probability of applying gaussian blur.
    gaussian_blur_prob = None
    # Number of augmentation repetitions (distributed training only) (default: 0)
    aug_repeats = 0
    # Number of augmentation splits (default: 0, valid: 0 or >=2)
    aug_splits = 0
    # Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.
    jsd_loss = False
    # Enable BCE loss w/ Mixup/CutMix use.
    bce_loss = False
    # Sum over classes when using BCE loss.
    bce_sum = False
    # Threshold for binarizing softened BCE targets (default: None, disabled).
    bce_target_thresh = None
    # Positive weighting for BCE loss.
    bce_pos_weight = None
    # Random erase prob (default: 0.)
    reprob = 0.
    # Random erase mode (default: "pixel")
    remode = 'pixel'
    # Random erase count (default: 1)
    recount = 1
    # Do not random erase first (clean) augmentation split
    resplit = False
    # cutmix alpha, cutmix enabled if > 0. (default: 0.)
    cutmix = 0.0
    # cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)
    cutmix_minmax = None
    # Probability of performing mixup or cutmix when either/both is enabled
    mixup_prob = 1.0
    # Probability of switching to cutmix when both mixup and cutmix enabled
    mixup_switch_prob = 0.5
    # How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
    mixup_mode = 'batch'
    # Turn off mixup after this epoch, disabled if 0 (default: 0)
    mixup_off_epoch = 0
    # Training interpolation (random, bilinear, bicubic default: "random")
    train_interpolation = 'random'
    # Dropout rate (default: 0.)
    drop = 0.0
    # Drop connect rate, DEPRECATED, use drop-path (default: None)
    drop_connect = None
    # Drop path rate (default: None)
    drop_path = None
    # Drop block rate (default: None)
    drop_block = None
    # BatchNorm momentum override (if not None)
    bn_momentum = None
    # BatchNorm epsilon override (if not None)
    bn_eps = None
    # Enable NVIDIA Apex or Torch synchronized BatchNorm.
    sync_bn = False
    # Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")
    dist_bn = 'reduce'
    # Enable separate BN layers per augmentation split.
    split_bn = False
    # Enable tracking moving average of model weights.
    model_ema = False
    # Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.
    model_ema_force_cpu = False
    # Enable warmup for model EMA decay.
    model_ema_warmup = False
    # Random seed (default: 42)
    seed = 42
    # worker seed mode (default: all)
    worker_seeding = 'all'
    # how many batches to wait before logging training status
    log_interval = 50
    # how many batches to wait before writing recovery checkpoint
    recovery_interval = 0
    # number of checkpoints to keep (default: 10)
    checkpoint_hist = 10
    # how many training processes to use (default: 4)
    workers = 4
    # save images of input batches every log interval for debugging
    save_images = False
    # Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
    pin_mem = False
    # disable fast prefetcher
    no_prefetcher = False
    # path to output folder (default: none, current dir)
    output = ''
    # name of train experiment, name of sub-folder for output
    experiment = ''
    # Best metric (default: "top1")
    eval_metric = 'top1'
    # Test/inference time augmentation (oversampling) factor. 0=None (default: 0)
    tta = 0
    # use the multi-epochs-loader to save time at the beginning of every epoch
    use_multi_epochs_loader = False
    # log training and validation metrics to wandb
    log_wandb = False
    # wandb project name
    wandb_project = None
    # wandb tags
    wandb_tags = []
    # If resuming a run, the id of the run in wandb
    wandb_resume_id = ''

#endregion

def main():
    utils.setup_default_logging()
    args  = Args()

    model = globals()[args.model]()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning('float16 is not recommended for training, for half precision bfloat16 is recommended.')

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.weight_decay  # this isn't stored in optimizer.defaults
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
        )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type in ('cuda',) and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
            trust_remote_code=args.dataset_trust_remote_code,
        )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=eval_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            img_dtype=model_dtype or torch.float32,
            device=device,
            use_prefetcher=args.prefetcher,
        )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        # with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        #     f.write(args_text)

        if args.log_wandb:
            if has_wandb:
                assert not args.wandb_resume_id or args.resume
                wandb.init(
                    project=args.wandb_project,
                    name=exp_name,
                    config=args,
                    tags=args.wandb_tags,
                    resume="must" if args.wandb_resume_id else None,
                    id=args.wandb_resume_id if args.wandb_resume_id else None,
                )
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        if args.warmup_prefix:
            sched_explain = '(warmup_epochs + epochs + cooldown_epochs). Warmup added to total when warmup_prefix=True'
        else:
            sched_explain = '(epochs + cooldown_epochs). Warmup within epochs when warmup_prefix=False'
        _logger.info(
            f'Scheduled epochs: {num_epochs} {sched_explain}. '
            f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                    model_dtype=model_dtype,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            latest_results = {
                'epoch': epoch,
                'train': train_metrics,
            }
            if eval_metrics is not None:
                latest_results['validation'] = eval_metrics
            results.append(latest_results)

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        # log best metric as tracked by checkpoint saver
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if utils.is_primary(args):
        # for parsable results display, dump top-10 summaries to avoid excess console spam
        display_results = sorted(
            results,
            key=lambda x: x.get('validation', x.get('train')).get(eval_metric, 0),
            reverse=decreasing_metric,
        )
        print(f'--result\n{json.dumps(display_results[-10:], indent=4)}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        model_ema=None,
        mixup_fn=None,
        num_updates_total=None,
  ):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    # -- Tambahan: rekam waktu mulai epoch
    epoch_start_time = time.time()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // args.grad_accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device=device, dtype=model_dtype), target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()

        time_now = time.time()
        update_time_m.update(time_now - update_start_time)
        update_start_time = time_now

        # -- Log progress, termasuk estimasi waktu
        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            if args.distributed:
                # synchronize current step and avg loss, each process keeps its own running avg
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()
                update_sample_count *= args.world_size

            # -- Hitung waktu yang telah berjalan & sisa waktu (ETA)
            waktu_terpakai = time.time() - epoch_start_time
            progress = (update_idx + 1) / updates_per_epoch  # 0.0 - 1.0
            if progress > 0:
                estimasi_total = waktu_terpakai / progress
                estimasi_sisa = estimasi_total - waktu_terpakai
            else:
                estimasi_sisa = 0.0

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})  '
                    f'Elapsed/ETA: {waktu_terpakai:.1f}s / {estimasi_sisa:.1f}s'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, f'train-batch-{batch_idx}.jpg'),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        # synchronize avg loss, each process keeps its own running avg
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()

    return OrderedDict([('loss', loss_avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        model_dtype=None,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == "npu":
                torch.npu.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


main()

#endregion
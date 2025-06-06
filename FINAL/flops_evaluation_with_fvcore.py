num_classes=1000

#@title DEFINISI ARSITEKTUR MODEL (CoAtNet)
#region DEFINISI ARSITEKTUR MODEL (CoAtNet)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryEfficientSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return MemoryEfficientSwish.apply(x)


class MemoryEfficientMish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1./h.cosh().pow_(2)
        grad_hx = i.sigmoid()
        grad_gx = grad_gh *  grad_hx
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx
        return grad_output * grad_f


class Mish(nn.Module):
    def forward(self, x):
        return MemoryEfficientMish.apply(x)



import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

act_fn_map = {
    'swish': 'silu'
}

memory_efficient_map = {
    'swish': Swish,
    'mish': Mish
}


def get_act_fn(act_fn, prefer_memory_efficient=True):
    if isinstance(act_fn, str):
        if prefer_memory_efficient and act_fn in memory_efficient_map:
            return memory_efficient_map[act_fn]()
        if act_fn in act_fn_map:
            act_fn = act_fn_map[act_fn]
        return getattr(F, act_fn)
    return act_fn


def drop_connect(x, drop_ratio, training=True):
    if not training or drop_ratio == 0:
        return x
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules:
            print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: {}".format(total_num_params))


def weight_init(m):
    '''
    Usage:
        model.apply(weight_init)
    '''
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        init.normal_(m.weight)
        if m.bias is not None:
            init.normal_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param)
            else:
                init.normal_(param)



import torch.nn as nn
import math


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, ksize, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=ksize, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            half_pad_h = pad_h >> 1
            half_pad_w = pad_w >> 1
            self.static_padding = nn.ZeroPad2d((half_pad_w, pad_w - half_pad_w, half_pad_h, pad_h - half_pad_h))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, bias=True, momentum=0.1, eps=1e-5, act_fn=None, static_padding=False, image_size=None):
        super().__init__()
        if static_padding:
            self.conv = Conv2dStaticSamePadding(in_channels, out_channels, ksize, image_size=image_size, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum, eps=eps)
        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class DepthwiseConv2dBlock(nn.Module):
    def __init__(self, in_channels, ksize, stride=1, bias=False, momentum=0.1, eps=1e-5, act_fn=None, static_padding=False, image_size=None):
        super().__init__()
        if static_padding:
            self.conv = Conv2dStaticSamePadding(in_channels, in_channels, ksize, image_size=image_size, stride=stride, groups=in_channels, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, stride=stride, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum, eps=eps)
        self.act_fn = get_act_fn(act_fn)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x



import torch.nn as nn
import numpy as np


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, bias=True, act_fn='mish', static_padding=False, image_size=None):
        super().__init__()
        squeezed_channels = max(int(in_channels * se_ratio), 1)
        if static_padding:
            self.reduce = Conv2dStaticSamePadding(in_channels, squeezed_channels, 1, image_size=image_size, bias=bias)
        else:
            self.reduce = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1, bias=bias)
        self.act_fn = get_act_fn(act_fn)
        if static_padding:
            self.expand = Conv2dStaticSamePadding(squeezed_channels, in_channels, 1, image_size=image_size, bias=bias)
        else:
            self.expand = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.expand(x_se)
        x = x * x_se.sigmoid()
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 momentum=0.1, eps=1e-5, ksize=3,
                 stride=1, expand_ratio=1, se_ratio=0.25,
                 drop_ratio=0.2, act_fn='mish', image_size=224):
        super().__init__()
        self.skip_connect = stride == 1 and in_channels == out_channels
        self.drop_ratio = drop_ratio
        self.act_fn = get_act_fn(act_fn)
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        expand_channels = in_channels * expand_ratio
        if not isinstance(image_size, int):
            image_size = np.array(image_size)
        if expand_ratio != 1:
            self.expand_conv = Conv2dBlock(in_channels, expand_channels, 1, bias=False, momentum=momentum, eps=eps, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        self.depthwise_conv = DepthwiseConv2dBlock(expand_channels, ksize, stride=stride, momentum=momentum, eps=eps, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        image_size = np.ceil(image_size / stride).astype(int)
        if se_ratio is not None:
            self.se = SqueezeExcitation(expand_channels, se_ratio=se_ratio, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        self.project_conv = Conv2dBlock(expand_channels, out_channels, 1, bias=False, momentum=momentum, eps=eps, static_padding=True, image_size=image_size)

    def forward(self, x):
        x_in = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se_ratio is not None:
            x = self.se(x)
        x = self.project_conv(x)
        if self.skip_connect:
            x = drop_connect(x, self.drop_ratio, training=self.training) + x_in
        return x


class MBConvForRelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, out_channels,
                 momentum=0.1, eps=1e-5, ksize=3, expand_ratio=1,
                 se_ratio=0.25, drop_ratio=0.1, act_fn='mish',
                 use_downsampling=False, **kwargs):
        super().__init__()
        self.use_downsampling = use_downsampling
        self.drop_ratio = drop_ratio
        self.norm = nn.BatchNorm2d(in_channels)
        self.mbconv = MBConv(in_channels, out_channels, momentum=momentum, eps=eps,
                             ksize=ksize, stride=2 if use_downsampling else 1,
                             expand_ratio=expand_ratio, se_ratio=se_ratio,
                             drop_ratio=drop_ratio, act_fn=act_fn, image_size=(inp_h, inp_w))
        if use_downsampling:
            self.pool = nn.MaxPool2d((2, 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.use_downsampling:
            x_downsample = self.pool(x)
            x_downsample = self.conv(x_downsample)
        else:
            x_downsample = x
        x = self.norm(x)
        x = self.mbconv(x)
        x = drop_connect(x, self.drop_ratio, training=self.training)
        x = x_downsample + x
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FeedForwardRelativeAttention(nn.Module):
    def __init__(self, in_dim, expand_dim, drop_ratio=0.1, act_fn='gelu'):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dim, expand_dim, kernel_size=1)
        self.act_fn = get_act_fn(act_fn)
        self.fc2 = nn.Conv2d(expand_dim, in_dim, kernel_size=1)
        self.drop_ratio = drop_ratio

    def forward(self, x):
        x_in = x
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = drop_connect(x, drop_ratio=self.drop_ratio, training=self.training) + x_in
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn='mish', ff_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(ff_dropout)
        self.norm = nn.LayerNorm(in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TransformerWithRelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, n_head, d_k=None, d_v=None,
                 out_channels=None, attn_dropout=0.1, ff_dropout=0.1,
                 act_fn='gelu', attn_bias=False, expand_ratio=4,
                 use_downsampling=False, **kwargs):
        super().__init__()
        self.use_downsampling = use_downsampling
        self.dropout = ff_dropout
        out_channels = out_channels or in_channels
        d_k = d_k or out_channels // n_head
        d_v = d_v or out_channels // n_head
        if use_downsampling:
            self.pool = nn.MaxPool2d((2, 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = RelativeAttention(inp_h, inp_w, in_channels, n_head, d_k, d_v, out_channels, attn_dropout=attn_dropout, ff_dropout=ff_dropout, attn_bias=attn_bias)
        self.ff = FeedForwardRelativeAttention(out_channels, out_channels * expand_ratio, drop_ratio=ff_dropout, act_fn=act_fn)

    def forward(self, x):
        if self.use_downsampling:
            x_stem = self.pool(x)
            x_stem = self.conv(x_stem)
        else:
            x_stem = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        if self.use_downsampling:
            x = self.pool(x)
        x = self.attention(x)
        x = drop_connect(x, self.dropout, training=self.training)
        x = x_stem + x
        x_attn = x
        x = self.ff(x)
        x = drop_connect(x, self.dropout, training=self.training)
        x = x_attn + x
        return x


import torch.nn as nn
import torch.nn.functional as F

configs = {
    'coatnet-0': {
        'num_blocks': [2, 2, 3, 5, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-1': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-2': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [128, 128, 256, 512, 1026],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-3': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-4': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-5': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 256, 512, 1280, 2048],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 64,
        'block_types': ['C', 'C', 'T', 'T']
    },
    # Something's not right with this one
    # 'coatnet-6': {
    #     'num_blocks': [2, 2, 4, [8, 42], 2],
    #     'num_channels': [192, 192, 384, [768, 1536], 2048],
    #     'expand_ratio': [4, 4, 4, 4, 4],
    #     'n_head': 128,
    #     'block_types': ['C', 'C', 'C-T', 'T']
    # },
    # 'coatnet-7': {
    #     'num_blocks': [2, 2, 4, [8, 42], 2],
    #     'num_channels': [192, 256, 512, [1024, 2048], 3072],
    #     'expand_ratio': [4, 4, 4, 4, 4],
    #     'n_head': 128,
    #     'block_types': ['C', 'C', 'C-T', 'T']
    # }
}

blocks = {
    'C': MBConvForRelativeAttention,
    'T': TransformerWithRelativeAttention
}


class CoAtNet(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, config='coatnet-0', num_classes=None, head_act_fn='mish', head_dropout=0.1):
        super().__init__()
        self.config = configs[config]
        block_types = self.config['block_types']
        self.s0 = self._make_stem(in_channels)
        self.s1 = self._make_block(block_types[0], inp_h >> 2, inp_w >> 2,
                                   self.config['num_channels'][0],
                                   self.config['num_channels'][1],
                                   self.config['num_blocks'][1],
                                   self.config['expand_ratio'][0])
        self.s2 = self._make_block(block_types[1], inp_h >> 3, inp_w >> 3,
                                   self.config['num_channels'][1],
                                   self.config['num_channels'][2],
                                   self.config['num_blocks'][2],
                                   self.config['expand_ratio'][1])
        self.s3 = self._make_block(block_types[2], inp_h >> 4, inp_w >> 4,
                                   self.config['num_channels'][2],
                                   self.config['num_channels'][3],
                                   self.config['num_blocks'][3],
                                   self.config['expand_ratio'][2])
        self.s4 = self._make_block(block_types[3], inp_h >> 5, inp_w >> 5,
                                   self.config['num_channels'][3],
                                   self.config['num_channels'][4],
                                   self.config['num_blocks'][4],
                                   self.config['expand_ratio'][3])
        self.include_head = num_classes is not None
        if self.include_head:
            if isinstance(num_classes, int):
                self.single_head = True
                num_classes = [num_classes]
            else:
                self.single_head = False
            self.heads = nn.ModuleList([ProjectionHead(self.config['num_channels'][-1], nc, act_fn=head_act_fn, ff_dropout=head_dropout) for nc in num_classes])

    def _make_stem(self, in_channels):
        return nn.Sequential(*[
            nn.Conv2d(
                in_channels if i == 0 else self.config['num_channels'][0],
                self.config['num_channels'][0], kernel_size=3, padding=1,
                stride=2 if i == 0 else 1
            ) for i in range(self.config['num_blocks'][0])
        ])

    def _make_block(self, block_type, inp_h, inp_w, in_channels, out_channels, depth, expand_ratio):
        block_list = []
        if not isinstance(in_channels, int):
            in_channels = in_channels[-1]
        if block_type in blocks:
            block_cls = blocks[block_type]
            block_list.extend([
                block_cls(
                    inp_h, inp_w, in_channels if i == 0 else out_channels,
                    n_head=self.config['n_head'], out_channels=out_channels,
                    expand_ratio=expand_ratio, use_downsampling=i == 0
                ) for i in range(depth)
            ])
        else:
            for i, _block_type in enumerate(block_type.split('-')):
                block_cls = blocks[_block_type]
                block_list.extend(
                    block_cls(
                        inp_h, inp_w, in_channels if i == 0 and j == 0 else out_channels[i - 1] if j == 0 else out_channels[i],
                        n_head=self.config['n_head'], out_channels=out_channels[i],
                        expand_ratio=expand_ratio, use_downsampling=j == 0
                    ) for j in range(depth[i])
                )
        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        if self.include_head:
            if self.single_head:
                return self.heads[0](x)
            return [head(x) for head in self.heads]
        return x

#endregion

#@title DEFINISI ARSITEKTUR MODEL (Mobile-Former)
#region DEFINISI ARSITEKTUR MODEL (Mobile-Former)
import math
import torch
import random
import numpy as np
import torch.nn as nn


class MyDyRelu(nn.Module):
    def __init__(self, k):
        super(MyDyRelu, self).__init__()
        self.k = k

    def forward(self, inputs):
        x, relu_coefs = inputs
        # BxCxHxW -> HxWxBxCx1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # h w b c 1 -> h w b c k
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCxk -> BxCxHxW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result


def mixup_data(x, y, alpha, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    b = x.size()[0]
    if use_cuda:
        index = torch.randperm(b).cuda()
    else:
        index = torch.randperm(b)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix(input, target, beta):
    lam = np.random.beta(beta, beta)
    b = input.size()[0]
    rand_index = torch.randperm(b).cuda()
    target_a = target
    target_b = target[rand_index]
    bx1, by1, bx2, by2 = rand_box(input.size(), lam)
    input[:, :, bx1:bx2, by1:by2] = input[rand_index, :, bx1:bx2, by1:by2]
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


def cutmix_criterion(criterion, output, target_a, target_b, lam):
    return lam * criterion(output, target_a) + (1. - lam) * criterion(output, target_b)


def rand_box(size, lam):
    _, _, h, w = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    # 在图片上随机取一点作为cut的中心点
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bx1 = np.clip(cx - cut_w // 2, 0, w)
    by1 = np.clip(cy - cut_h // 2, 0, h)
    bx2 = np.clip(cx + cut_w // 2, 0, w)
    by2 = np.clip(cy + cut_h // 2, 0, h)
    return bx1, by1, bx2, by2


'''
for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
'''


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            # 计算图片面积
            # c h w
            area = img.size()[1] * img.size()[2]
            # 比率范围
            target_area = random.uniform(self.sl, self.sh) * area
            # 宽高比
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, inp, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        se = self.avg_pool(x)
        b, c, _, _ = se.size()
        se = se.view(b, c)
        se = self.se(se).view(b, c, 1, 1)
        return x * se.expand_as(x)


class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.conv1 = nn.Conv2d(inp, hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.act1 = MyDyRelu(2)

        self.conv2 = nn.Conv2d(hid, hid, kernel_size=ks, stride=stride,
                               padding=ks // 2, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.act2 = MyDyRelu(2)

        self.conv3 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.bn1(self.conv1(x))
        out_ = [out, relu_coefs]
        out = self.act1(out_)

        out = self.bn2(self.conv2(out))
        out_ = [out, relu_coefs]
        out = self.act2(out_)

        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(MobileDown, self).__init__()
        self.dim = dim
        self.hid, self.out = hid, out
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.dw_conv1 = nn.Conv2d(inp, hid, kernel_size=ks, stride=stride,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(hid)
        self.dw_act1 = MyDyRelu(2)

        self.pw_conv1 = nn.Conv2d(hid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(inp)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(inp, hid, kernel_size=ks, stride=1,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(hid)
        self.dw_act2 = MyDyRelu(2)

        self.pw_conv2 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn2 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.dw_bn1(self.dw_conv1(x))
        out_ = [out, relu_coefs]
        out = self.dw_act1(out_)
        out = self.pw_act1(self.pw_bn1(self.pw_conv1(out)))

        out = self.dw_bn2(self.dw_conv2(out))
        out_ = [out, relu_coefs]
        out = self.dw_act2(out_)
        out = self.pw_bn2(self.pw_conv2(out))

        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  # 2,65,1024 batch,patch+cls_token,dim (每个patch相当于一个token)
        b, n, _, h = *x.shape, self.heads
        # 输入x每个token的维度为1024，在注意力中token被映射16个64维的特征（head*dim_head），
        # 最后再把所有head的特征合并为一个（16*1024）的特征，作为每个token的输出
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv)  # 2,65,(16*64) -> 2,16,65,64 ,16个head，每个head维度64
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,16,65,64 @ b,16,64*65 -> b,16,65,65 : q@k.T
        attn = self.attend(dots)  # 注意力 2,16,65,65  16个head，注意力map尺寸65*65，对应token（patch）[i,j]之间的注意力
        # 每个token经过每个head的attention后的输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # atten@v 2,16,65,65 @ 2,16,65,64 -> 2,16,65,64
        out = rearrange(out, 'b h n d -> b n (h d)')  # 合并所有head的输出(16*64) -> 1024 得到每个token当前的特征
        return self.to_out(out)


# inputs: n L C
# output: n L C
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3):
        super(Former, self).__init__()
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        # dim_head = dim // heads
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


import torch
from torch import nn, einsum
from einops import rearrange


# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        q = self.to_q(z).view(b, self.heads, m, c)
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ x
        out = rearrange(out, 'b h m c -> b m (h c)')
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, 'b h l c -> b l (h c)')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out

config_52 = {
    'name': 'mf52',
    'token': 3,  # num tokens
    'embed': 128,  # embed dim
    'stem': 8,
    'bneck': {'e': 24, 'o': 12, 's': 2},  # exp out stride
    'body': [
        # stage2
        {'inp': 12, 'exp': 36, 'out': 12, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 12, 'exp': 72, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 72, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 48, 'exp': 288, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 64, 'exp': 384, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1024,  # hid_layer
    'fc2': num_classes  # num_clasess
    ,
}

config_294 = {
    'name': 'mf294',
    'token': 6,  # tokens
    'embed': 192,  # embed_dim
    'stem': 16,
    # stage1
    'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
    'body': [
        # stage2
        {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': num_classes  # num_clasess
    ,
}

config_508 = {
    'name': 'mf508',
    'token': 6,  # tokens and embed_dim
    'embed': 192,
    'stem': 24,
    'bneck': {'e': 48, 'o': 24, 's': 1},
    'body': [
        {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': num_classes  # num_clasess
    ,
}

config = {
    'mf52': config_52,
    'mf294': config_294,
    'mf508': config_508
}


import time
import torch
import torch.nn as nn

from torch.nn import init

class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]


class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )
        # bneck
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], 3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))
        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            hswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        for m in self.block:
            x, z = m([x, z])
        # x, z = self.block([x, z])
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        z = z[:, 0, :].view(b, -1)
        out = torch.cat((x, z), -1)
        return self.head(out)
        # return x, z

#endregion

#@title DEFINISI ARSITEKTUR MODEL (Mobile-Fromer-RA)
#region DEFINISI ARSITEKTUR MODEL (Mobile-Fromer-RA)
import math
import torch
import random
import numpy as np
import torch.nn as nn


class MyDyRelu(nn.Module):
    def __init__(self, k):
        super(MyDyRelu, self).__init__()
        self.k = k

    def forward(self, inputs):
        x, relu_coefs = inputs
        # BxCxHxW -> HxWxBxCx1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # h w b c 1 -> h w b c k
        output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
        # HxWxBxCxk -> BxCxHxW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result


def mixup_data(x, y, alpha, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    b = x.size()[0]
    if use_cuda:
        index = torch.randperm(b).cuda()
    else:
        index = torch.randperm(b)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix(input, target, beta):
    lam = np.random.beta(beta, beta)
    b = input.size()[0]
    rand_index = torch.randperm(b).cuda()
    target_a = target
    target_b = target[rand_index]
    bx1, by1, bx2, by2 = rand_box(input.size(), lam)
    input[:, :, bx1:bx2, by1:by2] = input[rand_index, :, bx1:bx2, by1:by2]
    lam = 1 - ((bx2 - bx1) * (by2 - by1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


def cutmix_criterion(criterion, output, target_a, target_b, lam):
    return lam * criterion(output, target_a) + (1. - lam) * criterion(output, target_b)


def rand_box(size, lam):
    _, _, h, w = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    # 在图片上随机取一点作为cut的中心点
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bx1 = np.clip(cx - cut_w // 2, 0, w)
    by1 = np.clip(cy - cut_h // 2, 0, h)
    bx2 = np.clip(cx + cut_w // 2, 0, w)
    by2 = np.clip(cy + cut_h // 2, 0, h)
    return bx1, by1, bx2, by2


'''
for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
'''


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            # 计算图片面积
            # c h w
            area = img.size()[1] * img.size()[2]
            # 比率范围
            target_area = random.uniform(self.sl, self.sh) * area
            # 宽高比
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(inp, inp // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inp // reduction, inp, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        se = self.avg_pool(x)
        b, c, _, _ = se.size()
        se = se.view(b, c)
        se = self.se(se).view(b, c, 1, 1)
        return x * se.expand_as(x)


class Mobile(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(Mobile, self).__init__()
        self.hid = hid
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.conv1 = nn.Conv2d(inp, hid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid)
        self.act1 = MyDyRelu(2)

        self.conv2 = nn.Conv2d(hid, hid, kernel_size=ks, stride=stride,
                               padding=ks // 2, groups=hid, bias=False)
        self.bn2 = nn.BatchNorm2d(hid)
        self.act2 = MyDyRelu(2)

        self.conv3 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.bn1(self.conv1(x))
        out_ = [out, relu_coefs]
        out = self.act1(out_)

        out = self.bn2(self.conv2(out))
        out_ = [out, relu_coefs]
        out = self.act2(out_)

        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileDown(nn.Module):
    def __init__(self, ks, inp, hid, out, se, stride, dim, reduction=4, k=2):
        super(MobileDown, self).__init__()
        self.dim = dim
        self.hid, self.out = hid, out
        self.k = k
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, 2 * k * hid)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())
        self.stride = stride
        # self.se = DyReLUB(channels=out, k=1) if dyrelu else se
        self.se = se

        self.dw_conv1 = nn.Conv2d(inp, hid, kernel_size=ks, stride=stride,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn1 = nn.BatchNorm2d(hid)
        self.dw_act1 = MyDyRelu(2)

        self.pw_conv1 = nn.Conv2d(hid, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn1 = nn.BatchNorm2d(inp)
        self.pw_act1 = nn.ReLU()

        self.dw_conv2 = nn.Conv2d(inp, hid, kernel_size=ks, stride=1,
                                  padding=ks // 2, groups=inp, bias=False)
        self.dw_bn2 = nn.BatchNorm2d(hid)
        self.dw_act2 = MyDyRelu(2)

        self.pw_conv2 = nn.Conv2d(hid, out, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn2 = nn.BatchNorm2d(out)

        self.shortcut = nn.Identity()
        if stride == 1 and inp != out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out),
            )

    def get_relu_coefs(self, z):
        theta = z[:, 0, :]
        # b d -> b d//4
        theta = self.fc1(theta)
        theta = self.relu(theta)
        # b d//4 -> b 2*k
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        # b 2*k
        return theta

    def forward(self, x, z):
        theta = self.get_relu_coefs(z)
        # b 2*k*c -> b c 2*k                                     2*k            2*k
        relu_coefs = theta.view(-1, self.hid, 2 * self.k) * self.lambdas + self.init_v

        out = self.dw_bn1(self.dw_conv1(x))
        out_ = [out, relu_coefs]
        out = self.dw_act1(out_)
        out = self.pw_act1(self.pw_bn1(self.pw_conv1(out)))

        out = self.dw_bn2(self.dw_conv2(out))
        out_ = [out, relu_coefs]
        out = self.dw_act2(out_)
        out = self.pw_bn2(self.pw_conv2(out))

        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., relative_pos=True, num_tokens=6):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.relative_pos = relative_pos
        self.num_tokens = num_tokens

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # Tambahkan relative position bias table
        if self.relative_pos:
            # Tabel posisi relatif: (2*M-1, heads)
            self.pos_bias_table = nn.Parameter(torch.randn(2 * num_tokens - 1, heads))

            # Buat koordinat dan indeks relatif
            coords = torch.arange(num_tokens)
            relative_indices = coords.unsqueeze(1) - coords.unsqueeze(0)  # M x M
            relative_indices += num_tokens - 1  # shift to start from 0
            self.register_buffer("relative_indices", relative_indices)

    def forward(self, x):  # x: (B, M, D), M = num_tokens
        B, M, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # (B, H, M, M)

        # Tambahkan relative position bias
        if self.relative_pos:
            # Ambil bias berdasarkan indeks relatif: shape [M, M] -> [M*M] -> ambil dari tabel
            pos_bias = self.pos_bias_table[self.relative_indices.view(-1)]  # (M*M, H)
            pos_bias = pos_bias.view(1, self.heads, M, M).expand(B, -1, -1, -1)  # (B, H, M, M)
            dots = dots + pos_bias

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (B, H, M, D_h)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# inputs: n L C
# output: n L C
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=32, dropout=0.3, num_tokens=6):
        super(Former, self).__init__()
        mlp_dim = dim * 2
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                       relative_pos=True, num_tokens=num_tokens)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


import torch
from torch import nn, einsum
from einops import rearrange


# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        q = self.to_q(z).view(b, self.heads, m, c)
        dots = q @ x.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ x
        out = rearrange(out, 'b h m c -> b m (h c)')
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q =  x.reshape(b, c, h*w).transpose(1,2).unsqueeze(1)
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, 'b h l c -> b l (h c)')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out

config_52 = {
    'name': 'mf52',
    'token': 3,  # num tokens
    'embed': 128,  # embed dim
    'stem': 8,
    'bneck': {'e': 24, 'o': 12, 's': 2},  # exp out stride
    'body': [
        # stage2
        {'inp': 12, 'exp': 36, 'out': 12, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 12, 'exp': 72, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 72, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 48, 'exp': 288, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 64, 'exp': 384, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1024,  # hid_layer
    'fc2': num_classes  # num_clasess
    ,
}

config_294 = {
    'name': 'mf294',
    'token': 6,  # tokens
    'embed': 192,  # embed_dim
    'stem': 16,
    # stage1
    'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
    'body': [
        # stage2
        {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': num_classes  # num_clasess
    ,
}

config_508 = {
    'name': 'mf508',
    'token': 6,  # tokens and embed_dim
    'embed': 192,
    'stem': 24,
    'bneck': {'e': 48, 'o': 24, 's': 1},
    'body': [
        {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': num_classes  # num_clasess
    ,
}

config = {
    'mf52': config_52,
    'mf294': config_294,
    'mf508': config_508
}


import time
import torch
import torch.nn as nn

from torch.nn import init

class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim, num_tokens=6):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim, heads=heads, num_tokens=num_tokens)  # Kirim num_tokens
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]


class MobileFormerRA(nn.Module):
    def __init__(self, cfg):
        super(MobileFormerRA, self).__init__()
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )
        # bneck
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], 3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed'], num_tokens=cfg['token']))
        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            hswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x = self.bneck(self.stem(x))
        for m in self.block:
            x, z = m([x, z])
        # x, z = self.block([x, z])
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        z = z[:, 0, :].view(b, -1)
        out = torch.cat((x, z), -1)
        return self.head(out)
        # return x, z

#endregion

#@title GetResult
import torch
from fvcore.nn import parameter_count, FlopCountAnalysis

input = torch.randn(1, 3, 224, 224)
model_image_size = (3, 224, 224)

mf_flops = FlopCountAnalysis(MobileFormer(config_294), input).total()
co_flops = FlopCountAnalysis(CoAtNet(model_image_size[1], model_image_size[2], model_image_size[0], config=f'coatnet-3', num_classes=num_classes), input).total()
mfra_flops = FlopCountAnalysis(MobileFormerRA(config_294), input).total()

#@title Result
print(f'Mobile-Former FLOPs : {mf_flops}')
print(f'CoAtNet FLOPs : {co_flops}')
print(f'Mobile-Former-RA FLOPs : {mfra_flops}')
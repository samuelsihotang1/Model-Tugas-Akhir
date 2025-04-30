import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., inp_h=14, inp_w=14, attn_bias=False):
        super(RelativeAttention, self).__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.inp_h = inp_h
        self.inp_w = inp_w
        
        # Q, K, V Linear transformations
        self.to_qkv = nn.Linear(dim, dim_head * 3 * heads, bias=attn_bias)
        
        # Output projection layer
        self.to_out = nn.Sequential(
            nn.Linear(heads * dim_head, dim),
            nn.Dropout(dropout)
        )
        
        # Initialize relative bias parameter
        self.relative_bias = nn.Parameter(torch.randn(heads, (2 * inp_h - 1) * (2 * inp_w - 1)), requires_grad=True)
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
        relative_bias = self.relative_bias.view(1, self.heads, (self.inp_h << 1) - 1, -1)
        relative_bias = F.interpolate(relative_bias, size=((height << 1) - 1, (width << 1) - 1), mode='bilinear', align_corners=True)
        return relative_bias.view(self.heads, -1)
    
    def forward(self, x):
        b, n, c = x.shape
        h = self.heads
        len_x = n
        
        # Q, K, V computation
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, len_x, h, self.dim_head).transpose(1, 2), qkv)
        
        # Get relative bias and indices for the current size of the input
        relative_indices = self.relative_indices
        relative_bias = self._interpolate_relative_bias(self.inp_h, self.inp_w)
        relative_indices = relative_indices.view(1, 1, *relative_indices.size()).expand(b, h, -1, -1)
        relative_bias = relative_bias.view(1, relative_bias.size(0), 1, relative_bias.size(1)).expand(b, -1, len_x, -1)
        
        # Add the relative bias to the attention scores
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)
        
        # Scaled dot-product attention with relative bias
        dots = torch.matmul(q, k.transpose(-1, -2)) + relative_biases
        attn = dots.softmax(dim=-1)
        
        # Attention output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, -1, h * self.dim_head)
        
        # Final projection layer
        return self.to_out(out)


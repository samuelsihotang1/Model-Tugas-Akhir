
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


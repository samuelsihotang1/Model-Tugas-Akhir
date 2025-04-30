class RelativeAttentionMobileFormer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., inp_h=16, inp_w=16):
        super(RelativeAttentionMobileFormer, self).__init__()
        inner_dim = heads * dim_head  # Total number of features from all heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attn_dropout = nn.Dropout(dropout)

        # Linear layers for Q, K, and V
        self.Q = nn.Linear(dim, inner_dim * heads, bias=False)
        self.K = nn.Linear(dim, inner_dim * heads, bias=False)
        self.V = nn.Linear(dim, inner_dim * heads, bias=False)

        # Final output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * heads, dim),
            nn.Dropout(dropout)
        )

        # Relative bias parameters
        self.relative_bias = nn.Parameter(
            torch.randn(heads, (inp_h << 1) - 1, (inp_w << 1) - 1),
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
        relative_bias = self.relative_bias.view(1, self.heads, (self.inp_h << 1) - 1, -1)
        relative_bias = F.interpolate(relative_bias, size=((height << 1) - 1, (width << 1) - 1), mode='bilinear', align_corners=True)
        return relative_bias.view(self.heads, -1)

    def update_relative_bias_and_indices(self, height, width):
        self.relative_indices = self._get_relative_indices(height, width)
        self.relative_bias = self._interpolate_relative_bias(height, width)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads
        len_x = n  # Number of tokens (patches) in the input
        qkv = self.Q(x).chunk(3, dim=-1)  # [batch, num_tokens, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Relative attention bias
        relative_indices = self.relative_indices.view(1, 1, *self.relative_indices.size()).expand(b, h, -1, -1)
        relative_bias = self._interpolate_relative_bias(q.shape[2], k.shape[2]).to(x.device)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)

        # Attention scores computation
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        similarity = dots + relative_biases
        attn = similarity.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Combine heads
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., height=224, width=224):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.height, self.width = height, width

        # Query, Key, and Value projection layers
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Relative positional encoding components
        self.relative_bias = nn.Parameter(
            torch.randn(heads, (height * 2 - 1) * (width * 2 - 1)), requires_grad=True
        )
        self.relative_indices = self._get_relative_indices(height, width)

        # Output projection layer
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.attend = nn.Softmax(dim=-1)

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
        return out.long()

    def forward(self, x):  # 2,65,1024 batch, patch+cls_token, dim
        b, n, _, h = *x.shape, self.heads
        len_x = n
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # b,h,n,d
        
        # Calculate similarity between queries and keys
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,h,i,j
        relative_indices = self.relative_indices.view(1, 1, *self.relative_indices.size()).expand(b, h, -1, -1)
        
        # Compute relative bias for positional encoding
        relative_bias = self.relative_bias.view(1, self.heads, (self.height * 2 - 1), (self.width * 2 - 1))
        relative_bias = F.interpolate(relative_bias, size=((self.height * 2 - 1), (self.width * 2 - 1)), mode='bilinear', align_corners=True)
        relative_biases = relative_bias.gather(dim=-1, index=relative_indices)
        
        similarity = dots + relative_biases  # Add relative bias to the similarity
        attn = self.attend(similarity)  # Softmax to get attention scores

        # Apply attention to values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # b,h,i,d
        out = rearrange(out, 'b h n d -> b n (h d)')  # Combine heads

        return self.to_out(out)

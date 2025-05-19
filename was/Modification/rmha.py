
class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, token_dim, num_heads, token_num, scale=True):
        super(RelativeMultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.token_dim = token_dim
        self.scale = scale
        self.head_dim = token_dim // num_heads
        self.token_num = token_num
        
        # Linear layers for Q, K, V
        self.q = nn.Linear(token_dim, token_dim)
        self.k = nn.Linear(token_dim, token_dim)
        self.v = nn.Linear(token_dim, token_dim)

        # Relative positional bias parameter
        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, token_num, token_num))  # shape: (num_heads, token_num, token_num)
        
        # Output projection layer
        self.out_proj = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        T, bs, C = x.shape  # T = token_num, bs = batch_size, C = token_dim

        # Project Q, K, V
        q = self.q(x).view(T, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # bs x N x T x Ct/N
        k = self.k(x).view(T, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # bs x N x T x Ct/N
        v = self.v(x).view(T, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # bs x N x T x Ct/N

        # Compute attention scores (dot-product)
        attn_scores = torch.einsum("bntd,bnqd->bnqt", k, q)  # bs x N x T x T (dot-product)

        # Add relative position bias: shape (num_heads, T, T)
        # Ensure that we are slicing bias correctly to match sequence length T
        attn_scores = attn_scores + self.relative_position_bias[:, :T, :T]  # Shape: (num_heads, T, T)

        if self.scale:
            attn_scores = attn_scores / (self.head_dim ** 0.5)

        # Apply softmax to get attention weights
        attn_probs = torch.softmax(attn_scores, dim=-1)  # Softmax along the last dimension (T)

        # Compute attention output
        attention_output = torch.einsum("bnqt,bnvd->bntd", attn_probs, v)  # bs x N x T x Ct/N
        
        # Concatenate the output of all heads
        attention_output = attention_output.permute(2, 0, 1, 3).contiguous().view(T, bs, -1)  # T x bs x C

        # Project the output to the original token dimension
        output = self.out_proj(attention_output)

        return output

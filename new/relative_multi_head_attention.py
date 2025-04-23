import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, token_dim, num_heads, scale=True):
        super(RelativeMultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.token_dim = token_dim
        self.scale = scale
        self.head_dim = token_dim // num_heads
        
        self.q = nn.Linear(token_dim, token_dim)
        self.k = nn.Linear(token_dim, token_dim)
        self.v = nn.Linear(token_dim, token_dim)

        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, 1000))  # bias untuk posisi relatif
        
        self.out_proj = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        T, bs, C = x.shape  # T = token_num, bs = batch_size, C = token_dim

        q = self.q(x).view(T, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # bs x N x T x Ct/N
        k = self.k(x).view(T, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # bs x N x T x Ct/N
        v = self.v(x).view(T, bs, self.num_heads, self.head_dim).permute(1, 2, 0, 3)  # bs x N x T x Ct/N

        # Menghitung dot-product attention
        attn_scores = torch.einsum("bntd,bnqd->bnqt", k, q)  # bs x N x T x T (dot-product)
        
        # Menambahkan bias posisi relatif
        attn_scores = attn_scores + self.relative_position_bias[:self.num_heads, :T, :T]

        if self.scale:
            attn_scores = attn_scores / (self.head_dim ** 0.5)

        attn_probs = torch.softmax(attn_scores, dim=-1)  # Softmax untuk mendapatkan bobot perhatian
        
        # Menghitung hasil perhatian
        attention_output = torch.einsum("bnqt,bnvd->bntd", attn_probs, v)  # bs x N x T x Ct/N
        
        # Menggabungkan hasil dari semua head
        attention_output = attention_output.permute(2, 0, 1, 3).contiguous().view(T, bs, -1)  # T x bs x C

        output = self.out_proj(attention_output)

        return output


class GlobalBlock(nn.Module):
    def __init__(self,
                 block_type='mlp',
                 token_dim=128,
                 token_num=6,
                 mlp_token_exp=4,
                 attn_num_heads=4,
                 use_dynamic=False,
                 use_ffn=False,
                 norm_pos='post',
                 drop_path_rate=0.):
        super(GlobalBlock, self).__init__()

        print(f'G2G: {attn_num_heads} heads')

        self.block = block_type
        self.num_heads = attn_num_heads
        self.token_num = token_num
        self.norm_pos = norm_pos
        self.use_dynamic = use_dynamic
        self.use_ffn = use_ffn
        self.ffn_exp = 2

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
                nn.Sigmoid(),
            )

        if 'mlp' in self.block:
            self.token_mlp = nn.Sequential(
                nn.Linear(token_num, token_num * mlp_token_exp),
                nn.GELU(),
                nn.Linear(token_num * mlp_token_exp, token_num),
            )

        # Menggunakan RelativeMultiHeadAttention menggantikan Multi-Head Attention biasa
        if 'attn' in self.block:
            self.attn = RelativeMultiHeadAttention(token_dim=token_dim, num_heads=attn_num_heads)

        self.channel_mlp = nn.Linear(token_dim, token_dim)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        tokens = x
        T, bs, C = tokens.shape

        if 'mlp' in self.block:
            # Use post norm, token.shape: token_num x bs x channel
            t = self.token_mlp(tokens.permute(1, 2, 0))  # bs x channel x token_num
            t_sum = t.permute(2, 0, 1)  # token_num x bs x channel

        if 'attn' in self.block:
            # Menggunakan Relative Multi-Head Attention
            t_a = self.attn(tokens)  # Menghitung perhatian dengan RelativeMultiHeadAttention
            t_sum = t_sum + t_a if 'mlp' in self.block else t_a

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

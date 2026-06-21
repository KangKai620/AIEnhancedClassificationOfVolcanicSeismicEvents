# モジュールのインポート
import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, er_len=297, batch_size=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.Er = nn.Parameter(torch.rand((er_len, self.head_dim), dtype=torch.float32))
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_gradients = None
    
    def save_attn_gradientes(self, attn_grads):
        self.attn_gradients = attn_grads

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        Er_expanded = self.Er.unsqueeze(0).repeat(b, 1, 1)
        rpr_mat = self._get_valid_embedding(Er_expanded, q.shape[1], k.shape[1])
        qe = torch.einsum('bhld,bmd->bhlm', q, rpr_mat)
        srel = self._skew(qe)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = mask[:, None, None, :].to(device)
            dots = dots.masked_fill(~mask, mask_value)
            del mask
        dots = dots + srel
        attn = dots.softmax(dim=-1)
        # attn.register_hook(self.save_attn_gradientes)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        return out, attn

    def _get_valid_embedding(self, Er, len_q, len_k):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets valid embeddings based on max length of RPR attention
        ----------
        """

        len_e = Er.shape[1]
        # start = max(0, len_e - len_q)
        start = 0
        return Er

    def _skew(self, qe):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Performs the skew optimized RPR computation (https://arxiv.org/abs/1809.04281)
        ----------
        """

        sz = qe.shape[2]
        mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)
        qe = mask * qe
        qe = F.pad(qe, (1,0, 0,0, 0,0))
        qe = torch.reshape(qe, (-1, qe.shape[1], qe.shape[3], qe.shape[2]))
        srel = qe[:, :, 1:, :]
        del mask, qe
        return srel

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        x, attn = self.attn(src, mask=mask)
        x = self.norm1(src + self.dropout1(x))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x, attn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, depth, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderBlock(dim, num_heads, hidden_dim, dropout))
    def forward(self, x, mask=None):
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_list.append(attn)
        return x, attn_list

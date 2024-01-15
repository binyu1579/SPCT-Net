import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as ops
from einops import rearrange


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FFN(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, hidden_dim, weight_init="normal", bias_init="zeros", has_bias=True),
            nn.GELU(),
            nn.Dropout(keep_prob=1-dropout),
            nn.Dense(hidden_dim, hidden_dim, weight_init="normal", bias_init="zeros", has_bias=True),
            nn.Dropout(keep_prob=1-dropout)
        )

    def construct(self, x):
        return self.net(x)


class SelfAttention(nn.Cell):
    """
    dim:
        Token's dimension, EX: word embedding vector size
    """
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.to_qvk = nn.Dense(dim, dim * 3, weight_init="normal", has_bias=False)
        self.scale_factor = dim ** -0.5

        # Final linear transformation layer
        self.w_out = nn.Dense(dim, dim, weight_init="normal", has_bias=False)

    def construct(self, x):
        q, k, v = ops.Split(3, 3)(self.to_qvk(x))

        dots = ops.MatMul()(q, ops.Transpose()(k, perm=(0, 2, 1))) * self.scale_factor
        attn = ops.Softmax(axis=-1)(dots)
        out = ops.MatMul()(attn, v)
        return self.w_out(out)


class Transformer(nn.Cell):
    def __init__(self, dim, depth, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.SequentialCell()
        for _ in range(depth):
            self.layers.append(nn.SequentialCell(
                PreNorm(dim, SelfAttention(dim)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ))

    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerBlock(nn.Cell):
    def __init__(self, in_channels, layers, mlp_dim, patch_size=4, drop_out=0.0):
        super(TransformerBlock, self).__init__()
        self.patch_size = patch_size
        self.transformer = Transformer(in_channels, layers, mlp_dim=mlp_dim, dropout=drop_out)
        self.reshape1 = ops.Reshape()
        self.reshape2 = ops.Reshape()
        self.squeeze = ops.Squeeze(-1)
        self.mul = ops.Mul()
        self.concat = ops.Concat(-1)

    def construct(self, x):
        _, _, h, w, s = x.shape
        global_repr = self.mul(x, Tensor(1.0))
        global_repr = self.reshape1(global_repr, (0, -1), (2, 3, 4),
                                   (h // self.patch_size, w // self.patch_size, s // self.patch_size))
        global_repr = self.transformer(global_repr)
        global_repr = self.reshape2(global_repr, (0, 1, 2, 3), (h, w, s, -1))
        global_repr = self.mul(global_repr, Tensor(1.0))
        return global_repr


if __name__ == '__main__':
    x = Tensor(np.random.randn(2, 64, 24, 48, 48).astype(np.float32))
    net = TransformerBlock(in_channels=64, layers=2, mlp_dim=128)
    y = net(x)
    print(y.shape)
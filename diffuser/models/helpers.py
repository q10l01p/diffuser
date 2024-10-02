import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- 模块 ------------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 存储维度信息

    def forward(self, x):
        device = x.device  # 获取输入张量的设备信息
        half_dim = self.dim // 2  # 计算一半的维度
        emb = math.log(10000) / (half_dim - 1)  # 计算位置编码的比例因子
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # 生成指数衰减的编码
        emb = x[:, None] * emb[None, :]  # 计算位置编码
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 将sin和cos编码拼接
        return emb  # 返回位置编码

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)  # 定义一维卷积层用于下采样

    def forward(self, x):
        return self.conv(x)  # 应用卷积层进行下采样

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)  # 定义一维反卷积层用于上采样

    def forward(self, x):
        return self.conv(x)  # 应用反卷积层进行上采样

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
        一维卷积 --> 组归一化 --> Mish激活
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),  # 一维卷积层
            Rearrange('batch channels horizon -> batch channels 1 horizon'),  # 重排张量维度
            nn.GroupNorm(n_groups, out_channels),  # 组归一化层
            Rearrange('batch channels 1 horizon -> batch channels horizon'),  # 恢复张量维度
            nn.Mish(),  # Mish激活函数
        )

    def forward(self, x):
        return self.block(x)  # 通过顺序层应用块

#-----------------------------------------------------------------------------#
#--------------------------------- 注意力 ------------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 存储函数

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x  # 返回函数结果加上输入，实现残差连接

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps  # 设置epsilon值用于数值稳定性
        self.g = nn.Parameter(torch.ones(1, dim, 1))  # 初始化可学习参数g
        self.b = nn.Parameter(torch.zeros(1, dim, 1))  # 初始化可学习参数b

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)  # 计算方差
        mean = torch.mean(x, dim=1, keepdim=True)  # 计算均值
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b  # 返回归一化结果

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn  # 存储函数
        self.norm = LayerNorm(dim)  # 实例化LayerNorm

    def forward(self, x):
        x = self.norm(x)  # 先进行归一化
        return self.fn(x)  # 然后应用函数

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5  # 计算缩放因子
        self.heads = heads  # 存储头数
        hidden_dim = dim_head * heads  # 计算隐藏维度
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)  # 定义用于生成qkv的卷积层
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)  # 定义输出卷积层

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 计算qkv并分块
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)  # 重排qkv
        q = q * self.scale  # 缩放q

        k = k.softmax(dim=-1)  # 对k进行softmax
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # 计算上下文

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)  # 计算输出
        out = einops.rearrange(out, 'b h c d -> b (h c) d')  # 重排输出
        return self.to_out(out)  # 返回输出

#-----------------------------------------------------------------------------#
#---------------------------------- 采样 ------------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape  # 获取批次大小
    out = a.gather(-1, t)  # 从a中提取t对应的值
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 调整输出形状

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    余弦调度
    参考: https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1  # 计算步数
    x = np.linspace(0, steps, steps)  # 生成线性空间
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2  # 计算累积乘积的余弦值
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 归一化
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 计算beta值
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)  # 剪辑beta值
    return torch.tensor(betas_clipped, dtype=dtype)  # 返回beta张量

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()  # 应用条件到x
    return x

#-----------------------------------------------------------------------------#
#---------------------------------- 损失 ------------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)  # 注册权重为缓冲区
        self.action_dim = action_dim  # 存储动作维度

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
            预测值和目标值 : 张量
                [ 批次大小 x 时间跨度 x 转换维度 ]
        '''
        loss = self._loss(pred, targ)  # 计算损失
        weighted_loss = (loss * self.weights).mean()  # 计算加权损失
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()  # 计算初始动作损失
        return weighted_loss, {'a0_loss': a0_loss}  # 返回加权损失和初始动作损失信息

class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()  # 计算均方损失

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]  # 计算预测值和目标值的相关系数
        else:
            corr = np.NaN  # 如果预测值长度小于等于1，相关系数为NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),  # 计算预测值和目标值的均值
            'min_pred': pred.min(), 'min_targ': targ.min(),  # 计算预测值和目标值的最小值
            'max_pred': pred.max(), 'max_targ': targ.max(),  # 计算预测值和目标值的最大值
            'corr': corr,  # 相关系数
        }

        return loss, info  # 返回损失和信息

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}

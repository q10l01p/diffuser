import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        # 初始化卷积模块列表，包含两个一维卷积块
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),  # 第一个卷积块
            Conv1dBlock(out_channels, out_channels, kernel_size),  # 第二个卷积块
        ])

        # 初始化时间嵌入的多层感知机（MLP）
        self.time_mlp = nn.Sequential(
            nn.Mish(),  # 激活函数 Mish
            nn.Linear(embed_dim, out_channels),  # 线性层，将嵌入维度转换为输出通道数
            Rearrange('batch t -> batch t 1'),  # 重新排列张量的形状
        )

        # 初始化残差卷积层，如果输入通道数与输出通道数不同，则使用1x1卷积进行调整，否则使用恒等映射
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]  # 输入张量 x，形状为 [批次大小 x 输入通道数 x 时间步长]
            t : [ batch_size x embed_dim ]  # 时间嵌入张量 t，形状为 [批次大小 x 嵌入维度]
            returns:
            out : [ batch_size x out_channels x horizon ]  # 输出张量 out，形状为 [批次大小 x 输出通道数 x 时间步长]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)  # 第一个卷积块的输出加上时间嵌入的输出
        out = self.blocks[1](out)  # 通过第二个卷积块
        return out + self.residual_conv(x)  # 加上残差连接后的输出


class TemporalUnet(nn.Module):
    """
    时间 U-Net 模型，用于时间序列预测。
    """

    def __init__(
        self,
        horizon,  # 预测时间范围
        transition_dim,  # 输入特征维度
        cond_dim,  # 条件特征维度
        dim=32,  # 模型基础维度
        dim_mults=(1, 2, 4, 8),  # 各层维度倍数
        attention=False,  # 是否使用注意力机制
    ):
        """
        模型初始化。

        参数：
            horizon: 预测时间范围
            transition_dim: 输入特征维度
            cond_dim: 条件特征维度
            dim: 模型基础维度
            dim_mults: 各层维度倍数
            attention: 是否使用注意力机制
        """
        super().__init__()

        # 计算各层维度
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # 时间嵌入维度
        time_dim = dim
        # 时间嵌入模块
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),  # 正弦位置嵌入
            nn.Linear(dim, dim * 4),  # 线性层
            nn.Mish(),  # Mish 激活函数
            nn.Linear(dim * 4, dim),  # 线性层
        )

        # 下采样模块列表
        self.downs = nn.ModuleList([])
        # 上采样模块列表
        self.ups = nn.ModuleList([])
        # 分辨率层数
        num_resolutions = len(in_out)

        print(in_out)
        # 构建下采样模块
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # 是否为最后一层
            is_last = ind >= (num_resolutions - 1)

            # 下采样模块
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),  # 残差时间块
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),  # 残差时间块
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),  # 线性注意力机制
                Downsample1d(dim_out) if not is_last else nn.Identity()  # 下采样
            ]))

            # 更新预测时间范围
            if not is_last:
                horizon = horizon // 2

        # 中间层维度
        mid_dim = dims[-1]
        # 中间层模块
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)  # 残差时间块
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()  # 线性注意力机制
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)  # 残差时间块

        # 构建上采样模块
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # 是否为最后一层
            is_last = ind >= (num_resolutions - 1)

            # 上采样模块
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),  # 残差时间块
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),  # 残差时间块
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),  # 线性注意力机制
                Upsample1d(dim_in) if not is_last else nn.Identity()  # 上采样
            ]))

            # 更新预测时间范围
            if not is_last:
                horizon = horizon * 2

        # 最终卷积层
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),  # 卷积块
            nn.Conv1d(dim, transition_dim, 1),  # 卷积层
        )

    def forward(self, x, cond, time):
        """
        模型前向传播。

        参数：
            x: 输入序列 [batch x horizon x transition]
            cond: 条件特征 [batch x cond_dim]
            time: 时间戳 [batch x 1]
        """

        # 重排输入序列
        x = einops.rearrange(x, 'b h t -> b t h')

        # 时间嵌入
        t = self.time_mlp(time)
        # 隐藏层列表
        h = []

        # 下采样模块
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)  # 残差时间块
            x = resnet2(x, t)  # 残差时间块
            x = attn(x)  # 线性注意力机制
            h.append(x)  # 保存隐藏层
            x = downsample(x)  # 下采样

        # 中间层
        x = self.mid_block1(x, t)  # 残差时间块
        x = self.mid_attn(x)  # 线性注意力机制
        x = self.mid_block2(x, t)  # 残差时间块

        # 上采样模块
        for resnet, resnet2, attn, upsample in self.ups:
            # 拼接隐藏层
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)  # 残差时间块
            x = resnet2(x, t)  # 残差时间块
            x = attn(x)  # 线性注意力机制
            x = upsample(x)  # 上采样

        # 最终卷积层
        x = self.final_conv(x)

        # 重排输出序列
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class ValueFunction(nn.Module):
    """
    价值函数模型，用于估计状态的价值。
    """

    def __init__(
        self,
        horizon,  # 预测时间范围
        transition_dim,  # 输入特征维度
        cond_dim,  # 条件特征维度
        dim=32,  # 模型基础维度
        dim_mults=(1, 2, 4, 8),  # 各层维度倍数
        out_dim=1,  # 输出维度
    ):
        """
        模型初始化。

        参数：
            horizon: 预测时间范围
            transition_dim: 输入特征维度
            cond_dim: 条件特征维度
            dim: 模型基础维度
            dim_mults: 各层维度倍数
            out_dim: 输出维度
        """
        super().__init__()

        # 计算各层维度
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 时间嵌入维度
        time_dim = dim
        # 时间嵌入模块
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),  # 正弦位置嵌入
            nn.Linear(dim, dim * 4),  # 线性层
            nn.Mish(),  # Mish 激活函数
            nn.Linear(dim * 4, dim),  # 线性层
        )

        # 模块列表
        self.blocks = nn.ModuleList([])
        # 分辨率层数
        num_resolutions = len(in_out)

        print(in_out)
        # 构建模块
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # 是否为最后一层
            is_last = ind >= (num_resolutions - 1)

            # 模块
            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),  # 残差时间块
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),  # 残差时间块
                Downsample1d(dim_out)  # 下采样
            ]))

            # 更新预测时间范围
            if not is_last:
                horizon = horizon // 2

        # 中间层维度
        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)  # 残差时间块
        self.mid_down1 = Downsample1d(mid_dim_2)  # 下采样
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)  # 残差时间块
        self.mid_down2 = Downsample1d(mid_dim_3)  # 下采样
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)  # 全连接层输入维度

        # 最终层
        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),  # 线性层
            nn.Mish(),  # Mish 激活函数
            nn.Linear(fc_dim // 2, out_dim),  # 线性层
        )

    def forward(self, x, cond, time, *args):
        """
        模型前向传播。

        参数：
            x: 输入序列 [batch x horizon x transition]
            cond: 条件特征 [batch x cond_dim]
            time: 时间戳 [batch x 1]
        """

        # 重排输入序列
        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        # 时间嵌入
        t = self.time_mlp(time)

        # 模块
        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)  # 残差时间块
            x = resnet2(x, t)  # 残差时间块
            x = downsample(x)  # 下采样

        ##
        x = self.mid_block1(x, t)  # 残差时间块
        x = self.mid_down1(x)  # 下采样
        ##
        x = self.mid_block2(x, t)  # 残差时间块
        x = self.mid_down2(x)  # 下采样
        ##
        x = x.view(len(x), -1)  # 展平
        # 最终层
        out = self.final_block(torch.cat([x, t], dim=-1))  # 拼接时间嵌入并输入最终层
        return out

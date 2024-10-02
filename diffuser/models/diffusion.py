from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,  # 导入余弦beta调度函数
    extract,  # 导入提取函数
    apply_conditioning,  # 导入应用条件函数
    Losses,  # 导入损失类
)

Sample = namedtuple('Sample', 'trajectories values chains')  # 定义命名元组Sample，包含轨迹、值和链

@torch.no_grad()  # 禁用梯度计算以节省内存
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)  # 从模型获取均值和对数方差
    model_std = torch.exp(0.5 * model_log_variance)  # 计算模型标准差

    # 当t == 0时无噪声
    noise = torch.randn_like(x)  # 生成与x形状相同的随机噪声
    noise[t == 0] = 0  # 当t为0时，噪声置为0

    values = torch.zeros(len(x), device=x.device)  # 初始化值为零
    return model_mean + model_std * noise, values  # 返回加噪后的均值和值

def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)  # 根据值降序排序索引
    x = x[inds]  # 根据排序索引重新排列x
    values = values[inds]  # 根据排序索引重新排列values
    return x, values  # 返回排序后的x和values

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)  # 创建一个全为i的长整型张量
    return t  # 返回时间步张量

class GaussianDiffusion(nn.Module):  # 定义高斯扩散类，继承自nn.Module
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()  # 调用父类的初始化方法
        self.horizon = horizon  # 时间范围
        self.observation_dim = observation_dim  # 观测维度
        self.action_dim = action_dim  # 动作维度
        self.transition_dim = observation_dim + action_dim  # 转换维度为观测加动作维度
        self.model = model  # 模型

        betas = cosine_beta_schedule(n_timesteps)  # 计算余弦beta调度
        alphas = 1. - betas  # 计算alpha值
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # 计算alpha累积乘积
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # 计算前一个alpha累积乘积

        self.n_timesteps = int(n_timesteps)  # 时间步数
        self.clip_denoised = clip_denoised  # 是否裁剪去噪
        self.predict_epsilon = predict_epsilon  # 是否预测epsilon

        self.register_buffer('betas', betas)  # 注册betas为缓冲区
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # 注册alphas_cumprod为缓冲区
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # 注册alphas_cumprod_prev为缓冲区

        # 计算扩散q(x_t | x_{t-1})及其他
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # 注册平方根alphas_cumprod
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  # 注册平方根(1 - alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))  # 注册对数(1 - alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))  # 注册平方根倒数alphas_cumprod
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # 注册平方根(倒数alphas_cumprod - 1)

        # 计算后验q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 计算后验方差
        self.register_buffer('posterior_variance', posterior_variance)  # 注册后验方差为缓冲区

        ## 因为扩散链开始时后验方差为0，所以对数计算被裁剪
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))  # 注册裁剪后的后验对数方差
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # 注册后验均值系数1
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))  # 注册后验均值系数2

        ## 获取损失系数并初始化目标
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)  # 获取损失权重
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)  # 初始化损失函数

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            设置轨迹的损失系数

            action_weight   : float
                第一个动作损失的系数
            discount   : float
                将轨迹损失的第t步乘以discount**t
            weights_dict    : dict
                { i: c } 将观测损失的第i维乘以c
        '''
        self.action_weight = action_weight  # 设置动作损失权重

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)  # 初始化维度权重为1

        ## 设置观测维度的损失系数
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w  # 根据权重字典调整维度权重

        ## 随着轨迹时间步衰减损失: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)  # 计算每个时间步的折扣
        discounts = discounts / discounts.mean()  # 标准化折扣
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)  # 计算损失权重

        ## 手动设置第一个动作的权重
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights  # 返回损失权重

    #------------------------------------------ 采样 ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            如果self.predict_epsilon为True，模型输出为(缩放的)噪声;
            否则，模型直接预测x0
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )  # 根据噪声和x_t预测x0
        else:
            return noise  # 直接返回噪声

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # 计算后验均值
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)  # 提取后验方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)  # 提取裁剪后的后验对数方差
        return posterior_mean, posterior_variance, posterior_log_variance_clipped  # 返回后验均值和方差

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))  # 从噪声预测x0

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)  # 如果需要，裁剪去噪后的x_recon
        else:
            assert RuntimeError()  # 否则抛出运行时错误

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)  # 计算模型均值和后验方差
        return model_mean, posterior_variance, posterior_log_variance  # 返回模型均值和方差

    @torch.no_grad()  # 禁用梯度计算以节省内存
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device  # 获取设备信息

        batch_size = shape[0]  # 获取批次大小
        x = torch.randn(shape, device=device)  # 生成随机噪声
        x = apply_conditioning(x, cond, self.action_dim)  # 应用条件

        chain = [x] if return_chain else None  # 如果需要返回链，则初始化链

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()  # 初始化进度条
        for i in reversed(range(0, self.n_timesteps)):  # 反向遍历时间步
            t = make_timesteps(batch_size, i, device)  # 创建时间步张量
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)  # 调用采样函数
            x = apply_conditioning(x, cond, self.action_dim)  # 再次应用条件

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})  # 更新进度条
            if return_chain: chain.append(x)  # 如果需要返回链，则将当前x添加到链中

        progress.stamp()  # 打印进度

        x, values = sort_by_values(x, values)  # 根据值排序x
        if return_chain: chain = torch.stack(chain, dim=1)  # 如果需要返回链，将链堆叠
        return Sample(x, values, chain)  # 返回采样结果

    @torch.no_grad()  # 禁用梯度计算以节省内存
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
            条件 : [ (时间, 状态), ... ]
        '''
        device = self.betas.device  # 获取设备信息
        batch_size = len(cond[0])  # 获取批次大小
        horizon = horizon or self.horizon  # 如果未提供horizon，则使用默认的self.horizon
        shape = (batch_size, horizon, self.transition_dim)  # 定义采样形状

        return self.p_sample_loop(shape, cond, **sample_kwargs)  # 调用采样循环并返回结果

    #------------------------------------------ 训练 ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)  # 如果未提供噪声，则生成与x_start形状相同的随机噪声

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )  # 根据噪声和x_start生成样本

        return sample  # 返回生成的样本

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)  # 生成与x_start形状相同的随机噪声

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 对x_start进行噪声采样
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)  # 应用条件

        x_recon = self.model(x_noisy, cond, t)  # 通过模型预测重构的x
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)  # 再次应用条件

        assert noise.shape == x_recon.shape  # 确保噪声和重构的x形状相同

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)  # 计算损失
        else:
            loss, info = self.loss_fn(x_recon, x_start)  # 计算损失

        return loss, info  # 返回损失和信息

    def loss(self, x, *args):
        batch_size = len(x)  # 获取批次大小
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()  # 随机生成时间步
        return self.p_losses(x, *args, t)  # 计算损失并返回

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)  # 前向传播，执行条件采样


class ValueDiffusion(GaussianDiffusion):  # 定义ValueDiffusion类，继承自GaussianDiffusion

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)  # 生成与x_start形状相同的随机噪声

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 对x_start进行噪声采样
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)  # 应用条件

        pred = self.model(x_noisy, cond, t)  # 通过模型预测

        loss, info = self.loss_fn(pred, target)  # 计算损失
        return loss, info  # 返回损失和信息

    def forward(self, x, cond, t):
        return self.model(x, cond, t)  # 前向传播，通过模型预测

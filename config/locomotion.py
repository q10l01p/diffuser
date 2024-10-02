import socket
from diffuser.utils import watch

#------------------------ 基础配置 ------------------------#

## 自动为规划实验命名
## 通过使用这些参数标记文件夹

args_to_watch = [
    ('prefix', ''),  # 前缀
    ('horizon', 'H'),  # 时间步长
    ('n_diffusion_steps', 'T'),  # 扩散步数
    ## 值函数相关参数
    ('discount', 'd'),  # 折扣因子
]

logbase = 'logs'  # 日志基础路径

base = {
    'diffusion': {
        ## 模型
        'model': 'models.TemporalUnet',  # 时序U-Net模型
        'diffusion': 'models.GaussianDiffusion',  # 高斯扩散模型
        'horizon': 32,  # 时间步长
        'n_diffusion_steps': 20,  # 扩散步数
        'action_weight': 10,  # 动作权重
        'loss_weights': None,  # 损失权重
        'loss_discount': 1,  # 损失折扣
        'predict_epsilon': False,  # 是否预测噪声
        'dim_mults': (1, 2, 4, 8),  # 维度乘数
        'attention': False,  # 是否使用注意力机制
        'renderer': 'utils.MuJoCoRenderer',  # 渲染器

        ## 数据集
        'loader': 'datasets.SequenceDataset',  # 序列数据集加载器
        'normalizer': 'GaussianNormalizer',  # 高斯归一化器
        'preprocess_fns': [],  # 预处理函数列表
        'clip_denoised': False,  # 是否裁剪去噪结果
        'use_padding': True,  # 是否使用填充
        'max_path_length': 1000,  # 最大路径长度

        ## 序列化
        'logbase': logbase,  # 日志基础路径
        'prefix': 'diffusion/defaults',  # 前缀
        'exp_name': watch(args_to_watch),  # 实验名称

        ## 训练
        'n_steps_per_epoch': 10000,  # 每轮步数
        'loss_type': 'l2',  # 损失类型
        'n_train_steps': 1e6,  # 训练总步数
        'batch_size': 32,  # 批次大小
        'learning_rate': 2e-4,  # 学习率
        'gradient_accumulate_every': 2,  # 梯度累积步数
        'ema_decay': 0.995,  # EMA衰减率
        'save_freq': 20000,  # 保存频率
        'sample_freq': 20000,  # 采样频率
        'n_saves': 5,  # 保存次数
        'save_parallel': False,  # 是否并行保存
        'n_reference': 8,  # 参考数量
        'bucket': None,  # 桶
        'device': 'cuda',  # 设备
        'seed': None,  # 随机种子
    },

    'values': {
        'model': 'models.ValueFunction',  # 值函数模型
        'diffusion': 'models.ValueDiffusion',  # 值函数扩散模型
        'horizon': 32,  # 时间步长
        'n_diffusion_steps': 20,  # 扩散步数
        'dim_mults': (1, 2, 4, 8),  # 维度乘数
        'renderer': 'utils.MuJoCoRenderer',  # 渲染器

        ## 值函数特定参数
        'discount': 0.99,  # 折扣因子
        'termination_penalty': -100,  # 终止惩罚
        'normed': False,  # 是否归一化

        ## 数据集
        'loader': 'datasets.ValueDataset',  # 值函数数据集加载器
        'normalizer': 'GaussianNormalizer',  # 高斯归一化器
        'preprocess_fns': [],  # 预处理函数列表
        'use_padding': True,  # 是否使用填充
        'max_path_length': 1000,  # 最大路径长度

        ## 序列化
        'logbase': logbase,  # 日志基础路径
        'prefix': 'values/defaults',  # 前缀
        'exp_name': watch(args_to_watch),  # 实验名称

        ## 训练
        'n_steps_per_epoch': 10000,  # 每轮步数
        'loss_type': 'value_l2',  # 损失类型
        'n_train_steps': 200e3,  # 训练总步数
        'batch_size': 32,  # 批次大小
        'learning_rate': 2e-4,  # 学习率
        'gradient_accumulate_every': 2,  # 梯度累积步数
        'ema_decay': 0.995,  # EMA衰减率
        'save_freq': 1000,  # 保存频率
        'sample_freq': 0,  # 采样频率
        'n_saves': 5,  # 保存次数
        'save_parallel': False,  # 是否并行保存
        'n_reference': 8,  # 参考数量
        'bucket': None,  # 桶
        'device': 'cuda',  # 设备
        'seed': None,  # 随机种子
    },

    'plan': {
        'guide': 'sampling.ValueGuide',  # 值函数引导
        'policy': 'sampling.GuidedPolicy',  # 引导策略
        'max_episode_length': 1000,  # 最大回合长度
        'batch_size': 64,  # 批次大小
        'preprocess_fns': [],  # 预处理函数列表
        'device': 'cuda',  # 设备
        'seed': None,  # 随机种子

        ## 采样参数
        'n_guide_steps': 2,  # 引导步数
        'scale': 0.1,  # 缩放因子
        't_stopgrad': 2,  # 停止梯度时间步
        'scale_grad_by_std': True,  # 是否按标准差缩放梯度

        ## 序列化
        'loadbase': None,  # 加载基础路径
        'logbase': logbase,  # 日志基础路径
        'prefix': 'plans/',  # 前缀
        'exp_name': watch(args_to_watch),  # 实验名称
        'vis_freq': 100,  # 可视化频率
        'max_render': 8,  # 最大渲染数量

        ## 扩散模型
        'horizon': 32,  # 时间步长
        'n_diffusion_steps': 20,  # 扩散步数

        ## 值函数
        'discount': 0.997,  # 折扣因子

        ## 加载
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',  # 扩散模型加载路径
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',  # 值函数加载路径

        'diffusion_epoch': 'latest',  # 扩散模型加载轮次
        'value_epoch': 'latest',  # 值函数加载轮次

        'verbose': True,  # 是否详细输出
        'suffix': '0',  # 后缀
    },
}


#------------------------ 覆盖配置 ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,  # 缩放因子
        't_stopgrad': 4,  # 停止梯度时间步
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,  # 时间步长
        'dim_mults': (1, 4, 8),  # 维度乘数
        'attention': True,  # 是否使用注意力机制
    },
    'values': {
        'horizon': 4,  # 时间步长
        'dim_mults': (1, 4, 8),  # 维度乘数
    },
    'plan': {
        'horizon': 4,  # 时间步长
        'scale': 0.001,  # 缩放因子
        't_stopgrad': 4,  # 停止梯度时间步
    },
}

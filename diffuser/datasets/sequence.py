from collections import namedtuple  # 导入namedtuple，用于创建简单的类
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库
import pdb  # 导入Python调试器模块

from .preprocessing import get_preprocess_fn  # 从当前包导入get_preprocess_fn函数
from .d4rl import load_environment, sequence_dataset  # 从d4rl模块导入相关函数
from .normalization import DatasetNormalizer  # 从normalization模块导入DatasetNormalizer类
from .buffer import ReplayBuffer  # 从buffer模块导入ReplayBuffer类

# 定义Batch和ValueBatch数据结构
Batch = namedtuple('Batch', 'trajectories conditions')  # 用于存储轨迹和条件的批次
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')  # 用于存储轨迹、条件和值的批次

class SequenceDataset(torch.utils.data.Dataset):
    # 序列数据集类，继承自PyTorch的数据集类

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        # 初始化方法，设置数据集的各项参数
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)  # 获取预处理函数
        self.env = env = load_environment(env)  # 加载环境
        self.env.seed(seed)  # 设置随机种子
        self.horizon = horizon  # 设置时间跨度
        self.max_path_length = max_path_length  # 设置最大路径长度
        self.use_padding = use_padding  # 设置是否使用填充
        itr = sequence_dataset(env, self.preprocess_fn)  # 获取序列数据集

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)  # 创建回放缓冲区
        for i, episode in enumerate(itr):
            fields.add_path(episode)  # 添加路径到缓冲区
        fields.finalize()  # 完成缓冲区的初始化

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])  # 创建数据归一化器
        self.indices = self.make_indices(fields.path_lengths, horizon)  # 创建采样索引

        self.observation_dim = fields.observations.shape[-1]  # 获取观察维度
        self.action_dim = fields.actions.shape[-1]  # 获取动作维度
        self.fields = fields  # 保存字段数据
        self.n_episodes = fields.n_episodes  # 保存剧集数量
        self.path_lengths = fields.path_lengths  # 保存路径长度
        self.normalize()  # 对数据进行归一化

        print(fields)  # 打印字段信息
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            归一化将被扩散模型预测的字段
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)  # 将数据展平
            normed = self.normalizer(array, key)  # 对数据进行归一化
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)  # 保存归一化结果

    def make_indices(self, path_lengths, horizon):
        '''
            创建用于从数据集中采样的索引；
            每个索引映射到一个数据点
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)  # 计算最大起始点
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)  # 如果不使用填充，调整最大起始点
            for start in range(max_start):
                end = start + horizon  # 计算结束点
                indices.append((i, start, end))  # 添加索引
        indices = np.array(indices)  # 转换为NumPy数组
        return indices  # 返回索引数组

    def get_conditions(self, observations):
        '''
            为规划获取当前观测的条件
        '''
        return {0: observations[0]}  # 返回字典，键为0，值为当前观测

    def __len__(self):
        return len(self.indices)  # 返回数据集的长度

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]  # 获取索引对应的路径、起始点和结束点

        observations = self.fields.normed_observations[path_ind, start:end]  # 获取归一化的观测数据
        actions = self.fields.normed_actions[path_ind, start:end]  # 获取归一化的动作数据

        conditions = self.get_conditions(observations)  # 获取条件
        trajectories = np.concatenate([actions, observations], axis=-1)  # 将动作和观测数据拼接成轨迹
        batch = Batch(trajectories, conditions)  # 创建Batch对象
        return batch  # 返回Batch对象

class GoalDataset(SequenceDataset):
    # 目标数据集类，继承自SequenceDataset类

    def get_conditions(self, observations):
        '''
            为规划获取当前观测和计划中最后一个观测的条件
        '''
        return {
            0: observations[0],  # 当前观测
            self.horizon - 1: observations[-1],  # 最后一个观测
        }

class ValueDataset(SequenceDataset):
    '''
        为数据点添加值字段以训练值函数
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.discount = discount  # 设置折扣因子
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]  # 计算折扣因子数组
        self.normed = False  # 设置是否归一化为False
        if normed:
            self.vmin, self.vmax = self._get_bounds()  # 获取值的上下界
            self.normed = True  # 设置归一化为True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)  # 打印信息
        vmin = np.inf  # 初始化最小值为正无穷
        vmax = -np.inf  # 初始化最大值为负无穷
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()  # 获取值
            vmin = min(value, vmin)  # 更新最小值
            vmax = max(value, vmax)  # 更新最大值
        print('✓')  # 打印完成标志
        return vmin, vmax  # 返回最小值和最大值

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)  # 归一化到[0, 1]
        ## [-1, 1]
        normed = normed * 2 - 1  # 映射到[-1, 1]
        return normed  # 返回归一化结果

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)  # 获取批次数据
        path_ind, start, end = self.indices[idx]  # 获取索引对应的路径、起始点和结束点
        rewards = self.fields['rewards'][path_ind, start:]  # 获取奖励数据
        discounts = self.discounts[:len(rewards)]  # 获取折扣因子
        value = (discounts * rewards).sum()  # 计算折扣奖励的总和
        if self.normed:
            value = self.normalize_value(value)  # 如果归一化，归一化值
        value = np.array([value], dtype=np.float32)  # 转换为NumPy数组
        value_batch = ValueBatch(*batch, value)  # 创建ValueBatch对象
        return value_batch  # 返回ValueBatch对象

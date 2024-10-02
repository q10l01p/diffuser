from collections import namedtuple  # 导入命名元组，用于创建简单的类
import torch  # 导入 PyTorch 库
import einops  # 导入 einops 库，用于张量操作
import pdb  # 导入 Python 调试器

import diffuser.utils as utils  # 导入 diffuser.utils 模块
from diffuser.datasets.preprocessing import get_policy_preprocess_fn  # 导入获取策略预处理函数

# 定义 Trajectories 命名元组，包含动作、观察值和价值
Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:
    """
    引导策略类，用于生成动作及其相应的观察值和价值。
    """

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        """
        初始化 GuidedPolicy。

        参数：
            guide: 引导模型
            diffusion_model: 扩散模型
            normalizer: 归一化器
            preprocess_fns: 预处理函数
            **sample_kwargs: 其他采样参数
        """
        self.guide = guide  # 保存引导模型
        self.diffusion_model = diffusion_model  # 保存扩散模型
        self.normalizer = normalizer  # 保存归一化器
        self.action_dim = diffusion_model.action_dim  # 动作维度
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)  # 获取预处理函数
        self.sample_kwargs = sample_kwargs  # 保存其他采样参数

    def __call__(self, conditions, batch_size=1, verbose=True):
        """
        调用方法，生成动作及其轨迹。

        参数：
            conditions: 条件信息
            batch_size: 批大小
            verbose: 是否打印详细信息

        返回：
            action: 第一个动作
            trajectories: 动作、观察值和价值的轨迹
        """
        # 对条件信息进行预处理
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        # 格式化条件信息
        conditions = self._format_conditions(conditions, batch_size)

        ## 执行反向扩散过程
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)  # 将轨迹转换为 NumPy 数组

        ## 提取动作 [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]  # 获取动作部分
        actions = self.normalizer.unnormalize(actions, 'actions')  # 反归一化动作

        ## 提取第一个动作
        action = actions[0, 0]  # 获取第一个批次的第一个动作

        normed_observations = trajectories[:, :, self.action_dim:]  # 获取观察值部分
        observations = self.normalizer.unnormalize(normed_observations, 'observations')  # 反归一化观察值

        # 创建 Trajectories 实例
        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories  # 返回第一个动作和轨迹

    @property
    def device(self):
        """
        获取模型所在的设备。
        """
        parameters = list(self.diffusion_model.parameters())  # 获取模型参数
        return parameters[0].device  # 返回第一个参数的设备

    def _format_conditions(self, conditions, batch_size):
        """
        格式化条件信息。

        参数：
            conditions: 条件字典
            batch_size: 批大小

        返回：
            conditions: 格式化后的条件字典
        """
        # 对观察值进行归一化
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        # 将条件转换为 PyTorch 张量
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        # 重复条件以匹配批大小
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions  # 返回格式化后的条件
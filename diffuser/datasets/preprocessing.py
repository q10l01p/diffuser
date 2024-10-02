import gym  # 导入OpenAI Gym库，用于创建和操作强化学习环境
import numpy as np  # 导入NumPy库，用于数值计算
import einops  # 导入einops库，用于数组的重新排列
from scipy.spatial.transform import Rotation as R  # 从SciPy库中导入Rotation类，用于旋转转换
import pdb  # 导入Python调试器模块

from .d4rl import load_environment  # 从d4rl模块导入load_environment函数

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def compose(*fns):
    # 组合多个函数为一个函数

    def _fn(x):
        for fn in fns:  # 依次应用每个函数
            x = fn(x)
        return x  # 返回最终结果

    return _fn  # 返回组合后的函数

def get_preprocess_fn(fn_names, env):
    # 获取预处理函数的组合
    fns = [eval(name)(env) for name in fn_names]  # 动态获取函数并应用环境
    return compose(*fns)  # 返回组合后的函数

def get_policy_preprocess_fn(fn_names):
    # 获取策略预处理函数的组合
    fns = [eval(name) for name in fn_names]  # 动态获取函数
    return compose(*fns)  # 返回组合后的函数

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

#------------------------ @TODO: remove some of these ------------------------#

def arctanh_actions(*args, **kwargs):
    # 对动作应用反双曲正切

    epsilon = 1e-4  # 定义一个小的偏移量

    def _fn(dataset):
        actions = dataset['actions']  # 获取动作数据
        assert actions.min() >= -1 and actions.max() <= 1, \
            f'applying arctanh to actions in range [{actions.min()}, {actions.max()}]'  # 确保动作在[-1, 1]范围内
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)  # 将动作限制在范围内
        dataset['actions'] = np.arctanh(actions)  # 对动作应用反双曲正切
        return dataset  # 返回处理后的数据集

    return _fn  # 返回处理函数

def add_deltas(env):
    # 添加状态变化量

    def _fn(dataset):
        deltas = dataset['next_observations'] - dataset['observations']  # 计算状态变化量
        dataset['deltas'] = deltas  # 将变化量添加到数据集中
        return dataset  # 返回处理后的数据集

    return _fn  # 返回处理函数

def maze2d_set_terminals(env):
    # 设置迷宫环境中的终止状态
    env = load_environment(env) if type(env) == str else env  # 加载环境
    goal = np.array(env._target)  # 获取目标位置
    threshold = 0.5  # 设置距离阈值

    def _fn(dataset):
        xy = dataset['observations'][:, :2]  # 提取x和y坐标
        distances = np.linalg.norm(xy - goal, axis=-1)  # 计算到目标的距离
        at_goal = distances < threshold  # 判断是否到达目标
        timeouts = np.zeros_like(dataset['timeouts'])  # 初始化超时数组

        ## 在时间t超时当且仅当
        ## 在时间t到达目标并且
        ## 在时间t+1没有到达目标
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]  # 找到超时的步数
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]  # 计算路径长度

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts  # 更新数据集中的超时信息
        return dataset  # 返回处理后的数据集

    return _fn  # 返回处理函数

#-------------------------- block-stacking --------------------------#

def blocks_quat_to_euler(observations):
    '''
        将方块的四元数转换为欧拉角
        输入 : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1

        返回 : [ N x robot_dim + n_blocks * 10] = [ N x 47 ]
            xyz: 3
            sin: 3
            cos: 3
            contact: 1
    '''
    robot_dim = 7  # 机器人维度
    block_dim = 8  # 方块维度
    n_blocks = 4  # 方块数量
    assert observations.shape[-1] == robot_dim + n_blocks * block_dim  # 确保输入维度正确

    X = observations[:, :robot_dim]  # 提取机器人部分的数据

    for i in range(n_blocks):  # 遍历每个方块
        start = robot_dim + i * block_dim  # 计算方块数据的起始索引
        end = start + block_dim  # 计算方块数据的结束索引

        block_info = observations[:, start:end]  # 提取方块数据

        xpos = block_info[:, :3]  # 提取位置数据
        quat = block_info[:, 3:-1]  # 提取四元数数据
        contact = block_info[:, -1:]  # 提取接触数据

        euler = R.from_quat(quat).as_euler('xyz')  # 将四元数转换为欧拉角
        sin = np.sin(euler)  # 计算欧拉角的正弦值
        cos = np.cos(euler)  # 计算欧拉角的余弦值

        X = np.concatenate([
            X,
            xpos,
            sin,
            cos,
            contact,
        ], axis=-1)  # 将转换后的数据拼接到结果中

    return X  # 返回转换后的数据

def blocks_euler_to_quat_2d(observations):
    # 将欧拉角转换为四元数
    robot_dim = 7  # 机器人维度
    block_dim = 10  # 方块维度
    n_blocks = 4  # 方块数量

    assert observations.shape[-1] == robot_dim + n_blocks * block_dim  # 确保输入维度正确

    X = observations[:, :robot_dim]  # 提取机器人部分的数据

    for i in range(n_blocks):  # 遍历每个方块
        start = robot_dim + i * block_dim  # 计算方块数据的起始索引
        end = start + block_dim  # 计算方块数据的结束索引

        block_info = observations[:, start:end]  # 提取方块数据

        xpos = block_info[:, :3]  # 提取位置数据
        sin = block_info[:, 3:6]  # 提取正弦值
        cos = block_info[:, 6:9]  # 提取余弦值
        contact = block_info[:, 9:]  # 提取接触数据

        euler = np.arctan2(sin, cos)  # 计算欧拉角
        quat = R.from_euler('xyz', euler, degrees=False).as_quat()  # 将欧拉角转换为四元数

        X = np.concatenate([
            X,
            xpos,
            quat,
            contact,
        ], axis=-1)  # 将转换后的数据拼接到结果中

    return X  # 返回转换后的数据

def blocks_euler_to_quat(paths):
    # 将多个路径中的欧拉角转换为四元数
    return np.stack([
        blocks_euler_to_quat_2d(path)  # 对每个路径应用转换函数
        for path in paths
    ], axis=0)  # 将结果堆叠成一个数组

def blocks_process_cubes(env):
    # 处理方块数据

    def _fn(dataset):
        for key in ['observations', 'next_observations']:  # 遍历需要处理的键
            dataset[key] = blocks_quat_to_euler(dataset[key])  # 将四元数转换为欧拉角
        return dataset  # 返回处理后的数据集

    return _fn  # 返回处理函数

def blocks_remove_kuka(env):
    # 移除Kuka机器人的数据

    def _fn(dataset):
        for key in ['observations', 'next_observations']:  # 遍历需要处理的键
            dataset[key] = dataset[key][:, 7:]  # 移除前7个维度的数据
        return dataset  # 返回处理后的数据集

    return _fn  # 返回处理函数

def blocks_add_kuka(observations):
    '''
        为观测数据添加Kuka机器人的数据
        observations : [ batch_size x horizon x 32 ]
    '''
    robot_dim = 7  # 机器人维度
    batch_size, horizon, _ = observations.shape  # 获取批量大小和时间跨度
    observations = np.concatenate([
        np.zeros((batch_size, horizon, 7)),  # 添加全零的Kuka数据
        observations,
    ], axis=-1)  # 将结果拼接到观测数据中
    return observations  # 返回添加后的数据

def blocks_cumsum_quat(deltas):
    '''
        计算方块的四元数的累积和
        deltas : [ batch_size x horizon x transition_dim ]
    '''
    robot_dim = 7  # 机器人维度
    block_dim = 8  # 方块维度
    n_blocks = 4  # 方块数量
    assert deltas.shape[-1] == robot_dim + n_blocks * block_dim  # 确保输入维度正确

    batch_size, horizon, _ = deltas.shape  # 获取批量大小和时间跨度

    cumsum = deltas.cumsum(axis=1)  # 计算累积和
    for i in range(n_blocks):  # 遍历每个方块
        start = robot_dim + i * block_dim + 3  # 计算四元数数据的起始索引
        end = start + 4  # 计算四元数数据的结束索引

        quat = deltas[:, :, start:end].copy()  # 提取四元数数据

        quat = einops.rearrange(quat, 'b h q -> (b h) q')  # 重新排列数据
        euler = R.from_quat(quat).as_euler('xyz')  # 将四元数转换为欧拉角
        euler = einops.rearrange(euler, '(b h) e -> b h e', b=batch_size)  # 重新排列数据
        cumsum_euler = euler.cumsum(axis=1)  # 计算欧拉角的累积和

        cumsum_euler = einops.rearrange(cumsum_euler, 'b h e -> (b h) e')  # 重新排列数据
        cumsum_quat = R.from_euler('xyz', cumsum_euler).as_quat()  # 将累积和转换为四元数
        cumsum_quat = einops.rearrange(cumsum_quat, '(b h) q -> b h q', b=batch_size)  # 重新排列数据

        cumsum[:, :, start:end] = cumsum_quat.copy()  # 更新累积和结果

    return cumsum  # 返回累积和结果

def blocks_delta_quat_helper(observations, next_observations):
    '''
        计算方块的四元数变化量
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1
    '''
    robot_dim = 7  # 机器人维度
    block_dim = 8  # 方块维度
    n_blocks = 4  # 方块数量
    assert observations.shape[-1] == next_observations.shape[-1] == robot_dim + n_blocks * block_dim  # 确保输入维度正确

    deltas = (next_observations - observations)[:, :robot_dim]  # 计算机器人部分的变化量

    for i in range(n_blocks):  # 遍历每个方块
        start = robot_dim + i * block_dim  # 计算方块数据的起始索引
        end = start + block_dim  # 计算方块数据的结束索引

        block_info = observations[:, start:end]  # 提取当前方块数据
        next_block_info = next_observations[:, start:end]  # 提取下一个方块数据

        xpos = block_info[:, :3]  # 提取位置数据
        next_xpos = next_block_info[:, :3]  # 提取下一个位置数据

        quat = block_info[:, 3:-1]  # 提取四元数数据
        next_quat = next_block_info[:, 3:-1]  # 提取下一个四元数数据

        contact = block_info[:, -1:]  # 提取接触数据
        next_contact = next_block_info[:, -1:]  # 提取下一个接触数据

        delta_xpos = next_xpos - xpos  # 计算位置变化量
        delta_contact = next_contact - contact  # 计算接触变化量

        rot = R.from_quat(quat)  # 创建旋转对象
        next_rot = R.from_quat(next_quat)  # 创建下一个旋转对象

        delta_quat = (next_rot * rot.inv()).as_quat()  # 计算四元数变化量
        w = delta_quat[:, -1:]  # 提取w分量

        ## 确保w为正以避免 [0, 0, 0, -1]
        delta_quat = delta_quat * np.sign(w)  # 调整四元数

        ## 应用旋转和变化量以确保我们到达next_rot
        ## delta * rot = next_rot * rot' * rot = next_rot
        next_euler = next_rot.as_euler('xyz')  # 计算下一个欧拉角
        next_euler_check = (R.from_quat(delta_quat) * rot).as_euler('xyz')  # 检查计算结果
        assert np.allclose(next_euler, next_euler_check)  # 确保计算结果一致

        deltas = np.concatenate([
            deltas,
            delta_xpos,
            delta_quat,
            delta_contact,
        ], axis=-1)  # 将所有变化量拼接到一起

    return deltas  # 返回变化量

def blocks_add_deltas(env):
    # 为方块数据添加变化量

    def _fn(dataset):
        deltas = blocks_delta_quat_helper(dataset['observations'], dataset['next_observations'])  # 计算变化量
        # deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas  # 将变化量添加到数据集中
        return dataset  # 返回处理后的数据集

    return _fn  # 返回处理函数

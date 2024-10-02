import os  # 导入操作系统相关模块
import collections  # 导入集合模块，用于处理数据集合
import numpy as np  # 导入NumPy库，用于数值计算
import gym  # 导入OpenAI Gym库，用于创建和操作强化学习环境
import pdb  # 导入Python调试器模块

from contextlib import (  # 从contextlib模块中导入上下文管理器相关功能
    contextmanager,  # 导入contextmanager装饰器，用于定义上下文管理器
    redirect_stderr,  # 导入redirect_stderr，用于重定向标准错误输出
    redirect_stdout,  # 导入redirect_stdout，用于重定向标准输出
)

@contextmanager
def suppress_output():
    """
        一个上下文管理器，用于将标准输出和标准错误重定向到devnull
        参考：https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:  # 打开系统的devnull文件，以写模式
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:  # 重定向stderr和stdout到fnull
            yield (err, out)  # 生成重定向后的stderr和stdout

with suppress_output():
    ## d4rl库会打印出各种警告信息
    import d4rl  # 导入d4rl库，用于处理离线强化学习数据集

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:  # 如果name不是字符串类型
        ## name已经是一个环境对象
        return name  # 直接返回name
    with suppress_output():  # 使用suppress_output上下文管理器
        wrapped_env = gym.make(name)  # 创建一个Gym环境
    env = wrapped_env.unwrapped  # 获取未包装的环境对象
    env.max_episode_steps = wrapped_env._max_episode_steps  # 设置最大步数
    env.name = name  # 设置环境名称
    return env  # 返回环境对象

def get_dataset(env):
    dataset = env.get_dataset()  # 从环境中获取数据集

    if 'antmaze' in str(env).lower():  # 如果环境名称中包含'antmaze'
        ## antmaze-v0环境存在各种与轨迹分段相关的错误
        ## 手动重置终止和超时字段
        dataset = antmaze_fix_timeouts(dataset)  # 修复超时问题
        dataset = antmaze_scale_rewards(dataset)  # 调整奖励值
        get_max_delta(dataset)  # 获取最大变化量

    return dataset  # 返回处理后的数据集

def sequence_dataset(env, preprocess_fn):
    """
    返回一个遍历轨迹的迭代器。
    参数:
        env: 一个OfflineEnv对象。
        dataset: 一个可选的数据集用于处理。如果为None，
            则数据集默认为env.get_dataset()
        **kwargs: 传递给env.get_dataset()的参数。
    返回:
        一个字典的迭代器，字典包含以下键：
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)  # 获取数据集
    dataset = preprocess_fn(dataset)  # 预处理数据集

    N = dataset['rewards'].shape[0]  # 获取奖励的数量
    data_ = collections.defaultdict(list)  # 使用defaultdict初始化数据集合

    # 新版本的数据集添加了一个显式的超时字段。保持旧方法以便向后兼容。
    use_timeouts = 'timeouts' in dataset  # 检查数据集中是否存在超时字段

    episode_step = 0  # 初始化步数
    for i in range(N):  # 遍历数据集中的每个元素
        done_bool = bool(dataset['terminals'][i])  # 检查当前步是否为终止状态
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]  # 获取超时状态
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)  # 检查是否达到最大步数

        for k in dataset:  # 遍历数据集中的每个键
            if 'metadata' in k: continue  # 跳过元数据
            data_[k].append(dataset[k][i])  # 将当前步的数据添加到集合中

        if done_bool or final_timestep:  # 如果步数达到终止或超时
            episode_step = 0  # 重置步数
            episode_data = {}  # 初始化当前轨迹的数据
            for k in data_:  # 遍历数据集合
                episode_data[k] = np.array(data_[k])  # 转换为NumPy数组
            if 'maze2d' in env.name:  # 如果环境名称包含'maze2d'
                episode_data = process_maze2d_episode(episode_data)  # 处理maze2d轨迹数据
            yield episode_data  # 生成当前轨迹数据
            data_ = collections.defaultdict(list)  # 重置数据集合

        episode_step += 1  # 增加步数


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        为轨迹添加`next_observations`字段
    '''
    assert 'next_observations' not in episode  # 确保轨迹中没有`next_observations`字段
    length = len(episode['observations'])  # 获取观察值的长度
    next_observations = episode['observations'][1:].copy()  # 复制下一个观察值
    for key, val in episode.items():  # 遍历轨迹中的每个键值对
        episode[key] = val[:-1]  # 去掉最后一个元素
    episode['next_observations'] = next_observations  # 添加`next_observations`字段
    return episode  # 返回处理后的轨迹

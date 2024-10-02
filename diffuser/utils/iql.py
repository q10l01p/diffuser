import os  # 导入操作系统模块，用于文件操作
import numpy as np  # 导入 NumPy 模块，用于数值计算
import jax  # 导入 JAX 模块，用于自动微分和加速计算
import jax.numpy as jnp  # 导入 JAX 的 NumPy 扩展，用于在 JAX 环境中使用 NumPy 函数
import functools  # 导入 functools 模块，用于使用偏函数
import pdb  # 导入 pdb 模块，用于调试

from diffuser.iql.common import Model  # 导入 Model 类，用于创建和加载模型
from diffuser.iql.value_net import DoubleCritic  # 导入 DoubleCritic 类，用于创建双重评论家网络

def load_q(env, loadpath, hidden_dims=(256, 256), seed=42):  # 定义函数加载 Q 网络
    print(f'[ utils/iql ] Loading Q: {loadpath}')  # 打印加载 Q 网络的信息
    observations = env.observation_space.sample()[np.newaxis]  # 从环境中获取一个随机观测值
    actions = env.action_space.sample()[np.newaxis]  # 从环境中获取一个随机动作值

    rng = jax.random.PRNGKey(seed)  # 初始化 JAX 随机数生成器
    rng, key = jax.random.split(rng)  # 分割随机数生成器，获取两个新的生成器

    critic_def = DoubleCritic(hidden_dims)  # 创建双重评论家网络
    critic = Model.create(critic_def,  # 创建模型
                          inputs=[key, observations, actions])  # 传入随机数生成器、观测值和动作值作为输入

    ## allows for relative paths  # 注释说明：允许使用相对路径
    loadpath = os.path.expanduser(loadpath)  # 扩展路径，将 ~ 替换为用户主目录
    critic = critic.load(loadpath)  # 加载训练好的模型
    return critic  # 返回加载好的 Q 网络

class JaxWrapper:  # 定义一个 JAX 包装类

    def __init__(self, env, loadpath, *args, **kwargs):  # 初始化函数
        self.model = load_q(env, loadpath)  # 加载 Q 网络

    @functools.partial(jax.jit, static_argnames=('self'), device=jax.devices('cpu')[0])  # 使用 jax.jit 进行 JIT 编译
    def forward(self, xs):  # 定义 forward 函数，用于计算 Q 值
        Qs = self.model(*xs)  # 使用 Q 网络计算多个 Q 值
        Q = jnp.minimum(*Qs)  # 获取所有 Q 值中的最小值
        return Q  # 返回 Q 值

    def __call__(self, *xs):  # 定义调用函数，用于计算 Q 值
        Q = self.forward(xs)  # 使用 forward 函数计算 Q 值
        return np.array(Q)  # 将 Q 值转换为 NumPy 数组并返回
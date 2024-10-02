import os  # 导入os模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数值计算
from gym import utils  # 从Gym库中导入utils模块
from gym.envs.mujoco import mujoco_env  # 从Gym库中导入mujoco_env模块

class HalfCheetahFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # 定义HalfCheetahFullObsEnv类，继承自MujocoEnv和EzPickle

    def __init__(self):
        # 初始化函数
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/half_cheetah.xml')  # 获取半猎豹模型的路径
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5)  # 初始化MujocoEnv，帧跳跃为5
        utils.EzPickle.__init__(self)  # 初始化EzPickle

    def step(self, action):
        # 定义环境在执行一个动作后的变化
        xposbefore = self.sim.data.qpos[0]  # 获取动作前x坐标
        self.do_simulation(action, self.frame_skip)  # 执行动作
        xposafter = self.sim.data.qpos[0]  # 获取动作后x坐标
        ob = self._get_obs()  # 获取观测
        reward_ctrl = -0.1 * np.square(action).sum()  # 计算控制成本
        reward_run = (xposafter - xposbefore) / self.dt  # 计算前进奖励
        reward = reward_ctrl + reward_run  # 总奖励
        done = False  # 半猎豹环境一般不会结束
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)  # 返回观测、奖励、完成标志和奖励字典

    def _get_obs(self):
        # 定义获取观测的方法
        return np.concatenate([
            self.sim.data.qpos.flat,  # 关节位置
            self.sim.data.qvel.flat,  # 关节速度
        ])  # 返回观测

    def reset_model(self):
        # 定义重置模型的方法
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)  # 初始化关节位置，添加随机扰动
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1  # 初始化关节速度，添加随机扰动
        self.set_state(qpos, qvel)  # 设置状态
        return self._get_obs()  # 返回观测

    def viewer_setup(self):
        # 定义查看器设置的方法
        self.viewer.cam.distance = self.model.stat.extent * 0.5  # 设置摄像机距离

    def set(self, state):
        # 定义设置状态的方法
        qpos_dim = self.sim.data.qpos.size  # 获取关节位置的维度
        qpos = state[:qpos_dim]  # 从状态中提取关节位置
        qvel = state[qpos_dim:]  # 从状态中提取关节速度
        self.set_state(qpos, qvel)  # 设置状态
        return self._get_obs()  # 返回观测

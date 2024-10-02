import os  # 导入os模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数值计算
from gym import utils  # 从Gym库中导入utils模块
from gym.envs.mujoco import mujoco_env  # 从Gym库中导入mujoco_env模块

class HopperFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # 定义HopperFullObsEnv类，继承自MujocoEnv和EzPickle

    def __init__(self):
        # 初始化函数
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/hopper.xml')  # 获取单足跳跃者模型的路径
        mujoco_env.MujocoEnv.__init__(self, asset_path, 4)  # 初始化MujocoEnv，帧跳跃为4
        utils.EzPickle.__init__(self)  # 初始化EzPickle

    def step(self, a):
        # 定义环境在执行一个动作后的变化
        posbefore = self.sim.data.qpos[0]  # 获取动作前的x坐标
        self.do_simulation(a, self.frame_skip)  # 执行动作
        posafter, height, ang = self.sim.data.qpos[0:3]  # 获取动作后的x坐标、高度和角度
        alive_bonus = 1.0  # 生存奖励
        reward = (posafter - posbefore) / self.dt  # 计算前进奖励
        reward += alive_bonus  # 加上生存奖励
        reward -= 1e-3 * np.square(a).sum()  # 减去控制成本
        s = self.state_vector()  # 获取状态向量
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))  # 判断是否完成
        ob = self._get_obs()  # 获取观测
        return ob, reward, done, {}  # 返回观测、奖励、完成标志和空字典

    def _get_obs(self):
        # 定义获取观测的方法
        return np.concatenate([
            self.sim.data.qpos.flat,  # 关节位置
            np.clip(self.sim.data.qvel.flat, -10, 10)  # 裁剪后的关节速度
        ])  # 返回观测

    def reset_model(self):
        # 定义重置模型的方法
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)  # 初始化关节位置，添加小随机扰动
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)  # 初始化关节速度，添加小随机扰动
        self.set_state(qpos, qvel)  # 设置状态
        return self._get_obs()  # 返回观测

    def viewer_setup(self):
        # 定义查看器设置的方法
        self.viewer.cam.trackbodyid = 2  # 设置摄像机跟踪的身体ID
        self.viewer.cam.distance = self.model.stat.extent * 0.75  # 设置摄像机距离
        self.viewer.cam.lookat[2] = 1.15  # 设置摄像机的观察点
        self.viewer.cam.elevation = -20  # 设置摄像机的仰角

    def set(self, state):
        # 定义设置状态的方法
        qpos_dim = self.sim.data.qpos.size  # 获取关节位置的维度
        qpos = state[:qpos_dim]  # 从状态中提取关节位置
        qvel = state[qpos_dim:]  # 从状态中提取关节速度
        self.set_state(qpos, qvel)  # 设置状态
        return self._get_obs()  # 返回观测

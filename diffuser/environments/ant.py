import os  # 导入os模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数值计算
from gym import utils  # 从Gym库中导入utils模块
from gym.envs.mujoco import mujoco_env  # 从Gym库中导入mujoco_env模块

'''
    qpos : 15  # 关节位置的维度为15
    qvel : 14  # 关节速度的维度为14
    0-2: root x, y, z  # 根部的x, y, z坐标
    3-7: root quat  # 根部的四元数
    7  : front L hip  # 前左髋关节
    8  : front L ankle  # 前左踝关节
    9  : front R hip  # 前右髋关节
    10 : front R ankle  # 前右踝关节
    11 : back  L hip  # 后左髋关节
    12 : back  L ankle  # 后左踝关节
    13 : back  R hip  # 后右髋关节
    14 : back  R ankle  # 后右踝关节
'''

class AntFullObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # 定义AntFullObsEnv类，继承自MujocoEnv和EzPickle

    def __init__(self):
        # 初始化函数
        asset_path = os.path.join(
            os.path.dirname(__file__), 'assets/ant.xml')  # 获取蚂蚁模型的路径
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5)  # 初始化MujocoEnv，帧跳跃为5
        utils.EzPickle.__init__(self)  # 初始化EzPickle

    def step(self, a):
        # 定义环境在执行一个动作后的变化
        xposbefore = self.get_body_com("torso")[0]  # 获取动作前躯干的x坐标
        self.do_simulation(a, self.frame_skip)  # 执行动作
        xposafter = self.get_body_com("torso")[0]  # 获取动作后躯干的x坐标
        forward_reward = (xposafter - xposbefore) / self.dt  # 计算前进奖励
        ctrl_cost = 0.5 * np.square(a).sum()  # 计算控制成本
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )  # 计算接触成本
        survive_reward = 1.0  # 生存奖励
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward  # 总奖励
        state = self.state_vector()  # 获取状态向量
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0  # 判断是否未完成
        done = not notdone  # 判断是否完成
        ob = self._get_obs()  # 获取观测
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )  # 返回观测、奖励、完成标志和奖励字典

    def _get_obs(self):
        # 定义获取观测的方法
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],  # 关节位置（去掉根部x, y）
                self.sim.data.qvel.flat,  # 关节速度
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,  # 接触力，裁剪到[-1, 1]
            ]
        )  # 返回观测

    def reset_model(self):
        # 定义重置模型的方法
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )  # 初始化关节位置，添加随机扰动
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1  # 初始化关节速度，添加随机扰动
        self.set_state(qpos, qvel)  # 设置状态
        return self._get_obs()  # 返回观测

    def viewer_setup(self):
        # 定义查看器设置的方法
        self.viewer.cam.distance = self.model.stat.extent * 0.5  # 设置摄像机距离

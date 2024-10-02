import os  # 导入操作系统模块，用于文件操作
import json  # 导入 JSON 模块，用于处理 JSON 数据

class Logger:  # 定义一个日志记录器类

    def __init__(self, renderer, logpath, vis_freq=10, max_render=8):  # 初始化函数
        self.renderer = renderer  # 保存渲染器对象
        self.savepath = logpath  # 保存日志保存路径
        self.vis_freq = vis_freq  # 保存可视化频率
        self.max_render = max_render  # 保存最大渲染帧数

    def log(self, t, samples, state, rollout=None):  # 定义日志记录函数
        if t % self.vis_freq != 0:  # 如果当前步数不是可视化频率的倍数，则直接返回
            return

        ## render image of plans  # 注释说明：渲染计划图像
        self.renderer.composite(  # 使用渲染器绘制图像
            os.path.join(self.savepath, f'{t}.png'),  # 指定图像保存路径
            samples.observations,  # 传入观测值作为图像数据
        )

        ## render video of plans  # 注释说明：渲染计划视频
        self.renderer.render_plan(  # 使用渲染器绘制视频
            os.path.join(self.savepath, f'{t}_plan.mp4'),  # 指定视频保存路径
            samples.actions[:self.max_render],  # 传入动作序列作为视频数据
            samples.observations[:self.max_render],  # 传入观测值序列作为视频数据
            state,  # 传入状态信息作为视频数据
        )

        if rollout is not None:  # 如果存在轨迹数据
            ## render video of rollout thus far  # 注释说明：渲染当前轨迹视频
            self.renderer.render_rollout(  # 使用渲染器绘制视频
                os.path.join(self.savepath, f'rollout.mp4'),  # 指定视频保存路径
                rollout,  # 传入轨迹数据作为视频数据
                fps=80,  # 指定视频帧率
            )

    def finish(self, t, score, total_reward, terminal, diffusion_experiment, value_experiment):  # 定义日志结束函数
        json_path = os.path.join(self.savepath, 'rollout.json')  # 指定 JSON 文件保存路径
        json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,  # 构建 JSON 数据
            'epoch_diffusion': diffusion_experiment.epoch, 'epoch_value': value_experiment.epoch}  # 构建 JSON 数据
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)  # 将 JSON 数据保存到文件
        print(f'[ utils/logger ] Saved log to {json_path}')  # 打印日志保存信息
# 导入必要的库
import os
import numpy as np
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm

# 尝试导入一些可能用于Colab环境的库
try:
    import io
    import base64
    from IPython.display import HTML
    from IPython import display as ipythondisplay
except:
    print('[ utils/colab ] Warning: not importing colab dependencies')

# 从当前包中导入一些自定义函数
from .serialization import mkdir
from .arrays import to_torch, to_np
from .video import save_video


# 定义运行扩散模型的函数
def run_diffusion(model, dataset, obs, n_samples=1, device='cuda:0', **diffusion_kwargs):
    # 对观察值进行归一化处理
    obs = dataset.normalizer.normalize(obs, 'observations')

    # 添加批次维度并重复以生成多个样本
    obs = obs[None].repeat(n_samples, axis=0)

    # 为模型准备条件输入
    conditions = {
        0: to_torch(obs, device=device)
    }

    # 使用模型生成条件样本
    samples, diffusion = model.conditional_sample(conditions,
                                                  return_diffusion=True, verbose=False, **diffusion_kwargs)

    # 将扩散结果转换为NumPy数组
    diffusion = to_np(diffusion)

    # 提取观察值
    normed_observations = diffusion[:, :, :, dataset.action_dim:]

    # 对模型输出的观察值样本进行反归一化
    observations = dataset.normalizer.unnormalize(normed_observations, 'observations')

    # 重新排列维度
    observations = einops.rearrange(observations,
                                    'batch steps horizon dim -> steps batch horizon dim')

    return observations


# 定义显示扩散过程的函数
def show_diffusion(renderer, observations, n_repeat=100, substep=1, filename='diffusion.mp4',
                   savebase='/content/videos'):
    # 创建保存目录
    mkdir(savebase)
    savepath = os.path.join(savebase, filename)

    # 对观察值进行子采样
    subsampled = observations[::substep]

    # 渲染每一步的图像
    images = []
    for t in tqdm(range(len(subsampled))):
        observation = subsampled[t]
        img = renderer.composite(None, observation)
        images.append(img)
    images = np.stack(images, axis=0)

    # 在视频末尾添加暂停效果
    images = np.concatenate([
        images,
        images[-1:].repeat(n_repeat, axis=0)
    ], axis=0)

    # 保存并显示视频
    save_video(savepath, images)
    show_video(savepath)


# 定义显示单个样本的函数
def show_sample(renderer, observations, filename='sample.mp4', savebase='/content/videos'):
    # 创建保存目录
    mkdir(savebase)
    savepath = os.path.join(savebase, filename)

    # 渲染每个观察值序列
    images = []
    for rollout in observations:
        img = renderer._renders(rollout, partial=True)
        images.append(img)

    # 将多个序列的图像水平拼接
    images = np.concatenate(images, axis=2)

    # 保存并显示视频
    save_video(savepath, images)
    show_video(savepath, height=200)


# 定义显示多个样本的函数
def show_samples(renderer, observations_l, figsize=12):
    images = []
    for observations in observations_l:
        path = observations[-1]
        img = renderer.composite(None, path)
        images.append(img)
    images = np.concatenate(images, axis=0)

    # 显示图像
    plt.imshow(images)
    plt.axis('off')
    plt.gcf().set_size_inches(figsize, figsize)


# 定义在Jupyter notebook中显示视频的函数
def show_video(path, height=400):
    video = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: {0}px;">
                <source src="data:video/mp4;base64,{1}" type="video/mp4" />
             </video>'''.format(height, encoded.decode('ascii'))))
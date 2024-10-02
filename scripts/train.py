import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#----------------------------------- 设置 -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'  # 设置数据集名称
    config: str = 'config.locomotion'  # 设置配置文件名称

args = Parser().parse_args('diffusion')  # 解析命令行参数

#-----------------------------------------------------------------------------#
#---------------------------------- 数据集 ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),  # 设置数据集配置保存路径
    env=args.dataset,  # 设置环境名称
    horizon=args.horizon,  # 设置时间步长
    normalizer=args.normalizer,  # 设置归一化方法
    preprocess_fns=args.preprocess_fns,  # 设置预处理函数
    use_padding=args.use_padding,  # 是否使用填充
    max_path_length=args.max_path_length,  # 设置最大路径长度
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),  # 设置渲染配置保存路径
    env=args.dataset,  # 设置环境名称
)

dataset = dataset_config()  # 创建数据集实例
renderer = render_config()  # 创建渲染器实例

observation_dim = dataset.observation_dim  # 获取观察空间维度
action_dim = dataset.action_dim  # 获取动作空间维度

#-----------------------------------------------------------------------------#
#------------------------------ 模型和训练器 ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),  # 设置模型配置保存路径
    horizon=args.horizon,  # 设置时间步长
    transition_dim=observation_dim + action_dim,  # 设置转移维度
    cond_dim=observation_dim,  # 设置条件维度
    dim_mults=args.dim_mults,  # 设置维度乘数
    attention=args.attention,  # 设置注意力机制
    device=args.device,  # 设置设备
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),  # 设置扩散模型配置保存路径
    horizon=args.horizon,  # 设置时间步长
    observation_dim=observation_dim,  # 设置观察空间维度
    action_dim=action_dim,  # 设置动作空间维度
    n_timesteps=args.n_diffusion_steps,  # 设置扩散步数
    loss_type=args.loss_type,  # 设置损失类型
    clip_denoised=args.clip_denoised,  # 是否裁剪去噪结果
    predict_epsilon=args.predict_epsilon,  # 是否预测噪声
    ## 损失权重设置
    action_weight=args.action_weight,  # 设置动作权重
    loss_weights=args.loss_weights,  # 设置损失权重
    loss_discount=args.loss_discount,  # 设置损失折扣
    device=args.device,  # 设置设备
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),  # 设置训练器配置保存路径
    train_batch_size=args.batch_size,  # 设置训练批次大小
    train_lr=args.learning_rate,  # 设置学习率
    gradient_accumulate_every=args.gradient_accumulate_every,  # 设置梯度累积步数
    ema_decay=args.ema_decay,  # 设置EMA衰减率
    sample_freq=args.sample_freq,  # 设置采样频率
    save_freq=args.save_freq,  # 设置保存频率
    label_freq=int(args.n_train_steps // args.n_saves),  # 设置标签频率
    save_parallel=args.save_parallel,  # 是否并行保存
    results_folder=args.savepath,  # 设置结果保存文件夹
    bucket=args.bucket,  # 设置桶
    n_reference=args.n_reference,  # 设置参考数量
)

#-----------------------------------------------------------------------------#
#-------------------------------- 实例化 --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()  # 创建模型实例

diffusion = diffusion_config(model)  # 创建扩散模型实例

trainer = trainer_config(diffusion, dataset, renderer)  # 创建训练器实例

#-----------------------------------------------------------------------------#
#------------------------ 测试前向传播和反向传播 -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)  # 报告模型参数

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])  # 将数据集的第一个样本转换为批次
loss, _ = diffusion.loss(*batch)  # 计算损失
loss.backward()  # 反向传播
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- 主循环 ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)  # 计算总轮数

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)  # 训练一个轮次

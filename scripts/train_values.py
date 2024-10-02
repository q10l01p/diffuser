import diffuser.utils as utils
import pdb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # 设置默认数据集名称
    dataset: str = 'walker2d-medium-replay-v2'
    # 设置默认配置名称
    config: str = 'config.locomotion'

# 实例化解析器并解析命令行参数
args = Parser().parse_args('values')

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

# 配置数据集加载器
dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    ## value-specific kwargs
    # 设置价值相关的参数
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
)

# 配置渲染器
render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

# 实例化数据集加载器和渲染器
dataset = dataset_config()
renderer = render_config()

# 获取观测空间维度和动作空间维度
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

# 配置模型
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

# 配置扩散模型
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    device=args.device,
)

# 配置训练器
trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

# 实例化模型
model = model_config()

# 实例化扩散模型
diffusion = diffusion_config(model)

# 实例化训练器
trainer = trainer_config(diffusion, dataset, renderer)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# 测试前向和反向传播
print('Testing forward...', end=' ', flush=True)
# 从数据集中获取一个批次的数据
batch = utils.batchify(dataset[0])

# 计算损失并进行反向传播
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

# 计算训练轮数
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

# 开始训练循环
for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    # 进行训练
    trainer.train(n_train_steps=args.n_steps_per_epoch)

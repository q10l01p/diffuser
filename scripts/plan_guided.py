import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # 定义默认数据集名称
    dataset: str = 'walker2d-medium-replay-v2'
    # 定义默认配置名称
    config: str = 'config.locomotion'

# 实例化解析器并解析命令行参数
args = Parser().parse_args('plan')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
# 从磁盘加载扩散模型和价值函数
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
# 确保扩散模型和价值函数兼容
utils.check_compatibility(diffusion_experiment, value_experiment)

# 获取扩散模型、数据集和渲染器
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
# 初始化价值引导器
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

# 初始化日志记录器
logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
# 策略是无条件扩散模型和价值引导器的包装器
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    # 采样参数
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

# 实例化日志记录器和策略
logger = logger_config()
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

# 获取环境
env = dataset.env
# 重置环境
observation = env.reset()

## observations for rendering
# 渲染的观测值
rollout = [observation.copy()]

# 初始化总奖励
total_reward = 0
# 主循环
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    # 保存状态用于渲染
    state = env.state_vector().copy()

    ## format current observation for conditioning
    # 格式化当前观测值用于条件化
    conditions = {0: observation}
    # 使用策略生成动作和样本
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    # 在环境中执行动作
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    # 打印奖励和分数
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    ## update rollout observations
    # 更新回滚观测值
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    # 每 `args.vis_freq` 步渲染一次
    logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation

## write results to json file at `args.savepath`
# 将结果写入 `args.savepath` 的 JSON 文件
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)

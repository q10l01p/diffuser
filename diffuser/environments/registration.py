import gym  # 导入Gym库

ENVIRONMENT_SPECS = (
    # 定义环境规格的元组
    {
        'id': 'HopperFullObs-v2',  # 环境的ID
        'entry_point': ('diffuser.environments.hopper:HopperFullObsEnv'),  # 入口点，指向环境类
    },
    {
        'id': 'HalfCheetahFullObs-v2',  # 环境的ID
        'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahFullObsEnv'),  # 入口点，指向环境类
    },
    {
        'id': 'Walker2dFullObs-v2',  # 环境的ID
        'entry_point': ('diffuser.environments.walker2d:Walker2dFullObsEnv'),  # 入口点，指向环境类
    },
    {
        'id': 'AntFullObs-v2',  # 环境的ID
        'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),  # 入口点，指向环境类
    },
)

def register_environments():
    # 定义注册环境的方法
    try:
        for environment in ENVIRONMENT_SPECS:
            # 遍历环境规格
            gym.register(**environment)  # 使用解包语法注册环境

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in ENVIRONMENT_SPECS)  # 提取所有环境的ID，生成元组

        return gym_ids  # 返回环境ID的元组
    except:
        # 如果发生异常
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')  # 打印警告信息
        return tuple()  # 返回空元组

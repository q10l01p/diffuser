import numpy as np  # 导入NumPy库，用于数值计算
import scipy.interpolate as interpolate  # 导入SciPy库的插值模块，用于插值计算
import pdb  # 导入Python调试器模块

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']  # 定义点质量数据集的关键字段

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:
    # 数据集归一化器类

    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths)  # 将数据集展平

        self.observation_dim = dataset['observations'].shape[1]  # 获取观察值的维度
        self.action_dim = dataset['actions'].shape[1]  # 获取动作的维度

        if type(normalizer) == str:  # 如果normalizer是字符串类型
            normalizer = eval(normalizer)  # 动态评估字符串为对象

        self.normalizers = {}  # 初始化归一化器字典
        for key, val in dataset.items():  # 遍历数据集中的每个键值对
            try:
                self.normalizers[key] = normalizer(val)  # 创建归一化器
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')  # 打印跳过的信息

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():  # 遍历归一化器
            string += f'{key}: {normalizer}]\n'  # 添加归一化器的字符串表示
        return string  # 返回归一化器的字符串表示

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)  # 调用normalize方法

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)  # 调用特定键的归一化器的normalize方法

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)  # 调用特定键的归一化器的unnormalize方法

    def get_field_normalizers(self):
        return self.normalizers  # 返回所有字段的归一化器

def flatten(dataset, path_lengths):
    '''
        将数据集从 { key: [ n_episodes x max_path_length x dim ] }
        展平到 { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}  # 初始化展平后的数据集
    for key, xs in dataset.items():  # 遍历数据集中的每个键值对
        assert len(xs) == len(path_lengths)  # 确保数据集长度与路径长度一致
        flattened[key] = np.concatenate([  # 将数据沿第0维连接
            x[:length]  # 取每个路径的有效长度
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened  # 返回展平后的数据集

#-----------------------------------------------------------------------------#
#------------------------------- @TODO: remove? ------------------------------#
#-----------------------------------------------------------------------------#

class PointMassDatasetNormalizer(DatasetNormalizer):
    # 点质量数据集归一化器类

    def __init__(self, preprocess_fns, dataset, normalizer, keys=POINTMASS_KEYS):

        reshaped = {}  # 初始化重塑后的数据集
        for key, val in dataset.items():  # 遍历数据集中的每个键值对
            dim = val.shape[-1]  # 获取最后一维的大小
            reshaped[key] = val.reshape(-1, dim)  # 重塑数据

        self.observation_dim = reshaped['observations'].shape[1]  # 获取观察值的维度
        self.action_dim = reshaped['actions'].shape[1]  # 获取动作的维度

        if type(normalizer) == str:  # 如果normalizer是字符串类型
            normalizer = eval(normalizer)  # 动态评估字符串为对象

        self.normalizers = {
            key: normalizer(reshaped[key])  # 创建归一化器
            for key in keys
        }

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        父类，通过定义`normalize`和`unnormalize`方法进行子类化
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)  # 将数据类型转换为float32
        self.mins = X.min(axis=0)  # 计算每个维度的最小值
        self.maxs = X.max(axis=0)  # 计算每个维度的最大值

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)  # 调用normalize方法

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()  # 未实现normalize方法

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()  # 未实现unnormalize方法


class DebugNormalizer(Normalizer):
    '''
        恒等函数
    '''

    def normalize(self, x, *args, **kwargs):
        return x  # 返回输入值

    def unnormalize(self, x, *args, **kwargs):
        return x  # 返回输入值


class GaussianNormalizer(Normalizer):
    '''
        归一化为零均值和单位方差
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.means = self.X.mean(axis=0)  # 计算均值
        self.stds = self.X.std(axis=0)  # 计算标准差
        self.z = 1  # 初始化缩放因子

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / self.stds  # 归一化数据

    def unnormalize(self, x):
        return x * self.stds + self.means  # 反归一化数据


class LimitsNormalizer(Normalizer):
    '''
        将 [ xmin, xmax ] 映射到 [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class SafeLimitsNormalizer(LimitsNormalizer):
    '''
        类似于LimitsNormalizer，但可以处理某维度为常数的数据
    '''

    def __init__(self, *args, eps=1, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        for i in range(len(self.mins)):  # 遍历每个维度
            if self.mins[i] == self.maxs[i]:  # 如果最大值等于最小值
                print(f'''
                    [ utils/normalization ] Constant data in dimension {i} | '''
                    f'''max = min = {self.maxs[i]}'''
                )
                self.mins -= eps  # 调整最小值
                self.maxs += eps  # 调整最大值

#-----------------------------------------------------------------------------#
#------------------------------- CDF normalizer ------------------------------#
#-----------------------------------------------------------------------------#

class CDFNormalizer(Normalizer):
    '''
        通过边际CDF将训练数据变为均匀分布（在每个维度上）
    '''

    def __init__(self, X):
        super().__init__(atleast_2d(X))  # 调用父类构造函数
        self.dim = self.X.shape[1]  # 获取数据的维度
        self.cdfs = [
            CDFNormalizer1d(self.X[:, i])  # 为每个维度创建CDF归一化器
            for i in range(self.dim)
        ]

    def __repr__(self):
        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape  # 获取输入数据的形状
        ## reshape to 2d
        x = x.reshape(-1, self.dim)  # 重塑数据为二维
        out = np.zeros_like(x)  # 初始化输出数组
        for i, cdf in enumerate(self.cdfs):  # 遍历每个CDF归一化器
            fn = getattr(cdf, fn_name)  # 获取归一化器的方法
            out[:, i] = fn(x[:, i])  # 调用方法并存储结果
        return out.reshape(shape)  # 重塑输出为原始形状

    def normalize(self, x):
        return self.wrap('normalize', x)  # 调用wrap方法进行归一化

    def unnormalize(self, x):
        return self.wrap('unnormalize', x)  # 调用wrap方法进行反归一化

class CDFNormalizer1d:
    '''
        单维度的CDF归一化器
    '''

    def __init__(self, X):
        assert X.ndim == 1  # 确保输入是一维数据
        self.X = X.astype(np.float32)  # 将数据类型转换为float32
        quantiles, cumprob = empirical_cdf(self.X)  # 计算经验CDF
        self.fn = interpolate.interp1d(quantiles, cumprob)  # 创建插值函数
        self.inv = interpolate.interp1d(cumprob, quantiles)  # 创建反插值函数

        self.xmin, self.xmax = quantiles.min(), quantiles.max()  # 获取最小和最大分位数
        self.ymin, self.ymax = cumprob.min(), cumprob.max()  # 获取最小和最大累积概率

    def __repr__(self):
        return (
            f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}'
        )

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)  # 限制x在xmin和xmax之间
        ## [ 0, 1 ]
        y = self.fn(x)  # 计算插值
        ## [ -1, 1 ]
        y = 2 * y - 1  # 映射到[-1, 1]
        return y

    def unnormalize(self, x, eps=1e-4):
        '''
            X : [ -1, 1 ]
        '''
        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
                f'''[{x.min()}, {x.max()}] | '''
                f'''x : [{self.xmin}, {self.xmax}] | '''
                f'''y: [{self.ymin}, {self.ymax}]'''
            )

        x = np.clip(x, self.ymin, self.ymax)  # 限制x在ymin和ymax之间

        y = self.inv(x)  # 计算反插值
        return y

def empirical_cdf(sample):
    ## https://stackoverflow.com/a/33346366

    # 找到唯一值及其对应的计数
    quantiles, counts = np.unique(sample, return_counts=True)

    # 计算计数的累积和并除以样本大小，以获得0到1之间的累积概率
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def atleast_2d(x):
    if x.ndim < 2:  # 如果x的维度小于2
        return x.reshape(-1, 1)  # 重塑为二维数组
    return x  # 返回原始数组

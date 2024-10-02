import numpy as np

def atleast_2d(x):
    # 确保输入数组至少是二维的
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        # 初始化replay buffer
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=np.int),
        }
        self._count = 0  # 记录当前存储的episode数量
        self.max_n_episodes = max_n_episodes  # 最大episode数量
        self.max_path_length = max_path_length  # 每个episode的最大长度
        self.termination_penalty = termination_penalty  # 提前终止的惩罚

    def __repr__(self):
        # 定义对象的字符串表示
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        # 获取指定key的数据
        return self._dict[key]

    def __setitem__(self, key, val):
        # 设置指定key的数据
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        # 返回当前存储的episode数量
        return self._count

    @property
    def n_steps(self):
        # 返回所有episode的总步数
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        # 如果还没有keys属性,则根据path添加
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        # 将字典中的项添加为对象的属性
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        # 返回除'path_lengths'外的所有项
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        # 为新的key分配内存
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)

    def add_path(self, path):
        # 添加一个新的轨迹
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length

        # 如果是第一个添加的轨迹,设置keys
        self._add_keys(path)

        # 添加轨迹中的所有tracked keys
        for key in self.keys:
            array = atleast_2d(path[key])
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

        # 对提前终止的episode进行惩罚
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty

        # 记录轨迹长度
        self._dict['path_lengths'][self._count] = path_length

        # 增加episode计数
        self._count += 1

    def truncate_path(self, path_ind, step):
        # 截断指定轨迹到指定步数
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        # 完成buffer的构建,移除多余的空间
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

import os  # 导入操作系统模块，用于文件操作
import collections  # 导入集合模块，用于使用字典类型
import importlib  # 导入导入模块，用于动态导入类
import pickle  # 导入序列化模块，用于保存配置对象

def import_class(_class):  # 定义一个函数，用于动态导入类
    if type(_class) is not str:  # 如果输入的类不是字符串，则直接返回
        return _class
    ## 'diffusion' on standard installs  # 注释说明：在标准安装中，仓库名称为 'diffusion'
    repo_name = __name__.split('.')[0]  # 获取仓库名称，例如 'diffusion'
    ## eg, 'utils'  # 注释说明：模块名称，例如 'utils'
    module_name = '.'.join(_class.split('.')[:-1])  # 获取模块名称，例如 'diffusion.utils'
    ## eg, 'Renderer'  # 注释说明：类名称，例如 'Renderer'
    class_name = _class.split('.')[-1]  # 获取类名称，例如 'diffusion.utils.Renderer'
    ## eg, 'diffusion.utils'  # 注释说明：导入模块的完整路径，例如 'diffusion.utils'
    module = importlib.import_module(f'{repo_name}.{module_name}')  # 动态导入模块，例如 import diffusion.utils
    ## eg, diffusion.utils.Renderer  # 注释说明：获取导入模块中的类
    _class = getattr(module, class_name)  # 获取类，例如 diffusion.utils.Renderer
    print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')  # 打印导入的信息
    return _class  # 返回导入的类

class Config(collections.Mapping):  # 定义一个配置类，继承自 collections.Mapping，使其具有字典的功能

    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):  # 初始化函数
        self._class = import_class(_class)  # 导入配置类
        self._device = device  # 保存设备信息
        self._dict = {}  # 初始化一个空字典，用于存储配置参数

        for key, val in kwargs.items():  # 遍历传入的配置参数
            self._dict[key] = val  # 将参数添加到字典中

        if verbose:  # 如果 verbose 为 True
            print(self)  # 打印配置信息

        if savepath is not None:  # 如果指定了保存路径
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath  # 处理保存路径
            pickle.dump(self, open(savepath, 'wb'))  # 保存配置对象到指定路径
            print(f'[ utils/config ] Saved config to: {savepath}\n')  # 打印保存信息

    def __repr__(self):  # 定义字符串表示方法，用于打印配置信息
        string = f'\n[utils/config ] Config: {self._class}\n'  # 构建字符串信息
        for key in sorted(self._dict.keys()):  # 遍历配置参数
            val = self._dict[key]  # 获取参数值
            string += f'    {key}: {val}\n'  # 将参数信息添加到字符串中
        return string  # 返回字符串信息

    def __iter__(self):  # 定义迭代方法，使其可以被循环遍历
        return iter(self._dict)  # 返回字典的迭代器

    def __getitem__(self, item):  # 定义取值方法，使其可以通过索引访问配置参数
        return self._dict[item]  # 返回字典中对应索引的值

    def __len__(self):  # 定义长度方法，获取配置参数的个数
        return len(self._dict)  # 返回字典的长度

    def __getattr__(self, attr):  # 定义属性访问方法，用于获取配置参数
        if attr == '_dict' and '_dict' not in vars(self):  # 如果属性为 '_dict' 且字典未初始化
            self._dict = {}  # 初始化字典
            return self._dict  # 返回字典
        try:  # 尝试获取配置参数
            return self._dict[attr]  # 返回对应属性的值
        except KeyError:  # 如果属性不存在
            raise AttributeError(attr)  # 抛出属性错误

    def __call__(self, *args, **kwargs):  # 定义调用方法，用于创建配置类的实例
        instance = self._class(*args, **kwargs, **self._dict)  # 创建实例，传入配置参数
        if self._device:  # 如果指定了设备
            instance = instance.to(self._device)  # 将实例移动到指定设备
        return instance  # 返回实例
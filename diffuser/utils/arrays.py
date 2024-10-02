import collections  # 导入 collections 模块，用于创建特殊的容器数据类型
import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库
import pdb  # 导入 Python 调试器

# 定义数据类型和设备
DTYPE = torch.float  # 默认数据类型为 float
DEVICE = 'cuda:0'  # 默认设备为 GPU 0

#-----------------------------------------------------------------------------#
#------------------------------ numpy <--> torch -----------------------------#
#-----------------------------------------------------------------------------#

def to_np(x):
    """
    将 PyTorch 张量转换为 NumPy 数组。

    参数：
        x: 输入张量

    返回：
        转换后的 NumPy 数组
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()  # 从 GPU 转移到 CPU 并转换为 NumPy 数组
    return x

def to_torch(x, dtype=None, device=None):
    """
    将 NumPy 数组或其他数据结构转换为 PyTorch 张量。

    参数：
        x: 输入数据
        dtype: 指定数据类型
        device: 指定设备

    返回：
        转换后的 PyTorch 张量
    """
    dtype = dtype or DTYPE  # 如果未指定数据类型，则使用默认类型
    device = device or DEVICE  # 如果未指定设备，则使用默认设备
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}  # 递归处理字典
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)  # 转换现有张量的设备和类型
    return torch.tensor(x, dtype=dtype, device=device)  # 转换为 PyTorch 张量

def to_device(x, device=DEVICE):
    """
    将数据转换为指定设备的张量。

    参数：
        x: 输入数据
        device: 指定设备

    返回：
        转换后的数据
    """
    if torch.is_tensor(x):
        return x.to(device)  # 直接将张量转移到指定设备
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}  # 递归处理字典
    else:
        raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')  # 抛出错误

def batchify(batch):
    """
    将单个数据集项转换为适合模型输入的批次。

    1) 将 NumPy 数组转换为 PyTorch 张量
    2) 确保所有数据都有批处理维度

    参数：
        batch: 输入的单个数据集项

    返回：
        转换后的批次数据
    """
    fn = lambda x: to_torch(x[None])  # 将输入添加新的维度

    batched_vals = []  # 存储批处理值
    for field in batch._fields:  # 遍历数据集项的字段
        val = getattr(batch, field)  # 获取字段值
        val = apply_dict(fn, val) if type(val) is dict else fn(val)  # 处理字典或直接调用函数
        batched_vals.append(val)  # 添加到批处理值列表
    return type(batch)(*batched_vals)  # 返回新的批处理实例

def apply_dict(fn, d, *args, **kwargs):
    """
    对字典中的每个值应用指定的函数。

    参数：
        fn: 应用的函数
        d: 输入字典
        *args: 传递给函数的其他参数
        **kwargs: 传递给函数的关键字参数

    返回：
        处理后的字典
    """
    return {
        k: fn(v, *args, **kwargs)  # 对字典中的每个值应用函数
        for k, v in d.items()
    }

def normalize(x):
    """
    将数据 x 归一化到 [0, 1] 范围。

    参数：
        x: 输入数据

    返回：
        归一化后的数据
    """
    x = x - x.min()  # 减去最小值
    x = x / x.max()  # 除以最大值
    return x

def to_img(x):
    """
    将张量转换为图像格式。

    参数：
        x: 输入张量

    返回：
        转换后的图像数组
    """
    normalized = normalize(x)  # 归一化
    array = to_np(normalized)  # 转换为 NumPy 数组
    array = np.transpose(array, (1, 2, 0))  # 转置数组
    return (array * 255).astype(np.uint8)  # 转换为 8 位无符号整数

def set_device(device):
    """
    设置默认设备。

    参数：
        device: 要设置的设备
    """
    global DEVICE  # 声明 DEVICE 为全局变量
    DEVICE = device
    if 'cuda' in device:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # 设置默认张量类型为 CUDA

def batch_to_device(batch, device='cuda:0'):
    """
    将批次数据转换到指定设备。

    参数：
        batch: 输入批次
        device: 指定设备

    返回：
        转换到指定设备的批次
    """
    vals = [
        to_device(getattr(batch, field), device)  # 将每个字段的值转换到设备
        for field in batch._fields
    ]
    return type(batch)(*vals)  # 返回新的批处理实例

def _to_str(num):
    """
    将数字转换为可读字符串格式。

    参数：
        num: 输入数字

    返回：
        格式化后的字符串
    """
    if num >= 1e6:
        return f'{(num/1e6):.2f} M'  # 大于百万的数以 M 表示
    else:
        return f'{(num/1e3):.2f} k'  # 大于千的数以 k 表示

#-----------------------------------------------------------------------------#
#----------------------------- parameter counting ----------------------------#
#-----------------------------------------------------------------------------#

def param_to_module(param):
    """
    从参数名称获取模块名称。

    参数：
        param: 参数名称

    返回：
        模块名称
    """
    module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]  # 反转字符串并提取模块名称
    return module_name

def report_parameters(model, topk=10):
    """
    报告模型参数的数量和详细信息。

    参数：
        model: 输入模型
        topk: 需要报告的前 k 个参数

    返回：
        总参数数量
    """
    counts = {k: p.numel() for k, p in model.named_parameters()}  # 获取每个参数的数量
    n_parameters = sum(counts.values())  # 计算总参数数量
    print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')  # 打印总参数数量

    modules = dict(model.named_modules())  # 获取模型模块
    sorted_keys = sorted(counts, key=lambda x: -counts[x])  # 按参数数量排序
    max_length = max([len(k) for k in sorted_keys])  # 找到最长的参数名称
    for i in range(topk):
        key = sorted_keys[i]  # 获取前 k 个参数
        count = counts[key]  # 获取参数数量
        module = param_to_module(key)  # 获取模块名称
        print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')  # 打印参数详细信息

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])  # 计算剩余参数数量
    print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')  # 打印剩余参数信息
    return n_parameters  # 返回总参数数量
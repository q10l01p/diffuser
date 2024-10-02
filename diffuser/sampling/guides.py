import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import pdb  # 导入 Python 调试器


class ValueGuide(nn.Module):
    """
    价值引导模块，用于计算价值函数的输出和梯度。
    """

    def __init__(self, model):
        """
        初始化 ValueGuide 模型。

        参数：
            model: 传入的模型实例
        """
        super().__init__()  # 调用父类的初始化方法
        self.model = model  # 保存传入的模型

    def forward(self, x, cond, t):
        """
        前向传播方法。

        参数：
            x: 输入数据
            cond: 条件信息
            t: 时间步

        返回：
            输出数据，去掉最后一维
        """
        output = self.model(x, cond, t)  # 使用模型进行前向传播
        return output.squeeze(dim=-1)  # 去掉最后一维

    def gradients(self, x, *args):
        """
        计算输入 x 的梯度。

        参数：
            x: 输入数据
            *args: 其他参数

        返回：
            y: 模型输出
            grad: 输入 x 的梯度
        """
        x.requires_grad_()  # 设置 x 需要计算梯度
        y = self(x, *args)  # 调用前向传播方法
        # 计算 y 相对于 x 的梯度
        grad = torch.autograd.grad([y.sum()], [x])[0]
        x.detach()  # 断开 x 的梯度计算
        return y, grad  # 返回模型输出和梯度
import torch  # 导入 PyTorch 库

from diffuser.models.helpers import (
    extract,  # 从模型中提取参数的辅助函数
    apply_conditioning,  # 应用条件信息的辅助函数
)


@torch.no_grad()  # 禁用梯度计算
def n_step_guided_p_sample(
    model,  # 模型实例
    x,  # 输入数据
    cond,  # 条件信息
    t,  # 时间步
    guide,  # 指导模型，用于计算梯度
    scale=0.001,  # 梯度缩放系数
    t_stopgrad=0,  # 停止梯度的时间步
    n_guide_steps=1,  # 指导步骤的数量
    scale_grad_by_std=True,  # 是否根据标准差缩放梯度
):
    # 从模型中提取对数方差，并进行裁剪
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    # 计算标准差
    model_std = torch.exp(0.5 * model_log_variance)
    # 计算方差
    model_var = torch.exp(model_log_variance)

    # 循环进行 n_guide_steps 次指导
    for _ in range(n_guide_steps):
        with torch.enable_grad():  # 启用梯度计算
            # 计算指导模型的输出和梯度
            y, grad = guide.gradients(x, cond, t)

        # 如果需要，根据标准差缩放梯度
        if scale_grad_by_std:
            grad = model_var * grad

        # 对于 t < t_stopgrad 的时间步，将梯度置为 0
        grad[t < t_stopgrad] = 0

        # 更新输入 x
        x = x + scale * grad
        # 应用条件信息
        x = apply_conditioning(x, cond, model.action_dim)

    # 从模型中计算均值、方差和对数方差
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # 当 t == 0 时，不添加噪声
    noise = torch.randn_like(x)  # 生成与 x 相同形状的随机噪声
    noise[t == 0] = 0  # 在 t == 0 时噪声为 0

    # 返回模型均值加上噪声，和指导模型的输出 y
    return model_mean + model_std * noise, y

# 导入必要的模块
import shlex  # 用于将字符串分割成shell命令的参数列表
import subprocess  # 用于执行系统命令
import pdb  # Python调试器,用于调试代码


# 定义同步日志的函数
def sync_logs(logdir, bucket, background=False):
    # 移除Google Cloud存储中的'logs'前缀
    # 通过分割logdir路径并取最后一部分,然后在前面加上'logs'
    destination = 'logs' + logdir.split('logs')[-1]

    # 调用upload_blob函数上传日志
    upload_blob(logdir, destination, bucket, background)


# 定义上传文件到Google Cloud Storage的函数
def upload_blob(source, destination, bucket, background):
    # 构建gsutil命令
    # -m: 启用并行传输
    # -o GSUtil:parallel_composite_upload_threshold=150M: 设置并行复合上传阈值为150MB
    # rsync -r: 递归同步目录
    command = f'gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M rsync -r {source} {bucket}/{destination}'

    # 打印同步命令
    print(f'[ utils/cloud ] Syncing bucket: {command}')

    # 使用shlex.split()将命令字符串分割成参数列表
    command = shlex.split(command)

    # 根据background参数决定是否在后台运行命令
    if background:
        # 在后台运行命令
        subprocess.Popen(command)
    else:
        # 在前台运行命令并等待其完成
        subprocess.call(command)

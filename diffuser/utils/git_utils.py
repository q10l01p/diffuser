import os  # 导入操作系统模块，用于文件操作
import git  # 导入 git 模块，用于获取 git 仓库信息
import pdb  # 导入 pdb 模块，用于调试

PROJECT_PATH = os.path.dirname(  # 定义项目路径
    os.path.realpath(os.path.join(__file__, '..', '..')))  # 获取当前文件所在目录的父目录的父目录

def get_repo(path=PROJECT_PATH, search_parent_directories=True):  # 定义函数获取 git 仓库对象
    repo = git.Repo(  # 初始化 git 仓库对象
        path, search_parent_directories=search_parent_directories)  # 指定仓库路径和是否搜索父目录
    return repo  # 返回 git 仓库对象

def get_git_rev(*args, **kwargs):  # 定义函数获取 git 版本号
    try:  # 使用 try-except 语句处理异常
        repo = get_repo(*args, **kwargs)  # 获取 git 仓库对象
        if repo.head.is_detached:  # 判断是否处于分离头指针状态
            git_rev = repo.head.object.name_rev  # 获取分离头指针的版本号
        else:  # 否则，获取当前分支的版本号
            git_rev = repo.active_branch.commit.name_rev  # 获取当前分支的版本号
    except:  # 处理异常
        git_rev = None  # 将版本号设置为 None

    return git_rev  # 返回版本号

def git_diff(*args, **kwargs):  # 定义函数获取 git 差异
    repo = get_repo(*args, **kwargs)  # 获取 git 仓库对象
    diff = repo.git.diff()  # 获取仓库差异
    return diff  # 返回差异

def save_git_diff(savepath, *args, **kwargs):  # 定义函数保存 git 差异
    diff = git_diff(*args, **kwargs)  # 获取 git 差异
    with open(savepath, 'w') as f:  # 打开文件写入模式
        f.write(diff)  # 将差异写入文件

if __name__ == '__main__':  # 定义主函数，当脚本直接运行时执行

    git_rev = get_git_rev()  # 获取 git 版本号
    print(git_rev)  # 打印版本号

    save_git_diff('diff_test.txt')  # 保存 git 差异到文件 'diff_test.txt'
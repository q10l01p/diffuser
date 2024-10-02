import os
import glob
import numpy as np
import json
import pdb

import diffuser.utils as utils

# 定义要分析的数据集列表
DATASETS = [
    f'{env}-{buffer}-v2'
    for env in ['hopper', 'walker2d', 'halfcheetah']
    for buffer in ['medium-replay', 'medium', 'medium-expert']
]

# 定义日志文件路径
LOGBASE = 'logs/pretrained/'
# 定义要分析的试验路径
TRIAL = '*'
# 定义要分析的实验名称
EXP_NAME = 'plans*/*'
# 设置是否打印详细信息
verbose = False


def load_results(paths):
    '''
        paths : path to directory containing experiment trials
    '''
    # 初始化一个列表用于存储分数
    scores = []
    # 遍历每个试验路径
    for i, path in enumerate(sorted(paths)):
        # 加载试验结果
        score = load_result(path)
        # 打印试验路径和分数
        if verbose: print(path, score)
        # 如果分数为空，则跳过该试验
        if score is None:
            # print(f'Skipping {path}')
            continue
        # 将分数添加到列表中
        scores.append(score)

        # 获取试验路径的最后一个部分
        suffix = path.split('/')[-1]
        # 打印试验路径的最后一个部分、路径和分数
        # print(suffix, path, score)

    # 如果分数列表不为空，则计算分数的均值和标准误
    if len(scores) > 0:
        mean = np.mean(scores)
    else:
        mean = np.nan

    if len(scores) > 1:
        err = np.std(scores) / np.sqrt(len(scores))
    else:
        err = 0
    # 返回分数的均值、标准误和所有分数
    return mean, err, scores


def load_result(path):
    '''
        path : path to experiment directory; expects `rollout.json` to be in directory
    '''
    # 获取试验结果文件的完整路径
    fullpath = os.path.join(path, 'rollout.json')

    # 如果结果文件不存在，则返回 None
    if not os.path.exists(fullpath):
        return None

    # 加载结果文件
    results = json.load(open(fullpath, 'rb'))
    # 获取分数并将其乘以 100
    score = results['score'] * 100
    # 返回分数
    return score


#######################
######## setup ########
#######################


if __name__ == '__main__':

    class Parser(utils.Parser):
        # 定义一个可选参数，用于指定要分析的数据集
        dataset: str = None

    # 实例化解析器并解析命令行参数
    args = Parser().parse_args()

    # 遍历每个数据集
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        # 获取数据集路径下所有实验路径
        subdirs = sorted(glob.glob(os.path.join(LOGBASE, dataset, EXP_NAME)))

        # 遍历每个实验路径
        for subdir in subdirs:
            # 获取实验路径的最后一个部分
            reldir = subdir.split('/')[-1]
            # 获取实验路径下所有试验路径
            paths = glob.glob(os.path.join(subdir, TRIAL))
            # 对试验路径进行排序
            paths = sorted(paths)

            # 加载试验结果
            mean, err, scores = load_results(paths)
            # 如果分数为空，则跳过该实验
            if np.isnan(mean):
                continue
            # 获取实验路径的父路径和名称
            path, name = os.path.split(subdir)
            # 打印实验结果
            print(f'{dataset.ljust(30)} | {name.ljust(50)} | {path.ljust(50)} | {len(scores)} scores \n    {mean:.1f} +/- {err:.2f}')
            # 打印所有分数
            if verbose:
                print(scores)
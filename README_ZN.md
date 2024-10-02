# 基于扩散模型的规划 &nbsp;&nbsp; [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing)

本项目涉及[使用扩散进行规划以实现灵活行为合成（Planning with Diffusion for Flexible Behavior Synthesis）](https://diffusion-planning.github.io/)中扩散模型的训练与可视化。

[main 分支](https://github.com/jannerm/diffuser/tree/main)包含在 D4RL 运动环境中训练扩散模型和通过基于值函数的引导采样进行规划的代码。
[kuka 分支](https://github.com/jannerm/diffuser/tree/kuka)包含积木堆叠实验。
[maze2d 分支](https://github.com/jannerm/diffuser/tree/maze2d)包含通过图像修复在 Maze2D 环境中实现目标到达的内容。

<p align="center">
    <img src="https://diffusion-planning.github.io/images/diffuser-card.png" width="60%" title="Diffuser 模型">
</p>

**更新**
- 2022年12月9日：Diffuser（RL 模型）已集成到 🤗 Diffusers（Hugging Face 扩散模型库）中！查看[这些文档](https://huggingface.co/docs/diffusers/using-diffusers/rl)了解如何使用他们的管道运行 Diffuser。
- 2022年10月17日：值函数缩放中的一个 bug 已在[此提交](https://github.com/jannerm/diffuser/commit/3d7361c2d028473b601cc04f5eecd019e14eb4eb)中修复。感谢 [Philemon Brakel](https://scholar.google.com/citations?user=Q6UMpRYAAAAJ&hl=en) 发现这个问题！

## 快速入门

使用 [scripts/diffuser-sample.ipynb](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing) 在浏览器中在线加载预训练的扩散模型并从中采样。

## 安装步骤

\textbf{请在命令行中依次执行以下命令：}
```
$ conda env create -f environment.yml
$ conda activate diffuser
$ pip install -e .
```

## 使用预训练模型

### 下载模型权重

使用以下命令下载预训练的扩散模型（diffusion model）和值函数（value function）：
```
$ ./scripts/download_pretrained.sh
```

此命令会下载并解压一个[压缩文件](https://drive.google.com/file/d/1wc1m4HLj7btaYDN8ogDIAV9QzgWEckGy/view?usp=share_link)，其中包含[此目录](https://drive.google.com/drive/folders/1ie6z3toz9OjcarJuwjQwXXzDwh1XnS02?usp=sharing)的内容，并将其解压到`logs/pretrained`目录下。解压后的模型文件按以下结构组织：
```
└── logs/pretrained
    ├── ${environment_1}
    │   ├── diffusion （扩散模型）
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       ├── sample-${epoch}-*.png
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   ├── values （值函数）
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   └── plans （规划结果）
    │       └── defaults
    │           ├── 0
    │           ├── 1
    │           ├── ...
    │           └── 149
    │
    ├── ${environment_2}
    │   └── ...
```

其中，`state_${epoch}.pt`文件包含网络权重，`config.pkl`文件包含相关类的实例化参数。
png文件包含扩散模型训练过程中不同时间点生成的样本。
在`plans`子文件夹中，存储了每个环境使用默认超参数进行的150次评估试验的结果。

<details>
<summary>要汇总`logs`文件夹中的评估结果，请运行以下命令：`python scripts/read_results.py`。（点击展开可查看从Google Drive下载的计划运行此命令后的输出结果。）
</summary>

```
hopper-medium-replay-v2        | defaults   | logs/pretrained/hopper-medium-replay-v2/plans      | 150 scores
    93.6 +/- 0.37
hopper-medium-v2               | defaults   | logs/pretrained/hopper-medium-v2/plans             | 150 scores
    74.3 +/- 1.36
hopper-medium-expert-v2        | defaults   | logs/pretrained/hopper-medium-expert-v2/plans      | 150 scores
    103.3 +/- 1.30
walker2d-medium-replay-v2      | defaults   | logs/pretrained/walker2d-medium-replay-v2/plans    | 150 scores
    70.6 +/- 1.60
walker2d-medium-v2             | defaults   | logs/pretrained/walker2d-medium-v2/plans           | 150 scores
    79.6 +/- 0.55
walker2d-medium-expert-v2      | defaults   | logs/pretrained/walker2d-medium-expert-v2/plans    | 150 scores
    106.9 +/- 0.24
halfcheetah-medium-replay-v2   | defaults   | logs/pretrained/halfcheetah-medium-replay-v2/plans | 150 scores
    37.7 +/- 0.45
halfcheetah-medium-v2          | defaults   | logs/pretrained/halfcheetah-medium-v2/plans        | 150 scores
    42.8 +/- 0.32
halfcheetah-medium-expert-v2   | defaults   | logs/pretrained/halfcheetah-medium-expert-v2/plans | 150 scores
    88.9 +/- 0.25
```
</details>

<details>
<summary>要创建论文中的离线强化学习结果表格，请运行`python plotting/table.py`。这将生成一个可以直接复制到Latex文档中的表格。（点击展开可查看表格的LaTeX源代码。）</summary>

```
\definecolor{tblue}{HTML}{1F77B4}
\definecolor{tred}{HTML}{FF6961}
\definecolor{tgreen}{HTML}{429E9D}
\definecolor{thighlight}{HTML}{000000}
\newcolumntype{P}{>{\raggedleft\arraybackslash}X}
\begin{table*}[hb!]
\centering
\small
\begin{tabularx}{\textwidth}{llPPPPPPPPr}
\toprule
\multicolumn{1}{r}{\bf \color{black} 数据集} & \multicolumn{1}{r}{\bf \color{black} 环境} & \multicolumn{1}{r}{\bf \color{black} BC} & \multicolumn{1}{r}{\bf \color{black} CQL} & \multicolumn{1}{r}{\bf \color{black} IQL} & \multicolumn{1}{r}{\bf \color{black} DT} & \multicolumn{1}{r}{\bf \color{black} TT} & \multicolumn{1}{r}{\bf \color{black} MOPO} & \multicolumn{1}{r}{\bf \color{black} MOReL} & \multicolumn{1}{r}{\bf \color{black} MBOP} & \multicolumn{1}{r}{\bf \color{black} Diffuser} \\ 
\midrule
Medium-Expert（中等专家） & HalfCheetah & $55.2$ & $91.6$ & $86.7$ & $86.8$ & $95.0$ & $63.3$ & $53.3$ & $\textbf{\color{thighlight}105.9}$ & $88.9$ \scriptsize{\raisebox{1pt}{$\pm 0.3$}} \\ 
Medium-Expert（中等专家） & Hopper & $52.5$ & $\textbf{\color{thighlight}105.4}$ & $91.5$ & $\textbf{\color{thighlight}107.6}$ & $\textbf{\color{thighlight}110.0}$ & $23.7$ & $\textbf{\color{thighlight}108.7}$ & $55.1$ & $103.3$ \scriptsize{\raisebox{1pt}{$\pm 1.3$}} \\ 
Medium-Expert（中等专家） & Walker2d & $\textbf{\color{thighlight}107.5}$ & $\textbf{\color{thighlight}108.8}$ & $\textbf{\color{thighlight}109.6}$ & $\textbf{\color{thighlight}108.1}$ & $101.9$ & $44.6$ & $95.6$ & $70.2$ & $\textbf{\color{thighlight}106.9}$ \scriptsize{\raisebox{1pt}{$\pm 0.2$}} \\ 
\midrule
Medium（中等） & HalfCheetah & $42.6$ & $44.0$ & $\textbf{\color{thighlight}47.4}$ & $42.6$ & $\textbf{\color{thighlight}46.9}$ & $42.3$ & $42.1$ & $44.6$ & $42.8$ \scriptsize{\raisebox{1pt}{$\pm 0.3$}} \\ 
Medium（中等） & Hopper & $52.9$ & $58.5$ & $66.3$ & $67.6$ & $61.1$ & $28.0$ & $\textbf{\color{thighlight}95.4}$ & $48.8$ & $74.3$ \scriptsize{\raisebox{1pt}{$\pm 1.4$}} \\ 
Medium（中等） & Walker2d & $75.3$ & $72.5$ & $\textbf{\color{thighlight}78.3}$ & $74.0$ & $\textbf{\color{thighlight}79.0}$ & $17.8$ & $\textbf{\color{thighlight}77.8}$ & $41.0$ & $\textbf{\color{thighlight}79.6}$ \scriptsize{\raisebox{1pt}{$\pm 0.55$}} \\ 
\midrule
Medium-Replay（中等重放） & HalfCheetah & $36.6$ & $45.5$ & $44.2$ & $36.6$ & $41.9$ & $\textbf{\color{thighlight}53.1}$ & $40.2$ & $42.3$ & $37.7$ \scriptsize{\raisebox{1pt}{$\pm 0.5$}} \\ 
Medium-Replay（中等重放） & Hopper & $18.1$ & $\textbf{\color{thighlight}95.0}$ & $\textbf{\color{thighlight}94.7}$ & $82.7$ & $\textbf{\color{thighlight}91.5}$ & $67.5$ & $\textbf{\color{thighlight}93.6}$ & $12.4$ & $\textbf{\color{thighlight}93.6}$ \scriptsize{\raisebox{1pt}{$\pm 0.4$}} \\ 
Medium-Replay（中等重放） & Walker2d & $26.0$ & $77.2$ & $73.9$ & $66.6$ & $\textbf{\color{thighlight}82.6}$ & $39.0$ & $49.8$ & $9.7$ & $70.6$ \scriptsize{\raisebox{1pt}{$\pm 1.6$}} \\ 
\midrule
\multicolumn{2}{c}{\bf 平均} & 51.9 & \textbf{\color{thighlight}77.6} & \textbf{\color{thighlight}77.0} & 74.7 & \textbf{\color{thighlight}78.9} & 42.1 & 72.9 & 47.8 & \textbf{\color{thighlight}77.5} \hspace{.6cm} \\ 
\bottomrule
\end{tabularx}
\vspace{-.0cm}
\caption{
}
\label{table:locomotion}
\end{table*}
```

![](https://github.com/diffusion-planning/diffusion-planning.github.io/blob/master/images/table.png)
</details>

### 规划

要使用引导采样（guided sampling）进行规划，请运行以下命令：
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained
```

`--logbase`标志指向[实验加载器](scripts/plan_guided.py#L22-L30)到包含预训练模型的文件夹。
你可以使用标志覆盖规划超参数，例如`--batch_size 8`，但默认超参数通常是一个很好的起点。

## 从头开始训练

1. 使用以下命令训练扩散模型（diffusion model）：
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

默认超参数列在[locomotion:diffusion](config/locomotion.py#L22-L65)中。
你可以使用标志覆盖任何参数，例如，`--n_diffusion_steps 100`。

2. 使用以下命令训练值函数（value function）：
```
python scripts/train_values.py --dataset halfcheetah-medium-expert-v2
```
相应的默认超参数见[locomotion:values](config/locomotion.py#L67-L108)。

3. 使用你新训练的模型进行规划，命令与预训练规划部分相同，只需将logbase替换为指向你的新模型：
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs
```
相应的默认超参数见[locomotion:plans](config/locomotion.py#L110-L149)。

**延迟f-字符串（Deferred f-strings）。** 注意，一些规划脚本参数，如`--n_diffusion_steps`（扩散步数）或`--discount`（折扣因子），
实际上并不会在规划期间改变任何逻辑，而只是使用延迟f-字符串加载不同的模型。
例如，以下标志：
```
---horizon 32 --n_diffusion_steps 20 --discount 0.997
--value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}'
```
将解析为值检查点路径`values/defaults_H32_T20_d0.997`。可以在训练后更改扩散模型的时间范围（horizon）（参见[此处](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing)的示例），
但不能更改值函数的时间范围。

\textbf{注意：}延迟f-字符串是一种动态构建字符串的方法，允许在运行时根据参数值生成文件路径或其他字符串。这在处理不同配置的模型时特别有用。

## Docker 使用指南

Docker 可以帮助我们创建一个一致的环境来运行我们的项目。以下是使用 Docker 的步骤：

1. 构建 Docker 镜像：
```
docker build -f Dockerfile . -t diffuser
```
这个命令会根据 Dockerfile 中的指令创建一个名为 "diffuser" 的 Docker 镜像。

2. 测试 Docker 镜像：
```
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$HOME/.d4rl,target=/root/.d4rl \
    diffuser \
    bash -c \
    "export PYTHONPATH=$PYTHONPATH:/home/code && \
    python /home/code/scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```
这个命令会运行刚才创建的 Docker 镜像，并在其中执行训练脚本。
注意：
- `--gpus all` 选项允许容器使用所有可用的 GPU。
- `--mount` 选项用于将主机的目录挂载到容器中。

## Singularity 使用指南

Singularity 是另一个容器化解决方案，特别适用于高性能计算环境。以下是使用 Singularity 的步骤：

1. 构建 Singularity 镜像：
```
singularity build --fakeroot diffuser.sif Singularity.def
```
这个命令会根据 Singularity.def 文件创建一个名为 diffuser.sif 的 Singularity 镜像。
注意：`--fakeroot` 选项允许非 root 用户创建镜像。

2. 测试 Singularity 镜像：
```
singularity exec --nv --writable-tmpfs diffuser.sif \
        bash -c \
        "pip install -e . && \
        python scripts/train.py --dataset halfcheetah-medium-expert-v2 --logbase logs"
```

## 在Azure云平台上运行

### 设置

1. 标记Docker镜像（在`Docker`部分中构建）并将其推送到Docker Hub：
```
export DOCKER_USERNAME=$(docker info | sed '/Username:/!d;s/.* //')
docker tag diffuser ${DOCKER_USERNAME}/diffuser:latest
docker image push ${DOCKER_USERNAME}/diffuser
```

2. 更新`azure/config.py`，可以直接修改文件或设置相关的环境变量（参见`azure/config.py#L47-L52`）。要设置`AZURE\_STORAGE\_CONNECTION`变量，导航到存储账户的"Access keys"（访问密钥）部分。点击"Show keys"（显示密钥）并复制"Connection string"（连接字符串）。

3. 下载[`azcopy`](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)：
```
./azure/download.sh
```

### 使用方法

使用以下命令启动训练作业：
```
python azure/launch.py
```
启动脚本不接受命令行参数；相反，它会为`params\_to\_sweep`（参见`azure/launch_train.py`#L36-L38）中的每种超参数组合启动一个作业。

### 查看结果

要从Azure存储容器同步数据，运行：
```
./azure/sync.sh
```

要挂载存储容器：
1. 使用以下命令创建blobfuse配置：
```
./azure/make_fuse_config.sh
```

2. 运行以下命令将存储容器挂载到`~/azure\_mount`：
```
./azure/mount.sh
```

要卸载容器，运行：
```
sudo umount -f ~/azure_mount; rm -r ~/azure_mount
```
（此命令将强制卸载挂载点并删除挂载目录）


## 引用
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}

## 致谢

扩散模型的实现代码基于Phil Wang的`denoising-diffusion-pytorch`代码库（\url{https://github.com/lucidrains/denoising-diffusion-pytorch}）。

本代码库的项目结构和远程启动器基于`trajectory-transformer`代码库（\url{https://github.com/jannerm/trajectory-transformer}）。
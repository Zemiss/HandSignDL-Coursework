# 机器视觉原理课程作业：深度学习手势识别

![Python >=3.10](https://img.shields.io/badge/Python->=3.10-blue.svg)
![PyTorch >=2.2](https://img.shields.io/badge/PyTorch->=2.2-yellow.svg)

这是一个基于 PyTorch 的手势姿态识别项目，采用标准 `src/` 目录结构。模型使用 torchvision 的 ResNet50，并默认加载 ImageNet 预训练权重进行微调。

## 项目结构

- `src/`：核心库代码
- `scripts/`：训练和推理入口
- `configs/`：默认配置
- `data/`：数据集目录
- `outputs/`：训练输出目录
- `docs/`：文档

## 输出约定

默认输出目录为 `./outputs/`，其中：

- `outputs/checkpoints/`：模型文件
- `outputs/results/`：训练历史和曲线图

默认模型路径为 `./outputs/checkpoints/best_model.pth`。

切换到 ResNet50 后，旧 CNN 结构训练出的 `best_model.pth` 不能直接用于当前推理脚本，需要先重新训练生成新的 ResNet50 checkpoint。

## 测试代码与模型文件位置

单独用于测试/推理的代码在 `scripts/evaluate.py`，把需要测试的图片放在`./test_images`下即可。

该测试代码默认加载的训练好的分类模型文件在 `outputs/checkpoints/best_model.pth`。如果模型文件放在其他位置，可以通过 `--model_path` 指定。

## 使用方法

训练：

```powershell
conda run -n myenv python scripts/train.py --data_dir ./data/Hand_Posture_Hard_Stu --epochs 5 --device cuda --progress_interval 10
```

训练脚本默认训练 5 轮、要求使用 GPU 训练，并会每隔 `--progress_interval` 个 batch 打印一次训练进度。可以先检查当前环境是否支持 CUDA：

```powershell
conda run -n myenv python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，需要先安装 CUDA 版本的 PyTorch，否则训练脚本会直接报错停止。训练脚本默认调用 ResNet50 进行微调。如果当前环境无法下载 ImageNet 预训练权重，可以加 `--no_pretrained` 从随机初始化开始训练。

推理：

```powershell
conda run -n myenv python scripts/evaluate.py
```

直接运行 `scripts/evaluate.py` 会自动加载 `outputs/checkpoints/best_model.pth`，并递归检测 `test_images/` 下的常见图片格式。

可编辑安装：

```powershell
python -m pip install -e .
```

## 说明

根目录下不再保留 `train.py`、`test.py`、`utils.py` 这类薄包装文件，统一使用 `scripts/` 下的入口脚本。

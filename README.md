# 机器视觉原理课程作业：深度学习手势识别

![Python >=3.10](https://img.shields.io/badge/Python->=3.10-blue.svg)
![PyTorch >=2.2](https://img.shields.io/badge/PyTorch->=2.2-yellow.svg)

这是一个基于 PyTorch 的手势姿态识别项目，采用标准 `src/` 目录结构。模型使用 torchvision 的 ResNet50，并默认加载 ImageNet 预训练权重进行微调。


训练和测试的默认参数都集中在 `main/configs/default.yaml`，包括训练数据路径、测试数据路径、模型路径、设备、轮数和进度打印频率。命令行参数仍然可以覆盖这些默认值，例如临时加 `--epochs 7`。
## 项目结构

- `main/src/core.py`：核心库代码
训练脚本默认行为由 `main/configs/default.yaml` 决定，并会每隔 `--progress_interval` 个 batch 打印一次训练进度。可以先检查当前环境是否支持 CUDA：
- `main/test.py`：推理入口
- `main/configs/`：默认配置
- `data/`：数据集目录
- `main/model/`：模型文件目录
- `main/outputs/`：训练输出目录

## 输出约定

在 `main/` 目录下运行脚本时，默认输出目录为 `./outputs/`，其中：

- `model/`：模型文件
- `outputs/`：训练历史和曲线图。`training_history.csv` 会按 batch 记录 `batch_loss`、`running_train_loss`，并按 epoch 记录验证指标；`training_curves.png` 会绘制 batch loss、累计训练 loss 和验证 loss。

直接在 `main/` 目录运行 `test.py` 会自动加载配置里的默认模型路径，并递归检测配置里的默认测试目录下的常见图片格式。

切换到 ResNet50 后，旧 CNN 结构训练出的 `best_model.pth` 不能直接用于当前推理脚本，需要先重新训练生成新的 ResNet50 checkpoint。

## 测试代码与模型文件位置
```powershell
python -m main.train
```

如果要覆盖默认值，可以继续追加参数，例如：

```powershell

单独用于测试/推理的代码在 `main/test.py`，把需要测试的图片放在 `main/test_images` 下即可。

该测试代码默认加载的训练好的分类模型文件在 `model/best_model.pth`。如果模型文件放在其他位置，可以通过 `--input_model_path` 指定。

## 使用方法
python -m main.test
```

如果要覆盖默认值，可以继续追加参数，例如：

```powershell
python -m main.test --test_data_dir data/Hand_Posture_Hard_Stu --input_model_path main/model/best_model.pth --device cuda
训练：

```powershell
cd main
conda run -n myenv python train.py `
  --train_data_dir ../data/Hand_Posture_Hard_Stu `
  --output_model_path ./model/best_model.pth `
  --pretrained_weights_path ./model/resnet50-11ad3fa6.pth `
  --output_dir ./outputs `
  --epochs 5 `
  --device cuda `
  --progress_interval 10
```

训练脚本默认训练 5 轮、要求使用 GPU 训练，并会每隔 `--progress_interval` 个 batch 打印一次训练进度。可以先检查当前环境是否支持 CUDA：

```powershell
conda run -n myenv python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，需要先安装 CUDA 版本的 PyTorch，否则训练脚本会直接报错停止。训练脚本默认调用 ResNet50 进行微调，并会优先读取 `./model/resnet50-11ad3fa6.pth` 作为本地 ImageNet 预训练权重；如果该文件不存在，则使用 torchvision 默认逻辑下载权重。如果当前环境无法下载 ImageNet 预训练权重，也可以加 `--no_pretrained` 从随机初始化开始训练。

推理：

```powershell
cd main
conda run -n myenv python test.py `
  --test_data_dir ./test_images `
  --input_model_path ./model/best_model.pth
```

直接在 `main/` 目录运行 `test.py` 会自动加载 `model/best_model.pth`，并递归检测 `test_images/` 下的常见图片格式。
训练脚本也兼容旧参数名 `--data_dir` 和 `--model_path`，测试脚本兼容旧参数名 `--image_dir` 和 `--model_path`。

可编辑安装：

```powershell
python -m pip install -e .
```

## 常用完整命令

```
conda activate myenv
cd C:\Users\12445\Desktop\3359_2411273_谢博_课程设计Part2
```

**训练：**

python -m main.train

```
python -m main.train --train_data_dir data/Hand_Posture_Hard_Stu --epochs 7 --device cuda --progress_interval 1
```

**测试：**

python -m main.test

```
python -m main.test --test_data_dir data/Hand_Posture_Hard_Stu --input_model_path main/model/best_model.pth --device cuda
```
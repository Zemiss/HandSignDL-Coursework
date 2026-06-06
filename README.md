# 机器视觉原理课程作业：深度学习手势识别

![Python >=3.10](https://img.shields.io/badge/Python->=3.10-blue.svg)
![PyTorch >=2.2](https://img.shields.io/badge/PyTorch->=2.2-yellow.svg)

这是一个基于 PyTorch 的手势姿态识别项目，采用 `src/` 目录结构。模型使用 torchvision 的 ResNet50，并默认加载 ImageNet 预训练权重进行微调。

训练和测试的默认参数集中在 `configs/default.yaml`，按 `paths`、`data`、`train` 分组管理。命令行参数可以覆盖默认值，例如临时加 `--epochs 7`。

## 项目结构

- `src/core.py`：核心库代码
- `train.py`：训练入口
- `test.py`：推理入口
- `configs/`：默认配置
- `data/`：数据集目录
- `model/`：模型文件目录
- `outputs/`：训练输出目录

## 输出约定

在项目根目录运行脚本时，默认输出目录为 `./outputs/`，其中：

- `model/best_model.pth`：训练得到的最佳模型
- `outputs/training_history.csv`：训练历史，按 batch 记录 `batch_loss`、`running_train_loss`，并按 epoch 记录验证指标
- `outputs/training_curves.png`：训练曲线，绘制 batch loss、累计训练 loss 和验证 loss

切换到 ResNet50 后，旧 CNN 结构训练出的 `best_model.pth` 不能直接用于当前推理脚本，需要先重新训练生成新的 ResNet50 checkpoint。

## 环境

```powershell
conda activate myenv
cd C:\Users\12445\Desktop\3359_2411273_谢博_课程设计Part2
python -m pip install -r requirements.txt
```

训练脚本默认要求使用 CUDA。可以先检查当前环境是否支持 CUDA：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，需要安装 CUDA 版本的 PyTorch，或把训练放到支持 CUDA 的机器上执行。

## 训练

使用配置默认值：

```powershell
python train.py
```

覆盖默认值：

```powershell
python train.py `
  --train_data_dir ./data/Hand_Posture_Hard_Stu `
  --output_model_path ./model/best_model.pth `
  --pretrained_weights_path ./model/resnet50-11ad3fa6.pth `
  --output_dir ./outputs `
  --epochs 5 `
  --device cuda `
  --progress_interval 10
```

训练脚本默认调用 ResNet50 进行微调，并会优先读取 `./model/resnet50-11ad3fa6.pth` 作为本地 ImageNet 预训练权重；如果该文件不存在，则使用 torchvision 默认逻辑下载权重。如果当前环境无法下载 ImageNet 预训练权重，也可以加 `--no_pretrained` 从随机初始化开始训练。

## 推理

使用配置默认值：

```powershell
python test.py
```

覆盖默认值：

```powershell
python test.py `
  --test_data_dir ./data/Hand_Posture_Hard_Stu `
  --input_model_path ./model/best_model.pth `
  --device cuda
```

`test.py` 会递归检测测试目录下的常见图片格式：`.png`、`.jpg`、`.jpeg`、`.bmp`、`.webp`。

训练脚本兼容旧参数名 `--data_dir` 和 `--model_path`，测试脚本兼容旧参数名 `--image_dir` 和 `--model_path`。

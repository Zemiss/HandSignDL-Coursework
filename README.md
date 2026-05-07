# 机器视觉原理课程作业：深度学习手势识别

这是一个基于 PyTorch 的手势姿态识别项目，采用标准 `src/` 目录结构。

## 项目结构

- `src/hand_posture_recognition/`：核心库代码
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

## 使用方法

训练：

```powershell
python scripts/train.py --data_dir ./data/Hand_Posture_Hard_Stu
```

推理：

```powershell
python scripts/evaluate.py --image_dir ./test_images --model_path ./outputs/checkpoints/best_model.pth
```

可编辑安装：

```powershell
python -m pip install -e .
```

## 说明

根目录下不再保留 `train.py`、`test.py`、`utils.py` 这类薄包装文件，统一使用 `scripts/` 下的入口脚本。

# Hand Posture Recognition with PyTorch

这是一个基于 PyTorch 的六分类手势识别项目，用于课程设计 Part 2。项目输入 PNG 手势图片，输出以下六类之一：

```text
A, B, C, Five, Point, V
```

## 仓库结构

```text
.
├─ data/Hand_Posture_Hard_Stu/   # 本地数据集，不建议提交
├─ dataset.py                    # 数据集读取、增强与划分
├─ model.py                      # CNN 模型定义
├─ train.py                      # 训练入口
├─ test.py                       # 独立推理入口
├─ utils.py                      # 设备选择、模型保存与加载
├─ requirements.txt              # pip 依赖
├─ environment.yml               # conda 环境文件
├─ README.md
└─ 课程设计Part2_提交材料/
   └─ 实验报告.md
```

训练后会生成：

```text
best_model.pth
result/training_history.csv
result/training_curves.png
```

这些属于运行产物，已经写入 `.gitignore`。

## 环境要求

推荐使用 Python 3.10+。本地验证过的版本组合如下：

```text
torch 2.2.2
numpy 1.26.4
matplotlib 3.10.8
Pillow 10.x
```

如果你已经有现成环境，直接安装依赖即可：

```powershell
python -m pip install -r requirements.txt
```

如果需要新建 conda 环境：

```powershell
conda env create -f environment.yml
conda activate myenv
```

## 数据集准备

把数据放到：

```text
data/Hand_Posture_Hard_Stu/
```

目录下需要包含 6 个类别子目录：

```text
A/
B/
C/
Five/
Point/
V/
```

每个子目录中放对应的 `.png` 图片即可。

## 训练

直接运行：

```powershell
python train.py --data_dir ./data/Hand_Posture_Hard_Stu --epochs 20 --batch_size 64 --model_path ./best_model.pth
```

常用参数如下：

```text
--data_dir      数据集目录，默认 ./data/Hand_Posture_Hard_Stu
--output_dir    训练日志输出目录，默认 ./result
--model_path    最优模型保存路径，默认 ./best_model.pth
--epochs        训练轮数，默认 20
--batch_size    批大小，默认 64
--lr            学习率，默认 1e-3
--image_size    输入尺寸，默认 64
--val_ratio     验证集比例，默认 0.2
--device        auto / cpu / cuda / cuda:0，默认 auto
```

本地 20 epoch 的验证集最高准确率为：

```text
96.52%
```

## 测试

对 `test_images` 目录下所有 PNG 图片做推理：

```powershell
python test.py --image_dir ./test_images --model_path ./best_model.pth
```

输出示例：

```text
image_001.png: A
image_002.png: Five
image_003.png: Point
```

## 代码说明

- `dataset.py`：定义类别顺序、训练/验证预处理、数据集加载和分层划分。
- `model.py`：定义轻量 CNN。
- `train.py`：完成训练、验证、记录历史和保存最优权重。
- `test.py`：独立加载模型，对文件夹内 PNG 图片逐张预测。
- `utils.py`：提供设备选择、checkpoint 保存与加载。

## Git 提交建议

建议提交这些文件：

```text
.gitignore
README.md
requirements.txt
environment.yml
dataset.py
model.py
train.py
test.py
utils.py
课程设计Part2_提交材料/实验报告.md
```

不建议提交这些文件或目录：

```text
data/
result/
test_images/
best_model.pth
*.zip
__pycache__/
```

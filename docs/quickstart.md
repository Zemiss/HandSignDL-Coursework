# Quickstart

## Train

```powershell
python scripts/train.py --data_dir ./data/Hand_Posture_Hard_Stu
```

## Evaluate

```powershell
python scripts/evaluate.py --image_dir ./test_images --model_path ./outputs/checkpoints/best_model.pth
```

## Dataset layout

```text
data/Hand_Posture_Hard_Stu/
├── A/
├── B/
├── C/
├── Five/
├── Point/
└── V/
```


import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_posture_recognition import (  # noqa: E402
    CLASS_NAMES,
    EvalTransform,
    HandPostureDataset,
    TrainTransform,
    build_model,
    get_device,
    save_checkpoint,
    stratified_split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand posture classifier.")
    parser.add_argument("--data_dir", default="./data/Hand_Posture_Hard_Stu")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--model_path", default="./outputs/checkpoints/best_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def plot_history(history_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(history_path.open(encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_source = HandPostureDataset(args.data_dir, transform=TrainTransform(args.image_size))
    eval_source = HandPostureDataset(args.data_dir, transform=EvalTransform(args.image_size))
    train_indices, val_indices = stratified_split_indices(
        train_source.samples,
        num_classes=len(CLASS_NAMES),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(
        Subset(train_source, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_source, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_path = results_dir / "training_history.csv"
    best_acc = 0.0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step()
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
            print(
                f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(args.model_path, model, CLASS_NAMES, args.image_size, best_acc)

    plot_path = results_dir / "training_curves.png"
    plot_history(history_path, plot_path)
    print(f"best validation accuracy: {best_acc:.4f}")
    print(f"saved model: {args.model_path}")
    print(f"saved curves: {plot_path}")


if __name__ == "__main__":
    main()

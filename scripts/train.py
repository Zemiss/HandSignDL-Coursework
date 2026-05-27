import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import build_model  # noqa: E402
from preprocessing import (  # noqa: E402
    CLASS_NAMES,
    EvalTransform,
    HandPostureDataset,
    TrainTransform,
    stratified_split_indices,
)
from utils import get_device, save_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand posture classifier.")
    parser.add_argument("--data_dir", default="./data/Hand_Posture_Hard_Stu")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--model_path", default="./outputs/checkpoints/best_model.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="cuda or cuda:0. Training requires GPU by default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_pretrained", action="store_true", help="Train ResNet50 without ImageNet weights.")
    parser.add_argument("--progress_interval", type=int, default=10, help="Print training progress every N batches.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_device(preferred: str) -> torch.device:
    device = get_device(preferred)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training requires CUDA. Install a CUDA-enabled PyTorch build or pass training to a CUDA machine."
        )
    return device


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress_interval: int = 0,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader, start=1):
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
        if training and progress_interval > 0 and (batch_idx % progress_interval == 0 or batch_idx == total_batches):
            epoch_label = f"{epoch:03d}/{total_epochs:03d}" if epoch and total_epochs else "???/???"
            avg_loss = total_loss / total_count
            avg_acc = total_correct / total_count
            print(
                f"epoch {epoch_label} batch {batch_idx:03d}/{total_batches:03d}: "
                f"train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}",
                flush=True,
            )

    return total_loss / total_count, total_correct / total_count


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, float, list[float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    class_correct = [0 for _ in class_names]
    class_total = [0 for _ in class_names]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            matches = predictions == labels

            total_loss += loss.item() * labels.size(0)
            total_correct += matches.sum().item()
            total_count += labels.size(0)

            for class_idx in range(len(class_names)):
                class_mask = labels == class_idx
                class_count = class_mask.sum().item()
                if class_count:
                    class_total[class_idx] += class_count
                    class_correct[class_idx] += (predictions[class_mask] == labels[class_mask]).sum().item()

    class_acc = [
        (class_correct[idx] / class_total[idx]) if class_total[idx] else 0.0
        for idx in range(len(class_names))
    ]
    return total_loss / total_count, total_correct / total_count, class_acc


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

    device = resolve_training_device(args.device)
    print(f"training device: {device}", flush=True)
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

    model = build_model(num_classes=len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_path = results_dir / "training_history.csv"
    best_acc = 0.0
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                *[f"val_acc_{class_name}" for class_name in CLASS_NAMES],
            ]
        )
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                progress_interval=args.progress_interval,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            val_loss, val_acc, class_acc = evaluate_validation(model, val_loader, criterion, device, CLASS_NAMES)
            scheduler.step()
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, *class_acc])
            class_summary = " ".join(
                f"{class_name}={acc:.4f}" for class_name, acc in zip(CLASS_NAMES, class_acc)
            )
            print(
                f"epoch {epoch:03d}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            print(f"  val per-class: {class_summary}")

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

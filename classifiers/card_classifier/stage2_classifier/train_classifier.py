"""
Train Stage 2 Multi-Head Card Attribute Classifier
===================================================
Trains multi-task CNN (MobileNetV3-Small / MobileNetV3-Large / ResNet18) to predict:
- Rank (14 classes)
- Suit (5 classes)
- Edition (4 classes)
- Enhancement (9 classes)
- Seal (5 classes)
Guarantees mutual exclusivity per attribute via Softmax heads.
Saves checkpoints and copies best weights to weights/classifier_best.pt.
"""

import sys
import shutil
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
CLASSIFIER_ROOT = CURRENT_DIR.parent
DATASET_DIR = CLASSIFIER_ROOT / "dataset"
RUNS_DIR = CURRENT_DIR / "runs"
WEIGHTS_DIR = CLASSIFIER_ROOT / "weights"

try:
    from .config import RANKS, SUITS, EDITIONS, ENHANCEMENTS, SEALS
    from .dataset import CardCropDataset
    from .model import MultiHeadCardClassifier
except ImportError:
    from config import RANKS, SUITS, EDITIONS, ENHANCEMENTS, SEALS
    from dataset import CardCropDataset
    from model import MultiHeadCardClassifier


def get_next_run_dir(base_name="card_classifier", exist_ok=False) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if exist_ok:
        d = RUNS_DIR / base_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    idx = 1
    target = RUNS_DIR / base_name
    while target.exists():
        idx += 1
        target = RUNS_DIR / f"{base_name}-{idx}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_attribute_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 0.6), max(4, len(class_names) * 0.5)), dpi=120)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Normalize by true label count
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        im = ax.imshow(norm_cm, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            if val > 0:
                color = "white" if norm_cm[i, j] > 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_train_classifier(
    epochs=25,
    batch_size=64,
    lr=1e-3,
    backbone="mobilenet_v3_small",
    run_name="card_classifier",
    exist_ok=False,
    num_workers=0
):
    print("\n" + "=" * 60)
    print("Stage 2: Multi-Head Card Attribute Classifier Training")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[CUDA OK] GPU Device : {gpu_name} ({vram_gb:.2f} GB VRAM)")
        device = torch.device("cuda:0")
    else:
        print("[WARNING] CUDA not available, training on CPU.")
        device = torch.device("cpu")

    run_dir = get_next_run_dir(run_name, exist_ok=exist_ok)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    weights_subdir = run_dir / "weights"
    weights_subdir.mkdir(parents=True, exist_ok=True)

    print(f"Run Output Dir: {run_dir}")
    print(f"Backbone Model: {backbone}")
    print(f"Target Epochs : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print("=" * 60 + "\n")

    # Load datasets
    train_dataset = CardCropDataset(DATASET_DIR, split="train", val_ratio=0.20)
    val_dataset = CardCropDataset(DATASET_DIR, split="val", val_ratio=0.20)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = MultiHeadCardClassifier(backbone_name=backbone, pretrained=True, dropout=0.2).to(device)

    # Loss criteria
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_exact_match = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            logits = model(images)

            l_rank = criterion_ce(logits["rank"], targets["rank"])
            l_suit = criterion_ce(logits["suit"], targets["suit"])
            l_edition = criterion_ce(logits["edition"], targets["edition"])
            l_enh = criterion_ce(logits["enhancement"], targets["enhancement"])
            l_seal = criterion_ce(logits["seal"], targets["seal"])
            l_deb = criterion_bce(logits["debuffed"], targets["debuffed"])

            loss = l_rank + l_suit + l_edition + l_enh + l_seal + 0.5 * l_deb
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)

        train_loss /= len(train_dataset)
        scheduler.step()

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        correct_counts = {"rank": 0, "suit": 0, "edition": 0, "enhancement": 0, "seal": 0, "debuffed": 0}
        exact_matches = 0
        total_val = len(val_dataset)

        # Confusion matrix accumulators
        cms = {
            "rank": np.zeros((len(RANKS), len(RANKS)), dtype=int),
            "suit": np.zeros((len(SUITS), len(SUITS)), dtype=int),
            "edition": np.zeros((len(EDITIONS), len(EDITIONS)), dtype=int),
            "enhancement": np.zeros((len(ENHANCEMENTS), len(ENHANCEMENTS)), dtype=int),
            "seal": np.zeros((len(SEALS), len(SEALS)), dtype=int),
        }

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}

                logits = model(images)

                l_rank = criterion_ce(logits["rank"], targets["rank"])
                l_suit = criterion_ce(logits["suit"], targets["suit"])
                l_edition = criterion_ce(logits["edition"], targets["edition"])
                l_enh = criterion_ce(logits["enhancement"], targets["enhancement"])
                l_seal = criterion_ce(logits["seal"], targets["seal"])
                l_deb = criterion_bce(logits["debuffed"], targets["debuffed"])

                loss = l_rank + l_suit + l_edition + l_enh + l_seal + 0.5 * l_deb
                val_loss += loss.item() * len(images)

                p_rank = logits["rank"].argmax(dim=-1)
                p_suit = logits["suit"].argmax(dim=-1)
                p_ed = logits["edition"].argmax(dim=-1)
                p_enh = logits["enhancement"].argmax(dim=-1)
                p_seal = logits["seal"].argmax(dim=-1)
                p_deb = (torch.sigmoid(logits["debuffed"]) > 0.5).float()

                # Correctness masks
                m_rank = (p_rank == targets["rank"])
                m_suit = (p_suit == targets["suit"])
                m_ed = (p_ed == targets["edition"])
                m_enh = (p_enh == targets["enhancement"])
                m_seal = (p_seal == targets["seal"])
                m_deb = (p_deb == targets["debuffed"])

                correct_counts["rank"] += m_rank.sum().item()
                correct_counts["suit"] += m_suit.sum().item()
                correct_counts["edition"] += m_ed.sum().item()
                correct_counts["enhancement"] += m_enh.sum().item()
                correct_counts["seal"] += m_seal.sum().item()
                correct_counts["debuffed"] += m_deb.sum().item()

                # Exact Card Match (all core attributes + debuffed simultaneously correct)
                exact = m_rank & m_suit & m_ed & m_enh & m_seal & m_deb
                exact_matches += exact.sum().item()

                # Accumulate confusion matrices
                for t_i, p_i in zip(targets["rank"].cpu().numpy(), p_rank.cpu().numpy()):
                    cms["rank"][t_i, p_i] += 1
                for t_i, p_i in zip(targets["suit"].cpu().numpy(), p_suit.cpu().numpy()):
                    cms["suit"][t_i, p_i] += 1
                for t_i, p_i in zip(targets["edition"].cpu().numpy(), p_ed.cpu().numpy()):
                    cms["edition"][t_i, p_i] += 1
                for t_i, p_i in zip(targets["enhancement"].cpu().numpy(), p_enh.cpu().numpy()):
                    cms["enhancement"][t_i, p_i] += 1
                for t_i, p_i in zip(targets["seal"].cpu().numpy(), p_seal.cpu().numpy()):
                    cms["seal"][t_i, p_i] += 1

        val_loss /= total_val
        accs = {k: v / total_val for k, v in correct_counts.items()}
        exact_match_acc = exact_matches / total_val

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "exact_match_acc": exact_match_acc,
            "rank_acc": accs["rank"],
            "suit_acc": accs["suit"],
            "edition_acc": accs["edition"],
            "enhancement_acc": accs["enhancement"],
            "seal_acc": accs["seal"],
            "debuffed_acc": accs["debuffed"]
        }
        history.append(epoch_stats)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | "
            f"Exact Match: {exact_match_acc * 100:.1f}% | "
            f"Rank: {accs['rank'] * 100:.1f}% | Suit: {accs['suit'] * 100:.1f}% | "
            f"Ed: {accs['edition'] * 100:.1f}% | Enh: {accs['enhancement'] * 100:.1f}% | Seal: {accs['seal'] * 100:.1f}% | Deb: {accs['debuffed'] * 100:.1f}%"
        )

        # Save last checkpoint
        last_ckpt = weights_subdir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "stats": epoch_stats}, last_ckpt)

        # Check and save best
        if exact_match_acc > best_exact_match:
            best_exact_match = exact_match_acc
            best_ckpt = weights_subdir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "stats": epoch_stats}, best_ckpt)

            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_ckpt, WEIGHTS_DIR / "classifier_best.pt")

            # Save clean confusion matrices for each attribute
            plot_attribute_confusion_matrix(cms["rank"], RANKS, "Rank Confusion Matrix", fig_dir / "cm_rank.png")
            plot_attribute_confusion_matrix(cms["suit"], SUITS, "Suit Confusion Matrix", fig_dir / "cm_suit.png")
            plot_attribute_confusion_matrix(cms["edition"], EDITIONS, "Edition Confusion Matrix", fig_dir / "cm_edition.png")
            plot_attribute_confusion_matrix(cms["enhancement"], ENHANCEMENTS, "Enhancement Confusion Matrix", fig_dir / "cm_enhancement.png")
            plot_attribute_confusion_matrix(cms["seal"], SEALS, "Seal Confusion Matrix", fig_dir / "cm_seal.png")

    # Save metrics table and training curves
    df_history = pd.DataFrame(history)
    df_history.to_csv(run_dir / "results.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    ax1.plot(df_history["epoch"], df_history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(df_history["epoch"], df_history["val_loss"], label="Val Loss", color="crimson")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_history["epoch"], df_history["exact_match_acc"] * 100, label="Exact Match", color="forestgreen", lw=2)
    ax2.plot(df_history["epoch"], df_history["rank_acc"] * 100, label="Rank", linestyle="--")
    ax2.plot(df_history["epoch"], df_history["suit_acc"] * 100, label="Suit", linestyle="--")
    ax2.plot(df_history["epoch"], df_history["enhancement_acc"] * 100, label="Enhancement", linestyle="--")
    ax2.set_title("Validation Accuracy Trends (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "results.png")
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"Training Complete! Peak Exact Card Match: {best_exact_match * 100:.2f}%")
    print(f"Best Weights Saved To: {WEIGHTS_DIR / 'classifier_best.pt'}")
    print(f"Metrics & Figures In : {fig_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 Multi-Head Card Classifier")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"], help="Backbone model")
    parser.add_argument("--name", type=str, default="card_classifier", help="Run directory name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run")
    args = parser.parse_args()

    run_train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        run_name=args.name,
        exist_ok=args.exist_ok
    )

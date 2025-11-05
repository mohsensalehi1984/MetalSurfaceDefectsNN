import os
import argparse
from tqdm import tqdm
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import Config
from dataset import make_loaders
from model import SmallCNN, resnet18_gray
from utils import seed_everything, save_checkpoint, evaluate


# ----------------------------------------------------------------------
# 🔹 Utility functions
# ----------------------------------------------------------------------
def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_result(model, val_acc, csv_path="model_results.csv"):
    """Append model result to a CSV for comparison."""
    total_params = count_parameters(model)
    model_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Parameters", "Val_Acc"])
        writer.writerow([timestamp, model_name, total_params, round(val_acc, 4)])

    print(f"✅ Logged results to {csv_path} ({model_name}, {total_params:,} params, acc={val_acc:.4f})")


# ----------------------------------------------------------------------
# 🔹 Training function
# ----------------------------------------------------------------------
def train(cfg: Config):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print("Device is:", device)

    train_loader, val_loader, classes = make_loaders(
        cfg.data_root, cfg.train_dir, cfg.valid_dir,
        img_size=cfg.img_size, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )

    # Choose model
    # model = resnet18_gray(num_classes=cfg.num_classes, pretrained=True)
    model = SmallCNN(num_classes=cfg.num_classes)
    model = model.to(device)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    writer = SummaryWriter(log_dir=cfg.log_dir)
    best_val = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.epochs}')

        for xb, yb in loop:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        print(f'Epoch {epoch} | Train loss: {avg_loss:.4f} | Val acc: {val_acc:.4f}')

        # Save checkpoints
        os.makedirs(cfg.out_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.out_dir, f'epoch{epoch:03d}_acc{val_acc:.4f}.pt')
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': val_acc
        }, ckpt_path)

        # Save best model and log results
        if val_acc > best_val:
            best_val = val_acc
            best_path = os.path.join(cfg.out_dir, 'best_model.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes
            }, best_path)

            log_model_result(model, val_acc)

    writer.close()


# ----------------------------------------------------------------------
# 🔹 Entry point
# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='optional config file (not implemented)')
    args = parser.parse_args()

    cfg = Config()
    train(cfg)

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from model import SmallCNN,resnet18_gray
from config import Config
import os


def predict_single_image(model, image_path, device, classes):
    from PIL import Image
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image)
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
    print(f"Predicted: {classes[pred.item()]}")


def evaluate_full_dataset(model, cfg, device):
    test_dir = os.path.join(cfg.data_root, cfg.test_dir)
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    test_ds = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=test_ds.classes))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_ds.classes, yticklabels=test_ds.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", help="Path to a single image for inference")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on the full test set")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = resnet18_gray(num_classes=cfg.num_classes, pretrained=False).to(device)
    model = SmallCNN(num_classes=cfg.num_classes)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    print(f"Loaded model from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    if args.evaluate:
        evaluate_full_dataset(model, cfg, device)
    elif args.image:
        predict_single_image(model, args.image, device, [
                             "Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"])
    else:
        print("Please specify either --image or --evaluate")

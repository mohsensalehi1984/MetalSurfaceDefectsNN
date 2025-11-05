import torch
from model import SmallCNN,resnet18_gray
from config import Config

# Load config
cfg = Config()

ckpt_path = "checkpoints/best_model.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# model = resnet18_gray(num_classes=6)
model = SmallCNN(num_classes=cfg.num_classes)
model.load_state_dict(checkpoint["model_state"])

print("✅ Loaded model successfully\n")

# Print model summary
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

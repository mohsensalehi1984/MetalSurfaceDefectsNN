from dataclasses import dataclass
import os

@dataclass
class Config:
    # --- Dataset path ---
    # Change it to a proper path on your own system 
    data_root: str = '/data/DataSets/MetalSurfaceDefects/fantacher/neu-metal-surface-defects-data/versions/1/NEU Metal Surface Defects Data'
    
    train_dir: str = 'train'
    valid_dir: str = 'valid'
    test_dir: str = 'test'

    # --- Training parameters ---
    seed: int = 42
    num_workers: int = 4
    batch_size: int = 64
    img_size: int = 224
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = 'cuda'  # or 'cpu'

    save_every: int = 1
    num_classes: int = 6

    # --- Output directories (one level above current working directory) ---
    out_dir: str = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    log_dir: str = os.path.join(os.path.dirname(__file__), '..', 'logs')

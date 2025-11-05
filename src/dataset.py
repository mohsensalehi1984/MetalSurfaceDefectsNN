import os
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_transforms(img_size=224, train=True):
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=1),   # <--- NEW LINE
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=1),   # <--- NEW LINE
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])



def make_loaders(data_root, train_dir='train', valid_dir='valid',
                 img_size=224, batch_size=64, num_workers=4):
    train_path = os.path.join(data_root, train_dir)
    val_path = os.path.join(data_root, valid_dir)

    train_ds = ImageFolder(train_path, transform=get_transforms(img_size, train=True))
    val_ds = ImageFolder(val_path, transform=get_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.classes

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np


def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        test_dir,
        test_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=16,
        pin_memory=True):
    train_ds = RoadDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, shuffle=True)

    val_ds = RoadDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, shuffle=False)

    test_ds = RoadDataset(image_dir=test_dir, mask_dir=test_mask_dir, transform=val_transform)

    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory, shuffle=False)

    return {"train_dl": train_loader, "val_dl": val_loader, "test_dl": test_loader,
            "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds}


class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        return image, mask.unsqueeze(0)

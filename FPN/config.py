import albumentations as album
from albumentations.pytorch import ToTensorV2
from dataset import get_loaders

TRAIN_IMG_DIR = "../input/tiff/train/"
TRAIN_MASK_DIR = "../input/tiff/train_labels/"
VAL_IMG_DIR = "../input/tiff/val/"
VAL_MASK_DIR = "../input/tiff/val_labels/"
TEST_IMG_DIR = "../input/tiff/test/"
TEST_MASK_DIR = "../input/tiff/test_labels"
LEARNING_RATE = 1e-3
BATCH_SIZE = 12
NUM_WORKERS = 4
IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
EPOCHS = 40
ARCH, ENCODER, IN_C, OUT_C = "FPN", "resnext50_32x4d", 3, 1

train_transforms = album.Compose(
    [
        album.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, always_apply=True),
        album.Rotate(limit=35, p=0.8),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.8),
        album.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),

        ToTensorV2()
    ],
)

val_transforms = album.Compose(
    [
        album.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, always_apply=True),
        album.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ],
)

ds_ld = get_loaders(TRAIN_IMG_DIR,
                    TRAIN_MASK_DIR,
                    VAL_IMG_DIR,
                    VAL_MASK_DIR,
                    TEST_IMG_DIR,
                    TEST_MASK_DIR,
                    BATCH_SIZE,
                    train_transforms,
                    val_transforms,
                    NUM_WORKERS,
                    True)

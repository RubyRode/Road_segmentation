from pprint import pprint
import torch
from model import RoadSegmentation
import config
from utils import pred_visualize
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

# print(f"Train size: {len(train_ds)}")
# print(f"Valid size: {len(val_ds)}")
# print(f"Test size: {len(tst_ds)}")

# aug_visualize(train_ds, "train_ds")
# aug_visualize(val_ds, "val_ds")
# aug_visualize(tst_ds, "tst_ds")

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    logger = TensorBoardLogger("../tb_logs", name="RoadSegmentation_Unet_v0")
    model = RoadSegmentation(arch=config.ARCH,
                             encoder_name=config.ENCODER,
                             in_channels=config.IN_C,
                             out_classes=config.OUT_C)

    trainer = pl.Trainer(
        logger=logger,
        gpus=1,
        max_epochs=config.EPOCHS,
    )
    trainer.fit(
        model,
        train_dataloaders=config.ds_ld["train_dl"],
        val_dataloaders=config.ds_ld["val_dl"],
        # ckpt_path="../tb_logs/RoadSegmentation_v0/version_4/checkpoints/epoch=39-step=3719.ckpt",
    )

    valid_metrics = trainer.validate(model, dataloaders=config.ds_ld["val_dl"], verbose=False)
    pprint(valid_metrics)

    test_metrics = trainer.test(model, dataloaders=config.ds_ld["test_dl"], verbose=False)
    pprint(test_metrics)

    pred_visualize(model, config.ds_ld["test_dl"])

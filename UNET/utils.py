import matplotlib.pyplot as plt
import numpy as np
import torch


def aug_visualize(ds, title):
    sample = ds[0]
    plt.subplot(1, 2, 1)
    plt.title('Image')
    plt.imshow(np.transpose(sample[0], (1, 2, 0)))  # for visualization we have to transpose back to HWC
    plt.subplot(1, 2, 2)
    plt.title('Ground truth')
    plt.imshow(sample[1].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.suptitle(title + "  " + str(sample[0].shape))
    plt.show()


def pred_visualize(model, test_dataloader):
    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch[0])
    pr_masks = logits.sigmoid()

    for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
        plt.title("Prediction")
        plt.axis("off")

        plt.show()

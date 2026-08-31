import torch
import numpy as np

class RandPatchDataset(Dataset):
    def __init__(
            self,
            image, #np array memmapped #shape (C, H, W)
            labels, #np (C, H, W)
            patch_size=256,
            samples_per_epoch=10000,
    ):
        self.image = image
        self.labels = labels
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        H, W = self.labels.shape
        p = self.patch_size
        y = np.random.randint(0, H - p + 1)
        x = np.random.randint(0, W - p + 1)

        image_patch = self.image[:, y:y+p, x:x+p]
        label_patch = self.labels[y:y+p, x:x+p]

        image_patch = torch.from_numpy(
            np.ascontiguousarray(image_patch)
        ).float()

        label_patch = torch.from_numpy(
            np.ascontiguousarray(label_patch)
        ).long()

        return image_patch, label_patch

        
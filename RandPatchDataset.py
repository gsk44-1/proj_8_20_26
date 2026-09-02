import torch
from torch.utils.data import Dataset
import numpy as np

class RandPatchDataset(Dataset):
    def __init__(
            self,
            image, #np array memmapped #shape (C, H, W)
            labels, #np (C, H, W)
            blocks, #contains coordinates of large blocks
            patch_size=256,
            block_size=4096,
            samples_per_epoch=10000,
            min_cell = 0.05, #the minimum required content of a patch
    ):
        self.image = image
        self.labels = labels
        self.blocks = blocks
        self.block_size = block_size
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.min_cell = min_cell



    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        _, H, W = self.labels.shape
        p = self.patch_size
        b = self.block_size
    
        while True:
            block_id = np.random.randint(len(self.blocks))
            b_y, b_x = self.blocks[block_id]

            y = np.random.randint(b_y, b_y + b - p + 1)
            x = np.random.randint(b_x, b_x + b - p + 1)

            label_patch = self.labels[
                0,
                y:y+p, 
                x:x+p
                ]

            if np.mean(label_patch != 0) >= self.min_cell:
                break

        image_patch = self.image[:, y:y+p, x:x+p]

        image_patch = torch.from_numpy(
            np.ascontiguousarray(image_patch)
        ).float()

        label_patch = torch.from_numpy(
            np.ascontiguousarray(label_patch)
        ).float()

        label_patch = label_patch.unsqueeze(0)

        return image_patch, label_patch



"""
test
def make_blocks(H, W, b_size):
    blocks = []
    for y in range(0, H-b_size+1, b_size):
        for x in range(0, W-b_size+1, b_size):
            blocks.append((y, x))
    return blocks

    
then shuffle the blocks and do
rng.shuffle()

n = len(blocks)
n_train = int(0.7*n)
n_val = int(0.15*n)
etc
train_blocks = blocks[:n_train]
etc
"""
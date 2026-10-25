import torch
from torch.utils.data import Dataset, DataLoader

class SliceDataset(Dataset):
    def __init__(self, volume_dataset):
        self.volume_dataset = volume_dataset

        # Map each slice index to (volume_index, slice_index)
        self.index = []

        for v in range(len(volume_dataset)):
            images, _ = volume_dataset[v]

            for z in range(images.shape[0]):
                self.index.append((v, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        volume_idx, slice_idx = self.index[idx]

        images, labels = self.volume_dataset[volume_idx]

        return (
            images[slice_idx],
            labels[slice_idx]
        )
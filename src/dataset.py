import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from config import device
import scipy as sp
import numpy as np

IMG_SIZE = [3,400,400]

class ChessPositionDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)

        image = image.cpu().detach().numpy()
        factors = [IMG_SIZE[i] / image.shape[i] for i in range(len(IMG_SIZE))]
        image = sp.ndimage.zoom(image, factors)
        image = torch.from_numpy(image)

        patches = image.unfold(1,50,50).unfold(2,50,50).permute(1,2,0,3,4)
        patches = patches / 255
        label = os.path.splitext(os.path.basename(img_path))[0]
        if self.transform:
            patches = self.transform(patches)
        if self.target_transform:
            label = self.target_transform(label)
        return patches.to(device), label.to(device)

import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from config import device
import scipy as sp
import numpy as np
import cv2


class ImageLoader():
    def __init__(self, img_size = [3,400,400], transform=None):
        self.img_size = img_size
        self.transform = transform

    def load(self, img_path):
        self.img_path = img_path
        image = cv2.imread('board.png')
        image = np.transpose(image,(2,0,1))

        factors = [self.img_size[i] / image.shape[i] for i in range(len(self.img_size))]
        image = sp.ndimage.zoom(image, factors)
        image = torch.from_numpy(image)

        patches = image.unfold(1,50,50).unfold(2,50,50).permute(1,2,0,3,4)
        patches = patches / 255
        if self.transform:
            patches = self.transform(patches)
        return patches.to(device)

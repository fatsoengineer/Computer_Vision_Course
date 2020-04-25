import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensor
import cv2
import os
import numpy as np
from torch.utils.data import Dataset


class AlbumTransforms():

    def __init__(self, mean, std, trainset_transforms = [], training_flag = False):
        self.mean = mean 
        self.std = std
        self.transforms_base = [
                                Normalize(
                                    mean=self.mean,
                                    std= self.std
                                ),
                                ToTensor()
        ]
        self.trainset_transforms = trainset_transforms
        self.trainset_transforms.extend(self.transforms_base)
        self.training_flag = training_flag

        
    def __call__(self, img):
        img = np.array(img)
        transformations = Compose(self.trainset_transforms)  if self.training_flag is True else Compose(self.transforms_base)
        img = transformations(image=img)['image']
        return img



class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label



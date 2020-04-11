import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensor

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
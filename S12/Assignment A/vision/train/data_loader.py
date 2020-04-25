import torch
from .utils.transformations import AlbumTransforms, AlbumentationsDataset
from albumentations import *
from albumentations.pytorch import ToTensor
import numpy as np
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split



class TrainsetLoader:

    def __init__(self, trainset_config_list):
        self.use_cuda = trainset_config_list.get('use_cuda')
        self.dataset = trainset_config_list.get('dataset')
        self.mean = trainset_config_list.get('mean')
        self.std = trainset_config_list.get('std')
        self.trainset_transforms = trainset_config_list.get('trainset_transforms', [])
        self.batch_size = trainset_config_list.get('batch_size', [])
        self.num_workers = trainset_config_list.get('num_workers', [])
        self.dataloader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)  if self.use_cuda else dict(batch_size=self.batch_size)

    def __call__(self):
        train_album_transforms = AlbumTransforms(
                                                mean=self.mean,
                                                std=self.std,
                                                trainset_transforms=self.trainset_transforms,
                                                training_flag=True 
                                            )
        test_album_transforms = AlbumTransforms(
                                                mean=self.mean,
                                                std=self.std,
                                            )

        train_set = self.dataset(root='./data', train=True,
                                            download=True, transform=train_album_transforms)
        
        test_set = self.dataset(root='./data', train=False,
                                            download=True, transform=test_album_transforms)
        

        train_loader = torch.utils.data.DataLoader(train_set,shuffle=True, **self.dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False,  **self.dataloader_args)
        return train_loader, test_loader



class CustomDatasetLoader:
    def __init__(self, trainset_config_list):
        self.use_cuda = trainset_config_list.get('use_cuda')
        self.dataset = trainset_config_list.get('dataset')
        self.mean = trainset_config_list.get('mean')
        self.std = trainset_config_list.get('std')
        self.transforms_base = [
                                Normalize(
                                    mean=self.mean,
                                    std= self.std
                                ),
                                ToTensor()
        ]
        self.trainset_transforms = trainset_config_list.get('trainset_transforms', [])
        self.trainset_transforms.extend(self.transforms_base)

        self.batch_size = trainset_config_list.get('batch_size', [])
        self.num_workers = trainset_config_list.get('num_workers', [])
        self.dataloader_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)  if self.use_cuda else dict(batch_size=self.batch_size)

    def __call__(self):

        train_dataset_x, test_dataset_x, train_y, test_y  = train_test_split([_[0] for _ in self.dataset],[_[1] for _ in self.dataset], test_size=0.30, random_state=42)
        albumentations_transform_train = Compose(self.trainset_transforms)
        albumentations_transform_test = Compose(self.transforms_base)

        albumentations_pil_dataset_train = AlbumentationsDataset(
            file_paths=train_dataset_x,
            labels=train_y,
            transform=albumentations_transform_train,
        )

        albumentations_pil_dataset_test = AlbumentationsDataset(
            file_paths=test_dataset_x,
            labels=test_y,
            transform=albumentations_transform_test,
        )

        train_loader = torch.utils.data.DataLoader(albumentations_pil_dataset_train,shuffle=True, **self.dataloader_args)
        test_loader = torch.utils.data.DataLoader(albumentations_pil_dataset_test,shuffle=False, **self.dataloader_args)
        return train_loader, test_loader    
    

def compile_data(path):
    images = []
    labels = []

    class_ids = [line.strip() for line in open(os.path.join(path , 'wnids.txt'), 'r')]
    id_dict = {x:i for i, x in enumerate(class_ids)}
    all_classes = {line.split('\t')[0] : line.split('\t')[1].strip() for line in open( os.path.join(path, 'words.txt'), 'r')}
    class_names = [all_classes[x] for x in class_ids]

    # train data
    for value, key in enumerate(class_ids):
        img_path = os.path.join(path, "train", key, "images")
        images += [ os.path.join(img_path ,f"{key}_{i}.JPEG") for i in range(500)]
        labels += [value for i in range(500)]

    # validation data
    for line in open( os.path.join(path ,'val', 'val_annotations.txt')):
        img_name, class_id = line.split('\t')[:2]
        img_path = os.path.join(path, "val","images")
        images.append(os.path.join(img_path, f'{img_name}'))
        labels.append(id_dict[class_id])

    dataset = list(zip(images, labels))
    return dataset, class_names
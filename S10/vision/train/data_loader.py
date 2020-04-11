import torch
from .utils.transformations import AlbumTransforms

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
  
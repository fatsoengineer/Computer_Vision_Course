'''
Class to create custom layers for reusability
'''

import torch.nn as nn


class CustomLayers(nn.Module):

    def __init__(self):
        super(CustomLayers, self).__init__()
    
    #TODO: fix dropout parameters to 1
    def depth_wise_seperable_conv(self, 
                                n_in, 
                                n_out, 
                                kernel_size=3, 
                                kernels_per_layer=1, 
                                apply_batch_norm=True, 
                                apply_relu=True, 
                                padding=0, 
                                apply_dropout=True, 
                                dropout_value=0.1, 
                                use_bias=False
                                ):
        
        depthwise = nn.Conv2d(n_in, n_in * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=n_in, bias=use_bias)
        pointwise = nn.Conv2d(n_in * kernels_per_layer, n_out, kernel_size=1, bias=use_bias)
        relu = nn.ReLU() if apply_relu==True else None
        batch_norm = nn.BatchNorm2d(n_out) if apply_batch_norm==True else None
        dropout  = nn.Dropout(dropout_value) if apply_dropout == True else None

        # add any new parameter to the layers list
        layers = [
                  depthwise,
                  pointwise,
                  batch_norm,
                  relu,
                  dropout
        ]

        layers = [_ for _ in layers if _!= None]


        conv = nn.Sequential(
            *layers
        )
        return conv
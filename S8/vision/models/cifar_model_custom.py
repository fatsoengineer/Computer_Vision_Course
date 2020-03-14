from .custom_models import CustomLayers
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F


class Cifar(nn.Module):
    def __init__(self):
        super(Cifar, self).__init__()
        self.custom_layers  = CustomLayers()

        dropout_value = .1

        self.conv1 = nn.Sequential(
            self.custom_layers.depth_wise_seperable_conv(
                n_in=3,
                n_out=16,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),

            self.custom_layers.depth_wise_seperable_conv(
                n_in=16,
                n_out=16,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),

            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=2, bias=False, dilation=2),
            # nn.BatchNorm2d(16),
            # nn.Dropout(dropout_value),
            # nn.ReLU(),

            nn.MaxPool2d(2,2),
        )

        self.conv2 = nn.Sequential(
            self.custom_layers.depth_wise_seperable_conv(
                n_in=16,
                n_out=32,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),

            self.custom_layers.depth_wise_seperable_conv(
                n_in=32,
                n_out=32,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),


            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=2 ,bias=False, dilation=2),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            # self.custom_layers.depth_wise_seperable_conv(
            #     n_in=64,
            #     n_out=64,
            #     kernel_size=3,
            #     kernels_per_layer=1,
            #     apply_batch_norm=True,
            #     apply_relu=True,
            #     padding=1,
            #     apply_dropout=True,
            #     dropout_value=dropout_value),

            nn.MaxPool2d(2,2),
        )

        self.conv3 = nn.Sequential(
            self.custom_layers.depth_wise_seperable_conv(
                n_in=32,
                n_out=64,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),

            self.custom_layers.depth_wise_seperable_conv(
                n_in=64,
                n_out=64,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),

             self.custom_layers.depth_wise_seperable_conv(
                n_in=64,
                n_out=64,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),
            nn.MaxPool2d(2,2),

        )

        self.conv4 = nn.Sequential(
            self.custom_layers.depth_wise_seperable_conv(
                n_in=64,
                n_out=128,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=True,
                apply_relu=True,
                padding=1,
                apply_dropout=True,
                dropout_value=dropout_value),

            self.custom_layers.depth_wise_seperable_conv(
                n_in=128,
                n_out=128,
                kernel_size=3,
                kernels_per_layer=1,
                apply_batch_norm=False,
                apply_relu=False,
                padding=1,
                apply_dropout=False,
                dropout_value=dropout_value),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

      

        self.reducer = nn.Conv2d(128,10,1,bias=False)



    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.conv4(x)

        x= self.gap(x)
        x= self.reducer(x)
        x = x.view(-1, 10)
        return x



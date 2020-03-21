import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        dropout_value = .1
        print("Current Date/Time: ", datetime.now())
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 1), padding=0, bias=False), #26
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Dropout(dropout_value),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), #26
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Dropout(dropout_value),
        )

        self.conv21 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), #26
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        
        self.max_pool1 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding=0, bias=False), #26
        
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #26
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Dropout(dropout_value),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #26
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

        self.max_pool2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding=0, bias=False), #26
        
        )


        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #26
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), #26
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_value),
        )

       

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

      
        self.fc = nn.Linear(in_features=64, out_features=10, bias=False)
        # self.reducer = nn.Conv2d(64,10,1,bias=False)



    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv21(x1+x2)
        x4 = self.max_pool1(x1+x2+x3)
        x5 = self.conv3(x4)
        x6 = self.conv3(x4+x5)
        x7 = self.conv4(x4+x5+x6)
        x8 = self.max_pool2(x5+x6+x7)

        x9 = self.conv5(x8)
        x10 = self.conv5(x8+x9)
        x11 = self.conv6(x8+x9+x10)

        x12 = self.gap(x11)
        x12 = x12.view(-1, 64)
        x12 = self.fc(x12)
        

        return F.log_softmax(x12)

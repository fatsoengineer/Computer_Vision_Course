### FInal Accuracy
```

epoch: 20 
loss=0.008348315954208374 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 52.54it/s]

Test set: Average loss: 0.0192, Accuracy: 9946/10000 (99.46%)
```

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
       BatchNorm2d-7            [-1, 8, 28, 28]              16
         Dropout2d-8            [-1, 8, 28, 28]               0
         MaxPool2d-9            [-1, 8, 14, 14]               0
           Conv2d-10           [-1, 16, 12, 12]           1,152
             ReLU-11           [-1, 16, 12, 12]               0
      BatchNorm2d-12           [-1, 16, 12, 12]              32
        Dropout2d-13           [-1, 16, 12, 12]               0
           Conv2d-14           [-1, 16, 10, 10]           2,304
             ReLU-15           [-1, 16, 10, 10]               0
      BatchNorm2d-16           [-1, 16, 10, 10]              32
        Dropout2d-17           [-1, 16, 10, 10]               0
        MaxPool2d-18             [-1, 16, 5, 5]               0
           Conv2d-19             [-1, 24, 3, 3]           3,456
             ReLU-20             [-1, 24, 3, 3]               0
      BatchNorm2d-21             [-1, 24, 3, 3]              48
        Dropout2d-22             [-1, 24, 3, 3]               0
           Conv2d-23             [-1, 10, 3, 3]             240
        AvgPool2d-24             [-1, 10, 1, 1]               0
================================================================
Total params: 7,944
Trainable params: 7,944
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.52
Params size (MB): 0.03
Estimated Total Size (MB): 0.56
----------------------------------------------------------------
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:60: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.

```



### Log
```
  0%|          | 0/938 [00:00<?, ?it/s]epoch: 1 
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:60: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.2596367299556732 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.28it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0627, Accuracy: 9833/10000 (98.33%)

epoch: 2 
loss=0.031580567359924316 batch_id=937: 100%|██████████| 938/938 [00:13<00:00, 67.11it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0419, Accuracy: 9871/10000 (98.71%)

epoch: 3 
loss=0.012555301189422607 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 62.51it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0338, Accuracy: 9900/10000 (99.00%)

epoch: 4 
loss=0.013363540172576904 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.85it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0309, Accuracy: 9905/10000 (99.05%)

epoch: 5 
loss=0.13904759287834167 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.34it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0298, Accuracy: 9910/10000 (99.10%)

epoch: 6 
loss=0.2807323932647705 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.52it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0245, Accuracy: 9926/10000 (99.26%)

epoch: 7 
loss=0.07095541059970856 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.38it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0261, Accuracy: 9919/10000 (99.19%)

epoch: 8 
loss=0.033350005745887756 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.74it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0231, Accuracy: 9925/10000 (99.25%)

epoch: 9 
loss=0.052108779549598694 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.99it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0240, Accuracy: 9929/10000 (99.29%)

epoch: 10 
loss=0.05380229651927948 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 65.68it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0239, Accuracy: 9917/10000 (99.17%)

epoch: 11 
loss=0.031224608421325684 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 64.70it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0248, Accuracy: 9925/10000 (99.25%)

epoch: 12 
loss=0.055045679211616516 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 63.46it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0220, Accuracy: 9933/10000 (99.33%)

epoch: 13 
loss=0.2000570446252823 batch_id=937: 100%|██████████| 938/938 [00:14<00:00, 63.00it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0208, Accuracy: 9936/10000 (99.36%)

epoch: 14 
loss=0.08786655962467194 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 60.43it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0204, Accuracy: 9944/10000 (99.44%)

epoch: 15 
loss=0.041443005204200745 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.14it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0193, Accuracy: 9947/10000 (99.47%)

epoch: 16 
loss=0.031190738081932068 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 55.77it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0178, Accuracy: 9946/10000 (99.46%)

epoch: 17 
loss=0.12824541330337524 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.47it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0187, Accuracy: 9942/10000 (99.42%)

epoch: 18 
loss=0.22803552448749542 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 59.58it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0195, Accuracy: 9934/10000 (99.34%)

epoch: 19 
loss=0.019069239497184753 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 52.32it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0190, Accuracy: 9939/10000 (99.39%)

epoch: 20 
loss=0.008348315954208374 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 52.54it/s]

Test set: Average loss: 0.0192, Accuracy: 9946/10000 (99.46%)

```
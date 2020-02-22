## Step - 1

**Target:**

> Create base model

**Results**


> 
*   Total params: 10,368
*   Best Test Accuracy: 
```
EPOCH: 14
Loss=0.044927358627319336 Batch_id=468 Accuracy=99.47: 100%|██████████| 469/469 [00:10<00:00, 43.20it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0316, Accuracy: 9903/10000 (99.03%)
```


**Analysis**

> * Created a model with three blocks and tried to keep the number of parameters around 10000, so, it wont be too hard to again reduce number of layers or parameters.
* Because I chose three blocks, I used two max pooling, and keeping the size of image in mind, added paading in the first block
* Ran multiple times to confirm that the results are not fluke or not fluctuating from each training at large 





## Step 2

 **Target:**

> After Creation of Base Model, Reduce number of parameters


**Results**


> 
*   Total params: 9,648
*   Best Test Accuracy: 
```
EPOCH: 13
Loss=0.02335897646844387 Batch_id=468 Accuracy=99.34: 100%|██████████| 469/469 [00:12<00:00, 38.92it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0317, Accuracy: 9905/10000 (99.05%)
```


**Analysis**

> * Played around with no of output channels size to reduce parameters keeping in mind that that the results dont fall behind much from the step 1
* Ran multiple times to confirm that the results are not fluke or not fluctuating from each training at large 



## Step 3

 **Target:**

> After Reduction of parameters, To add batch normalisation and dropout to increase efficiency and decrease over-fitting


**Results**


> 
*   Total params: 9,792
*   Best Test Accuracy: 
```
EPOCH: 12
Loss=0.0010679016122594476 Batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:11<00:00, 41.99it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0215, Accuracy: 9938/10000 (99.38%)
```


**Analysis**

> * Test Accuracy increased significantly
* Tested for both Batch norm before relu and batch norm after relu. Batch norm before relu was giving results better
* Over-fitting decreased
* Ran multiple times to confirm that the results are not fluke or not fluctuating from each training at large 



## Step 4

 **Target:**

> After adding batch normalisation, GAP layer to be added and increase capacity


**Results**


> 
*   Total params: 9,496
*   Best Test Accuracy: 
```
EPOCH: 11
Loss=0.013936594128608704 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:12<00:00, 37.66it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0302, Accuracy: 9907/10000 (99.07%)
```


**Analysis**

> * Model performance reduced after ading GAP layer and increasing capacity
* Ran multiple times to confirm that the results were very fluctuating and was not reliable. Came to te conclusion that this model will not work and there is a need to change my model.


## Step 5

 **Target:**

> Restructure model as the model started fluctuating after adding GAP layer and increasing capacity


**Results**


> 
*   Total params: 6,376
*   Best Test Accuracy: 
```
EPOCH: 10
Loss=0.01986701786518097 Batch_id=937 Accuracy=98.65: 100%|██████████| 938/938 [00:15<00:00, 59.50it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0260, Accuracy: 9932/10000 (99.32%)
```


**Analysis**

> * Restructured the model to remove a block to increase depth. Removed padding to avoid increasing unnecessary depth. Got better and consistant result.
* Ran multiple times to confirm that the results are not fluke or not fluctuating from each training at large 



## Step 6

 **Target:**

> Add image Augmentation


**Results**


> 
*   Total params: 6,376
*   Best Test Accuracy: 
```
EPOCH: 14
Loss=0.0072428882122039795 Batch_id=937 Accuracy=98.68: 100%|██████████| 938/938 [00:15<00:00, 67.61it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0191, Accuracy: 9944/10000 (99.44%)
```


**Analysis**

> * Added RandomRotation, which helped increase accuracy.
* Also had added RandomErasing, but that reduced accuracy a lot, so removed it.
* Ran multiple times to confirm that the results are not fluke or not fluctuating from each training at large 



## Step 7

 **Target:**

> Adding LR scheduler


**Results**


> 
*   Total params: 6,376
*   Best Test Accuracy: 
```
EPOCH:14 | LR: [0.008679444396382234]
Loss=0.018181532621383667 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:16<00:00, 55.82it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0178, Accuracy: 9947/10000 (99.47%)
```


**Analysis**

> * Tried Step LR or ReduceLrOnPlateau schedulers to check the results, but got the best result with OneCycleLR.
* Ran multiple times to confirm that the results are not fluke or not fluctuating from each training at large 






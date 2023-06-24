**This is the submission for assigment number 8 of ERA V1 course.**

**Problem Statement** <br/>
The Task given was to use CIFAR 10 data and get the convolutional network with atleast 70% accuracy. 

The number of parameters had to be less than 50,000 parameters. 

It was also asked to use Batch Normalization, Group Normalization and Layer Normalization and observe the results. 

**File Structure** <br />
model.py - has different classes for batch normalization, group normalization and layer normalization. Also as asked, I have also provided the networks used for Assignment 6 and 7. The names are as follows: <br />
     Net_batch_normalization <br />
     Net_group_norm <br />
     Net_layer_normalization <br />
     Net_s6 <br />
     Net_s7 <br />

The description of the data is as follows:

Dataset CIFAR10
    Number of datapoints: 50000 <br />
    Root location: ./data <br />
    Split: Train <br />
    StandardTransform <br />
Transform: Compose( <br />
               RandomAutocontrast(p=0.1) <br />
               RandomRotation(degrees=[-7.0, 7.0], interpolation=nearest, expand=False, fill=1) <br />
               ToTensor() <br />
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) <br />
           ) <br />
Dataset CIFAR10 <br />
    Number of datapoints: 10000 <br />
    Root location: ./data <br />
    Split: Test <br />
    StandardTransform <br />
Transform: Compose( <br />
               ToTensor() <br />
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) <br />
           ) <br />

Following are the sample images of train dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/train_dataset.jpg" alt="alt text" width="600px">

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/test_dataset.jpg" alt="alt text" width="600px">




**PARAMETERS FOR BATCH NORMALIZARTION ARCHITECTURE**

---------------------------------------------------------------- <br />
|       Layer (type)     |&nbsp;&nbsp;          Output Shape   |&nbsp;&nbsp;     Param # <br />
|------------------------|-------------------------------------|---------------------------| <br />        
================================================================== <br />
|            Conv2d-1     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;         1,344 <br />

|              ReLU-2     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;             0 <br />
|       BatchNorm2d-3     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;            96 <br />
|         Dropout2d-4     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;             0 <br />
|            Conv2d-5     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;        20,784 <br />
|              ReLU-6     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;             0 <br />
|       BatchNorm2d-7     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;            96 <br />
|         Dropout2d-8     |&nbsp;&nbsp;      [-1, 48, 32, 32]  |&nbsp;&nbsp;             0 <br />
|            Conv2d-9     |&nbsp;&nbsp;      [-1, 32, 32, 32]  |&nbsp;&nbsp;         1,568 <br />
|        MaxPool2d-10     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-11     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;         9,248 <br />
|             ReLU-12     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;             0 <br />
|      BatchNorm2d-13     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;            64 <br />
|        Dropout2d-14     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-15     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;         9,248 <br />
|             ReLU-16     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;             0 <br />
|      BatchNorm2d-17     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;            64 <br />
|        Dropout2d-18     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-19     |&nbsp;&nbsp;      [-1, 32, 16, 16]  |&nbsp;&nbsp;         1,056 <br />
|        MaxPool2d-20     |&nbsp;&nbsp;        [-1, 32, 8, 8]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-21     |&nbsp;&nbsp;        [-1, 16, 8, 8]  |&nbsp;&nbsp;         4,624 <br />
|             ReLU-22     |&nbsp;&nbsp;        [-1, 16, 8, 8]  |&nbsp;&nbsp;             0 <br />
|      BatchNorm2d-23     |&nbsp;&nbsp;        [-1, 16, 8, 8]  |&nbsp;&nbsp;            32 <br />
|        Dropout2d-24     |&nbsp;&nbsp;        [-1, 16, 8, 8]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-25     |&nbsp;&nbsp;         [-1, 8, 8, 8]  |&nbsp;&nbsp;         1,160 <br />
|             ReLU-26     |&nbsp;&nbsp;         [-1, 8, 8, 8]  |&nbsp;&nbsp;             0 <br />
|      BatchNorm2d-27     |&nbsp;&nbsp;         [-1, 8, 8, 8]  |&nbsp;&nbsp;            16 <br />
|        Dropout2d-28     |&nbsp;&nbsp;         [-1, 8, 8, 8]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-29     |&nbsp;&nbsp;        [-1, 10, 8, 8]  |&nbsp;&nbsp;           730 <br />
|             ReLU-30     |&nbsp;&nbsp;        [-1, 10, 8, 8]  |&nbsp;&nbsp;             0 <br />
|      BatchNorm2d-31     |&nbsp;&nbsp;        [-1, 10, 8, 8]  |&nbsp;&nbsp;            20 <br />
|        Dropout2d-32     |&nbsp;&nbsp;        [-1, 10, 8, 8]  |&nbsp;&nbsp;             0 <br />
|AdaptiveAvgPool2d-33     |&nbsp;&nbsp;        [-1, 10, 1, 1]  |&nbsp;&nbsp;             0 <br />
|           Conv2d-34     |&nbsp;&nbsp;        [-1, 10, 1, 1]  |&nbsp;&nbsp;           110 <br />


================================================================ <br />
Total params: 50,260 <br />
Trainable params: 50,260 <br />
Non-trainable params: 0 <br />

---------------------------------------------------------------- <br />
Input size (MB): 0.01 <br />
Forward/backward pass size (MB): 3.96 <br />
Params size (MB): 0.19 <br />
Estimated Total Size (MB): 4.16 <br />

---------------------------------------------------------------- <br />


** TRAIN ACCURACY:  72.792 TRAIN LOSS:  0.8170133829116821 ** <br />
** TEST ACCURACY:  74.02 TEST LOSS:  0.7378784431934357  ** <br />

Following are the sample images of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/batch_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

**PARAMETERS FOR BATCH Group Normalization ARCHITECTURE** <br />

---------------------------------------------------------------- <br />
        Layer (type)    |&nbsp;&nbsp;           Output Shape   |&nbsp;&nbsp;      Param # <br />
        
================================================================ <br />
            Conv2d-1    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;        1,344 <br />
              ReLU-2    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;            0 <br />
         GroupNorm-3    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;           96 <br />
         Dropout2d-4    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;            0 <br />
            Conv2d-5    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;       20,784 <br />
              ReLU-6    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;            0 <br />
         GroupNorm-7    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;           96 <br />
         Dropout2d-8    |&nbsp;&nbsp;       [-1, 48, 32, 32]   |&nbsp;&nbsp;            0 <br />
            Conv2d-9    |&nbsp;&nbsp;       [-1, 32, 32, 32]   |&nbsp;&nbsp;        1,568 <br />
        MaxPool2d-10    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;            0 <br />
           Conv2d-11    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;        9,248 <br />
             ReLU-12    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;            0 <br />
        GroupNorm-13    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;           64 <br />
        Dropout2d-14    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;            0 <br />
           Conv2d-15    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;        9,248 <br />
             ReLU-16    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;            0 <br />
        GroupNorm-17    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;           64 <br />
        Dropout2d-18    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;            0 <br />
           Conv2d-19    |&nbsp;&nbsp;       [-1, 32, 16, 16]   |&nbsp;&nbsp;       1,056 <br />
        MaxPool2d-20    |&nbsp;&nbsp;         [-1, 32, 8, 8]   |&nbsp;&nbsp;            0 <br />
           Conv2d-21    |&nbsp;&nbsp;         [-1, 16, 8, 8]   |&nbsp;&nbsp;        4,624 <br />
             ReLU-22    |&nbsp;&nbsp;         [-1, 16, 8, 8]   |&nbsp;&nbsp;            0 <br />
        GroupNorm-23    |&nbsp;&nbsp;         [-1, 16, 8, 8]   |&nbsp;&nbsp;           32 <br />
        Dropout2d-24    |&nbsp;&nbsp;         [-1, 16, 8, 8]   |&nbsp;&nbsp;            0 <br />
           Conv2d-25    |&nbsp;&nbsp;          [-1, 8, 8, 8]   |&nbsp;&nbsp;        1,160 <br />
             ReLU-26    |&nbsp;&nbsp;          [-1, 8, 8, 8]   |&nbsp;&nbsp;            0 <br />
        GroupNorm-27    |&nbsp;&nbsp;          [-1, 8, 8, 8]   |&nbsp;&nbsp;           16 <br />
        Dropout2d-28    |&nbsp;&nbsp;          [-1, 8, 8, 8]   |&nbsp;&nbsp;            0 <br />
           Conv2d-29    |&nbsp;&nbsp;         [-1, 10, 8, 8]   |&nbsp;&nbsp;          730 <br />
             ReLU-30    |&nbsp;&nbsp;         [-1, 10, 8, 8]   |&nbsp;&nbsp;            0 <br />
        GroupNorm-31    |&nbsp;&nbsp;         [-1, 10, 8, 8]   |&nbsp;&nbsp;           20 <br />
        Dropout2d-32    |&nbsp;&nbsp;         [-1, 10, 8, 8]   |&nbsp;&nbsp;            0 <br />
AdaptiveAvgPool2d-33    |&nbsp;&nbsp;         [-1, 10, 1, 1]   |&nbsp;&nbsp;           0 <br />
           Conv2d-34    |&nbsp;&nbsp;         [-1, 10, 1, 1]   |&nbsp;&nbsp;          110 <br />
           
================================================================ <br />
Total params: 50,260 <br />
Trainable params: 50,260 <br />
Non-trainable params: 0 <br />

---------------------------------------------------------------- <br />
Input size (MB): 0.01 <br />
Forward/backward pass size (MB): 3.96 <br />
Params size (MB): 0.19 <br />
Estimated Total Size (MB): 4.16

---------------------------------------------------------------- <br />

** TRAIN ACCURACY:  71.612 TRAIN LOSS:  0.4181869626045227 ** <br />
** TEST ACCURACY:  71.23 TEST LOSS:  0.8142993453979492 ** <br />


Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------




**PARAMETERS FOR LAYER NORMALIZARTION ARCHITECTURE**

---------------------------------------------------------------- <br />
        Layer (type)               Output Shape         Param # <br />
        
================================================================ <br />
            Conv2d-1           [-1, 10, 32, 32]             280 <br />
              ReLU-2           [-1, 10, 32, 32]               0 <br />
         LayerNorm-3           [-1, 10, 32, 32]          20,480 <br />
         Dropout2d-4           [-1, 10, 32, 32]               0 <br />
            Conv2d-5            [-1, 8, 32, 32]             728 <br />
              ReLU-6            [-1, 8, 32, 32]               0 <br />
         LayerNorm-7            [-1, 8, 32, 32]          16,384 <br />
         Dropout2d-8            [-1, 8, 32, 32]               0 <br />
            Conv2d-9           [-1, 16, 32, 32]             144 <br />
        MaxPool2d-10           [-1, 16, 16, 16]               0 <br />
           Conv2d-11            [-1, 8, 16, 16]           1,160 <br />
             ReLU-12            [-1, 8, 16, 16]               0 <br />
        LayerNorm-13            [-1, 8, 16, 16]           4,096 <br />
        Dropout2d-14            [-1, 8, 16, 16]               0 <br />
           Conv2d-15           [-1, 16, 16, 16]             144 <br />
        MaxPool2d-16             [-1, 16, 8, 8]               0 <br />
           Conv2d-17              [-1, 8, 8, 8]           1,160 <br />
             ReLU-18              [-1, 8, 8, 8]               0 <br />
        LayerNorm-19              [-1, 8, 8, 8]           1,024 <br />
        Dropout2d-20              [-1, 8, 8, 8]               0 <br />
           Conv2d-21              [-1, 4, 8, 8]             292 <br />
             ReLU-22              [-1, 4, 8, 8]               0 <br />
        LayerNorm-23              [-1, 4, 8, 8]             512 <br />
        Dropout2d-24              [-1, 4, 8, 8]               0 <br />
           Conv2d-25             [-1, 10, 8, 8]             370 <br />
             ReLU-26             [-1, 10, 8, 8]               0 <br />
        LayerNorm-27             [-1, 10, 8, 8]           1,280 <br />
        Dropout2d-28             [-1, 10, 8, 8]               0 <br />
AdaptiveAvgPool2d-29             [-1, 10, 1, 1]               0 <br />
           Conv2d-30             [-1, 10, 1, 1]             110 <br />
           
================================================================ <br />
Total params: 48,164 <br />
Trainable params: 48,164 <br />
Non-trainable params: 0 <br />

---------------------------------------------------------------- <br />
Input size (MB): 0.01 <br />
Forward/backward pass size (MB): 0.86 <br />
Params size (MB): 0.18 <br />
Estimated Total Size (MB): 1.06 <br />

---------------------------------------------------------------- <br />


** TRAIN ACCURACY:  53.694 TRAIN LOSS:  1.3298343420028687 **
** TEST ACCURACY:  55.56 TEST LOSS:  1.2183823497772217   **

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

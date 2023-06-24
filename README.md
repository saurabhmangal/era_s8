<h1> This is the submission for assigment number 8 of ERA V1 course.</h1>

<h2>Problem Statement</h2>
The Task given was to use CIFAR 10 data and get the convolutional network with atleast 70% accuracy. 

The number of parameters had to be less than 50,000 parameters. 

It was also asked to use Batch Normalization, Group Normalization and Layer Normalization and observe the results. 

<h2>File Structure</h2>
model.py - has different classes for batch normalization, group normalization and layer normalization. Also as asked, I have also provided the networks used for Assignment 6 and 7. The names are as follows: <br />
     Net_batch_normalization <br />
     Net_group_norm <br />
     Net_layer_normalization <br />
     Net_s6 <br />
     Net_s7 <br />

<h2>The description of the data is as follows:</h2>

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

<h2>Following are the sample images of train dataset:</h2>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/train_dataset.jpg" alt="alt text" width="600px">

<h2>Following are the sample images of the test dataset:</h2>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/test_dataset.jpg" alt="alt text" width="600px">

<h2>PARAMETERS FOR BATCH NORMALIZARTION ARCHITECTURE</h2>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/batch_norm_param.jpg" alt="alt text" width="600px">

<h3>TRAIN ACCURACY:  72.792 TRAIN LOSS:  0.8170133829116821 </h3>
<h3> TEST ACCURACY:  74.02 TEST LOSS:  0.7378784431934357  </h3>

<h3>Following are the loss:</h3>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/batch_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

**PARAMETERS FOR BATCH Group Normalization ARCHITECTURE** <br />

<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm_param.jpg" alt="alt text" width="600px">

** TRAIN ACCURACY:  71.612 TRAIN LOSS:  0.4181869626045227 ** <br />
** TEST ACCURACY:  71.23 TEST LOSS:  0.8142993453979492 ** <br />


Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------


**PARAMETERS FOR LAYER NORMALIZARTION ARCHITECTURE**
<img src="https://github.com/saurabhmangal/era_s8/blob/master/layer_norm_param.jpg" alt="alt text" width="600px">

** TRAIN ACCURACY:  53.694 TRAIN LOSS:  1.3298343420028687 **
** TEST ACCURACY:  55.56 TEST LOSS:  1.2183823497772217   **

Following are the sample imagese of the test dataset:
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

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

<h4>TRAIN ACCURACY:  72.792 TRAIN LOSS:  0.81 </h4>
<h4> TEST ACCURACY:  74.02 TEST LOSS:  0.74  </h4>

<h4>Following are the loss and accuracies of train and test for batch  normalization:</h4>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/batch_norm.jpg" alt="alt text" width="600px">


<h4>Following are thee 10 images with miss classification. Here p--> predicted and a--> actual:</h4>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/mis_classified_image.jpg" alt="alt text">
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

<h2>PARAMETERS FOR GROUP NORMALIZARTION ARCHITECTURE</h2>

<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm_param.jpg" alt="alt text" width="600px">

 <h4>TRAIN ACCURACY:  71.6 TRAIN LOSS:  0.42 </h4>
 <h4>TEST ACCURACY:  71.23 TEST LOSS:  0.81 </h4>


<h4>Following are the loss and accuracies of train and test for group normalization:</h4>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

<h2>PARAMETERS FOR LAYER NORMALIZARTION ARCHITECTURE</h2>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/layer_norm_param.jpg" alt="alt text" width="600px">

<h4>TRAIN ACCURACY:  53.7  TRAIN LOSS:  1.33</h4>
<h4>TEST ACCURACY :  55.6  TEST LOSS:  1.23</h4>

<h4>Following are the loss and accuracies of train and test for layer normalization:</h4>
<img src="https://github.com/saurabhmangal/era_s8/blob/master/group_norm.jpg" alt="alt text" width="600px">

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

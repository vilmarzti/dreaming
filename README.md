# DREAMING

This repository is for my application of the "Few-Shot-Based-Training" style transfer repository. The application of that repository is straight forward.
As I wanted to create some rather complex video with it i needed to do segmentation. The biggest part of this repository is the segmentation part, where I use different models 
to segment the video in different parts. This segmentation masks are later used to train the style transfer on different parts of the video. And then also used to overlay all the parts of the model.

## Requirments

I used the requirements from the "Few-Shots-Based-Training" video. For any additional pyhton libraries I created I requirements.txt.

As this repo trains (rather simple) Neural Networks you need access to a graphics card. 

The `segmentation`folder contains the python library that I used to execute the scrips. Install them with `pip install -e .`

## Workflow

This is a rough overview of the workflow

* Find a Video
* Segment the Video
    * Create Segmentation Dataset
    * Use that dataset to create segmentations for your video
        * Train the various segmentation algorithms to create masks for all frames
        * Use ensemble learning to aggregate the different masks
* Train the style-transfer on your segments and create the corresponding video-frames
* Patch the different segment-videos together

### Find a Video

Find a video you like to apply the style transfer to and use ffmpeg to create the corresponding frames for the video. Best is to create a directory where you save all the results and intermidiate steps are saved. Apply Gamma corection and what else you need to get a clear baseline from the frames.

### Segment the Video

The segmentation works by using an ensemble of different segmentation techniques. Some ML based and some based on optical computation.

#### Create a segmentation dataset

First you need a small dataset on that we're able to train our segmentation NN. So pick a handful of frames of your video and create segmentation masks with your favourite Image-processing program. As a reference for ~1500 frames I created 15 segmentations. 

Having done that split the frames into two parts. One for training and validating the simple NN segmentations and one for the ensemble-learning. I used a slightly bigger dataset for ensemble as that will give me the final result.

For the ensemble-learning and NN methods create each a folder called `train_input train_output valid_input valid_output test_input test_output` with the corresponding train, valid and test dataset.

Also create a folder for the background-subtraction algorithm. This should include the the segmentation masks that are used for training the NN's.

The directories should have a similar pattern to this. Note the `backsub` folder which includes the above mentioned masks for the background subtraction algorithm.

```
.
├── backsub
├── ensemble
│   ├── test_input
│   ├── test_output
│   ├── train_input
│   ├── train_output
│   ├── valid_input
│   └── valid_output
└── nn
    ├── test_input
    ├── test_output
    ├── train_input
    ├── train_output
    ├── valid_input
    └── valid_output

```


#### Train the models on the segmentation dataset

##### Train the convolutional-NN

This is a a rather simple Convolution Neural networks with mulitple hyper-paramters. It is located at `segmentation/models/cnn.py` to find the right hyper-parameters use the script provided in 
`scripts/models/cnn/hyper_cnn.py`

##### Train UNet

An implementation of the UNET architecture, this has maximally 3 down and up convolutions but your are invited to try more. The model is located at `segmentation/models/unet.py`. Use the script provided in `scirpts/models/unet/hyper_unet.py`to find the right hyper-parameters and the right model.

#### Create masks using various models or algorithms.

After training the ConvNet and UNet we are ready to create the segmentation masks for our video

##### Background Subtraction

Here I use a background-subtraction algorithm to create masks. As I noticed that I have a rather static foreground. So I used a technique where I fed in the created foreground-masks back into the background-subtraction algorithm. See the script in `scripts/masks/backsub.py` for implementaion details and execute it to create the appropriate masks. This script uses reference segmentations to bootstrap the algorithm and keep in on track. Only use the segmentations you use for training UNet and ConvNet. Else the ensemble learning might place to much confidence in the backtground subtraction.

##### GMMs

GMMs are use to model the color distribution of the frames. Use the scirpt `scripts/masks/find_num_gmm.py` to find the appropriate number of components in your gmm. This algorithm is mostly applicable if your foreground and background have the same color distribution over the length of the video.

TODO: creating masks

##### GrabCut

GrabCut is an algorithm for segmentation, you give it a rough estimate of your segmentation and it executes a finer segmentation based on the pixel-values in your frame. This script is fine-tuned for my spefic video. You probably have to change the different parameters for it to work with a different video.

####  Use ensemble learning to aggregate the different masks

TODO

### Style-Transfer

TODO

### Patch it all together

TODO

## Structure of the segmentation folder

TODO

## Structure of the scirpts folder

TODO

## Various todo's

* Expand comments
* Complete README
* Evaluation of UNet and ConvNet
* Masks with UNet and ConvNet
* Aggregating masks to dataset
* Ensemble learning (Hyperparams/Model/evaluation)
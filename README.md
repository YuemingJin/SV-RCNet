# SV-RCNet
SV-RCNet: Workflow Recognition from Surgical Videos using Recurrent Convolutional Network (TMI 2017)

by Yueming Jin, Qi Dou, Hao Chen, Lequan Yu, Chi-Wing Fu, Jing Qin and Pheng Ann Heng

## Introduction
The SV-RCNet repository contains the codes used in 2016 M2CAI workflow challenge and our SV-RCNet paper. Our method ranks the first in the [M2CAI challenge](http://camma.u-strasbg.fr/m2cai2016/index.php/workflow-challenge-results/) and achieves a promising performance in one large surgical dataset, i.e., Cholec80 dataset.

The implementation is based on Ubuntu 14.04, CUDA 8.0, cuDNN 5.0, Anaconda 2.7 and Caffe.

## Installation
1. Clone the SV-RCNet repository
    ```shell
    git clone https://github.com/YuemingJin/SV-RCNet.git
    ```
2. Build
    ```shell
    cd SV-RCNet
    cp Makefile.config.example Makefile.config
    # Adjust Makefile.config
    # Or directly use provided 'Makefile.config' file in the folder, which we have adjusted the necessary configurations, such as setting "WITH_PYTHON_LAYER := 1".
    make all -j8
    make pycaffe
    ```
*Note:*
- Please first install Anaconda 2.7 following [official instructions](https://www.anaconda.com/). In addition, adjust path in your 'Makefile.config' file.
- For other installation issues, please follow the official instructions of [Caffe](http://caffe.berkeleyvision.org/installation.html).

## Step by Step Recognition
Most related codes are in 'surgicalVideo/' folder.

1. Download data

Cholec80 dataset or M2CAI dataset

1. Preprocess data

Download [ffmpeg](https://www.johnvansickle.com/ffmpeg/) and use ffmpeg to split the videos to image. We split the videos in 1 fps for Cholec80 and only split video01 as an example.
    ```shell
    cd surgicalVideo
    sh split_video_to_image.sh 
    ```
*Note: may need to modify the ground truth file (gt_file_Cholec80) according to the name of images you created.*

2. Training the network

- Download pre-trained ResNet-50 model at https://github.com/KaimingHe/deep-residual-networks.
Put it in 'models/ResNet-50/'.
- Enter 'models/ResNet-50' and modify path in 'ResNet-50-workflow-train-val.prototxt' and pre-trained model name in 'train_ResNet_50.sh'.
- Train ResNet-50
    ```shell
    sh train_ResNet_50.sh 
    ```
- The trained ResNet-50 will be saved in 'snapshot/' folder. Please choose and copy the model to the 'models/SV-RCNet/' folder as the next step pre-trained model when the loss does not decrease.
- Enter 'models/SV-RCNet' and modify pre-trained model name in 'train_SVRCNet.sh'.
- Train SV-RCNet
    ```shell
    sh train_SVRCNet.sh
    ```

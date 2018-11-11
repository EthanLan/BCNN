# [Bayesian Convolutional Neural Networks for Compressed Sensing Restoration]
Xinjie Lan, Xin Guo, and Kenneth E. Barner.

## Introduction
Bayesian Convolutional Neural Networks (BCNNs)is a new Compressed Sensing (CS) restoration algorithm that combining Convolutional Neural Networks (CNNs) and Bayesian inference method. In this paper, we show significant improvements in reconstruction results over classical Structured Compressed Sensing (SCS) algorithms and restoration methods based on neural networks, such as ReconNet, DR2Net, and LDAMP. The code provided here helps one to reproduce some of the results presented in the paper.

## Citation (BibTex):
If you are using this code, please cite the following paper.
```
@artical{BCNNs,
author = {Xinjie Lan and Xin Guo and Kenneth E. Barner},
title = {Bayesian Convolutional Neural Networks for Compressed Sensing Restoration},
booktitle = {arVix},
month = {Nov.},
year = {2018}
}
```
## System Requirements:
This software has been tested on Matlab R2018a.

## Reconstructing images from CS measurement using BCNNs:
The pre-trained models for measurement rates of 0.25, 0.1, 0.04 and 0.01 and the corresponding measurement matrices are provided in the ./ReconNet-master/test/caffemodels and ./ReconNet-master/test/phi directories respectively. ./ReconNet-master/test_images contains the test images (downloaded from two sources: http://dsp.rice.edu/software/DAMP-toolbox and http://see.xidian.edu.cn/faculty/wsdong/NLR_Exps.htm) used to produce the results in tables 1 and 2, and figures 3 and 4 in the paper. The reconstructed images for the different measurement rates are provided in ./ReconNet/test/reconstruction_results.

To reproduce the results, first select a measurement rate by editing ./ReconNet/test/test_everything.m accordingly and run test_everything.m. Please note the time complexity results provided in table 2 of the paper were produced using a NVIDIA GTX 980 GPU and hence may not necessarily match if you are using a different GPU. 

## Training models:
The network definition and parameters of the initial random weights of the network are provided in ./ReconNet-master/train/ReconNet_arch.prototxt and the optimization parameters in ./ReconNet-master/train/ReconNet_solver.prototxt.

1. Run generate_train.m from ./ReconNet-master/train/ directory in MATLAB to sample the image patches of size 33 by 33 which act as the training labels, and the corresponding  random Gaussian measurements (using a measurement matrix in ./ReconNet-master/phi directory) which act as training inputs for the network. The training inputs and labels will be saved in hdf5 format in ./ReconNet-master/train/train.h5. Similarly run ./ReconNet-master/train/generate_test.m to generate the validation set which will be saved in test.h5.

2. Open the terminal and run ./ReconNet-master/train/train.sh. A directory to save the caffemodels is created before the training begins.

## Contact:
Xinjie Lan, (lxjbit@udel.edu)

## Acknowledgements:
Our code is generated based on the [Field of Experts](https://www.visinf.tu-darmstadt.de/vi_research/code/index.en.jsp#foe) code.

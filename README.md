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
In the 'restoration' folder, there are four per-trained BCNN models. Each models have different parameters of BCNN. You can find the detailed information of these models in the experimental section of our paper. The 'demo_bcnns_cs.m' file uses these models to implement CS reconstruction. The default image dimension is 64 by 64, and CS measurement ratio is 0.25.

We compared BCNN model with two classical Structured Compressed Sensing methods and three neural network algorithms.

BCS (L. Yu, C. Wei, and G. Zheng, “Adaptive bayesian estimation with cluster structured sparsity,” Signal Proc. Letters, vol. 22, pp. 2309–2313, 2015.) 
https://sites.google.com/site/link2yulei/publications 

TV (C. Li, “An efficient algorithm for total variation regularization with applications to the single pixel camera and compressive sensing,” Master’s thesis, Rice University, Houston, Texas, 2009.) https://www.caam.rice.edu/~optimization/L1/TVAL3/

ReconNet (K. Kulkarni, S. Lohit, P. Turaga, R. Kerviche, and A. Ashok, “Reconnet: Non-iterative reconstruction of images from compressively sensed measurements,” in IEEE Conf. on CVPR, June 2016.)
 https://github.com/KuldeepKulkarni/ReconNet

DR2Net (H. Yao, F. Dai, D. Zhang, Y. Ma, S. Zhang, and Y. Zhang, “Dr2-net: Deep residual reconstruction network for image compressive sensing,” arXiv preprint arXiv:1702.05743, 2017.)
https://github.com/coldrainyht/caffe dr2/tree/master/DR2

LDAMP (C. A. Metzler, A. Mousavi, and R. G. Baraniuk, “Learned d-amp:principled neural network based compressive image recovery,” arxivpreprint arXiv:1704.06625, 2017.)
https://github.com/ricedsp/D-AMP_Toolbox

## Training models:
The network definition and parameters of the initial random weights of the network are provided in ./ReconNet-master/train/ReconNet_arch.prototxt and the optimization parameters in ./ReconNet-master/train/ReconNet_solver.prototxt.

1. Run generate_train.m from ./ReconNet-master/train/ directory in MATLAB to sample the image patches of size 33 by 33 which act as the training labels, and the corresponding  random Gaussian measurements (using a measurement matrix in ./ReconNet-master/phi directory) which act as training inputs for the network. The training inputs and labels will be saved in hdf5 format in ./ReconNet-master/train/train.h5. Similarly run ./ReconNet-master/train/generate_test.m to generate the validation set which will be saved in test.h5.

2. Open the terminal and run ./ReconNet-master/train/train.sh. A directory to save the caffemodels is created before the training begins.

## Contact:
Xinjie Lan, (lxjbit@udel.edu)

## Acknowledgements:
Our code is generated based on the [Field of Experts](https://www.visinf.tu-darmstadt.de/vi_research/code/index.en.jsp#foe) code.

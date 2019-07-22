## DMENet &mdash; Official TensorFlow Implementation
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.10](https://img.shields.io/badge/tensorflow-1.10-green.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-7.3.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

![Teaser image](./assets/figure.png)
**Picture:** *These people are not real &ndash; they were produced by our generator that allows control over different aspects of the image.*

This repository contains the official TensorFlow implementation of the following paper:

> **Deep Defocus Map Estimation using Domain Adaptation**<br>
> Junyong Lee (POSTECH), Sungkil Lee (Sungkyunkwan University), Sunghyun Cho (POSTECH) Seungyong Lee (POSTECH)<br>
> 
> http://cg.postech.ac.kr/papers/2019_CVPR_JY.pdf
>
> **Abstract:** *In this paper, we propose the first end-to-end convolutional neural network (CNN) architecture, Defocus Map Estimation Network (DMENet), for spatially varying defocus map estimation. To train the network, we produce a novel depth-of-field (DOF) dataset, SYNDOF, where each image is synthetically blurred with a ground-truth depth map. Due to the synthetic nature of SYNDOF, the feature characteristics of images in SYNDOF can differ from those of real defocused photos. To address this gap, we use domain adaptation that transfers the features of real defocused photos into those of synthetically blurred ones. Our DMENet consists of four subnetworks: blur estimation, domain adaptation, content preservation, and sharpness calibration networks. The subnetworks are connected to each other and jointly trained with their corresponding supervisions in an end-toend manner. Our method is evaluated on publicly available blur detection and blur estimation datasets and the results show the state-of-the-art performance.*

For any inquiries, please contact [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources

All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](http://cg.postech.ac.kr/papers/2019_CVPR_JY.pdf) |
| [Supplementary Files](http://stylegan.xyz/video) |
| Source code |
| SYNDOF Dataset |

## Getting Started
To train the network,
```bash
python main.py --is_train True --mode [model name]
```
To test the network,
```bash
python main.py --is_train False --mode [model name]
```
other options
```bash
--is_pretrain : ([True/False], pretrain the network with MSE loss first)
--delete_log : ([True/False], deletes checkpoint, summaries before start training)
```
modify config.py for more options

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition,ersonalization technology through userreference analysis is under study.  
    
Please checkout out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

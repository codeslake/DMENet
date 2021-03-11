# DMENet: Deep Defocus Map Estimation Network
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![TensorFlow 1.13.1](https://img.shields.io/badge/tensorflow-1.13.1-green.svg?style=plastic)
![TensorLayer 1.11.1](https://img.shields.io/badge/tensorlayer-1.11.1-green.svg?style=plastic)
![CUDA 10.0.130](https://img.shields.io/badge/CUDA-10.0.130-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

This repository contains the official TensorFlow implementation of the following paper:

> **[Deep Defocus Map Estimation using Domain Adaptation](http://cg.postech.ac.kr/papers/2019_CVPR_JY.pdf)**<br>
> Junyong Lee, Sungkil Lee, Sunghyun Cho and Seungyong Lee, CVPR2019

![Teaser image](./assets/figure.png)

## Getting Started
### Prerequisites
1. Download the docker image and run container: `docker pull codeslake/tensorflow-1.13.1:DME`.
> **Note:**
>
> The image is heavy as it is not organized.

2. Preparing datasets for training
Frist, download the dataset from [here](https://www.dropbox.com/s/s6ehxpvu9xxu9w8/datasets.zip?dl=0).

Initially, datasets should be placed in `./datasets` where each testing and training are separated as `./datasets/test` or `./datasets/train` (one may change the offset in `config.py`).

2. Preparing pretrained VGG19 for training
Download pretrained VGG19 checkpoint file from [here](https://www.dropbox.com/s/7ah1jwrmggog4q9/vgg19.zip?dl=0).
We need the pretrained VGG19 for the encoder of DMENet.

## Testing models of CVPR2019

Download pretrained weights for DMENet from [here](https://www.dropbox.com/s/p1xlr5jgj7oemt1/DMENet_BDCS.zip?dl=0).
Place the file under `./logs/DMENet_BDCS/checkpoint` (one may change the offset in `./config.py`).
Then, run the network by typing,

```bash
python main.py --is_train False --mode DMENet_BDCS
```

> **Note:**
>
> *Please note that due to server issue, checkpoint file used for the paper is lost.
The provided checkpoint file is the new checkpoint that shows the closest evaluation results as the checkpoint used in the paper.*

## Training & testing the network

To train the network:

```bash
python main.py --is_train True --mode DMENet_BDCS
```

To test the network:

```bash
python main.py --is_train False --mode DMENet_BDCS
```

* options
    * `--is_pretrain`: Pretrain the network with MSE loss (`True` | `False`). Default: `False`
    * `--delete_log`: Deletes logs such as checkpoints, scalar/image logs before training begins (`True` | `False`). Default: `False`


## Citation
If you find this code useful, please consider citing:

```
@InProceedings{Lee_2019_CVPR,
    author = {Lee, Junyong and Lee, Sungkil and Cho, Sunghyun and Lee, Seungyong},
    title = {Deep Defocus Map Estimation Using Domain Adaptation},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```

## Contact
Open an issue for any inquiries.
You may also have contact with [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources
All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://www.dropbox.com/s/pmjhh1ocugagwyh/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.pdf?dl=0) |
| [Supplementary Files](https://www.dropbox.com/s/van0beau0npq3de/supp.zip?dl=0) |
| [Checkpoint Files](https://www.dropbox.com/s/p1xlr5jgj7oemt1/DMENet_BDCS.zip?dl=0) |
| [Datasets](https://www.dropbox.com/s/s6ehxpvu9xxu9w8/datasets.zip?dl=0)|

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

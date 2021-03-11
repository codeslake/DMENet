# DMENet: Deep Defocus Map Estimation Network
![Python 3.6](https://img.shields.io/badge/Python-3.6.13-green.svg?style=plastic)
![TensorFlow 1.13.1](https://img.shields.io/badge/tensorflow-1.15.0-green.svg?style=plastic)
![TensorLayer 1.11.1](https://img.shields.io/badge/tensorlayer-1.11.1-green.svg?style=plastic)
![CUDA 10.0.130](https://img.shields.io/badge/CUDA-10.0.130-green.svg?style=plastic)
![CUDNN 7.6.](https://img.shields.io/badge/CUDNN-7.6.5-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

![Teaser image](./assets/figure.png)

This repository contains the official TensorFlow implementation of the following paper:

> **[Deep Defocus Map Estimation using Domain Adaptation](http://cg.postech.ac.kr/papers/2019_CVPR_JY.pdf)**<br>
> Junyong Lee, Sungkil Lee, Sunghyun Cho and Seungyong Lee, CVPR2019

<script src="https://anvil.works/embed.js" async></script>
<iframe style="width:100%;" data-anvil-embed src="https://2JI532DIZN4TSYWF.anvil.app/BIEWGFSFTYML53VXPQZBRNTX"></iframe>

## Getting Started
### Prerequisites
1. Setup Environment 
    * Option 1. Docker
        * Download the docker image and run container: `docker pull codeslake/ubuntu18.04-conda:DME`.
        * Inside container, activate `DMENet` environment (*i.e.*, `conda activate DMENet`).

    * Option 2. Manual (conda should be installed)
        ```bash
        $ conda create --name DMENet python=3.6
        $ conda activate DMENet
        $ conda install cudatoolkit=10.0
        $ conda install cudnn=7.6
        $ pip install tensorflow-gpu==1.13.1
        $ pip install tensorlayer==1.11.1
        ```

2. Install Requirement: `pip install -r requirments.txt`


3. Download [datasets](https://www.dropbox.com/s/s6ehxpvu9xxu9w8/datasets.zip?dl=1).
    * Initially, datasets should be placed in `./datasets` where each testing and training are separated as `./datasets/test` or `./datasets/train`

        > **Note:**
        >
        > The offset path can be configured by `offset` in `config.py`.

4. Download [pretrained weights of DMENet](https://www.dropbox.com/s/04lg03ogsto1fmw/DMENet_BDCS.zip?dl=1).
    * Place the file under `./logs/DMENet_BDCS/checkpoint`

        > **Note:**
        >
        > The offset path can be configured by `config.TRAIN.root_dir` in `config.py`.

5. Download [pretrained VGG19 weigths](https://www.dropbox.com/s/7ah1jwrmggog4q9/vgg19.zip?dl=1) (for training only).
    * Unzip the weight as `pretrained/vgg19.npy`.

## Testing models of CVPR2019

```bash
python main.py --is_train False --mode DMENet_BDCS
```

> **Note:**
>
> *Please note that due to server issue, the checkpoint used for the paper is lost.
> The provided checkpoint is the new checkpoint that shows the closest evaluation results as the checkpoint used in the paper.*

## Training & testing the network

To train the network:

```bash
python main.py --is_train True --mode DMENet_BDCS
```

To test the network:

```bash
python main.py --is_train False --mode DMENet_BDCS
```

* arguments
    * `--is_pretrain`: Pretrain the network with the MSE loss (`True` | `False`). Default: `False`
    * `--delete_log`: Deletes logs such as checkpoints and  scalar/image logs before training begins (`True` | `False`). Default: `False`


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
| [Paper PDF](https://www.dropbox.com/s/pmjhh1ocugagwyh/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.pdf?dl=1) |
| [Supplementary Files](https://www.dropbox.com/s/van0beau0npq3de/supp.zip?dl=1) |
| [Checkpoint Files](https://www.dropbox.com/s/p1xlr5jgj7oemt1/DMENet_BDCS.zip?dl=1) |
| [Datasets](https://www.dropbox.com/s/s6ehxpvu9xxu9w8/datasets.zip?dl=1)|

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

### Useful Links
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

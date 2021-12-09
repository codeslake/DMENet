# DMENet: Deep Defocus Map Estimation Network
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=flat)
[![License CC BY-NC](https://img.shields.io/badge/Anvil-Open_in_Anvil_(may_be_offline)-blue.svg?style=flat)](https://2JI532DIZN4TSYWF.anvil.app/BIEWGFSFTYML53VXPQZBRNTX)

This repository contains the official TensorFlow implementation of the following paper:

> **[Deep Defocus Map Estimation using Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.pdf)**<br>
> Junyong Lee, Sungkil Lee, Sunghyun Cho and Seungyong Lee, CVPR 2019

<p align="left">
  <a href="https://codeslake.github.io/publications/#DMENet">
    <img width=85% src="./assets/teaser_DMENet.gif" />
  </a>
</p>

## Getting Started
### Prerequisites
*Tested environment*

![Ubuntu](https://img.shields.io/badge/Ubuntu-16.04%20&%2018.04-blue.svg?style=plastic)
![Python 3.6](https://img.shields.io/badge/Python-3.6.13-green.svg?style=plastic)
![TensorFlow 1.15.0](https://img.shields.io/badge/tensorflow-1.15.0-green.svg?style=plastic)
![TensorLayer 1.11.1](https://img.shields.io/badge/tensorlayer-1.11.1-green.svg?style=plastic)
![CUDA 10.0.130](https://img.shields.io/badge/CUDA-10.0%20&%2011.1-green.svg?style=plastic)
![CUDNN 7.6.](https://img.shields.io/badge/CUDNN-7.6.5%20&%208.0.4-green.svg?style=plastic)

1. Setup environment
    ```bash
    $ git clone https://github.com/codeslake/DMENet.git
    $ cd DMENet

    # for CUDA10.0
    $ conda create -y --name DMENet python=3.6 && conda activate DMENet
    $ sh install_CUDA10.0.sh

    # for CUDA11.1 (the name of conda environment matters)
    $ conda create -y --name DMENet_CUDA11 python=3.6 && conda activate DMENet_CUDA11
    $ sh install_CUDA11.1.sh
    ```

3. Download and unzip datasets([option 1](https://www.dropbox.com/s/xkx1me8dvuv3xd0/datasets.zip?dl=1), [option 2](https://postechackr-my.sharepoint.com/:u:/g/personal/junyonglee_postech_ac_kr/EZsNnnayLAxNnT9UcM9GD8cBK65R8yXg9vyEd0lmKe88Zw)) under `[DATASET_ROOT]`.

    ```
    ├── [DATASET_ROOT]
    │   ├── train
    │   │   ├── SYNDOF
    │   │   ├── CUHK
    │   │   ├── Flickr
    │   ├── test
    │   │   ├── CUHK
    │   │   ├── RTF
    │   │   ├── SYNDOF
    ```

    > **Note:**
    >
    > * `[DATASET_ROOT]` is currently set to `./datasets/`. It can be specified by modifying [`config.data_offset`](https://github.com/codeslake/DMENet/blob/master/config.py#L35-L36) in `./config.py`.

4. Download [pretrained weights of DMENet](https://www.dropbox.com/s/04lg03ogsto1fmw/DMENet_BDCS.zip?dl=1) and unzip it as in `[LOG_ROOT]/DMENet_BDCS/checkpoint/DMENet_BDCS.npz` (`[LOG_ROOT]` is currently set to `./logs/`).

5. Download [pretrained VGG19 weights](https://www.dropbox.com/s/7ah1jwrmggog4q9/vgg19.zip?dl=1) and unzip as in `pretrained/vgg19.npy` (for training only).

### Logs
* Training and testing logs will be saved under `[LOG_ROOT]/[mode]/`:

    ```
    ├── [LOG_ROOT]
    │   ├── [mode]
    │   │   ├── checkpoint      # model checkpoint
    │   │   ├── log             # scalar/image log for tensorboard
    │   │   ├── sample          # sample images of training
    │   │   ├── result          # resulting images of evaluation
    ```

    > **Note:**
    >
    > * `[LOG_ROOT]` is currently set to `./logs/`. It can be specified by modifying [`config.root_offset`](https://github.com/codeslake/DMENet/blob/master/config.py#L73-L74) in `./config.py`.

## Testing final model of CVPR 2019
*Please note that due to the server issue, the checkpoint used for the paper is lost.
<br/>The provided checkpoint is the new checkpoint that shows the closest evaluation results as in the paper.*

*Check out [updated performance](/evaluation) with the new checkpoint.*

* Test the final model by:

    ```bash
    python main.py --mode DMENet_BDCS --test_set CUHK
    ```

    > **Note:**
    >
    > * Testing results will be saved in `[LOG_ROOT]/[mode]/result/[test_set]/`:
    >
    >   ```
    >   ...
    >   ├── [test_set]
    >   │   ├── image                     # input defocused images
    >   │   ├── defocus_map               # defocus images (network's direct output in range [0, 1])
    >   │   ├── defocus_map_min_max_norm  # min-max normalized defocus images in range [0, 1] for visualization
    >   │   ├── sigma_map_7_norm          # sigma maps containing normalized standard deviations (in range [0, 1]) for a Gaussian kernel. For the actual standard deviation value, one should multiply 7 to this map.
    >   ```
    > * Quantitative results are computed from matlab. (*e.g.*, [evaluation on the RTF dataset](https://github.com/codeslake/DMENet/tree/master/evaluation/RTF)).

    * Options
        * `--mode`: The name of a model to test. The logging folder named with the `[mode]` will be created as `[LOG_ROOT]/[mode]/`. Default: `DMENet_BDCS`
        * `--test_set`: The name of a dataset to evaluate. `CUHK` | `RTF0` | `RTF1` | `RTF1_6` | `random`. Default: `CUHK`
            * The folder structure can be modified in the function [`get_eval_path(..)`](https://github.com/codeslake/DMENet/blob/master/config.py#L85-L98) in `./config.py`.
            * `random` is for testing models with any images, which should be placed as `[DATASET_ROOT]/random/*.[jpg|png]`.

* Check out [the evaluation code for the RTF dataset](https://github.com/codeslake/DMENet/tree/master/evaluation/RTF), and [the deconvolution code](https://github.com/codeslake/DMENet/tree/master/deconvolution).



## Training & testing the network

* Train the network by:

    ```bash
    python main.py --is_train --mode [mode]
    ```

    > **Note:**
    >
    > * If you train DMENet with newly generated SYNDOF dataset from [this repo](https://github.com/codeslake/SYNDOF), comment [this line](https://github.com/codeslake/DMENet/blob/master/utils.py#L43) and uncomment [this line](https://github.com/codeslake/DMENet/blob/master/utils.py#L49) before the training.

* Test the network by:

    ```bash
    python main.py --mode [mode] --test_set [test_set]
    ```

    * arguments
        * `--mode`: The name of a model to train. The logging folder named with the `[mode]` will be created as `[LOG_ROOT]/[mode]/`. Default: `DMENet_BDCS`
        * `--is_pretrain`: Pretrain the network with the MSE loss (`True` | `False`). Default: `False`
        * `--delete_log`: Deletes `[LOG_ROOT]/[mode]/*` before training begins (`True` | `False`). Default: `False`


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

## Related Links
* CVPR 2021: Iterative Filter Adaptive Network for Single Image Defocus Deblurring \[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf)\]\[[code](https://github.com/codeslake/IFAN)\]
* ICCV 2021: Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions \[[paper](https://arxiv.org/pdf/2108.09108.pdf)\]\[[code](https://github.com/HyeongseokSon1/KPAC)\]

## Resources
All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.pdf) |
| [Supplementary Files](https://www.dropbox.com/s/van0beau0npq3de/supp.zip?dl=1) |
| [Checkpoint Files](https://www.dropbox.com/s/04lg03ogsto1fmw/DMENet_BDCS.zip?dl=1) |
| Datasets([option 1](https://www.dropbox.com/s/xkx1me8dvuv3xd0/datasets.zip?dl=1), [option 2](https://postechackr-my.sharepoint.com/:u:/g/personal/junyonglee_postech_ac_kr/EZsNnnayLAxNnT9UcM9GD8cBK65R8yXg9vyEd0lmKe88Zw))|
| [SYNDOF Generation Repo](https://github.com/codeslake/SYNDOF)|

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.

Please check out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

### Useful Links
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)


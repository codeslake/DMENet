## DMENet: Deep Defocus Map Estimation Network<br><sub>Official Implementation of the CVPR 2021 Paper</sub><br><sub>[Project](https://junyonglee.me/projects/DMENet) | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.pdf) | [Supp](https://www.dropbox.com/s/van0beau0npq3de/supp.zip?dl=1) | [Poster](https://www.dropbox.com/s/85umxea9uha3ptq/CVPR2019_poster.pdf?raw=1)</sub><br><sub><sub>[![License CC BY-NC](https://img.shields.io/badge/Anvil-Open_in_Anvil_(may_be_offline)-blue.svg?style=flat)](https://2JI532DIZN4TSYWF.anvil.app/BIEWGFSFTYML53VXPQZBRNTX)</sub></sub>

This repository contains the official matlab implementation of SYNDOF generation used in the following paper:

> [**Deep Defocus Map Estimation using Domain Adaptation**](https://junyonglee.me/projects/DMENet)<br>
> [Junyong Lee](https://junyonglee.me)<sup>1</sup>, [Sungkil Lee](http://cg.skku.edu/slee/)<sup>2</sup>, [Sunghyun Cho](https://www.scho.pe.kr/)<sup>3</sup>, and [Seungyong Lee](http://cg.postech.ac.kr/leesy/)<sup>1</sup><br>
> <sup>1</sup>POSTECH, <sup>2</sup>Sungkyunkwan University, <sup>3</sup>DGIST<br>
> *IEEE Computer Vision and Pattern Recognition (**CVPR**) 2019*<br>


<p align="left">
  <a href="https://junyonglee.me/projects/DMENet">
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
    * Option 1. install from scratch
        ```bash
        $ git clone https://github.com/codeslake/DMENet.git
        $ cd DMENet

        # for CUDA10
        $ conda create -y --name DMENet python=3.6 && conda activate DMENet
        $ sh install_CUDA10.0.sh

        # for CUDA11 (the name of conda environment matters)
        $ conda create -y --name DMENet_CUDA11 python=3.6 && conda activate DMENet_CUDA11
        $ sh install_CUDA11.1.sh
        ```

    * Option 2. docker
        ```bash
        $ nvidia-docker run --privileged --gpus=all -it --name DMENet --rm codeslake/dmenet:CVPR2019 /bin/zsh
        $ git clone https://github.com/codeslake/DMENet.git
        $ cd DMENet

        # for CUDA10
        $ conda activate DMENet

        # for CUDA11
        $ conda activate DMENet_CUDA11
        ```

3. Download and unzip datasets([Google Drive](https://drive.google.com/open?id=1M8Xt-T8jR8AQImpzEr6OCaXtCFGqj9sT&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/xkx1me8dvuv3xd0/datasets.zip?dl=1) | [OneDrive](https://postechackr-my.sharepoint.com/:u:/g/personal/junyonglee_postech_ac_kr/EZsNnnayLAxNnT9UcM9GD8cBK65R8yXg9vyEd0lmKe88Zw)) under `[DATASET_ROOT]`.

    ```
    [DATASET_ROOT]
     ├── train
     │   ├── SYNDOF
     │   ├── CUHK
     │   └── Flickr
     └── test
         ├── CUHK
         ├── RTF
         └── SYNDOF
    ```

    > **Note:**
    >
    > * `[DATASET_ROOT]` is currently set to `./datasets/`. It can be specified by modifying [`config.data_offset`](https://github.com/codeslake/DMENet/blob/master/config.py#L35-L36) in `./config.py`.

4. Download pretrained weights of DMENet ([Google Drive](https://drive.google.com/open?id=1MP47SAdcGkj5YnigOyBEAHJL76MgDiLi&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/04lg03ogsto1fmw/DMENet_BDCS.zip?dl=1)) and unzip it as in `[LOG_ROOT]/DMENet_CVPR2019/DMENet_BDCS/checkpoint/DMENet_BDCS.npz` (`[LOG_ROOT]` is currently set to `./logs/`).

5. Download pretrained VGG19 weights ([Google Drive](https://drive.google.com/open?id=1MnS3yFbWBZfQ6SW4kos73bxjKL7oeE-k&authuser=codeslake%40gmail.com&usp=drive_fs) | [Dropbox](https://www.dropbox.com/s/7ah1jwrmggog4q9/vgg19.zip?dl=1)) and unzip as in `pretrained/vgg19.npy` (for training only).

### Logs
* Training and testing logs will be saved under `[LOG_ROOT]/DMENet_CVPR2019/[mode]/`:

    ```
    [LOG_ROOT]
     └──DMENet_CVPR2019
        ├── [mode]
        │   ├── checkpoint      # model checkpoint
        │   ├── log             # scalar/image log for tensorboard
        │   ├── sample          # sample images of training
        │   └── result          # resulting images of evaluation
        └── ...
    ```
    > `[LOG_ROOT]` can be modified with [`config.root_offset`](https://github.com/codeslake/DMENet/blob/master/config.py#L73-L74) in `./config.py`.

## Testing final model of CVPR 2019
*Please note that due to the server issue, the checkpoint used for the paper is lost.
<br/>The provided checkpoint is the new checkpoint that shows the closest evaluation results as in the paper.*

*Check out [updated performance](/evaluation) with the new checkpoint.*

* Test the final model by:

    ```bash
    python main.py --mode DMENet_BDCS --test_set CUHK
    ```
    > Testing results will be saved in `[LOG_ROOT]/DMENet_CVPR2019/[mode]/result/[test_set]/`:
    >
    > ```
    > ...
    > [LOG_ROOT]/DMENet_CVPR2019/[mode]/result/
    >  └── [test_set]
    >      ├── image                     # input defocused images
    >      ├── defocus_map               # defocus images (network's direct output in range [0, 1])
    >      ├── defocus_map_min_max_norm  # min-max normalized defocus images in range [0, 1] for visualization
    >      └── sigma_map_7_norm          # sigma maps containing normalized standard deviations (in range [0, 1]) for a Gaussian kernel. For the actual standard deviation value, one should multiply 7 to this map.
    > ```

    > Quantitative results are computed from matlab. (*e.g.*, [evaluation on the RTF dataset](https://github.com/codeslake/DMENet/tree/master/evaluation/RTF)).

    * Options
        * `--mode`: The name of a model to test. The logging folder named with the `[mode]` will be created as `[LOG_ROOT]/DMENet_CVPR2019/[mode]/`. Default: `DMENet_BDCS`
        * `--test_set`: The name of a dataset to evaluate. `CUHK` | `RTF0` | `RTF1` | `RTF1_6` | `random`. Default: `CUHK`
            * The folder structure can be modified in the function [`get_eval_path(..)`](https://github.com/codeslake/DMENet/blob/master/config.py#L85-L98) in `./config.py`.
            * `random` is for testing models with any images, which should be placed as `[DATASET_ROOT]/test/random/*.[jpg|png]`.

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
        * `--mode`: The name of a model to train. The logging folder named with the `[mode]` will be created as `[LOG_ROOT]/DMENet_CVPR2019/[mode]/`. Default: `DMENet_BDCS`
        * `--is_pretrain`: Pretrain the network with the MSE loss (`True` | `False`). Default: `False`
        * `--delete_log`: Deletes `[LOG_ROOT]/DMENet_CVPR2019/[mode]/*` before training begins (`True` | `False`). Default: `False`


## Contact
Open an issue for any inquiries.
You may also have contact with [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Related Links
* CVPR 2021: Iterative Filter Adaptive Network for Single Image Defocus Deblurring \[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf)\]\[[code](https://github.com/codeslake/IFAN)\]
* ICCV 2021: Single Image Defocus Deblurring Using Kernel-Sharing Parallel Atrous Convolutions \[[paper](https://arxiv.org/pdf/2108.09108.pdf)\]\[[code](https://github.com/HyeongseokSon1/KPAC)\]
* SYNDOF dataset generation repo \[[link](https://github.com/codeslake/SYNDOF)\]

## License
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=flat)<br>
This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## Citation
If you find this code useful, please consider citing:

```
@InProceedings{Lee2019DMENet,
    author    = {Junyong Lee and Sungkil Lee and Sunghyun Cho and Seungyong Lee},
    title     = {Deep Defocus Map Estimation Using Domain Adaptation},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2019}
}
```


# Deconvolution using defocus map estimated from DMENet
* *This code is based on "[Fast Image Deconvolution using Hyper-Laplacian Priors](https://papers.nips.cc/paper/2009/file/3dd48ab31d016ffcbf3314df2b3cb9ce-Paper.pdf)", Krishnan *et al.*, In Proc. NIPS 2009.<br/>Refer [here](https://dilipkay.wordpress.com/fast-deconvolution/) for the original code.*

## Getting Started
1. Place your dataset as:

    ```
    ...
    ├── deconvolution
    │   ├── source
    │   │   ├── input                  
    │   │   │    ├── [DATASET]         # the name of a dataset [`CUHK` | `DPDD` | `RealDOF`]
    │   │   │    │    ├── *.[jpg|png]  # input images  
    │   │   ├── defocus map
    │   │   │    ├── [DATASET]         # the name of the dataset that DMENet ran on
    │   │   │    │    ├── *.[jpg|png]  # defocus maps (results of DMENet in `[LOG_ROOT]/[mode]/result/[test_set]/defocus_map`)
    ...
    ```

    > **Note:**
    > 
    > * For the DPDD dataset, refer [here](https://www.eecs.yorku.ca/~abuolaim/eccv_2020_dp_defocus_deblurring/dataset.html).
    > * The RealDOF test set is the test set that we used for the defocus deblurring paper, which is provisionally accepted to CVPR2021. We will release the test set soon.

2. In the matlab console, type follwoing for the evaluation.
    
    ```
    >> run_DMENet_deconv([DATASET], is_gpu, gpu_num)

    # gpu example
    >> run_DMENet_deconv('CUHK', true, 1)

    # cpu eample
    >> run_DMENet_deconv('CUHK')
    ```

    * Parameters
        * [DATASET]: the name of a dataset. `CUHK` | `DPDD` | `RealDOF`.
        * is_gpu: whether to use gpu. `true` | `false`. Default: `false`
        * gpu_num: the device number of a gpu.

    * Results will be saved as:

        ```
        ...
        ├── deconvolution
        ...
        │   ├── output
        │   │   ├── [DATASET]          # the name of the dataset used for deconvolution
        │   │   │    ├── *.[jpg|png]   # resulting deconvolution images
        ```

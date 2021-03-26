## Getting Started
1. Download and unzip [ground-truths of the RTF dataset and corresponding results of our method](https://www.dropbox.com/s/ph9pvj5g53vea6h/RTF_our_results_gt.zip?dl=1) under where the evaluation code is:

    ```
    ...
    ├── evaluation_RTF
    │   ├── RTF
    │   │   ├── gt
    │   │   ├── out
    │   │   │   ├── BDCS
    │   │   ├── *.m
    │   │   ├── ...
    ```

    * Images in the `out` directory is [`defocus_map`](https://github.com/codeslake/DMENet/blob/master/main.py#L481), which is the direct output of the network.
    
        > **Note:**
        >
        > Here is [the original zip file](https://www.dropbox.com/s/f2bkay9xykgmouc/Defocus_Blur_Dataset.zip?dl=1) of the RTF dataset provided by the author.

2. Type `run_quantitative_RTF` in the matlab console for the evaluation.
    * For evaluating the methods in Table 2 in the main paper,
        * Except [4], all defocus map results are converted to Gaussian PSF (which have the maximum standard deviation=3.275), using the code provided by [4].
        * For [40], we set `maxBlur` in their code as 3.275 (which was originally 3).
        * For [30], we computed standard deviations using Eq. (4) in their paper, then clipped the results to have the maximum value 3.275.
        * For [24], we clipped ground-truths to have the maximum value 2.0 (according to their paper).
        * For [13], we clipped their results to have the maximum value 3.275 (the results have values of maximum 5.0).
        * For Ours, we compute standard deviation maps (*i.e.*, `(out * 15) - 1) / 2`) and clipped them to have values between 0 and 3.275 ([`evaluation/RTF/quantitative_RTF.m`](/evaluation/RTF/quantitative_RTF.m#L45-L53)).

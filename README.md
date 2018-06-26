SYNDOF (Synthetic Defocus Blur Image Generator)
Matlab implementation of defocus image generation

<img src="./assets/figure.png" width="700">

## Prerequisites
-   matlab

## A typical top-level directory layout
    .
    ├── data                        # directory for input images
    ├── result                      # directory for output images
    ├── generate_blur_by_depth.m    # main function for SYNDOF
    ├── blur_by_depth.m
    ├── depth_read.m
    └── README.md

## Getting Started
-   On matlab console, type
```bash
generate_blur_by_depth(29, false, false, 0) # max_coc, is_random_gen, is_gpu, gpu_num
```

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition,ersonalization technology through userreference analysis is under study.  
    
Please checkout out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

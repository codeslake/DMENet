Deep Defocus Map Estimation
Tensorflow implementation of deep defocus map estimation

<img src="./assets/figure.png" width="700">

## Prerequisites
-   python 2.7
-   tensorflow 1.8.0

## A typical top-level directory layout
    .
    ├── data                        # directory for input images
    ├── result                      # directory for output images
    ├── main.py                     # main function for SYNDOF
    ├── model.py                     # main function for SYNDOF
    ├── config.py
    ├── utils.py
    └── README.md

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

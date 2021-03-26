#!/bin/bash
conda install -y cudatoolkit=10.0
conda install -y cudnn=7.6
pip install --no-cache tensorflow-gpu==1.15
pip install --no-cache tensorlayer==1.11.1
pip install --no-cache jupyterlab
pip install --no-cache -r requirements.txt

#!/bin/bash
pip install --no-cache --user nvidia-pyindex
conda install -y -c conda-forge openmpi
export PATH="~/.local/bin:$PATH"
export LD_LIBRARY_PATH="/anaconda3/envs/DMENet_CUDA11/lib/:$LD_LIBRARY_PATH"
pip install --no-cache --user 'nvidia-tensorflow[horovod]'
pip install --no-cache tensorlayer==1.11.1
pip install --no-cache --upgrade numpy==1.16.0
pip install --no-cache --upgrade warpt==1.11.1

pip install --no-cache jupyterlab
pip install --no-cache -r requirements.txt


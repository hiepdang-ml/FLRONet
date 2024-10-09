#/usr/bin/bash

cd ~/FLRONet

# Environment
conda init
source ~/.bashrc
conda env create -f env.yaml
conda activate FLRONet

# Install zip and unzip
sudo apt-get install zip unzip

# Download data
wget https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/bc.zip?download=true -O bc.zip && \
wget https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cylinder/prop.zip?download=true -O prop.zip  && \
unzip bc.zip -d bc && unzip prop.zip -d prop && rm *.zip



#!/bin/bash
sudo apt-get -y update
sudo apt-get -y install graphviz graphviz-dev
sudo apt-get -y install zip unzip
sudo apt-get install build-essential -y
sudo apt-get -y install git

pip install -r requirements.txt

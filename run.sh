#!/bin/bash

echo "Aspect Sentiment Triplet Extraction"

echo "Install dependancies"
pip install torch==1.9.0+cu111 -f  https://download.pytorch.org/whl/cu111/torch_stable.html
pip install transformers==4.8.2
pip install stanfordcorenlp==3.9.1.1

cd RoBMRC/

echo "Preprocess Data"
python ./tools/dataProcessV2.py

echo "Begin Training"
python ./tools/main.py --mode train 

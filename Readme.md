#

Code and data of the paper Aspect Sentiment Triplet Extraction based on Transformer model

Author: Emelda Kuete

#### Requirements:

```
  python==3.8.5
  torch==1.9.0+cu111
  transformers==4.8.2
  standfordcorenlp==3.9.1.1
```

#### Original Datasets:

You can download the 14-Res, 14-Lap, 15-Res, 16-Res datasets from https://github.com/xuuuluuu/SemEval-Triplet-data.
Put it into different directories (./data/original/[v1, v2]) according to the version of the dataset.

#### Data Preprocess:

You must download the stanfordNLP to get POS from https://stanfordnlp.github.io/CoreNLP/index.html by following these steps:

```
    - wget http://nlp.stanford.edu/software/stanford-corenlp-4.5.4.zip
    - unzip stanford-corenlp-4.5.4.zip
    - cd stanford-corenlp-4.5.4
    - for file in `find . -name "*.jar"`; do export
CLASSPATH="$CLASSPATH:`realpath $file`"; done
    - java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

Then run the following :

```
  python ./tools/dataProcessV1.py # Preprocess data from version 1 dataset
  python ./tools/dataProcessV2.py # Preprocess data from version 2 dataset
```

The results of data preprocessing will be placed in the ./data/preprocess/.

#### How to run:

```
  python ./tools/main.py --mode train # For training
  python ./tools/main.py --mode test # For testing
```

Training different versions of datasets can modify the value of dataset_version in main.py.

```
dataset_version = "v1/"
dataset_version = "v2/"
```

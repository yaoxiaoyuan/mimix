# Mimix: A Tool for Seq2seq learning framework based on Pytorch and Tensorflow

Mimix is a tool for Seq2seq learning framework based on Pytorch. Basically, I implement several generative models, including rnn-seq2seq and transformer. Some other models, such as Text CNN, Transformer-based classifier and Transformer-based Language model, are also provided. To facilitate the use of pretrained models, Mimix can also load bert model weights. Other pretrained models may be supported in the future.

I have trained some generative models which are listed in Demo section. You can follow the instructions and run the demo.

## Requirements

pytorch >= 1.6

numpy >= 1.15

python >= 3.5

## Model Config

You may refer to the example config file in test_data directory.

## Usage

train: python train.py --conf conf_file

interact: python interact.py --conf conf_file

batch predict: python predict.py --conf conf_file

## Demo

### Chinese Shi Ci Lian Generator

1. Download conf and model data: https://pan.baidu.com/s/1Fb3o0GPoyv5Yj6cQN5SPNQ code: hbl4
2. Put conf and model under mimix folder
3. cd src and run: python interact.py --conf ../conf/poet_base_conf

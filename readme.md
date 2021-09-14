# Mimix: A Tool for Seq2seq learning framework based on Pytorch and Tensorflow

Mimix is a tool for Seq2seq learning framework based on Pytorch. Basically, I implement several generative models, including rnn-seq2seq and transformer. Some other models, such as Text CNN, Transformer-based classifier and Transformer-based Language model, are also provided. To facilitate the use of pretrained models, Mimix can also load bert model weights. Other pretrained models may be supported in the future.

I have trained some generative models which are listed in Demo section. You can follow the instructions and run the demo.

## Requirements

pytorch >= 1.6

numpy >= 1.15

python >= 3.5

## Model Config

(TO DO)

## Usage

train: python train.py --conf conf_file

interact: python interact.py --conf conf_file

batch predict: python predict.py --conf conf_file

## Demo

Several pretrained models are listed below. You can use it follow the  instructions.

| model                                | architecture                    | n_params  | n_layers | d_model | n_heads | d_head |
| ------------------------------------ | ---- | ---- | :--- | ---- | ---- | ---- |
| Chinese chitchat generator           | transformer enc-dec | 100m | 12 | 512 | 8 | 64 |
| Chinese shi,ci,duilian generator     | transformer uni-lm | 50m  | 12 | 512 | 8 | 64 |
| Chinese news summarization generator | transformer enc-dec | 100m | 12 | 512 | 8 | 64 |

Training data details are listed below.

| model                                | n_samples | data size |
| ------------------------------------ | --------- | --------- |
| Chinese chitchat generator           | 400m      | 76GB      |
| Chinese shi,ci,qu,duilian generator  | 6m        | 1GB       |
| Chinese news summarization generator | 18m       | 42GB      |

### Chinese chitchat Generator

1. Download conf and model data: https://pan.baidu.com/s/1aQOGLOjNtabA7YyWOdH1Sg code: eegm
2. Put conf and model under mimix folder
3. cd src and run: python interact.py --conf ../conf/chat_base_conf

### Chinese Shi Ci Qu Lian Generator

1. Download conf and model data: https://pan.baidu.com/s/1aQOGLOjNtabA7YyWOdH1Sg code: eegm
2. Put conf and model under mimix folder
3. cd src and run: python interact.py --conf ../conf/shi_base_conf

### Chinese news summarization Generator

1. Download conf and model data: https://pan.baidu.com/s/1aQOGLOjNtabA7YyWOdH1Sg code: eegm
2. Put conf and model under mimix folder
3. cd src and run: python interact.py --conf ../conf/summ_base_conf

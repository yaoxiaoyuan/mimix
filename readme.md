# Mimix: A Tool for Seq2seq learning framework based on Pytorch and Tensorflow

Mimix is a tool for Seq2seq learning framework based on Pytorch. Basically, I implement several generative models, including rnn-seq2seq and transformer. Some other models, such as Text CNN, Transformer-based classifier and Transformer-based Language model, are also provided. To facilitate the use of pretrained models, Mimix can also load bert model weights. Other pretrained models may be supported in the future.

I have trained some Chinese Generative Models which are listed in Demo section. You can follow the instructions and run the demo.

## Updates

20220621 add more models

20220709 add more models

## Requirements

pytorch >= 1.6

numpy >= 1.15

python >= 3.5

## Model Config

(TO DO)

## Usage

train: python train_single.py --conf conf_file

interact: python interact.py --conf conf_file

batch predict: python predict.py --conf conf_file

## Demo

Several pretrained models are listed below. You can use it follow the  instructions.

| model                                | architecture                    | n_params  | n_layers | d_model | n_heads | n_samples | data size |
| ------------------------------------ | ---- | ---- | :--- | ---- | ---- | ------------------------------------ | ------------------------------------ |
| Chinese chitchat generator           | transformer enc-dec | 100m | 12 | 512 | 8 | 400m | 76GB |
| Chinese shi,ci,duilian generator     | transformer dec-only lm | 49m | 12 | 512 | 8 | 6m | 1GB |
| Chinese news summarization generator | transformer enc-dec | 100m | 12 | 512 | 8 | 18m | 42GB |
| Chinese shi,ci,duilian generator v2 | transformer dec-only lm  | 102m     | 12       | 768     | 12      | 6m | 1GB |
| Chinese news summarization generator v2 | transformer enc-dec | 216m     | 12       | 768     | 12 | 36m | 113GB |
| Chinese modern poet, lyric generator    | transformer dec-only lm | 103m     | 12       | 768     | 12      | 2m | 1GB |
| Chinese question paraphrase tool        | transformer enc-dec     | 216m     | 12       | 768     | 12      | 32m | 25GB |
| Chinese question similarity tool        | transformer enc         | 103m     | 12       | 768     | 12 | 32m | 25GB |

### HOW TO RUN

1. Download conf and model data: https://pan.baidu.com/s/1aQOGLOjNtabA7YyWOdH1Sg code: eegm

2. Put conf and model under mimix folder

3. cd src and run: 

   1. Chinese chitchat generator: python interact.py --conf ../conf/chat_base_conf

      1. input:?????????????????????

      2. output:

         ?????????????????????? -5.690431594848633

         ??????,???????????? -6.024079322814941

         ????????????????????????????????? -6.452165603637695

         ?????????????????????,?????? -6.467472553253174

         ????????? ?????? -6.556297302246094

         ?????????????????????????????? -7.752975940704346

         ?????????,?????????????????? -7.771842002868652

         ????????????????????? -7.823556900024414

         ?????? ???????????? -7.892529487609863

         ????????????,????????????????????? -8.928380966186523

         

   2. Chinese shiciqulian generator: python interact.py --conf ../conf/shi_base_conf

      1. input: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_
   
      2. output: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_ ???????????? \_content\_ ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
   
         
   
   3. Chinese summarization generator interact.py --conf ../conf/summ_base_conf
   
      1. input: 6???21?????????????????????????????????????????????????????????????????????2022???6???7????????????????????????43?????????4?????????????????????????????????????????????????????????41???????????????????????????????????????????????????????????????6???10????????????????????????5??????????????????4?????????????????????????????????????????????????????? ??????2???40???????????????????????????????????????????????????????????????31??????????????????????????????????????????????????????25?????????????????????????????????????????? ??????????????????29?????????4??????????????????2???47????????????2???55???4???????????????120?????????2???41??????????????????????????????????????????????????????????????????????????????????????????3???09????????????????????????????????????????????????2???40????????????????????????????????????2???41??????????????????????????????????????????1????????????????????????2???47????????????????????????????????????????????????????????????7?????????
   
      2. output:
   
         ?????????????????????????????????????????????????????? -11.3525390625
   
         ??????????????????????????????????????????????????? -11.6630859375
   
         ????????????????????????????????? -14.353515625
   
         ?????????????????????????????????????????????????????? 4???????????? -17.193359375
   
         ?????????????????????????????????????????????????????????4???????????? -17.4345703125
   
         ?????????????????????????????????????????????????????? 4????????? -18.19140625
   
         ?????????????????????????????????4??????????????????????????? -18.275390625
   
         ?????????????????????????????????4???????????????????????????????????? -18.3115234375
   
         ?????????????????????????????????4??????????????????????????? -18.455078125
   
         ?????????4??????????????????????????????????????? -18.4736328125
   
         ?????????????????????????????????4????????????????????????????????? -19.4404296875
   
         ?????????????????????????????????4???????????????????????????????????? -21.013671875
   
         ?????????4??????????????????????????????????????????7????????????????????? -21.6318359375
   
         ?????????4?????????????????????????????????????????????????????????????????? -23.00390625
   
         ?????????4?????????????????????????????????????????????????????????????????????????????? -23.678224563598633
   
         
   
   4. Chinese shiciqulian generator v2: python interact.py --conf ../conf/shi_base_conf_v2

      1. input: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_

      2. output: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_ ?????? \_content\_ ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

         

   5. Chinese summarization generator v2: interact.py --conf ../conf/summ_base_conf_v2

      1. input: 6???21?????????????????????????????????????????????????????????????????????2022???6???7????????????????????????43?????????4?????????????????????????????????????????????????????????41???????????????????????????????????????????????????????????????6???10????????????????????????5??????????????????4?????????????????????????????????????????????????????? ??????2???40???????????????????????????????????????????????????????????????31??????????????????????????????????????????????????????25?????????????????????????????????????????? ??????????????????29?????????4??????????????????2???47????????????2???55???4???????????????120?????????2???41??????????????????????????????????????????????????????????????????????????????????????????3???09????????????????????????????????????????????????2???40????????????????????????????????????2???41??????????????????????????????????????????1????????????????????????2???47????????????????????????????????????????????????????????????7?????????

      2. output:

         ??????????????????????????????????????????????????? -4.3955078125

         ????????????????????????????????????????????? -5.14453125

         ??????????????????????????????????????? -6.619140625

         ??????????????????????????????????????????????????? -6.9287109375

         ??????????????????????????????????????? -6.9638671875

         ?????????????????????????????????????????????????????? -7.060546875

         ???????????????????????????????????????????????? -7.927734375

         ????????????????????????????????????????????????4????????????????????????????????? -8.693359375

         ??????????????????????????????????????????????????????4????????????????????????????????? -9.5048828125

         ????????????????????????????????????????????????4??????????????????????????????????????????????????? -10.0068359375

         ????????????????????????????????????????????????4??????????????????????????? -10.1826171875

         ??????????????????????????????????????????????????????4???????????? -10.6962890625

         ??????????????????????????????????????????????????????4??????????????????????????? -10.759765625

         ??????????????????????????????????????????????????????4???????????????????????????????????????????????? -11.46523380279541

         ??????????????????????????????????????????????????????4????????????????????? -11.826171875
         
         

   6. Chinese modern poet, lyric generator : interact.py --conf ../conf/poet_base_conf

      1. input: \_poet\_ \_title\_ ????????? \_content\_

      2. output: \_poet\_ \_title\_ ????????? \_content\_ ?????? \_nl\_ ???????????? \_nl\_ ????????? \_nl\_ ?????? \_nl\_ ???????????? \_nl\_ ??? \_nl\_ ??????????????? \_nl\_ ?????? \_nl\_ ?????? \_nl\_ ?????? \_nl\_ ?????? _nl\_ ???????????? \_nl\_ ??? \_nl\_ ??? \_nl\_ ?????? ?????? \_nl\_ ??? \_nl\_ ?????? \_nl\_ ??????

         

   7. Chinese question paraphrase: interact.py --conf ../conf/aug_base_conf

      1. input: ??????????????????

      2. output:

         ???????????????????????? -4.042654037475586

         ???????????????????????? -7.5486650466918945

         ??????????????????????????? -8.162402153015137

         ???????????????????????????? -8.664673805236816

         ???????????????????????? -9.0930814743042

         ?????????????????? -9.227267265319824

         ????????????????????? -10.365612030029297

         ????????????????????? -11.004858016967773

         ?????????????????????????????? -11.71064281463623

         ????????????,?????????????????????????????????? -16.173974990844727

         

   8. Chinese question similarity tool: interact.py --conf ../conf/sim_base_conf

      1. input: ?????????????????????\t??????????????????\t?????????????????????\t????????????????????????

      2. output:

         ????????????????????? ?????????????????? 0.708049

         ????????????????????? ????????????????????? 0.11056142

         ????????????????????? ???????????????????????? 0.19936031
   
         ?????????????????? ????????????????????? 0.015660984
   
         ?????????????????? ???????????????????????? 0.046153657
   
         ????????????????????? ???????????????????????? 0.6877855
   
         

### Cite

```
@misc{mimix,
  title={mimix},
  author={Xiaoyuan Yao},
  year={2021}
}
```




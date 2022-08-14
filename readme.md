# Mimix: A Tool for Seq2seq learning framework based on Pytorch and Tensorflow

Mimix is a tool for Seq2seq learning framework based on Pytorch. Basically, I implement several generative models, including rnn-seq2seq and transformer. Some other models, such as Text CNN, Transformer-based classifier and Transformer-based Language model, are also provided. To facilitate the use of pretrained models, Mimix can also load bert model weights. Other pretrained models may be supported in the future.

I have trained some Chinese Generative Models which are listed in Demo section. You can follow the instructions and run the demo.

## Updates

20210913 release chiense chitchat, poet, summarization generation models

20220621 release poet v2, summarization v2, question paraphrase generation and question similarity models

20220709 release modern poet and lyric generation model

20220814 release question generation model

## Requirements

pytorch >= 1.6

numpy >= 1.15

python >= 3.5

## Usage

train: python run_train.py --model_conf conf_file  --train_conf conf_file

interact: python interact.py --model_conf conf_file

batch predict: python run_predict.py --model_conf conf_file

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
| Chinese question paraphrase generator | transformer enc-dec     | 216m     | 12       | 768     | 12      | 32m | 25GB |
| Chinese question similarity tool        | transformer enc         | 103m     | 12       | 768     | 12 | 32m | 25GB |
| Chinese question generation | transformer enc-dec | 216m | 12 | 768     | 12 | 0.5m | 0.5GB |

### HOW TO RUN

1. Download conf and model data: https://pan.baidu.com/s/1aQOGLOjNtabA7YyWOdH1Sg code: eegm

2. Put conf and model under mimix folder

3. cd src and run: 

   1. Chinese chitchat generator: python interact.py --model_conf ../conf/chat_base_conf

      1. input:今天天气好热啊

      2. output:

         你们那里不热吗? -5.690431594848633

         是啊,今天不热 -6.024079322814941

         没有空调的日子好无聊啊 -6.452165603637695

         来我这里吹空调,凉快 -6.467472553253174

         不热啊 凉快 -6.556297302246094

         我今天穿的是一件外套 -7.752975940704346

         哈哈哈,我是来看热的 -7.771842002868652

         这个鞋子很好看 -7.823556900024414

         冷啊 今天好热 -7.892529487609863

         今天风大,出门都没开空调 -8.928380966186523

         

   2. Chinese shiciqulian generator: python interact.py --model_conf ../conf/shi_base_conf

      1. input: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_
   
      2. output: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_ 题赠友人 \_content\_ 曾经万里隔寒霜，梦寐三更到谢娘。一朵白云飘故旧，千丝玉絮送清凉。天涯雁影南方去，海角归心北地伤。岁月蹉跎尘念久，年华虚度鬓如霜。
   
         
   
   3. Chinese summarization generator interact.py --model_conf ../conf/summ_base_conf
   
      1. input: 6月21日，河北省公安厅发布唐山打人案情况通报：经查，2022年6月7日，陈某亮（男，43岁）等4人从江苏驾车至河北唐山，与陈某志（男，41岁）等人合谋实施网络赌博洗钱违法犯罪活动。6月10日凌晨，陈某志等5人与陈某亮等4人在唐山市路北区某烧烤店聚餐饮酒。期 间，2时40分，陈某志对下班后在同店就餐的王某某（女，31岁）进行骚扰，遭拒后伙同马某齐（男，25岁）、陈某亮等人，对王某某、 刘某某（女，29岁）等4人进行殴打，2时47分逃离，2时55分4名被害人由120送医。2时41分接群众报警后，唐山市公安局路北分局机场路派出所民警率辅警于3时09分赶到现场开展处置工作。据通报，2时40分陈某志对女孩进行骚扰，2时41分警方接到群众报警，冲突发生1分钟即有人报警。2时47分陈某志及其同伙逃离，从骚扰到逃离，共计7分钟。
   
      2. output:
   
         唐山打人案情况通报 -5.3681640625
   河北省公安厅通报唐山打人案情况 -6.82421875
         河北警方通报唐山打人案情况 -7.1669921875
   唐山打人案情况通报：4人合谋实施网络赌博洗钱违法犯罪活动 -8.2666015625
         唐山打人案情况通报：4人合谋实施网络赌博洗钱违法犯罪 -9.4990234375
   唐山打人案情况通报：4人合谋实施网络赌博洗钱 -9.9794921875
         唐山打人案情况通报：4名被害人从120送医 -10.2021484375
   河北警方通报唐山打人案情况：4人合谋实施网络赌博洗钱违法犯罪活动 -10.27734375
         河北省公安厅通报唐山打人案情况：4人合谋实施网络赌博洗钱违法犯罪活动 -10.55828857421875
   河北警方通报唐山打人案情况：4名被害人由120送医 -11.2734375
         唐山打人案情况通报：4人合谋实施网络赌博洗钱犯罪活动 -11.3076171875
   河北警方通报唐山打人案情况：4人合谋实施网络赌博洗钱 -11.52734375
         唐山打人案情况通报：4人合谋实施网络赌博洗钱活动 -11.560546875
   河北省公安厅通报唐山打人案情况：4名被害人由120送医 -11.7265625
         唐山打人案情况通报：4名被害人被殴打7分钟 -11.966796875
   
         
   
   4. Chinese shiciqulian generator v2: python interact.py --model_conf ../conf/shi_base_conf_v2
   
      1. input: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_
   
      2. output: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_ 秋菊 \_content\_ 月冷霜寒百草黄，疏篱落木卸浓妆。清姿独傲冰风骨，雅韵闲生玉露香。翠色满园人未识，花容入梦夜初长。无须绿叶来陪衬，此处陶然笑夕阳。
   
         
   
   5. Chinese summarization generator v2: python interact.py --model_conf ../conf/summ_base_conf_v2
   
      1. input: 6月21日，河北省公安厅发布唐山打人案情况通报：经查，2022年6月7日，陈某亮（男，43岁）等4人从江苏驾车至河北唐山，与陈某志（男，41岁）等人合谋实施网络赌博洗钱违法犯罪活动。6月10日凌晨，陈某志等5人与陈某亮等4人在唐山市路北区某烧烤店聚餐饮酒。期 间，2时40分，陈某志对下班后在同店就餐的王某某（女，31岁）进行骚扰，遭拒后伙同马某齐（男，25岁）、陈某亮等人，对王某某、 刘某某（女，29岁）等4人进行殴打，2时47分逃离，2时55分4名被害人由120送医。2时41分接群众报警后，唐山市公安局路北分局机场路派出所民警率辅警于3时09分赶到现场开展处置工作。据通报，2时40分陈某志对女孩进行骚扰，2时41分警方接到群众报警，冲突发生1分钟即有人报警。2时47分陈某志及其同伙逃离，从骚扰到逃离，共计7分钟。
   
      2. output:
   
         河北省公安厅发布唐山打人案情况通报 -4.3955078125

         河北省公安厅通报唐山打人案情况 -5.14453125

         河北省公安厅通报唐山打人案 -6.619140625

         河北省公安厅通报唐山打人案情况通报 -6.9287109375

         河北警方通报唐山打人案情况 -6.9638671875

         河北省公安厅发布唐山打人案情况通报！ -7.060546875

         河北省公安厅通报唐山打人案情况！ -7.927734375

         河北省公安厅通报唐山打人案情况：4人合谋实施网络赌博洗钱 -8.693359375

         河北省公安厅发布唐山打人案情况通报：4人合谋实施网络赌博洗钱 -9.5048828125

         河北省公安厅通报唐山打人案情况：4人合谋实施网络赌博洗钱违法犯罪活动 -10.0068359375

         河北省公安厅通报唐山打人案情况：4人合谋实施网络赌博 -10.1826171875

         河北省公安厅发布唐山打人案情况通报：4人被殴打 -10.6962890625

         河北省公安厅发布唐山打人案情况通报：4人合谋实施网络赌博 -10.759765625

         河北省公安厅发布唐山打人案情况通报：4人合谋实施网络赌博洗钱违法犯罪活 -11.46523380279541

         河北省公安厅发布唐山打人案情况通报：4人凌晨聚餐饮酒 -11.826171875
      
         

   6. Chinese modern poet, lyric generator : python interact.py --model_conf ../conf/poet_base_conf

      1. input: \_poet\_ \_title\_ 寒江雪 \_content\_

      2. output: \_poet\_ \_title\_ 寒江雪 \_content\_ 冬夜 \_nl\_ 没有月亮 \_nl\_ 只剩下 \_nl\_ 一片 \_nl\_ 孤独的白 \_nl\_ 风 \_nl\_ 把满山野林 \_nl\_ 吹得 \_nl\_ 更加 \_nl\_ 清凉 \_nl\_ 而我 _nl\_ 正倚着窗 \_nl\_ 听 \_nl\_ 那 \_nl\_ 簌簌 落英 \_nl\_ 和 \_nl\_ 天地 \_nl\_ 对话

         

   7. Chinese question paraphrase generator: python interact.py --model_conf ../conf/aug_base_conf

      1. input: 孕妇吃什么好
   
      2. output:

         适合孕妇吃的食物 -4.042654037475586

         怀孕的人吃什么好 -7.5486650466918945

         最适合孕妇吃的食物 -8.162402153015137

         孕妇适合吃什么食物? -8.664673805236816

         哪些食物对孕妇好 -9.0930814743042

         产妇适合吃啥 -9.227267265319824

         女性怀孕吃啥好 -10.365612030029297

         妊娠期吃什么好 -11.004858016967773

         什么食物对孕妇有好处 -11.71064281463623

         在孕期时,吃什么食物对胎儿比较好? -16.173974990844727

         

   8. Chinese question generation: python interact.py --model_conf ../conf/qg_base_conf

      1. input: 《玫瑰之战》是由孙皓执导，郑仁湘、张涵编剧，袁泉、黄晓明、俞飞鸿领衔主演，代旭、于谨维、王鹤润主演、芦芳生、张艺上、隋俊波特邀主演，王志飞特约出演，王姬特别主演的都市剧。

      2. output:

         孙皓 \_sep\_ 玫瑰之战是谁执导的。 -6.365093231201172

         

### Cite

```
@misc{mimix,
  title={mimix},
  author={Xiaoyuan Yao},
  year={2021}
}
```




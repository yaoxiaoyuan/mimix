# Mimix: A Tool for Seq2seq learning framework based on Pytorch 

Mimix is a tool for Seq2seq learning framework based on Pytorch. Basically, I implement several generative models, including rnn-seq2seq and transformer. Some other models, such as Text CNN, Transformer-based classifier and Transformer-based Language model, are also provided. To facilitate the use of pretrained models, Mimix can also load bert model weights. Other pretrained models may be supported in the future.

I have trained some Chinese Generative Models which are listed in Demo section. You can follow the instructions and run the demo.  ~~**An application is required for model download.**~~ Now you can download the weights directly without sending me an email. **You must follow the agreement to use the model. Please scan the completed [agreement](https://github.com/yaoxiaoyuan/mimix/blob/main/agreement/Mimix%20Model%20Release%20Agreement.docx).** 

**UPDATE 2023.08**:  

1. The entire project's code has been refactored. Now it is easier to use. 

2. **The old model weights and config files are not compatible with new code. Please download new model data. Old weights links may become invalid in the future.**

3. I added some example code and config files for training. The training code is flexible and can be easily modified. If use training code, it is better to read the model kwargs in code. Documentation may be improved in the future. 

4. Now MIMIX support image classification,  Image caption and image-text match task. I have trained image classification model on ChineseFoodNet dataset. By using the pretrained VIT model, it can easily beats the single resnet152 model (VIT get top1 acc 0.7967 on test data set, ResNet152 get top1 acc 0.7900 on test data set) .  

5. I also add a streamlit script to better play with the text generation task and multimodal task. To run the demo, you can run streamlit run app.py -- --model_conf /your/model/conf/path![streamlit](image/streamlit.png)![streamlit](image/streamlit2.png)

   ![streamlit](image/streamlit3.png)

6. Although LLM achieve remarkable performance in recent years, this project focus on training model useful but not that large.  Training LLM cost too much for individuals. Also, there already exists many projects for finetune large LLM  at low cost. However, I may boost Chinese text generation models trained before by leveraging the new LLM model.  New models may be released in the future.

7. RNN-seq2seq code has been removed in new code. You can still check the code on branch v1. But it won't be maintained.



If you want to contact with me, please send me an email.

### Cite

```
@misc{mimix,
  title={mimix},
  author={Xiaoyuan Yao},
  year={2021}
}
```

## Models

Several pretrained models are listed below.  ~~**Some models are not available for download due to the data privacy. Please check the "open for download" column. You can follow the instructions to use the open-download model.**~~

| model name                           | architecture                    | n_params  | n_layers | d_model | n_heads | n_samples | data size | open for download |
| ------------------------------------ | ---- | ---- | :--- | ---- | ---- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| Chinese chitchat generation           | transformer enc-dec | 100m | 12 | 512 | 8 | 400m | 76GB | True |
| Chinese shi,ci,duilian generation     | transformer dec-only lm | 49m | 12 | 512 | 8 | 6m | 1GB | True |
| Chinese news summarization generation | transformer enc-dec | 100m | 12 | 512 | 8 | 18m | 42GB | True |
| Chinese qa generation | transformer enc-dec     | 100m     | 12       | 512     | 8       | 66m       | 12.8GB    | True |
| Chinese modern poet, lyric generation    | transformer dec-only lm | 103m     | 12       | 768     | 12      | 2m        | 1GB       | True          |
| Chinese question paraphrase generation   | transformer enc-dec     | 216m     | 12       | 768     | 12      | 32m       | 25GB      | True          |
| Chinese question similarity tool        | transformer enc         | 103m     | 12       | 768     | 12      | 32m       | 25GB      | True          |
| Chinese question generation             | transformer enc-dec     | 216m     | 12       | 768     | 12      | 0.5m      | 0.5GB     | True          |
| Chinese comment generation              | transformer enc-dec     | 216m     | 12       | 768     | 12      | 18m       | 1.8GB     | True          |
| Chinese essay generation                | transformer dec-only lm | 135m     | 12       | 768     | 12      | 480k      | 0.7GB     | True          |
| Chinese product description generation  | transformer enc-dec     | 216m     | 12       | 768     | 12      | 2m        | 0.7GB      | True          |
| Chinese product review generation       | transformer enc-dec     | 216m     | 12       | 768     | 12      | 10m       | 2.4GB     | True          |
| Chinese wuxia novel generation | transformer dec-only lm | 369m | 24 | 1024 | 16 | 830k | 1.2G | True |
| Chinese-English translation | transformer enc-dec | 216m | 12 | 768 | 12 | 60m | 16GB | True |
| Chinese paper generation                 | transformer enc-dec     | 216m     | 12       | 768     | 12      | 4m        | 4.8GB     | True              |
| Chinese tag generation                   | transformer enc-dec     | 216m     | 12       | 768     | 12      | 22m       | 24GB      | True              |
| Chinese medical qa           | transformer enc-dec     | 216m     | 12       | 768     | 12      | 2.7m      | 1.38GB    | True         |
| Chinese doc2query generation | transformer enc-dec | 216m | 12       | 768     | 12      | 1.3m      | 1.5GB     | True         |
| Chinese ancient translation              | transformer enc-dec     | 216m     | 12       | 768     | 12      | 6m        | 1GB       | True             |
| Chinese spelling correction | transformer enc-dec | 216m | 12 | 768 | 12 | 32m | 6GB | True |
| Chinese Food classification | VIT                     | 88m      | 12       | 768     | 12      | 180k      | 20G       | True |
| Chinese Traditional Medicine classification | VIT                     | 88m      | 12       | 768     | 12      | 267k      | 5G        | True |
| Chinese image caption | transformer enc-dec | 219m | 12 | 768 | 12 | 356k | 34G | True |
| Chinese CLIP | dual encoder | 192m | 12 | 768 | 12 | 3m | 300G | True |

~~Download link:  https://pan.baidu.com/s/18UmwOwbN2u_J0ym382SxAA?pwd=bxka~~ 

**DON'T USE OLD MODEL WEIGHTS AND CONFIG FILES ABOVE!** 

**DON'T USE OLD MODEL WEIGHTS AND CONFIG FILES ABOVE!** 

**DON'T USE OLD MODEL WEIGHTS AND CONFIG FILES ABOVE!** 

The old weights and configs are not compatible with new code. Please download new model data with below link. Old weights links may become invalid in the future.

Download link：https://pan.baidu.com/s/1BJ9we7rs9PYxA_0yqt91pw?pwd=hn7z 

### HOW TO RUN MODEL

1. Download conf and model data.  

2. Put conf and model under mimix folder

3. run: 

   1. Chinese chitchat generation: python interact.py --model_conf conf/chitchat_small_conf

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

         

   2. Chinese shiciqulian generation: python interact.py --model_conf conf/shi_small_conf

      1. input: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_

      2. output: \_shi\_ \_xinyun\_ \_7lv\_ \_title\_ 题赠友人 \_content\_ 曾经万里隔寒霜，梦寐三更到谢娘。一朵白云飘故旧，千丝玉絮送清凉。天涯雁影南方去，海角归心北地伤。岁月蹉跎尘念久，年华虚度鬓如霜。

         

   3. Chinese summarization generation interact.py --model_conf conf/summ_small_conf

      1. input: 6月21日，河北省公安厅发布唐山打人案情况通报：经查，2022年6月7日，陈某亮（男，43岁）等4人从江苏驾车至河北唐山，与陈某志（男，41岁）等人合谋实施网络赌博洗钱违法犯罪活动。6月10日凌晨，陈某志等5人与陈某亮等4人在唐山市路北区某烧烤店聚餐饮酒。期 间，2时40分，陈某志对下班后在同店就餐的王某某（女，31岁）进行骚扰，遭拒后伙同马某齐（男，25岁）、陈某亮等人，对王某某、 刘某某（女，29岁）等4人进行殴打，2时47分逃离，2时55分4名被害人由120送医。2时41分接群众报警后，唐山市公安局路北分局机场路派出所民警率辅警于3时09分赶到现场开展处置工作。据通报，2时40分陈某志对女孩进行骚扰，2时41分警方接到群众报警，冲突发生1分钟即有人报警。2时47分陈某志及其同伙逃离，从骚扰到逃离，共计7分钟。

      2. output:

         河北唐山警方通报唐山打人案情况：4人被刑拘  -12.504453659057617

         河北唐山警方通报唐山打人案情况：4人被拘  -13.885486602783203

         唐山警方通报唐山打人案情况：4人被刑拘  -14.24258041381836

         唐山警方通报唐山打人案情况：4人被刑拘 4人被警方刑拘  -20.783082962036133

         女孩遭4人骚扰后遭殴打 河北唐山警方通报打人案  -22.540132522583008

         女孩遭4人骚扰后遭殴打 河北唐山警方通报打人案情  -22.774932861328125

         

   4. Chinese modern poet, lyric generation : python interact.py --model_conf conf/poet_base_conf

      1. input: \_poet\_ \_title\_ 寒江雪 \_content\_

      2. output: \_poet\_ \_title\_ 寒江雪 \_content\_ 冬夜 \_nl\_ 没有月亮 \_nl\_ 只剩下 \_nl\_ 一片 \_nl\_ 孤独的白 \_nl\_ 风 \_nl\_ 把满山野林 \_nl\_ 吹得 \_nl\_ 更加 \_nl\_ 清凉 \_nl\_ 而我 _nl\_ 正倚着窗 \_nl\_ 听 \_nl\_ 那 \_nl\_ 簌簌 落英 \_nl\_ 和 \_nl\_ 天地 \_nl\_ 对话

         

   5. Chinese question paraphrase generation: python interact.py --model_conf conf/aug_base_conf

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

         

   6. Chinese question similarity tool: interact.py --conf conf/sim_base_conf

      1. input:  适合小孩听的歌\t推荐几首儿歌\t熬夜有什么坏处\t晚睡对身体的影响

      2. output:

         适合小孩听的歌 推荐几首儿歌 0.75041837

         适合小孩听的歌 熬夜有什么坏处 0.13408381

         适合小孩听的歌 晚睡对身体的影响 0.19240658

         推荐几首儿歌 熬夜有什么坏处 0.06708599

         推荐几首儿歌 晚睡对身体的影响 0.07781723

         熬夜有什么坏处 晚睡对身体的影响 0.65766805

         

   7. Chinese question generation: python interact.py --model_conf conf/qg_base_conf

      1. input: 《玫瑰之战》是由孙皓执导，郑仁湘、张涵编剧，袁泉、黄晓明、俞飞鸿领衔主演，代旭、于谨维、王鹤润主演、芦芳生、张艺上、隋俊波特邀主演，王志飞特约出演，王姬特别主演的都市剧。

      2. output:

         孙皓 \_sep\_ 玫瑰之战是谁执导的。 -6.365093231201172

         

   8. Chinese ancient translation: python interact.py --model_conf conf/a2m_base_conf

       1. input:  白日依山尽,黄河入海流。欲穷千里目,更上一层楼。

       2. output:

         夕阳依傍着终南山慢慢地西沉,滔滔黄河奔腾向东流入大海。想要把千里之外的景色看完,就应该再登上更高的一层城楼。 -5.487918376922607

         夕阳依傍着山峦慢慢地西沉,滔滔黄河奔腾向东流入大海。想要把千里之外的景色看完,就应该再登上更高的一层城楼。 -5.857359886169434

         夕阳依傍着高山慢慢地西沉,滔滔黄河奔腾向东流入大海。想要把千里之外的景色看完,就应该再登上更高的一层城楼。 -5.87897253036499

         夕阳依傍着终南山慢慢地西沉,滔滔黄河奔腾向东流入大海。想要把千里之外的景色尽情欣赏,就应该再登上更高的一层城楼。 -6.974221706390381

         夕阳依傍着终南山慢慢地西沉,滔滔黄河奔腾向东流入大海。想要把千里之外的景色尽情地观赏,就应该再登上更高的一层城楼。 -7.131397724151611
       
       
       
   9. Chinese comment generation: python interact.py --model_conf conf/comment_base_conf

       1. input:  今天（9月27日），中国空军新闻发言人申进科大校在第14届中国航展新闻发布会上表示，歼-20已经遍布东南西北中，列装越来越多、飞得越来越远。

       2. output: 

          中国军工加油！ -9.743780136108398

          中国军工加油 -9.822992324829102

          中国人民解放军万岁！ -10.73838996887207
       
       
       
   10. Chinese essay generation: python interact.py --model_conf conf/essay_base_conf

       1. input: 春天

       2. output:

          春天来了 \_sep\_ “春天在哪里呀，春天在哪？春天在那小朋友的眼睛里。”随着春风和歌声，春姑娘催促我们赶紧出门寻找春天来。一 走进花园，闻见浓浓的花香和泥土的芳香，在向前走几步，就能闻到鲜花的芳香，看到满地的小野花，有红色的、黄色的、蓝色的、紫的，五颜六色，美丽极了！早晨，坐上妈妈送我的自行车去上学，虽然很冷，但是在妈妈的带领下，我真的很开心。回到家，给妈妈说：“妈妈，春天在我家的小院里，我要到院子里去了。”她顺口答应了，去给我买小礼物了。到了院子里，我看见两边的杨树发芽了，枝头也长出了一片小嫩芽。忽然我们的院子里多了一种植物，爸爸说：“那是桃树。”我兴奋地说：“这块地就是桃树的家了！这里的桃树叫桃中四君子呢！”田野里的花真多呀，把山上的树木都覆盖住了；那些紫粉的花，像婴儿睡在妈妈的肚子里、还有那些展开小翅膀的花，像眼睛里放着光；这些花朵很大，很大，像一个个白色的大绒球，在风中翩翩起舞。田野里的油菜花更美了！大地被油菜花染成黄色，风一吹，油菜荚上的金光闪闪，好像在跳一支动人的芭蕾。田野里的春天真美呀！春天真神奇呀！我爱春天！ [-736.4401]

          
       
   11. Chinese qa generation: python interact.py --model_conf conf/qa_small_conf

       1. input: 失恋了怎么办

       2. output:

          找个新欢 -6.095271110534668

          失恋了就不要再想对方，这样会让自己很难受。 -6.133023738861084

          我也有过这样的经历，但是不要放弃。 -6.613738536834717

          失恋了，就该找一个懂得珍惜的人。 -6.870847702026367

          失恋了就该哭，这是最好的办法。 当然也要看你自己怎么想 -7.109080791473389

          失恋这个事情，是你的强求不来的。如果他真心爱着你，那么就会想办法让她幸福快乐！ -7.139413356781006

          去找一个人代替他，把这段感情埋在心里 -7.654685974121094

          我的经验：失恋很正常，分手后就好了。你现在可以做些事情转移注意力，或者去旅游散散心什么的，不要让自己再想起她 -7.7700090408325195

          向前看，别让自己再沉浸在悲伤中！ -8.159266471862793

          “失恋”的男人，不是真正爱你的那一个。 只会让自己陷入痛苦中、反思过去！ 如果他心里还有你就好好珍惜吧！ -9.232671737670898

          

   12. Chinese product description generation: python interact.py --model_conf conf/desc_base_conf

       1. input: 维多利亚的摩尔佧 性感连体泳衣女比基尼高级感时尚泡温泉泳装

       2. output:

          这款光滑的泳衣没有任何的剃裁，版型很好，上紧下松，上半身刚刚好，外面不卡脖子，夏天穿周身透气舒适，非常的性感 -137.44415283203125

          泳衣正反面均采用印花设计，色彩绚丽且超显气质，加身后完全就是胸部了，v字领口柔化脖颈曲线，给人更佳的锁骨效果，清爽的短裤，修饰出腰肢，舒适的长裤给人更加高挑的感觉，性感的露背设计岁月静好，时尚个性。 -221.9921417236328

          海边美女们都爱的游泳池，想去清凉一下心情，就带着奥特曼来到泳池边，度假的好伴侣就是它了。这款泳衣是连体的设计，一点也不显肥大，带来满满安 全感。高颜高性感的比基尼，性感不要不要的。腰间还有吊带，可以调整腰线。无论你是哪种身材都一样的好看。 -243.95460510253906

          

   13. Chinese product review generation: python interact.py --model_conf conf/review_base_conf

       1. input:  Cartier卡地亚LOVE戒指 玫瑰金黄金白金 窄版对戒

       2. output:

          还不错,挺好的 \_sep\_ 还是挺好的,这个价钱值了 -28.99207878112793

          刚买就降价了 \_sep\_ 东西还好,就是买完就降,郁闷 -32.52592468261719

          戒指感觉款式不错 \_sep\_ 其他都还可以,就是太小了 -37.16313934326172

          

   14. Chinese wuxia novel generation: python interact.py --model_conf conf/wx_base_conf

       1. input:

          小船转过一排垂柳，远远看见水边一丛花树映水而红，灿若云霞。段誉"啊"的一声低呼。

       2. output:

          小船转过一排垂柳，远远看见水边一丛花树映水而红，灿若云霞。段誉"啊"的一声低呼。木婉清和全金发对他似乎颇为悦眼,三人跨出小船,游到水边。木婉 清骂道:“死贼,又著了人家的道儿。”那花树旁边系著一只小船,段誉顺著花树,向左一看,只见花丛影里一个少女倚树而坐,正在倚树而歌,正是段仇世的女儿段天德。此时她歌声清脆,唱起了一首小调:南峰山舍忘词,曾经借问少白头。 \_nl\_ 三更双弦意满楼,征囊终不归。 \_nl\_ 美冠一行惟顾此,寒啸了一为身。 \_nl\_ 但愿人头云鬓和曳,何结同心千里随。 \_nl\_ 此情只在飞鸿迹,生死何必萦怀?只期玉箫一曲牵牛筋。 \_nl\_ 段誉一听歌声,脸上微微一红。只听吴天德叹 道:“我段天德生平以风流自炫,有时不免大意,却也只图一时快意。那是谁到了姑苏,听这歌词,果然是个一表人才的姑娘,我出来斟酒时,仍是将款,那里想得 到她会暗中加害于我了。侥幸令狐大哥吉人天相,报了大仇,咱们可得快些回来。” \_nl\_ 原来吴天德的妻子乃是江湖有名的“飞刀娘子”萧飞英,以一柄柳叶飞刀,在打中大名鼎鼎的“姑苏慕容”后,杀死了慕容复,雍和之位,秦晋云南,大享富贵,他夫妻双双归隐大理,直到廿四年之前,他夫妻两人路过姑苏,在姑苏最大的一间酒楼见到了段天德夫妇两个。那时他夫人仍穿着原服,段天德的妻子换上男装,他却略显道貌岸然,以示江湖儿女,不去理会他妻子。可是段天德夫妻 一看在他眼里,却心中暗暗纳罕:“她左颊明明是多刺了一粒小痣的,怎么这时忽然多了一颗大痣?”他虽情知萧飞英并不是什么好人,神态之间,总不免甚是小 心。 [-1582.4723]

          小船转过一排垂柳，远远看见水边一丛花树映水而红，灿若云霞。段誉"啊"的一声低呼。只听得段誉笑著道:“姊姊,快开船走吧,再耽一刻,天就要黑了。”柔声说道:“莫怕,莫怕……”忽听得呜咽之声,听那声音似是啼哭。段誉背向船舷,正好树丛中伸出一只犹似羊脂白玉的手 来。段誉道:“啊,姊姊,吓了你……”王玉燕紧紧搂住了他,斥道:“又瞎起上来!”段誉道:“姊姊……”王玉燕道:“嗯,是谁先开船的?说!”段誉叹道:“说 来话长,待我慢慢的跟你说。你先别走,我慢慢说。”王玉燕道:“些须小事,便说了了不成?”段誉道:“嗯,是……是许多事不便跟你动手。”玉燕突然想起, 他一个大男人跟著自己说话,半边身子情不自禁的移开了些,说道:“嗯,那你小心些不捱。”声音极是温柔。 \_nl\_ 段誉见她神态豁达,心下对她又增了几分爱慕,突然之间,觉得有一个出色的女子,即使是在镜中画图,在花间听风,也是和她一模一样,于是他心中又想:“王姑娘常说,这次金盆洗手,承玄难大师等各派英雄以为首领,算定那晦明禅师会乘机捣乱丐帮不成,故尔金盆洗手,将门派人物都归降了丐帮。此刻瞧他心情,倒像当教主的人,原是座下游客。倘若他当了帮主,那就……格格格,那可够瞧上老大一阵子啦。”从此他对王玉燕庄敬有加,十分顺从。 \_nl\_ 王玉燕见他又爱上了自己,不自禁耳根红透,两人虽是男女有别, 他这一句话都说了出来。王玉燕展眉微笑,道:“你瞧着我干么?”段誉道:“嗯,瞧著。 [-1303.9319]

          小船转过一排垂柳，远远看见水边一丛花树映水而红，灿若云霞。段誉"啊"的一声低呼。阿朱看得明白,原来是他在海中栖息,这才不觉奇怪。她笑道:“相公,怎么了?”段誉道:“没什么。” \_nl\_ 阿朱又道:“你这两日来很累了么?那你想打什么火?”段誉大口吸水,不答真文。阿朱也喝了两口,笑道:“这远处有座山,你可不由得和我拌嘴。”段誉抬头向她瞧了一眼,道:“你说那山。”阿朱道:“可不是么?隔山望见那山上山走来了两人,谁也不相识,原来是几个捉鬼的和尚。两人唠唠叨叨的说了一会儿子,也不知什么,两个和尚就跳到那座山上去。”段誉听到这里,突然手掌在水中一拍,高声叫道:“你瞧上了热闹没有?”阿朱 道:“什么热闹?”段誉道:“我在一张棺村之前撞见两条绿线……;什么绿线?快快给我招来。”他一面大叫,一面将这张形如巨棺的船板向外一推,便向那两艘绿线划去。阿朱伸手拦住了他,笑道:“不是两条,是四个!”她扯过段誉手中的短桨,运稳了桨桨,便催著船向山边驶去,果然越向山上去,那小径越形狭窄,好几次段誉都几乎要从船头跌将下去。阿朱又将船划得渐渐加快,呼呼风响,两岸山峦渐不相闻。 \_nl\_ 不片刻便到了阿湾,那阿湾只是零落的垂柳,有一条小径自 山坡蜿蜒而下,蜿蜒入丛林之中。阿朱把船划入丛林,驶入了一片石级小径,段誉负手站在钓摊之旁,一瞬不瞬的望著棋盘大声说那一集目两轸的棋局。他内功 虽然已经极有限,却也是个识货的行家,一见棋盘上黑白,便知阿朱兜了个大大的圈。 [-1382.7443]

          

   15. Chinese-English translation: python interact.py --model_conf conf/nmt_base_conf

       1. input:

          \_zhen\_ 2023年元旦刚过，菲律宾总统马科斯对中国进行了国事访问，他因此成为新年第一位访华的外国领导人。

       2. output:

          on new year's day, 2023, president marcos of the philippines paid a state visit to china, making him the first foreign leader to visit china in the new year.  -0.3184391975402832
       
          on new year's day, 2023, president marcos of the philippines paid a state visit to china, making him the first foreign leader to visit china.  -0.34901772605048287
       
          on new year's day, 2023, president marcos of the philippines made his first foreign visit to china.  -0.3811599291287936
       
          on new year's day, 2023, philippine president marcos made his first foreign visit to china.  -0.3870043357213338
       
          on new year's day 2023, philippine president marcos made his first foreign visit to china.  -0.41778796652088995
          
       3. input:
       
          \_enzh\_ New Zealand leader Jacinda Ardern announces shock resignation before upcoming election
       
       4. output:
       
          新西兰领导人雅辛达·阿尔德林在即将举行的选举前宣布辞职。  -0.2518916130065918
          
          新西兰领导人雅辛达·阿尔德林在即将到来的选举前宣布辞职。  -0.25208149285152043
          
          新西兰领导人雅辛达·阿尔德林在即将到来的选举前宣布了震惊辞职。  -0.2606537640094757
       
          新西兰领导人阿尔德林在即将举行的选举前宣布辞职。  -0.2731153869628906
       
          新西兰领导人阿尔德林在即将到来的选举前宣布辞职。  -0.2777881622314453
          
          
          
       
   16. Chinese paper generation: python interact.py --model_conf conf/paper_base_conf

       1. input:

          中日韩 \_sep\_ 自贸区

       2. output:

          中日韩自贸区发展的制约因素分析 \_sep\_ 发展中日韩自贸区的对策是自贸区的大幅提高。以签订《自由贸易协定》为契机,将各成员国 签订自贸协定作为中日韩自贸区发展的制约因素的内容,本文分析了中日韩自贸区制度的特点,提出了加强中日韩自贸区建设的建议。 -136.17562866210938
          
          发展自贸区:中日韩自贸区面临的机遇与挑战 \_sep\_ 金融危机以后,中日韩的自贸区得以提升。自贸区作为自贸区的重要组成部分,将有 力推动中日韩各国的经贸合作与交流。但是,由于自贸区自身存在内外部条件不协调、自贸区机制不完善、自贸区本身存在诸多问题,解 决自贸区发展问题成为全球一体化的关键问题。促进自贸区发展,中日韩自贸区的建设与发展需要充分调研,借鉴其成功经验,提出解决问题的对策。 -233.56361389160156
          
          中日韩自贸区与韩国新自贸区若干问题探析 \_sep\_ 自2005年日本政府最近在韩国举行调查示范活动后,由韩国自由贸易区(garthge aurantifiation,api)于2007年启动,正式于2008年建立了中日韩自贸区。在自贸区的建立、保护、协调等方面,中日韩三国及其相关部门在经贸领域取得了宝贵的成果。但是,由于签订协议不够透明,各自制度较为模糊,使中日韩自贸区的发展有今天的困境。 -316.9208068847656
          
       
   17. Chinese tag generation: python interact.py --model_conf conf/tag_base_conf

       1. input:

          近日，由国家感染性疾病临床医学研究中心、中国人民解放军总医院第五医学中心主办的国家感染性疾病临床医学研究中心2023年学术 年会在北京召开。《每日经济新闻》记者在现场注意到，中科院院士高福等专家学者就长新冠、疫苗和药物研发等话题展开交流。后新 冠时期既不能"恐冠"也不能"轻冠"针对新冠病毒，高福表示，在后新冠时期，大家开始讨论人类已经走出了新冠病毒的阴影，但我们不 能只看新冠病毒带来的挑战，也要看到它给我们提供的机遇，以及让人类如何重新思考。后新冠时期既不能"恐冠"也不能"轻冠"。

       2. output:

          医学 \_sep\_ 临床医学 \_sep\_ 新型冠状病毒 -7.32832670211792
          

   18. Chinese medical qa generation: python interact.py --model_conf conf/med_base_conf

       1. input:

          最近老是犯困全身乏力腰酸背痛

       2. output:

          就您的情况考虑是气血虚弱,植物神经功能紊乱导致的,建议用归脾丸治疗。 -29.775508880615234
          
          这种情况可能是由于多种原因引起的,例如疲劳、压力过大、饮食不规律等。建议您注意休息,适当进行体育锻炼,保持良好的心态和饮食习惯,同时也要注意补充足够的营养物质, 如蛋白质、维生素及矿物质等。如果症状持续存在或加重,建议您咨询医生进行检查和治疗。 -50.89771270751953
          
          这样的情况要考虑是否有肾虚或者腰椎疾病引起的,建议你去医院做一下检查看看。可以吃点六味地黄丸和金锁固精丸来治疗,注意休息,不要过度劳累 -53.82005310058594
          

   19. Chinese doc2query generation: python interact.py --model_conf conf/doc2query_base_conf

       1. input:

          "五一"假期过后，社交媒体上出现了许多关于二阳的讨论。据北京疾控公布的第19周 （5月8日-14日）的数据，全市共报告法定传染病16种18081例。新冠感染连续3周超越流感，重返法定传染病病种排名第一。

       2. output:

          北京二阳是什么意思 -4.803812503814697
          
          北京新冠病毒多少 -6.473724365234375
          
          北京新冠病毒感染数量 -6.521775245666504


   20. Chinese spelling correction: python interact.py --model_conf conf/csc_base_conf

       1. input:

          大家要努力鞋习aigc只是。

       2. output:

          大家要努力学习aigc知识。 -0.0025362607557326555


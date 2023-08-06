# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:54:44 2023

@author: Xiaoyuan Yao
"""

import streamlit as st
from mimix.predictor import EncDecGenerator,LMGenerator,TextEncoder
from mimix.predictor import ImageEncoder


def main():
    st.markdown(
        """
        ## Mimix DEMO
        """
    )
    usage = "usage: app.py --model_conf <file>"
    parser = OptionParser(usage)

    parser.add_option("--model_conf", action="store", type="string",
                      dest="model_config")
    parser.add_option("--mode", action="store", type="string",
                      dest="mode", default="demo")
    
    (options, args) = parser.parse_args(sys.argv)

    if not options.model_config:
        print(usage)
        sys.exit(0)

    conf_file = options.model_config
    config = load_model_config(real_path(conf_file))

    content = st.text_area("输入新闻正文", max_chars=512)
    if st.button("一键生成摘要"):
        start_message = st.empty()
        start_message.write("正在抽取，请等待...")
        start_time = time.time()
        titles = predict_one_sample(model, tokenizer, device, args, content)
        end_time = time.time()
        start_message.write("抽取完成，耗时{}s".format(end_time - start_time))
        for i, title in enumerate(titles):
            st.text_input("第{}个结果".format(i + 1), title)
    else:
        st.stop()
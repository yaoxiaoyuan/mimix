# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:54:44 2023

@author: Xiaoyuan Yao
"""
import time
import sys
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import io
import streamlit as st
from mimix.predictor import EncDecGenerator,LMGenerator,TextEncoder
from mimix.predictor import ImageEncoder, ClipMatcher, MAE
from mimix.utils import real_path, load_model_config

st.set_page_config(page_title="Mimix Demo", initial_sidebar_state="auto", layout="wide")
    
def text_gen_app(model):
    """
    """
    st.markdown(
        """
        ## Text Generation DEMO
        """
    )
     
    with st.form(key='my_form'):
        text = st.text_area("input text", max_chars=512)
        values = ('beam_search', 'sample')
        strategy = st.selectbox('strategy', values, index=values.index(model.strategy))
        beam_size = st.number_input("beam_size", min_value=0, max_value=10, value=model.beam_size)
        group_size = st.number_input("group_size", min_value=0, max_value=5, value=model.group_size)
        max_decode_steps = st.number_input("max_decode_steps", min_value=0, max_value=512, value=model.max_decode_steps, step=1)
        repetition_penalty = st.slider("repetition_penalty", min_value=-10.0, max_value=0.0, value=model.repetition_penalty, step=0.1)
        temperature = st.slider("temperature", min_value=0., max_value=10.0, value=model.top_p, step=0.01)
        top_k = st.number_input("top_k", min_value=0, max_value=100, value=model.top_k, step=1)
        top_p = st.slider("top_p", min_value=0., max_value=1.0, value=model.top_p, step=0.01)
    
        submit = st.form_submit_button("generate")
        
        if submit:
            model.strategy = strategy
            model.beam_size = beam_size
            model.group_size = group_size
            model.max_decode_steps = max_decode_steps
            model.repetition_penalty = repetition_penalty
            model.top_k = top_k
            model.top_p = top_p
            start_message = st.empty()
            if model.group_size > 0 and model.beam_size % model.group_size != 0:
                start_message.write("beam_size must be a multiple of group_size!")
            else:
                start_message.write("generating...")
                start_time = time.time()
                res = model.predict([text])
                end_time = time.time()
                start_message.write("done, cost{}s".format(end_time - start_time))
                for i, (text, score) in enumerate(res[0][1]):
                    st.text_area("the {} result".format(i + 1), text)


def image_classification_app(model):
    """
    """
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(image, width=224)
        res = model.predict_cls([image])
        for i, (label, score) in enumerate(res[0][1]):
            st.text_area("top {} result".format(i + 1), label + " " + str(score))


def image2text_app(model):
    """
    """
    st.markdown(
        """
        ## Text Generation DEMO
        """
    )
    
    with st.form(key='my_form'):
        uploaded_file = st.file_uploader("Choose a image file", type="jpg")
        prefix = st.text_area("input text, can be empty", max_chars=512)
        values = ('beam_search', 'sample')
        strategy = st.selectbox('strategy', values, index=values.index(model.strategy))
        beam_size = st.number_input("beam_size", min_value=0, max_value=10, value=model.beam_size)
        group_size = st.number_input("group_size", min_value=0, max_value=5, value=model.group_size)
        max_decode_steps = st.number_input("max_decode_steps", min_value=0, max_value=512, value=model.max_decode_steps, step=1)
        repetition_penalty = st.slider("repetition_penalty", min_value=-10.0, max_value=0.0, value=model.repetition_penalty, step=0.1)
        temperature = st.slider("temperature", min_value=0., max_value=10.0, value=model.top_p, step=0.01)
        top_k = st.number_input("top_k", min_value=0, max_value=100, value=model.top_k, step=1)
        top_p = st.slider("top_p", min_value=0., max_value=1.0, value=model.top_p, step=0.01)
    
        submit = st.form_submit_button("generate")
        
        if uploaded_file is not None and submit:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            st.image(image, width=224)
            model.strategy = strategy
            model.beam_size = beam_size
            model.group_size = group_size
            model.max_decode_steps = max_decode_steps
            model.repetition_penalty = repetition_penalty
            model.top_k = top_k
            model.top_p = top_p
            start_message = st.empty()
            if model.group_size > 0 and model.beam_size % model.group_size != 0:
                start_message.write("beam_size must be a multiple of group_size!")
            else:
                start_message.write("generating...")
                start_time = time.time()
                
                prefix_list = None
                if prefix is not None and len(prefix) > 0:
                    prefix_list = [prefix]
                res = model.predict([image], prefix_list)
                image.close()
        
                end_time = time.time()
                start_message.write("done, cost{}s".format(end_time - start_time))
                for i, (text, score) in enumerate(res[0][1]):
                    st.text_area("the {} result".format(i + 1), text)


def image_text_match_app(model):
    """
    """
    st.markdown(
        """
        ## CLIP DEMO
        """
    )
    
    with st.form(key='my_form'):
        uploaded_file = st.file_uploader("Choose a image file", type="jpg", key="1")
        uploaded_file_2 = st.file_uploader("Choose a image file", type="jpg", key="2")
        texts = st.text_area("input text", max_chars=512)
        submit = st.form_submit_button("compute image text match score") 
        submit_2 = st.form_submit_button("compute image image match score") 
        #submit_3 = st.form_submit_button("compute text text match score") 
        if submit:
            if uploaded_file is not None and len(texts) > 0:
                image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                texts = [text.strip() for text in texts.split("\n")]
                texts = [text for text in texts if len(text) > 0]
                st.image(image, width=224)
                res = model.predict_sim([image], texts)
                res = [[text,score,prob] for text,score,prob in zip(texts, res[0][0], res[1][0])]
                res.sort(key=lambda x:x[1], reverse=True)
                for text,score,prob in res:
                    info = "%s match score: %.2f, match prob: %.2f" % (text, score, prob)
                    st.text(info)
            else:
                info = "image and text must not be empty!"
                st.text(info)

        if submit_2:
            if uploaded_file is not None and uploaded_file_2 is not None:
                image = Image.open(io.BytesIO(uploaded_file.getvalue()))
                image_2 = Image.open(io.BytesIO(uploaded_file_2.getvalue()))
                st.image(image, width=224)
                st.image(image_2, width=224)
                res = model.predict_images_sim([image, image_2])
                info = "match score: %.2f" % (res[0][0][1])
                st.text(info)
            else:
                info = "image and image_2 must not be empty!"
                st.text(info) 
        
        '''
        if submit_3:
            if len(texts) > 0:
                res =  model.predict_texts_sim(texts)
                texts = [text.strip() for text in texts.split("\n")]
                texts = [text for text in texts if len(text) > 0]
                for i,text in enumerate(texts):
                    for j,text_2 in enumerate(texts):
                        if j > i:
                            info = "%s %s match score: %.2f" % (text, text_2, res[0][i][j])
                            st.text(info)
            else:
                info = "texts must not be empty!"
                st.text(info) 
        '''


def mae_app(model):
    """
    """
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        st.image(image, width=224)
        origin, reconstruct, im_masked, im_paste = model.visualize(image)
        
        origin = Image.fromarray(origin.astype(np.uint8))
        reconstruct = Image.fromarray(reconstruct.astype(np.uint8))
        im_masked = Image.fromarray(im_masked.astype(np.uint8))
        im_paste = Image.fromarray(im_paste.astype(np.uint8))
        
        st.text("masked")
        st.image(im_masked, width=224)
        st.text("reconstruct")
        st.image(reconstruct, width=224)
        st.text("reconstruction + visible")
        st.image(im_paste, width=224)

def run_app():
    """
    """
    parser = ArgumentParser()

    parser.add_argument("--model_conf", type=str)
    
    args = parser.parse_args(sys.argv[1:])

    model_config = load_model_config(real_path(args.model_conf))
        
    if model_config["task"] == "enc_dec":
        model = EncDecGenerator(model_config)
        text_gen_app(model)
    elif model_config["task"] == "lm":
        model = LMGenerator(model_config)
        text_gen_app(model)   
    elif model_config["task"] == "image_classification":
        model = ImageEncoder(model_config)
        image_classification_app(model)
    elif model_config["task"] == "image2text":
        model = EncDecGenerator(model_config)
        image2text_app(model)
    elif model_config["task"] == "image_text_match":
        model = ClipMatcher(model_config)
        image_text_match_app(model)
    elif model_config["task"] == "masked_auto_encoder":
        model = MAE(model_config)
        mae_app(model)
        
if __name__ == "__main__":
    run_app()
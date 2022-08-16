# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:54:03 2022

@author: Xiaoyuan Yao
"""
import models
import dataset
import loss
import process_data

def add_task_model(task, build_model_fn):
    """
    """
    models.model_builder_dict[task] = build_model_fn


def add_task_datset(task, build_dataset_fn):
    """
    """
    dataset.dataset_builder_dict[task] = build_dataset_fn


def add_task_loss(task, build_loss_fn):
    """
    """
    loss.loss_builder_dict[task] = build_loss_fn


def add_task_data_processor(task, build_data_processor_fn):
    """
    """
    process_data.data_processor_builder_dict[task] = build_data_processor_fn

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jasoncheung <jasoncheung@iZwz95ffbqqbe9pkek5f3tZ>
#
# Distributed under terms of the MIT license.

"""

"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModelForPreTraining
from transformers import TFElectraForSequenceClassification, ElectraConfig
from transformers import TFTrainer, TFTrainingArguments
from transformers import training_args

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

dir_path = '/home/jasoncheung/project/trans/trans_models/electra_chinese_small'

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-electra-180g-small-discriminator")

config = ElectraConfig.from_pretrained(
    'hfl/chinese-electra-180g-small-discriminator', num_labels=5,)

# model = TFAutoModelForPreTraining.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
# model = TFElectraForSequenceClassification.from_pretrained("hfl/chinese-electra-180g-small-discriminator")

# inputs = tokenizer("你听明白了吗", return_tensors="tf")
# outputs = model(**inputs)
# print(inputs, outputs)

# load datas
path_datas = '/home/jasoncheung/project/trans/trans_datas/weibo_senti_100k.csv'
df = pd.read_csv(path_datas)
datas = df.review.tolist()
labels = df.label.tolist()

train_datas, test_datas, train_labels, test_labels = train_test_split(datas, labels, test_size=0.1)
train_datas, val_datas, train_labels, val_labels = train_test_split(train_datas, train_labels, test_size=0.1)

train_encodings = tokenizer(train_datas, truncation=True, padding='max_length', max_length=180)
val_encodings = tokenizer(val_datas, truncation=True, padding='max_length', max_length=180)
test_encodings = tokenizer(test_datas, truncation=True, padding='max_length', max_length=180)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_labels))


# training
training_args = TFTrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    save_total_limit=5,
    evaluation_strategy='steps',
    eval_steps=250,
    load_best_model_at_end=True,
    disable_tqdm=False,
    max_steps=1000,

)

with training_args.strategy.scope(): 
    model = TFElectraForSequenceClassification.from_pretrained(dir_path, 
                                                               num_labels=2, 
                                                               )
    # model.load_weights('/home/jasoncheung/project/trans/results/checkpoint/ckpt-18.index')


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,    # ensorflow_datasets training dataset
    eval_dataset=val_dataset,       # tensorflow_datasets evaluation dataset
    compute_metrics=compute_metrics,

)

# trainer.train()


'''
dir_path = '/home/jasoncheung/project/trans/trans_models/electra_chinese_small/'
model.save_pretrained(dir_path)
config.save_pretrained(dir_path)
tokenizer.save_vocabulary(dir_path)
'''



#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jasoncheung <jasoncheung@iZwz95ffbqqbe9pkek5f3tZ>
#
# Distributed under terms of the MIT license.

"""

"""
import re
import json

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import logging


# 文本预处理规则
def extract(s):
    result = re.sub('回复(.*?):', '', s)
    return result


def corpus2label(datas):
    punc = r'~`!#$%^&*()_+-=|\\\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    datas = [extract(i) for i in datas]
    datas = [i.replace(' ', '').replace('\t', '').strip() for i in datas]
    datas = [re.sub(r"[%s]+" % punc, "", i) for i in datas]
    datas = [[char2idx[i] if i in char2idx else char2idx['<UNK>'] for i in j] for j in datas]
    datas = tf.keras.preprocessing.sequence.pad_sequences(datas, maxlen=MAX_LEN, padding='post', truncating='post')
    print('datas.shape', datas.shape)

    return datas


def Cnn_softmax(lens):# CNN best for ToB
    inputs = tf.keras.Input(shape=(lens, ), )
    embed = tf.keras.layers.Embedding(input_dim=len(char2idx), output_dim=EMBEDDING_DIM)(inputs)
    cnn_layer = tf.keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    pooling_layer = tf.keras.layers.MaxPool1D(pool_size=46)(cnn_layer)
    flatten_layer = tf.keras.layers.Flatten()(pooling_layer)
    dropout = tf.keras.layers.Dropout(0.3)(flatten_layer)
    y = tf.keras.layers.Dense(1, activation='softmax')(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=[y])
    print(model.summary())
    return model 


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


class JasonTools(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train_model(self, model, model_path, original_model):
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='adam', loss=binary_focal_loss(gamma=2, alpha=0.25), metrics=['accuracy'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1,
                                                     save_best_only=True, save_weights_only=1, period=1)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min',
                                                  restore_best_weights=True)
        model_history = model.fit(self.x_train, self.y_train, shuffle=True, epochs=EPOCHS, validation_split=0.1,
                                  callbacks=[checkpoint, earlystop])

        model = original_model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)

        logging.info('best_model_path:  %s' % model_path)
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        self.cal_pr(model, self.x_test, self.y_test)

        return model_history

    def finetune_model(self, model, model_path, x_datas, y_datas, class_weight=None):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                                     mode='min', verbose=1,
                                                     save_best_only=True, save_weights_only=1, period=1)
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min',
                                                  restore_best_weights=True)
        if class_weight:
            model.fit(x_datas, y_datas, shuffle=True, epochs=EPOCHS,
                      validation_split=0.1,
                      callbacks=[checkpoint, earlystop], class_weight=class_weight, batch_size=8)
        else:
            model.fit(x_datas, y_datas, shuffle=True, epochs=EPOCHS,
                      validation_split=0.1,
                      callbacks=[checkpoint, earlystop], batch_size=8)

        model.load_weights(model_path)
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        logging.info('best_finetune_model_path:  %s' % model_path)
        logging.info('best_model_path:  %s' % model_path)
        logging.info('test_loss: %.3f - test_acc: %.3f' % (test_loss, test_acc))
        self.cal_pr(model, self.x_test, self.y_test)

    @staticmethod
    def cal_pr(model, x_test, y_test):
        pred = model.predict(x_test)
        pred = [i[0] for i in pred]
        pred = [1 if i >= 0.5 else 0 for i in pred]
        true = y_test
        # pred = list(np.argmax(pred, axis=1))
        # true = list(np.argmax(self.y_test, axis=1))
        report = classification_report(true, pred)
        logging.info('classification_report: \n')
        logging.info(report)
        logging.info('confusion_matrix: \n')
        logging.info(confusion_matrix(true, pred))
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        logging.info('tn: %d, fp: %d, fn: %d, tp: %d' % (tn, fp, fn, tp))
        return report
    @staticmethod
    def plot_history(histories, path='model/acc_char.png', key='accuracy'):
        plt.figure(figsize=(16, 10))

        for name, history in histories:
            val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + ' Val')
            plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title() + ' Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.legend()

        plt.xlim([0, max(history.epoch)])
        plt.savefig(path)


# if __name__ == '__main__':
# load datas
MAX_LEN = 100
EMBEDDING_DIM = 100
EPOCHS = 20

path_datas = '/home/jasoncheung/project/trans/trans_datas/weibo_senti_100k.csv'
df = pd.read_csv(path_datas)
datas = df.review.tolist()
labels = df.label.tolist()
labels = np.array(labels)

dir_path = '/home/jasoncheung/project/trans/trans_models/'
idx2char = json.load(open(dir_path+'idx2char.json', 'r'))
char2idx = json.load(open(dir_path+'char2idx.json', 'r'))


print('before: ', len(datas), datas[0])
datas = corpus2label(datas)
print('after: ', len(datas), datas[0])


train_datas, test_datas, train_labels, test_labels = train_test_split(datas, labels, test_size=0.1)

# load model
trainer = JasonTools(train_datas, train_labels, test_datas, test_labels)
model = Cnn_softmax(MAX_LEN)

model_path = './model/cnn_char/cnn_char.h5'
record = []

record.append(('cnn_char', trainer.train_model(model, model_path, Cnn_softmax(MAX_LEN))))

trainer.plot_history(record, path='model/acc_char_emotion_positive.png')



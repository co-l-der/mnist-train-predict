#!/usr/bin/env python
# encoding: utf-8
__author__ = "han"

import numpy as np
from tensorflow import keras


class DataLoader(object):
    """
    作用： 加载原始数据类
    """
    num_classes = 10
    img_rows = 28
    img_cols = 28

    def __init__(self, path):
        self.path = path

    def load_data(self):
        """
        作用：加载mnist数据集，并处理成模型所需的数据
        :return:
        """

        if not self.path:
            raise Exception("data path not initialized.")
        with np.load(self.path, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
        input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train/255
        x_test = x_test/255

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return input_shape, x_train, x_test, y_train, y_test


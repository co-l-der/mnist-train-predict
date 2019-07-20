#!/usr/bin/env python
# encoding: utf-8
import cv2
import numpy as np
from tensorflow import keras

__author__ = "han"


class Predictor(object):
    """
    作用:用于模型预测
    """
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def load_input(input_x):
        img = cv2.imread(input_x, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = cv2.bitwise_not(img)
        img = img.reshape((1, 28, 28, 1))
        img = img.astype("float32")
        standard_input = img / 255
        return standard_input

    def predict(self, input_x):
        standard_input = Predictor.load_input(input_x)
        result = self.model.predict(standard_input, batch_size=1, verbose=0)
        result = np.array(result).flatten()
        result = result.tolist()
        result = result.index(max(result))
        return result
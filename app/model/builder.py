#!/usr/bin/env python
# encoding: utf-8
import os

__author__ = "han"

import tensorflow as tf
from tensorflow import keras
import keras.backend.tensorflow_backend as KTF


class ModelBuilder(object):
    """
    作用：用于模型构建, dense layer
    """
    batch_size = 128
    epochs = 1

    def __init__(self):
        self.gpu_device = None
        self.input_shape = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model_path = None
        self.model = None

    def init_resource(self, gpu_device=None, gpu_fraction=None):
        """
        功能：初始化训练所需设备信息
        :param gpu_device: 显卡编号，如“0”，“0,1”
        :param gpu_fraction:显存使用率
        :return:
        """
        if isinstance(gpu_device, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
        if isinstance(gpu_fraction, (float, int)):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
            KTF.set_session(tf.Session(config=config))

    def init_input(self, input_shape, x_train, x_test, y_train, y_test):
        self.input_shape = input_shape
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = None

    def init_output(self, model_path):
        self.model_path = model_path

    def build(self):
        """
        作用： 建立模型
        :return:
        """

        num_classes = self.y_train.shape[-1]

        image_input = keras.Input(shape=self.input_shape)

        layer1 = keras.layers.Conv2D(32, kernel_size=(3, 3),
                                     activation="relu")(image_input)

        layer2 = keras.layers.Conv2D(64, (3, 3), activation="relu")(layer1)

        pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(layer2)
        dp1 = keras.layers.Dropout(0.25)(pool1)

        fl1 = keras.layers.Flatten()(dp1)
        fc1 = keras.layers.Dense(128, activation="relu")(fl1)

        dp2 = keras.layers.Dropout(0.5)(fc1)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(dp2)

        self.model = keras.Model(inputs=image_input, outputs=output_layer)

    def train(self):
        """
        作用：模型训练
        :return:
        """
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        checkpoints = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            verbose=1
        )

        callbacks = [checkpoints]

        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test),
                       callbacks=callbacks)

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
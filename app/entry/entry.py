#!/usr/bin/env python
# encoding: utf-8
import os
import re

from app.client.predictor import Predictor
from app.data.loader import DataLoader
from app.entry import Entry
from app.model.builder import ModelBuilder
from app.utils.tool import ConfigTool, StringConvertTool

__author__ = "han"


class ModelEntry(Entry):
    """
    作用：训练模型入口
    """

    def __init__(self, config):
        super().__init__(config)
        # 获取当前类的配置
        self.config = ConfigTool.get_current_config(self.config, __class__)

    def run(self):
        train_data_path = self.config.get("train_data_path")
        model_path = self.config.get("model_path")
        data_loader = DataLoader(train_data_path)
        input_shape, x_train, x_test, y_train, y_test = data_loader.load_data()
        model_builder = ModelBuilder()
        model_builder.init_input(input_shape, x_train, x_test, y_train, y_test)
        model_builder.init_output(model_path)
        model_builder.build()
        model_builder.train()


class DataEntry(Entry):

    def __init__(self, config):
        super().__init__(config)
        self.config = ConfigTool.get_current_config(self.config, __class__)

    def run(self):
        # TODO:此处model包下要执行的模块
        for i in range(10):
            print("data entry run %s" % str(i))


class ClientEntry(Entry):

    def __init__(self, config):
        super().__init__(config)
        self.config = ConfigTool.get_current_config(self.config, __class__)

    def run(self):
        model_path = self.config.get("model_path")
        test_data_path = self.config.get("test_data_path")
        predictor = Predictor(model_path)
        imgs = os.listdir(test_data_path)
        imgs = StringConvertTool.sort(imgs)
        for img in imgs:
            result = predictor.predict(os.path.join(test_data_path, img))
            print(result)



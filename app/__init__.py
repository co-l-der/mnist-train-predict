#!/usr/bin/env python
# encoding: utf-8
from app.entry import *
from app.utils.config import Config
from app.utils.tool import ClassTool

__author__ = "han"


def create_app(config="../config/config.py"):
    conf = Config(__name__)
    conf.from_pyfile(config)
    # 获取单个entry
    # model_entry = ModelEntry(conf)
    # app = AppGenerator(model_entry, conf)
    # 获取所有entries, 无序
    # entries = ClassTool.get_all_classes(Entry)
    # app = AppGenerator(entries, conf)
    # 自定义entries组合
    # 训练时使用
    # select_entries = [ModelEntry, ClientEntry]
    # 预测时使用
    select_entries = [ClientEntry]
    app = AppGenerator(select_entries, conf)
    return app

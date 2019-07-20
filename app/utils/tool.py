#!/usr/bin/env python
# encoding: utf-8
import re

__author__ = "han"


class ClassTool(object):

    @staticmethod
    def get_all_classes(parent):
        """
        功能：获取其父类下的所有直接子类
        :param parent: 父类名称
        :return:
        """
        all_subclasses = list()
        for subclass in parent.__subclasses__():
            if (not subclass in all_subclasses):
                all_subclasses.append(subclass)
        return all_subclasses

    # 由文件获取当前父目录, 父目录不跟随调用改变
    # current_dir = os.path.dirname(os.path.realpath(__file__))


class ConfigTool(object):

    @staticmethod
    def get_current_config(config, current_class):
        current_config = config.get(current_class.__name__.upper().replace("ENTRY", "_ENTRY"))
        return current_config


class StringConvertTool(object):

    @staticmethod
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    @staticmethod
    def str2int(v_str):
        return [StringConvertTool.tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

    @staticmethod
    def sort(v_list):
        """
        功能：将含有数字的字符串列表按照其中数字排序
        :param v_list: 含有数字的字符串列表
        :return:
        """
        return sorted(v_list, key=StringConvertTool.str2int)
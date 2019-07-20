#!/usr/bin/env python
# encoding: utf-8

from app.entry import Entry

__author__ = "han"


class AppGenerator(object):

    def __init__(self, entries, config):
        self.entries = entries
        self.config = config

    def run(self):
        obj_list = self.get_entry_obj()
        runner = map(lambda obj: obj.run(), obj_list)
        list(runner)

    def run_async(self):
        # TODO:异步执行待调试
        obj_list = self.get_entry_obj()
        gevent_list = [gevent.spawn(obj.run()) for obj in obj_list]

    def get_entry_obj(self):
        obj_list = list()
        if isinstance(self.entries, Entry):
            obj_list.append(self.entries)

        if isinstance(self.entries, list):
            obj_list = [obj(self.config) for obj in self.entries]

        return obj_list
#!/usr/bin/env python
# encoding: utf-8
from app import create_app

__author__ = "han"

app = create_app("../config/config.py")

if __name__ == "__main__":
    app.run()
    # app.run_async()

"""
项目配置
"""

import os

APP_PATH = "/home/han/self/core_template_code/mnist-train-predict"

MODEL_ENTRY = {
    "train_data_path": os.path.join(APP_PATH, "data/mnist.npz"),
    "model_path": os.path.join(APP_PATH, "model/mnist.h5")
}

CLIENT_ENTRY = {
    "model_path": MODEL_ENTRY.get("model_path"),
    "test_data_path": os.path.join(APP_PATH, "test")
}
"""
项目配置
"""

import os

# 尽量设置成绝对路径
APP_PATH = "/home/han/self/core_template_code/mnist-train-predict"

MODEL_ENTRY = {
    "gpu_device": "0",
    "gpu_fraction": 0.6,
    "train_data_path": os.path.join(APP_PATH, "data/mnist.npz"),
    "model_path": os.path.join(APP_PATH, "model/mnist.h5")
}

CLIENT_ENTRY = {
    "model_path": MODEL_ENTRY.get("model_path"),
    "test_data_path": os.path.join(APP_PATH, "test")
}
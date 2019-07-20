
**简要描述：**

项目套用之前only-run-demo模板，实现ModelEntry和ClientEntry逻辑

1. 使用keras搭建cnn模型（2层卷积层+2层全连接层）,识别手写字

2. ClientEntry中实现了使用测试集进行批量手写字识别

**说明：**

1. 训练数据集为官方提供的mnist.npz,一共有60000条训练样本

2. 训练结果,loss:0.2744, acc:0.9151, val_loss:0.0718, val_acc:0.9752



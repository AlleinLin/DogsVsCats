# 猫狗分类项目

## 项目简介

这是一个基于改进AlexNet模型的猫狗分类项目，使用PyTorch实现。项目可以通过训练模型来识别图片中的猫和狗，并提供了预测功能。

## 项目结构

```
DogVsCat/
├── data/                # 数据目录
│   ├── train/           # 训练数据
│   │   ├── cat/         # 猫的图片
│   │   └── dog/         # 狗的图片
│   └── predict/         # 预测用的图片
├── model/               # 保存的模型
├── test/                # 测试数据
├── AlexNet_new.py       # 改进的AlexNet模型
├── DataLoader.py        # 数据加载器
├── predict.py           # 预测脚本
├── train.py             # 训练脚本
├── log.txt              # 训练日志
└── README.md            # 项目说明
```

## 技术栈

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib
- numpy

## 安装依赖

```bash
pip install torch torchvision pillow matplotlib numpy
```

## 数据准备

1. 在 `data/train/` 目录下创建 `cat/` 和 `dog/` 子目录
2. 将猫的图片放入 `data/train/cat/` 目录
3. 将狗的图片放入 `data/train/dog/` 目录
4. 对于预测，将需要分类的图片放入 `data/predict/` 目录

## 模型训练

运行训练脚本：

```bash
python train.py
```

训练过程中会输出每个epoch的训练和测试准确率，训练完成后模型会保存到 `model/` 目录。

## 模型预测

运行预测脚本：

```bash
python predict.py
```

预测脚本会加载训练好的模型，并对 `data/predict/` 目录中的图片进行分类，显示预测结果。

## 模型说明

项目使用了改进的AlexNet模型，主要改进包括：

1. 添加了BatchNorm层，加速训练并提高模型稳定性
2. 使用了Dropout层，防止过拟合
3. 支持Adam优化器（可在代码中切换）
4. 支持学习率衰减

## 训练参数

- 训练轮数：18（可在代码中修改）
- 批处理大小：32
- 学习率：0.001
- 优化器：SGD（带动量0.9）
- 损失函数：CrossEntropyLoss

## 预测结果

预测脚本会显示图片和对应的预测结果（猫或狗）。

## 注意事项

1. 确保数据目录结构正确
2. 训练过程中会使用GPU（如果可用）
3. 预测脚本默认使用CPU进行预测
4. 模型文件较大，请确保有足够的存储空间

## 性能指标

在测试集上，模型准确率可达99%以上。

## 参考资料

- AlexNet论文：[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- PyTorch官方文档：https://pytorch.org/docs/stable/index.html
- torchvision官方文档：https://pytorch.org/vision/stable/index.html

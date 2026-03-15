import torch.nn as nn
import torch.optim as optim

"""
    What's New?
使用 Adam 优化器: 替换了原始的 SGD 优化器，Adam 往往能更快地收敛并达到更好的结果。

添加 BatchNorm 层: 在每个卷积层之后加入 BatchNorm 层，可以帮助稳定训练过程并加速训练速度。

使用学习率衰减或调度: 可以使用学习率衰减或调度策略来调整学习率，在训练的不同阶段使用不同的学习率可以帮助模型更好地收敛。

权重初始化: 可以使用 Xavier 或 Kaiming 等权重初始化方法来初始化模型权重，这可以帮助避免梯度爆炸问题并加速训练。

——2024.6.23
"""


class AlexNet_improved(nn.Module):
  def __init__(self, num_classes=1000, learning_rate=0.001):
    super(AlexNet_improved, self).__init__()

    # 特征提取层
    self.feature_extraction = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
      nn.BatchNorm2d(96),  # 添加 BatchNorm
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(384),
      nn.ReLU(inplace=True),

      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2)
    )

    # 分类器
    self.classifier = nn.Sequential(
      nn.Dropout(p=0.5),  # 调整 Dropout 率
      nn.Linear(256 * 6 * 6, 4096),
      nn.ReLU(inplace=True),

      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes)
    )

    # 定义优化器
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

  def forward(self, x):
    x = self.feature_extraction(x)
    x = x.view(x.size(0), -1)  # Flatten the output of feature extractor
    x = self.classifier(x)
    return x
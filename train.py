import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from AlexNet_new import AlexNet_improved
from DataLoader import DogCatDataSet

random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

#epochs = 70 # 基准模型训练次数，在70次时间准确率在98%，使用SGD优化器模型，不存在 BatchNorm 层 和 Dropout

#epochs = 50

epochs = 18  # 训练次数 实际测试迭代100次在第20次时准确率已经达到100%，出现过拟合，Adam优化器的收敛速度明显快于基准模型的SGD ——2024.6.24
'''
测试中第15次训练存在：
train 15 epoch loss: 0.041  acc: 98.550 
test  15 epoch loss: 0.022  acc: 99.200 
尝试训练次数减少到13次 ——2024.6.25
'''
# epochs = 13 # 2024.6.25

#epochs = 11 #2024.6.26

batch_size = 32  # 批处理大小
num_workers = 22  # 多线程的数目
use_gpu = torch.cuda.is_available()

# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

])


train_dataset = DogCatDataSet(img_dir="./data/train", transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

test_dataset = DogCatDataSet(img_dir="./data/validation", transform=data_transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)



net = AlexNet_improved(num_classes=2)


if use_gpu:
    net = net.cuda()


# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #实践中使用Adam反而使得模型随着迭代次数增加逐渐欠拟合，遂改回SGD
#optimizer = optim.Adam(net.parameters(), lr=0.001) # 将优化器由SGD更换为Adam，提供更快的收敛速度

if __name__ == '__main__':

    # 开始训练
    net.train()
    for epoch in range(epochs):
        if((epoch + 1) % 20 == 0):
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * 0.1
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
        epoch + 1, running_loss / train_total * batch_size, 100 * train_correct / train_total))

        # 模型测试
        correct = 0
        test_loss = 0.0
        test_total = 0

        net.eval()
        for data in test_loader:
            images, labels = data
            images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total * batch_size, 100 * correct / test_total))
    torch.save(net, "./model/AlexNet_CatvsDog_improved.pth")

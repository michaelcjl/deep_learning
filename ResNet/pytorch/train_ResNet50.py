# import seaborn as sns
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST

from ResNet import *


# 处理训练集数据
def train_data_process():
    # 加载FashionMNIST数据集
    train_data = FashionMNIST(
        root="./data/FashionMNIST",  # 数据路径
        train=True,  # 只使用训练数据集
        transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]
        ),  # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型
        # 尺寸为Channel * Height * Width，数值范围缩小为[0.0, 1.0]
        download=True,  # 若本身没有下载相应的数据集，则选择True
    )

    train_loader = Data.DataLoader(
        dataset=train_data,  # 传入的数据集
        batch_size=64,  # 每个Batch中含有的样本数量
        shuffle=False,  # 不对数据集重新排序
        num_workers=2,  # 加载数据所开启的进程数量
    )
    print("The number of batch in train_loader:", len(train_loader))  # 一共有938个batch，每个batch含有64个训练样本

    # 获得一个Batch的数据
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组
    class_label = train_data.classes  # 训练集的标签
    class_label[0] = "T-shirt"
    print("the size of batch in train data:", batch_x.shape)

    # 可视化一个Batch的图像
    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()

    return train_loader, class_label


# 处理测试集数据
def test_data_process():
    test_data = FashionMNIST(
        root="./data/FashionMNIST",  # 数据路径
        train=False,  # 不使用训练数据集
        transform=transforms.Compose(
            [transforms.Resize(size=224), transforms.ToTensor()]
        ),  # 把PIL.Image或者numpy.array数据类型转变为torch.FloatTensor类型
        # 尺寸为Channel * Height * Width，数值范围缩小为[0.0, 1.0]
        download=True,  # 如果前面数据已经下载，这里不再需要重复下载
    )
    test_loader = Data.DataLoader(
        dataset=test_data,  # 传入的数据集
        batch_size=1,  # 每个Batch中含有的样本数量
        shuffle=True,  # 不对数据集重新排序
        num_workers=2,  # 加载数据所开启的进程数量
    )

    # 获得一个Batch的数据
    for step, (b_x, b_y) in enumerate(test_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组
    print("The size of batch in test data:", batch_x.shape)

    return test_loader


# 定义网络的训练过程
def train_model(model, traindataloader, train_rate, criterion, device, optimizer, num_epochs=100):
    """
    :param model: 网络模型
    :param traindataloader: 训练数据集，会切分为训练集和验证集
    :param train_rate: 训练集batch_size的百分比
    :param criterion: 损失函数
    :param device: 运行设备
    :param optimizer: 优化方法
    :param num_epochs: 训练的轮数
    """

    batch_num = len(traindataloader)  # batch数量
    train_batch_num = round(batch_num * train_rate)  # 将80%的batch用于训练，round()函数四舍五入
    best_model_wts = copy.deepcopy(model.state_dict())  # 复制当前模型的参数
    # 初始化参数
    best_acc = 0.0  # 最高准确度
    train_loss_all = []  # 训练集损失函数列表
    train_acc_all = []  # 训练集准确度列表
    val_loss_all = []  # 验证集损失函数列表
    val_acc_all = []  # 验证集准确度列表
    since = time.time()  # 当前时间
    # 进行迭代训练模型
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 初始化参数
        train_loss = 0.0  # 训练集损失函数
        train_corrects = 0  # 训练集准确度
        train_num = 0  # 训练集样本数量
        val_loss = 0.0  # 验证集损失函数
        val_corrects = 0  # 验证集准确度
        val_num = 0  # 验证集样本数量
        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(traindataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            if step < train_batch_num:  # 使用数据集的80%用于训练
                model.train()  # 设置模型为训练模式，启用Batch Normalization和Dropout
                output = model(b_x)  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
                loss = criterion(output, b_y)  # 计算每一个batch的损失函数
                optimizer.zero_grad()  # 将梯度初始化为0
                loss.backward()  # 反向传播计算
                optimizer.step()  # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
                train_loss += loss.item() * b_x.size(0)  # 对损失函数进行累加
                train_corrects += torch.sum(pre_lab == b_y.data)  # 如果预测正确，则准确度train_corrects加1
                train_num += b_x.size(0)  # 当前用于训练的样本数量
            else:  # 使用数据集的20%用于验证
                model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
                output = model(b_x)  # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
                loss = criterion(output, b_y)  # 计算每一个batch中64个样本的平均损失函数
                val_loss += loss.item() * b_x.size(0)  # 将验证集中每一个batch的损失函数进行累加
                val_corrects += torch.sum(pre_lab == b_y.data)  # 如果预测正确，则准确度val_corrects加1
                val_num += b_x.size(0)  # 当前用于验证的样本数量

        # 计算并保存每一次迭代的成本函数和准确率
        train_loss_all.append(train_loss / train_num)  # 计算并保存训练集的成本函数
        train_acc_all.append(train_corrects.double().item() / train_num)  # 计算并保存训练集的准确率
        val_loss_all.append(val_loss / val_num)  # 计算并保存验证集的成本函数
        val_acc_all.append(val_corrects.double().item() / val_num)  # 计算并保存验证集的准确率
        print("{} Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} Val Loss: {:.4f} Val Acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存当前最高准确度下的模型参数
        time_use = time.time() - since  # 计算耗费时间
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    model.load_state_dict(best_model_wts)  # 加载最高准确度下的模型参数
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )  # 将每一代的损失函数和准确度保存为DataFrame格式

    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    return model, train_process


# 测试模型
def test_model(model, testdataloader, device):
    """
    :param model: 网络模型
    :param testdataloader: 测试数据集
    :param device: 运行设备
    """

    # 初始化参数
    test_corrects = 0.0
    test_num = 0
    test_acc = 0.0
    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in testdataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
            output = model(test_data_x)  # 前向传播过程，输入为测试数据集，输出为对每个样本的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            test_corrects += torch.sum(pre_lab == test_data_y.data)  # 如果预测正确，则准确度val_corrects加1
            test_num += test_data_x.size(0)  # 当前用于训练的样本数量

    test_acc = test_corrects.double().item() / test_num  # 计算在测试集上的分类准确率
    print("test accuracy:", test_acc)


# 模型的训练和测试
def train_model_process(myconvnet):
    optimizer = torch.optim.Adam(myconvnet.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU加速
    train_loader, class_label = train_data_process()  # 加载训练集
    test_loader = test_data_process()  # 加载测试集

    myconvnet = myconvnet.to(device)
    myconvnet, train_process = train_model(
        myconvnet, train_loader, 0.8, criterion, device, optimizer, num_epochs=100
    )  # 开始训练模型
    test_model(myconvnet, test_loader, device)  # 使用测试集进行评估


if __name__ == "__main__":
    model = resnet50()
    train_model_process(model)

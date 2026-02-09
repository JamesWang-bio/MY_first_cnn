import torch
import torch.nn as nn
import torch.nn.functional as F
import data_import

'''===================================Chapter 1: 训练数据======================================================'''
'''从data_importer里面导入图片矩阵的信息，输出的信息，'''

img_tr1 = data_import.img_train_s
'''训练材料，图片矩阵'''
label_test1 = data_import.label_train
'''训练材料，图片标签'''
train_num = 10000
#data_mian = zip(img_tr, label_test)
img_tr = img_tr1[:train_num]
label_train = label_test1[:train_num]

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(img_tr, label_train)


# 输入的图片统一为28*28像素

'''==================================Chapter 2: 前向传播函数========================================================='''
'''关键函数1： 前向传播方程'''
'''Conv(卷积核模块)
    说明书：
    conv = nn.Conv2d(
    in_channels=3,      # 输入通道数
    out_channels=64,    # 输出通道数（卷积核数量）
    kernel_size=3,      # 卷积核尺寸
    stride=1,          # 步长
    padding=1,         # 填充
    dilation=1,        # 空洞率
    groups=1,          # 分组卷积
    bias=True,         # 是否包含偏置
    padding_mode='zeros' # 填充模式
)'''


class my_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.core_num = 1
        self.core_siz_1 = 3
        self.interlink_num = 64

        # 使用最简单的设置
        self.cov1 = nn.Conv2d(1, self.core_num, kernel_size=3, stride=1, padding=1)

        # 直接设置权重
        with torch.no_grad():
            self.cov1.weight.data = torch.tensor(
                [[[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]]], dtype=torch.float32)
            self.cov1.weight.data = self.cov1.weight.data.unsqueeze(0)
            self.cov1.bias.data.zero_()

        flattened_features = 28 * 28 * self.core_num
        self.fc1 = nn.Linear(flattened_features, self.interlink_num)
        self.fc2 = nn.Linear(self.interlink_num, 10)

    def forward(self, info):
        # 最简单的维度处理
        if info.dim() == 3:
            info = info.unsqueeze(1)

        info = F.relu(self.cov1(info.float()))
        info = info.view(info.size(0), -1)
        info = F.dropout(F.relu(self.fc1(info)), p=0.01, training=self.training)
        info = self.fc2(info)

        return info


'''info :# 形状: torch.Size([64, 10])
   # 内容示例:
tensor([[ 0.1234, -0.5678,  0.9012, ...,  0.3456],  # 第1个样本的10个类别分数
        [-0.2345,  0.6789, -0.1234, ...,  0.4567],  # 第2个样本的10个类别分数
        ...,
        [ 0.7890, -0.0123,  0.5678, ..., -0.2345]], # 第64个样本的10个类别分数
       device='cuda:0' or 'cpu')'''

forward_model = my_module()
'''压到一个函数名里面方便用'''

error_er = nn.CrossEntropyLoss()
'''传播的lost（误差值）'''

optimizer = torch.optim.Adam(forward_model.parameters(), lr=0.001)

'''optimizer = torch.optim.Adam(
    model.parameters(),  # 要优化的参数：所有卷积层和全连接层的权重
    lr=1e-3              # 学习率：每次参数更新的步长大小
)'''

'''===================================Chapter 3 训练主程序，最终的训练参数存储在 forward_model中================================================='''


def train():
    forward_model.train()

    correct_num = 0
    total_cost = 0.0

    for imgs, labels in train_dataset:
        # 确保数据是张量
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.tensor(imgs, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        if labels.dim() == 0:  # 标量（0维）
            labels = labels.unsqueeze(0)

        # 1. 清空之前的梯度
        optimizer.zero_grad()  # 准备工作

        # 2. 前向传播
        logits = forward_model(imgs)  # 计算预测

        # 3. 计算损失
        loss = error_er(logits, labels)  # 计算误差
        total_cost += loss.item()

        # 4. 反向传播（计算梯度）
        loss.backward()  # 计算每个参数的梯度，存在参数的 .grad 属性中

        # 5. 更新参数（关键步骤！）
        optimizer.step()  # ← 这里！根据梯度更新所有权重和偏置
        '''梯度决定参数的变化方向'''

    return total_cost


'''=================Chapter 4 训练的最终章，总控程序 ================================================'''
# 修改训练脚本 main_control.py 中的保存部分
def save_model(model, filepath):
    """保存模型的状态字典"""
    train()
    torch.save(model.state_dict(), filepath)  # 只保存参数，不保存类定义
    print(f"模型已保存到: {filepath}")

if __name__ == "__main__":
    save_model(forward_model, "D:/cnn_第一神经网络.pth")


from tensorflow.keras.datasets import mnist
import torch
import numpy as np

# 加载MNIST数据集
(img_train, label_train), (img_test, label_test) = mnist.load_data()
'''图像数据：x_train, x_test 是 uint8 格式的 3D 矩阵
形状：(60000, 28, 28) 和 (10000, 28, 28)，值范围 0~255（灰度像素值）

标签数据：y_train, y_test 是 uint8 格式的 1D 矩阵
形状：(60000,) 和 (10000,)，值范围 0~9（数字类别）'''

# 归一化到0-1范围
img_test_s = img_test.astype('float32') / 255.0
img_train_s = img_train.astype('float32') / 255.0

'''归一化，梯度下降减少计算量'''

# 转换为PyTorch张量
if not isinstance(img_train_s, torch.Tensor):
    img_train_s = torch.tensor(img_train_s, dtype=torch.float32)
if not isinstance(label_train, torch.Tensor):
    label_train = torch.tensor(label_train, dtype=torch.long)

# 检查并调整维度
print(f"原始img_train_s形状: {img_train_s.shape}")  # 应该是 (60000, 28, 28)

# 确保是4D张量 [batch_size, channels, height, width]
if img_train_s.dim() == 3:
    # 对于MNIST，添加通道维度（灰度图：1个通道）
    img_train_s = img_train_s.unsqueeze(1)  # 从 (60000, 28, 28) 变为 (60000, 1, 28, 28)

print(f"调整后img_train_s形状: {img_train_s.shape}")

# 对测试数据做同样的处理
if not isinstance(img_test_s, torch.Tensor):
    img_test_s = torch.tensor(img_test_s, dtype=torch.float32)
if not isinstance(label_test, torch.Tensor):
    label_test = torch.tensor(label_test, dtype=torch.long)

if img_test_s.dim() == 3:
    img_test_s = img_test_s.unsqueeze(1)  # 从 (10000, 28, 28) 变为 (10000, 1, 28, 28)

print(f"img_test_s形状: {img_test_s.shape}")
print(f"label_train形状: {label_train.shape}")
print(f"数据类型检查 - img_train_s: {img_train_s.dtype}, label_train: {label_train.dtype}")


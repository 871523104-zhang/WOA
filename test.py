import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from 生成bp import BPNet

# 使用此文件看模型的好坏


# 实例化模型
net = BPNet()

# 加载数据集
testRatio = 0.2
X = pd.read_excel('features.xlsx').values
Y = pd.read_excel('labels.xlsx').values

# 获取x和y的数据特征（最大最小值，均值等）作为归一化依据
# x和y分别有不同的归一化特征
scalerx = MinMaxScaler().fit(X)
scalery = MinMaxScaler().fit(Y)
x_m = scalerx.transform(X)
y_m = scalery.transform(Y)
# train和test占比
# 导出拟合图像时：这里由于数据基数大，为得出图像容易看，所以将比例设定很小
# 导出数据统计图像时：可适当增加test比例使其为8:2
x_train, x_test, y_train, y_test = train_test_split(x_m, y_m, test_size=testRatio)

# 读取模型
net.load_state_dict(torch.load('F:\yuri_nn\WOA\WOAUse.pth'))

# 评估模型
net.eval()
with torch.no_grad():
    # 输入并得出预测结果
    test_input = torch.from_numpy(x_test).float()
    test_output = net(test_input)
    # 将预测结果y值和对比y值反归一化至正常
    y_predict = scalery.inverse_transform(test_output)
    y_test = scalery.inverse_transform(y_test)
    loss_fn = nn.MSELoss()
    loss = loss_fn(torch.from_numpy(y_predict).float(), torch.from_numpy(y_test).float())
    print(f"最终测试损失: {loss.item()}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
# -------------------------------# 数据加载与预处理# -------------------------------
# 假设你已经有特征数据X和标签数据y
X = pd.read_excel('features.xlsx').values
y = pd.read_excel('labels.xlsx').values
# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
# -------------------------------# WOA算法实现# -------------------------------
class WhaleOptimization:
    def __init__(self, num_whales, max_iter, dim):
        self.num_whales = num_whales
        self.max_iter = max_iter
        self.dim = dim  # 维度：特征选择（64） + 超参数（例如3个）
        self.whales = np.random.rand(self.num_whales, self.dim)
        self.alpha_score = float('inf')
        self.alpha_pos = np.zeros(self.dim)

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.num_whales):
                # 解码鲸鱼的位置，得到特征选择和超参数
                features, params = self.decode(self.whales[i])
                # 评估适应度
                fitness = self.fitness_function(features, params)
                # 更新最佳个体
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.whales[i].copy()
            # 更新鲸鱼位置
            self.update_whales(t)
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.alpha_score}")
        # 返回最佳解
        return self.alpha_pos

    def decode(self, whale):
        # 前64维为特征选择，后面为超参数
        feature_selection = whale[:64] > 0.5  # 二进制特征选择
        params = whale[64:]
        # 解码超参数，例如卷积核大小、学习率等
        conv_units = int(params[0] * 100) + 10  # 取值范围10-110
        learning_rate = params[1] * 0.009 + 0.001  # 取值范围0.001-0.01
        batch_size = int(params[2] * 90) + 10     # 取值范围10-100
        return feature_selection, [conv_units, learning_rate, batch_size]

    def fitness_function(self, features, params):
        # 根据特征选择子集和超参数构建并训练CNN模型
        selected_features = features.nonzero()[0]
        if len(selected_features) == 0:
            return float('inf')  # 如果没有选择任何特征，适应度设为无穷大
        X_sub = X_train[:, selected_features]
        X_val = X_test[:, selected_features]

        model = build_cnn_model(input_dim=X_sub.shape[1], params=params)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params[1])

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(X_sub, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params[2], shuffle=True)

        # 训练模型（为了节省时间，这里只训练1个epoch，可以根据需要增加）
        model.train()
        for epoch in range(1):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 模型评估
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            acc = accuracy_score(y_test, predicted.numpy())

        # 以1 - 准确率作为适应度（越小越好），并加入特征数量的惩罚项
        fitness = 1 - acc + 0.01 * (len(selected_features) / 64)
        return fitness

    def update_whales(self, t):
        a = 2 - t * (2 / self.max_iter)
        for i in range(self.num_whales):
            r = random.random()
            A = 2 * a * r - a
            C = 2 * r
            p = random.random()
            if p < 0.5:
                D = abs(C * self.alpha_pos - self.whales[i])
                self.whales[i] = self.alpha_pos - A * D
            else:
                distance_to_alpha = abs(self.alpha_pos - self.whales[i])
                self.whales[i] = distance_to_alpha * np.exp(-1 * t) * np.cos(2 * np.pi * t) + self.alpha_pos
            # 边界检查
            self.whales[i] = np.clip(self.whales[i], 0, 1)
# -------------------------------# CNN模型定义# -------------------------------
class CNNModel(nn.Module):
    def __init__(self, input_dim, params):
        super(CNNModel, self).__init__()
        conv_units = params[0]
        self.fc1 = nn.Linear(input_dim, conv_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(conv_units, int(conv_units/2))
        self.fc3 = nn.Linear(int(conv_units/2), 2)  # 二分类问题，输出2个类

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
def build_cnn_model(input_dim, params):
    model = CNNModel(input_dim, params)
    return model
# -------------------------------# BP模型定义# -------------------------------
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
def build_bp_model(input)
# -------------------------------# 主函数执行WOA优化# -------------------------------
if __name__ == "__main__":
    num_whales = 5       # 鲸鱼数量
    max_iter = 10        # 最大迭代次数
    dim = 67             # 64个特征选择 + 3个超参数

    woa = WhaleOptimization(num_whales, max_iter, dim)
    best_solution = woa.optimize()

    # 解码最佳解
    best_features, best_params = woa.decode(best_solution)
    print("Best Features Selected:", np.where(best_features)[0])
    print("Best Hyperparameters:", best_params)

    # 用最佳解训练最终模型
    selected_features = best_features.nonzero()[0]
    X_best = X_train[:, selected_features]
    X_val_best = X_test[:, selected_features]

    final_model = build_cnn_model(input_dim=X_best.shape[1], params=best_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best_params[1])

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_best, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=best_params[2], shuffle=True)

    # 训练模型
    final_model.train()
    for epoch in range(5):  # 可以增加epoch数量以提高性能
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item()}")

    # 模型评估
    final_model.eval()
    with torch.no_grad():
        outputs = final_model(X_val_best)
        _, predicted = torch.max(outputs.data, 1)
        final_acc = accuracy_score(y_test, predicted.numpy())
    print("Final Model Accuracy:", final_acc)

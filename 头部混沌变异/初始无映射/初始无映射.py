import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from openpyxl import Workbook
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# 使用maxabsscaler区别不大，失去缩放的目的
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def tent_map(x, a):
    if x < a:
        return x / a
    else:
        return (1 - x) / (1 - a)
    
def initializePopulationWithTent(dim, pop_size, lb, ub, a):
    population = np.zeros((pop_size, dim))
    for i in range(pop_size):
        # 初始化
        x = np.random.uniform(0, 1, dim)
        for j in range(dim):
            x[j] = tent_map(x[j], a)
            x[j] = lb + x[j] * (ub - lb)# 映射到搜索空间
        population[i,:] = x
    return population


# 定义BP神经网络
class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(5, 23)  # 5个输入，10个隐藏神经元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(23, 1)  # 1个输出

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 提取神经网络的参数为一维向量
def get_network_params(net):
    params = []
    for param in net.parameters():
        params.extend(param.data.cpu().numpy().flatten())
    return np.array(params)

# 将一维向量的参数加载回神经网络
def set_network_params(net, params):
    pointer = 0
    for param in net.parameters():
        num_params = param.numel()
        new_param = params[pointer:pointer + num_params].reshape(param.size())
        param.data = torch.from_numpy(new_param).float()
        pointer += num_params

# 定义适应度函数（均方误差）
def fitness_function(net, X, Y):
    outputs = net(torch.from_numpy(X).float())
    loss = nn.MSELoss()
    return loss(outputs, torch.from_numpy(Y).float()).item()

# 鲸鱼优化算法
def WOA(net, X, Y, pop_size, max_iter):
    dim = len(get_network_params(net))  # 参数维度
    lb = -1  # 下界
    ub = 1   # 上界

    # 初始化鲸鱼群体
    a = 0.5
    population = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.zeros(pop_size)

    # 评估初始群体的适应度
    for i in range(pop_size):
        set_network_params(net, population[i])
        fitness[i] = fitness_function(net, X, Y)

    # 找到当前最优解
    best_idx = np.argmin(fitness)
    best_whale = population[best_idx].copy()
    best_score = fitness[best_idx]

    bestScoreHistory = []
    # 主循环
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)  # 线性递减参数

        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2

            p = np.random.rand()
            b = 1  # 常数，用于螺旋更新
            l = np.random.uniform(-1, 1)

            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_whale - population[i])
                    population[i] = best_whale - A * D
                else:
                    rand_idx = np.random.randint(0, pop_size)
                    rand_whale = population[rand_idx]
                    D = abs(C * rand_whale - population[i])
                    population[i] = rand_whale - A * D
            else:
                D = abs(best_whale - population[i])
                population[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_whale

            # 边界检查
            population[i] = np.clip(population[i], lb, ub)

            # 更新适应度
            set_network_params(net, population[i])
            fitness[i] = fitness_function(net, X, Y)

        # 更新全局最优解
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_score:
            best_score = fitness[current_best_idx]
            best_whale = population[current_best_idx].copy()

        # print(f"迭代 {t+1}/{max_iter}, 最佳适应度: {best_score}")
        bestScoreHistory.append(best_score)

    # 返回最优解
    return best_whale, bestScoreHistory

# 主函数
if __name__ == "__main__":   
    # 保存模型路径
    testRatio = 0.2
    popSize = 30
    maxIter = 100
    
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
    train_input = torch.from_numpy(x_train).float()
    test_input = torch.from_numpy(x_test).float()
    y_train_actual = torch.from_numpy(y_train).float()

    bestScoreTotal = []
    testLossTotal = []
    rmseTotal = []
    maeTotal = []
    wb = Workbook()
    # 30次WOA优化BP初始参数
    for i in range(30):
        BPLossTotal = []
        # 初始化网络
        net = BPNet()
        # 使用鲸鱼优化算法优化网络参数
        # max_iter：运行轮数
        # pop_size：优化过程中考虑的候选解的数量
        # best_params：性能最好的神经网络参数
        # bestScoreHistory：记录了每次迭代中全局最优解的适应度，可以用来分析算法的性能和收敛情况
        best_params, bestScoreHistory = WOA(net, x_train, y_train, pop_size=popSize, max_iter=maxIter)
        iterations = range(1, len(bestScoreHistory)+1)
        lastScore = bestScoreHistory[-1]
        bestScoreTotal.append(lastScore)
        # plt.figure(figsize=(10,5))
        # 绘制bestscorehistory
        # print(best_params)
        # plt.plot(iterations, bestScoreHistory)
        # plt.title('BsetScoreHistory')
        # plt.xlabel('iteration')
        # plt.ylabel('Best Score')
        # plt.grid(True)
        # plt.legend(['best score'])
        # plt.show()
        
        # 将最优参数设置到网络中
        set_network_params(net, best_params)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.1)
        
        # 训练模型
        num_epochs = 150  # 假设训练10个epoch
        for epoch in range(num_epochs):
            # 前向传播
            BPoutputs = net(train_input)
            BPloss = criterion(BPoutputs, y_train_actual)
            # 反向传播和优化
            optimizer.zero_grad()
            BPloss.backward()
            optimizer.step()
            # 打印统计信息
            # print(f'Epoch {epoch + 1}, Loss: {BPloss.item()}')
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {BPloss.item()}')
            BPLossTotal.append(BPloss.item())
        # print(BPLossTotal)
        # plt.figure()
        # plt.plot(BPLossTotal, label='fitness')
        # plt.show()

        
        path = f'头部混沌变异\\初始无映射\\WOAInit{i}.pth'
        # 保存模型权重
        torch.save(net.state_dict(), path)

        # 测试网络
        with torch.no_grad():
            test_output = net(test_input)
            y_predict = scalery.inverse_transform(test_output)
            y_test = scalery.inverse_transform(y_test)
            
            rmse_value = rmse(y_test, y_predict)
            mae_value = mae(y_test, y_predict)
            criterion = nn.MSELoss()
            loss = criterion(torch.from_numpy(y_predict).float(), torch.from_numpy(y_test).float())
            print(f"最终测试损失: {loss.item()},最终测试rmse：{rmse_value:.4f}，最终测试mae：{mae_value:.4f}")
            testLossTotal.append(loss.item())
            rmseTotal.append(rmse_value)
            maeTotal.append(mae_value)
            
        print(f'*****第{i+1}次WOA优化训练完成*****')
        
    # 将total结果存储到excel中
    total = [bestScoreTotal, testLossTotal, rmseTotal, maeTotal]
    df = pd.DataFrame(total, index=['bestscore', 'Loss', 'rmse', 'mae'])
    df.to_excel('头部混沌变异\\初始无映射\\total30-150.xlsx')
    #     # 最佳适应度
    #     plt.figure()
    #     plt.plot(bestScoreHistory, label='fitness')
    #     plt.show()
    #     # 折线图
    #     plt.plot(y_test, label='actual')
    #     plt.plot(y_predict, label='predict')
    #     plt.legend()
    #     plt.show()
    #     # 散点图
    #     plt.scatter(y_test, y_predict)
    #     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
    #     plt.xlabel('actual')
    #     plt.ylabel('predict')
    #     plt.show()
        
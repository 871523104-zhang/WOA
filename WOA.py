import numpy as np
import random
import math
import copy

def initialization(pop,ub,lb,dim):
    '''
    种群初始化函数
    
    pop:种群数量
    ub:每个维度的变量上边界,维度为[dim,l]
    lb:每个维度的变量下边界,维度为[dim,l]
    dim:每个个体的维度
    
    x:输出的种群,维度为[pop,dim]
    '''
    # 初始化种群的决策变量
    x = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            # 生成ub和lb之间的随机值
            x[i,j]=(ub[j]-lb[j])*np.random.random()+lb[j]
    # x：初始positions
    return x

def BorderCheck(x, ub, lb, pop, dim):
    '''
    边界检查函数:将每个维度变量限制在上下边界中
    
    x:输入数据，维度为[pop,dim]
    '''
    for i in range(pop):
        for j in range(dim):
            if x[i, j] > ub[j]:
                x[i, j] = ub[j]
            elif x[i, j] < lb[j]:
                x[i, j] = lb[j]
    return x

def CaculateFitness(x, fun):
    '''
    计算种群所有个体适应度的值
    
    fun:计算种群个体适应度的函数###########################################################
    '''
    pop = x.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(x[i, :])
    return fitness

def SortFitness(Fit):
    '''
    适应度排序
    
    Fit:适应度值(numpy数组)
    
    fitness:排序后适应度值
    index:排序后适应度索引
    '''
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index

def SortPosition(x, index):
    '''
    根据适应度值对位置进行排序:根据index重新排列x的行,并输出排列后的xnew
    '''
    xnew = np.zeros(x.shape)
    for i in range(x.shape[0]):
        xnew[i, :] = x[index[i], :]
    return xnew

def WOA(pop, inputnum, hiddennum, outputnum, lb, ub, MaxIter, fun):
    '''
    鲸鱼优化算法
    
    pop:种群数量
    ub:每个维度的变量上边界,维度为[dim,l]
    lb:每个维度的变量下边界,维度为[dim,l]
    dim:每个个体的维度（自变量个数）
    MaxIter:最大迭代次数
    fun:适应度函数接口(计算种群个体适应度)
    
    GbestScore:最优解对应的适应度值
    GbestPosition:最优解
    Curve:迭代曲线
    '''
    
    dim = inputnum * hiddennum + hiddennum +hiddennum*outputnum + outputnum
    
    x = initialization(pop, ub, lb, dim)
    fitness = CaculateFitness(x, fun)
    fitness, sortIndex = SortFitness(fitness)
    x = SortPosition(x, sortIndex)
    GbestScore = copy.copy(fitness[0])# 记录最优适应度值
    GbestPosition = np.zeros([1, dim])
    GbestPosition[0, :] = copy.copy(x[0, :]) #记录最优解
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        print(f'第{t}次迭代')
        Leader = x[0, :] #领头鲸
        a = 2 - t * (2 / MaxIter) #线性下降权重：随迭代次数增加线性减小的权重。探索解空间→开发解空间
                                  #变异率/交叉率：控制种群多样性和搜索能力
        for i in range(pop):
            r1 = random.random()
            r2 = random.random()
            
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = 2 * random.random() - 1 #-1和1之间的随机数
            
            for j in range(dim):
                p = random.random()
                if p < 0.5:
                    if np.abs(A) >= 1:#寻找猎物
                        #随机选择一个个体
                        rand_leader_index = min(int(np.floor(pop * random.random() + 1)), pop - 1)
                        x_rand = x[rand_leader_index, :]
                        D_x_rand = np.abs(C * x_rand[j] - x[i, j])
                        x[i, j] = x_rand[j] - A * D_x_rand
                    elif np.abs(A) < 1:# 包围猎物
                        D_Leader = np.abs(C * Leader[j] - x[i, j])
                        x[i, j] = Leader[j] - A * D_Leader
                elif p >= 0.5:#气泡网攻击
                    distance2Leader = np.abs(Leader[j] - x[i, j])
                    x[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 *math.pi) + Leader[j]
        
        x = BorderCheck(x, ub, lb, pop, dim)
        fitness = CaculateFitness(x, fun)
        fitness, sortIndex = SortFitness(fitness)
        x = SortPosition(x, sortIndex)
        # 更新全局最优
        if fitness[0] <= GbestScore:
            GbestScore = copy.copy(fitness[0])
            GbestPosition[0, :] = copy.copy(x[0, :])
        Curve[t] = GbestScore
        
    return GbestScore, GbestPosition, Curve
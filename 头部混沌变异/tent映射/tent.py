import matplotlib.pyplot as plt

def tent_map(x, a):
    if x < a:
        return x / a
    else:
        return (1 - x) / (1 - a)

# # 设置参数
# a = 0.7
# x0 = 0.2
# n = 1000

# # 生成混沌序列
# x = [x0]
# for i in range(n):
#     x.append(tent_map(x[-1], a))

# # 绘制混沌图像
# plt.plot(x)
# plt.title("Tent Map with a = {}".format(a))
# plt.xlabel("Iteration")
# plt.ylabel("Value")
# plt.show()
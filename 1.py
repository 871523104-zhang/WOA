import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = pd.read_excel('features.xlsx').values
y = pd.read_excel('labels.xlsx').values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

scaler = MinMaxScaler()
scaler.fit(y)
y_m = scaler.transform(y)

numpy_predict = scaler.inverse_transform(y_m)

print(f'初始X类型是{type(X)}')
print(f'初始y类型是{type(y)}')
print(f'初始X是{X}')
print(f'初始y是{y}')
print(f'归一化后y是{y_m}')
print(f'反归一化后y是{numpy_predict}')
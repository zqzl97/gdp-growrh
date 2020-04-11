# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from numpy import sqrt
from math import sqrt
import matplotlib.pyplot as plt



# header: 指定第几行作为列名(忽略注解行)，如果没有指定列名，默认header=0
dataset = pd.read_csv('mac_economic4.csv', header=0, index_col=0)
df = pd.DataFrame(dataset)
print(type(dataset))
print(dataset.shape[1])
n_vars = 1 if type(dataset) is list else dataset.shape[1]
  
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence(t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1)) for j in range(n_vars)]
    # put it all together (这是横着拼的)
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load dataset
dataset = pd.read_csv('mac_economic4.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[7, 8, 9, 10, 11]], axis=1, inplace=True)


# .values把每一行的数据变成了一个list列表[]
values = reframed.values
print(values)
n_train = 20
train = values[0:n_train, :]
test = values[n_train:, :]
# split into input and outputs
# [:, :-1]代表除最后一列的其他列
train_X = train[:, :-1]
train_y = train[:, -1]
test_X = test[:, :-1]
test_y = test[:, -1]

# reshape input to be 3D[samples, timesteps, features]
# 这里面reshape的只有训练集和测试集的X，没有y
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# 这里面的y只有一列，y的shape值像是行值
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
# 隐藏层有50个神经元，input_shape 选择的是后两维
model.add(LSTM(50, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(25, activation='relu'))
# 输出层一个神经元
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
# verbose:日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
history = model.fit(train_X, train_y, epochs=40, batch_size=20, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
# 利用reshape后的test_x进行预测
yhat = model.predict(test_X)
# 预测之后，test_X回到原来2D数据
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
# 为什么要把y和x(除第一列联系起来)
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
# 这里只取了第一列
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
# 把y还原到原来的数据结构
test_y = test_y.reshape(len(test_y), 1)
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(sum(((inv_y - inv_yhat) ** 2) / len(inv_y)))
print('Test RMSE: %.3f' % rmse)



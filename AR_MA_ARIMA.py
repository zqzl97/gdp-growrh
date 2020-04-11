# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设定画布大小
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')


# 设定画布大小
rcParams['figure.figsize'] = 15, 6
# data = pd.read_csv('GDP_growth.csv')
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y')
data = pd.read_csv('GDP_growth.csv', parse_dates=[0], header=[0], index_col=[0], date_parser=dateparse)


def test_stationarity(timeseries):
    # 这里以一年为一个窗口，每一个时间t的值由它前面12个月(包括自己)的均值替代，标准差同理
    # 均值
    rolmean = timeseries.rolling(window=12).mean()
    # rolmean = pd.rolling_mean(timeseries, window=12) 过时了
    rolstd = timeseries.rolling(window=12).std()
    # rolstd = pd.rolling_std(timeseries, window=12) 过时了
    # plot rolling statistics
    # 画子图
    fig = plt.figure()
    # fig.add_subplot(111) 参数1：子图总行数 参数2：子图总列数 参数3：子图位置
    fig.add_subplot()
    orig = plt.plot(timeseries, color='blue', label='Original')  # 原始数据
    mean = plt.plot(rolmean, color='red', label='rolling mean')  # 均值
    std = plt.plot(rolstd, color='black', label='Rolling standard deviation') # 标准差

    plt.legend(loc='best')  # 图例
    plt.title('Rolling Mean and Standard deviation')  # 标题
    plt.show(block=False)

    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    # dftest的输出前一项依次为检测值，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print(dfoutput)
    
ts = data['GDP growth (%)']
# 由于原数据值域范围比较大，为了缩小值域，同时保留其他信息，常用的方法是对数化，取log
ts_log = np.log(ts)
'''
# 平滑：以一个滑动窗口内的均值代替原来的值，为了使值之间的差距缩小
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log, color='blue')
plt.plot(moving_avg, color='red')
plt.show()

# 然后作差，取对数的减去平滑后的
ts_log_avg_diff = ts_log - moving_avg
ts_log_avg_diff.dropna(inplace=True)
# 红色为差值的均值，黑色为差值的标准差
test_stationarity(ts_log_avg_diff)
# 可以看到，做了处理之后的数据基本上没有了随时间变化的趋势，DFtest的结果告诉我们在95%的置信度下，数据是稳定的

# 上面的方法是将所有的时间平等看待，而在许多情况下，可以认为越近的时刻越重要
# 所以引入指数加权移动平均--Exponentially-weighted moving average.(pandas中通过evma()函数提供了此功能)
# halflife的值决定了衰减因子alpha: alpha = 1- exp(log(0.5)/halflife)，halflife:半衰期
expweighted_avg = ts_log.ewm(halflife=12).mean()
# print('****', expweighted_avg.dtype)
expweighted_avg = pd.ewma(ts_log, halflife=12)
ts_log_ewma_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff)

# 可以看到相比普通的Moving Average,新的数据平均标准差更小了。而且DFtest可以得出结论:数据在99%的置信度上是稳定的。(不知这99%的置信度是怎么计算的)


# 检测和去除季节性，有两种方法
# 1 差分化:以特定滞后数目的时刻的值作差。2 分解：对趋势和季节性分别建模再移除它们

# Differencing--差分
ts_log_diff = ts_log - ts_log.shift()
# 删掉有缺失的数据
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# 下面这一步ts_log_diff = ts_log.diff(1)和上面ts_log_diff = ts_log - ts_log.shift()得出的结果是一样的
ts_log_diff = ts_log.diff(1)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
plt.show()
# 从图中 可以看出相比MA方法，Differencing方法处理数据的均值和方差在时间轴上的振幅明显缩小了。DFtest的结论是在90%的置信度下，数据是稳定的
# 上图看出一阶差分大致已经具有周期性，不妨绘制二阶差分对比：
# diff()函数是求导数和差分的
ts_log_diff1 = ts_log.diff(1)
ts_log_diff2 = ts_log.diff(2)
ts_log_diff1.plot()
ts_log_diff2.plot()
plt.show()

ts_log_diff2 = ts_log.diff(2)
ts_log_diff2.dropna(inplace=True)
test_stationarity(ts_log_diff2)
plt.show()

# 3.Decomposing分解
# 分解可以用来把时序数据中的趋势和周期性数据都分离出来：
def decompose(timeseries):
    # 返回包括3个部分trend(趋势部分)，seasonal(季节性部分)，residual(残留部分)
    decomposition = seasonal_decompose(timeseries)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return trend, seasonal, residual

trend, seasonal, residual = decompose(ts_log)

# 如何可以明显的看到，将original数据拆成了三份。trend数据具有明显的趋势性
# Seasonality数据具有明显的周期性，Residuals是剩余的部分，可以认为数去除了趋势和季节性数据之后，稳定的数据，是我们所需要的
# 消除了trend和seasonal之后，只对residual部分作为想要的时序数据进行处理
residual.dropna(inplace=True)
test_stationarity(residual)
# 如图所示，数据的均值和方差趋于常数，几乎无波动(看上去比之前的陡峭，但是要注意它的值域只有[-0.05,0.05]之间)，
# 所以直观上可以认为是稳定的数据。另外DFtest的结果显示，Statistic值远小于1%时的Critical value,所以在99%的置信度下，数据是稳定的

# 6.对时序数据进行
# 假设经过处理，已经得到了稳定时序数据。接下来，我们使用ARIMA模型
# step1:通过ACF，PACF进行ARIMA(p, d, q)的p, q参数估计
# 由前文Differencing差分部分已知，一阶差分后数据已经稳定，所以d=1
# 所以用一阶差分化的ts_log_diff = ts_log - ts_log.shift()作为输入
# 等价于yt = Yt - Yt-1作为输入
# 先画出ACF,PACF的图像，代码如下：
# ACF and PACF plots
ts_log_diff = ts_log - ts_log.shift()
# dropna这一步真的很重要，没有dropna下面的图形会出错
ts_log_diff.dropna(inplace=True)
'''
ts_log_diff = ts_log - ts_log.shift()
# dropna这一步真的很重要，没有dropna下面的图形会出错
ts_log_diff.dropna(inplace=True)

lag_acf = acf(ts_log_diff, nlags=20)  # nlags表示要计算ACF的滞后数
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')  # ols 最小二乘法

# plot acf
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # 置信区间
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # 置信区间
plt.title('Autocorrelation Function')

# plot pacf
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # 置信区间
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')  # 置信区间
plt.title('Partial Autocorrelation Function')
# tight_layout会自动调整子图参数，使之填充整个图像区域。
plt.tight_layout()
plt.show()
# 图中，上下两条灰线之间是置信区间，p的值就是ACF第一次穿过上置信区间时的横轴值。
# q的值就是PACF第一次穿过上置信区间的横轴值。所以从图中可以得到p=2,q=2。


# step2: 得到参数估计值p, d, q之后，生成模型ARIMA(p, d, q)p=2 d=1 q=2
# 为了突出差别，用三种参数取值的三个模型作为对比
# 模型1：AR模型（ARIMA(2, 1, 0))from statsmodels.tsa.arima_model import ARIMA
# 模型1：AR模型(ARIMA(2, 1, 0))
model = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff, label='ts_log_diff')
plt.plot(results_AR.fittedvalues, color='red', label='fitted-values')
plt.legend(loc='best')
plt.title('RSS:%.4f' % sum((results_AR.fittedvalues - ts_log_diff)**2))
plt.show()


# 模型2：MA模型(ARIMA(0, 1, 2))
model = ARIMA(ts_log, order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff, label='ts_log_diff')
plt.plot(results_MA.fittedvalues, color='red', label='fitted-values')
plt.legend(loc='best')
plt.title('RSS:%.4f' % sum((results_MA.fittedvalues - ts_log_diff)**2))
plt.show()

# 模型3：ARIMA模型(ARIMA(2, 1, 2))
model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff, label='ts_log_diff')
plt.plot(results_ARIMA.fittedvalues, color='red', label='fitted-values')
plt.legend(loc='best')
plt.title('RSS:%.4f' % sum((results_ARIMA.fittedvalues - ts_log_diff)**2))
plt.show()

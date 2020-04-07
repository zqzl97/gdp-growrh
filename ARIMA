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
ts = data['GDP growth (%)']
ts_log = np.log(ts)
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




import statsmodels.api as sm
sm.tsa.arma_order_select_ic(ts, max_ar=6, max_ma=4, ic='aic')['aic_min_order']  # AIC
sm.tsa.arma_order_select_ic(ts, max_ar=6, max_ma=4, ic='bic')['bic_min_order']  # BIC
sm.tsa.arma_order_select_ic(ts, max_ar=6, max_ma=4, ic='hqic')['hqic_min_order']  # HQIC

from statsmodels.tsa.arima_model import ARIMA
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

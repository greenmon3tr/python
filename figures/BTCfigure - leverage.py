"""
the code is used for plotting a figure on returns of strategies with different leverages
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(__file__))

# 导入数据
df = pd.read_csv("BTCUSDT(30d).csv", parse_dates=['Date'], index_col='Date')#.tail(10000)
print('imported!')

# 格式转化
df = df[pd.to_datetime(df.index, errors='coerce').notna()]
df.index = pd.to_datetime(df.index)
df = df.asfreq('s')
df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce').dropna()

# 可调节参数
mem = 25 # 用户的峰值记忆力
srt = 2 # 用户的短期感知
buy_ratio = 0.3
threshold = 1e-9 # 阈值
bound = 12e-9 # 上限
fee = 0.00005 # 手续费
delay = 1 # 延迟
leverage1 = 1 
leverage2 = 5 
leverage3 = 20 
leverage4 = 100 
initial_cash = 1000000  # 初始资金（可以调整）

# 计算
df['bargain_price'] = df['Close'].shift(-delay)
df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1)).dropna().asfreq('s')
df['ret_std'] = df['log_returns'].rolling(window=30).std()
df['log_ret_prev'] = df['log_returns'].shift(1)
df['ret_mean'] = df['log_ret_prev'].rolling(window=srt).mean()
df['peak'] = df['ret_mean'].rolling(window=mem).max()
df['valley'] = df['ret_mean'].rolling(window=mem).min()
df['_score'] = np.where(df['log_ret_prev'] < 0,
                       df['valley'] * abs(df['log_ret_prev']*3.15),
                       df['peak'] * abs(df['log_ret_prev']))

df['main_scr'] = df['_score'].rolling(window=10).apply(lambda x: x.loc[x.abs().idxmax()], raw=False) / 3
df['score'] = np.where(np.sign(df['main_scr'] + df['_score']) != np.sign(df['_score']), 0, df['_score'])

# 对齐数据
df = df[['score', '_score', 'bargain_price', 'ret_std']].dropna()
print('cleaned!')

# 初始化策略变量
position1 = position2 = position3 = position4 = 0  # 初始仓位（0代表不持有）
equity1 = initial_cash
cash1 = initial_cash * leverage1
equity2 = initial_cash
cash2 = initial_cash * leverage2
equity3 = initial_cash
cash3 = initial_cash * leverage3
equity4 = initial_cash
cash4 = initial_cash * leverage4  # 初始资金
equity_curves_1 = []  # 记录每日的资产曲线
equity_curves_2 = []  
equity_curves_3 = []  
equity_curves_4 = []  

# 滚动预测并实施交易策略
for i in range(len(df['score'])):

    # 交易策略：如果预测的收益率为正，则买入；为负则卖出
    stat1 = df['score'].iloc[i]
    price = df['bargain_price'].iloc[i]

    if df['ret_std'].iloc[i] > 1e-6:

        if abs(stat1) > threshold and abs(stat1) < bound:

            if stat1 > 0:  # 如果预测为正且没有持仓
                buy = min(cash1, equity1 * buy_ratio * leverage1)
                position1 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash1 -= buy
                buy = min(cash2, equity2 * buy_ratio * leverage2)
                position2 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash2 -= buy
                buy = min(cash3, equity3 * buy_ratio * leverage3)
                position3 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash3 -= buy
                buy = min(cash4, equity4 * buy_ratio * leverage4)
                position4 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash4 -= buy  

            else:  # 如果预测为负且有持仓
                sold = min(position1, equity1 * buy_ratio / price * leverage1)
                cash1 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position1 -= sold
                sold = min(position2, equity2 * buy_ratio / price * leverage2)
                cash2 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position2 -= sold
                sold = min(position3, equity3 * buy_ratio / price * leverage3)
                cash3 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position3 -= sold
                sold = min(position4, equity4 * buy_ratio / price * leverage4)
                cash4 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position4 -= sold

    equity1 = (cash1 + position1 * price - initial_cash * (leverage1 - 1))
    equity_curves_1.append(equity1)
    equity2 = (cash2 + position2 * price - initial_cash * (leverage2 - 1))
    equity_curves_2.append(equity2)
    equity3 = (cash3 + position3 * price - initial_cash * (leverage3 - 1))
    equity_curves_3.append(equity3)
    equity4 = (cash4 + position4 * price - initial_cash * (leverage4 - 1))
    equity_curves_4.append(equity4)

    if i % 100000 == 0: print(i//100000)

    # 记录当天的资产价值


# 转换为Pandas Series以便于绘图
equity_series_1 = np.log(pd.Series(equity_curves_1, index = df.index))
equity_series_2 = np.log(pd.Series(equity_curves_2, index = df.index))
equity_series_3 = np.log(pd.Series(equity_curves_3, index = df.index))
equity_series_4 = np.log(pd.Series(equity_curves_4, index = df.index))

# 绘制资产曲线
plt.figure(figsize=(10, 6))
plt.plot(equity_series_1, label=f'leverage = {leverage1}', color='red', linewidth=1)
plt.plot(equity_series_2, label=f'leverage = {leverage2}', color='orange', linewidth=1)
plt.plot(equity_series_3, label=f'leverage = {leverage3}', color='yellow', linewidth=1)
plt.plot(equity_series_4, label=f'leverage = {leverage4}', color='green', linewidth=1)
import matplotlib.dates as mdates 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 
plt.title('Backtest Equity Curve - log')
plt.xlabel('Date')
plt.ylabel('log(Portfolio Value)')
plt.legend()
plt.show()

for equity_series in [equity_series_1, equity_series_2, equity_series_3, equity_series_4]:
    # 将数据聚合到日频
    daily_data = equity_series.resample('D').last()  # 使用最后的每一天数据，您也可以根据需要选择'ohlc'等聚合方式

    # 计算每日的收益率
    daily_returns = daily_data.pct_change().dropna()
    daily_return = (1 + daily_returns).prod() ** (1 / len(daily_returns)) - 1
    annual_return = (1 + daily_return) ** 365 - 1

    # 计算每日的夏普比率
    risk_free_rate = 0
    sharpe_ratio_daily = (daily_returns.mean() - risk_free_rate) / daily_returns.std()

    # 输出每日回报和夏普比率
    print()
    print(f"Full Return: {round((equity_series[-1]/initial_cash - 1)*100, 3)}%")
    print(f"Daily Return: {daily_return * 100:.4f}%")
    print(f"Annualized Return: {annual_return * 100:.4f}%")
    print("Daily Sharpe Ratio:", sharpe_ratio_daily)

    # 计算年化夏普比率
    sharpe_ratio_yearly = sharpe_ratio_daily * (365)**0.5  # 252是交易日的数量
    print("Yearly Sharpe Ratio:", round(sharpe_ratio_yearly, 3))
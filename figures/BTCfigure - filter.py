"""
"The code is used for plotting a figure of the returns of strategies, with or without a filter.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(__file__))

# 导入数据
df = pd.read_csv("BTCUSDT(30d).csv", parse_dates=['Date'], index_col='Date')#.tail(100000)
print('imported!')

# 格式转化
df = df[pd.to_datetime(df.index, errors='coerce').notna()]
df.index = pd.to_datetime(df.index)
df = df.asfreq('s')
df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce').dropna()

# 可调节参数
mem = 25 # 用户的峰值记忆力
srt = 2 # 用户的短期感知
#yxh = 10
buy_ratio = 0.3
threshold = 1e-9 # 阈值
bound = 12e-9
fee = 0.00005 # 手续费0.000063
delay = 1 # 延迟

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
initial_cash = 1000000  # 初始资金（可以调整）
position1 = position2 = position3 = position4 = 0  # 初始仓位（0代表不持有）
equity1 = cash1 = equity2 = cash2 = equity3 = cash3 = equity4 = cash4 = initial_cash  # 初始资金
equity_curves_none = []  # 记录每日的资产曲线
equity_curves_bound = []  
equity_curves_shift = []  
equity_curves_filtd = []  

# 滚动预测并实施交易策略
for i in range(len(df['score'])):

    # 交易策略：如果预测的收益率为正，则买入；为负则卖出
    _stat = df['_score'].iloc[i]
    stat1 = df['score'].iloc[i]
    price = df['bargain_price'].iloc[i]

    if df['ret_std'].iloc[i] > 1e-6:

        if abs(stat1) > threshold:   

            if stat1 > 0:  # 如果预测为正且没有持仓
                buy = min(cash1, equity1 * buy_ratio)
                position1 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash1 -= buy  
            else:  # 如果预测为负且有持仓
                sold = min(position1, equity1 * buy_ratio / price)
                cash1 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position1 -= sold

        if abs(_stat) > threshold:

            if _stat > 0:  # 如果预测为正且没有持仓
                buy = min(cash2, equity2 * buy_ratio)
                position2 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash2 -= buy  
            else:  # 如果预测为负且有持仓
                sold = min(position2, equity2 * buy_ratio / price)
                cash2 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position2 -= sold
   
        if abs(_stat) > threshold and abs(_stat) < bound:

            if _stat > 0:  # 如果预测为正且没有持仓
                buy = min(cash3, equity3 * buy_ratio)
                position3 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash3 -= buy  
            else:  # 如果预测为负且有持仓
                sold = min(position3, equity3 * buy_ratio / price)
                cash3 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position3 -= sold

        if abs(stat1) > threshold and abs(stat1) < bound:

            if stat1 > 0:  # 如果预测为正且没有持仓
                buy = min(cash4, equity4 * buy_ratio)
                position4 += ( buy / (price) / (1+fee) ) # 用现金买入比特币
                cash4 -= buy  
            else:  # 如果预测为负且有持仓
                sold = min(position4, equity4 * buy_ratio / price)
                cash4 += ( sold * (price) / (1+fee) ) # 卖出比特币
                position4 -= sold

    equity1 = cash1 + position1 * price
    equity_curves_shift.append(equity1)
    equity2 = cash2 + position2 * price
    equity_curves_none.append(equity2)
    equity3 = cash3 + position3 * price
    equity_curves_bound.append(equity3)
    equity4 = cash4 + position4 * price
    equity_curves_filtd.append(equity4)

    if i % 100000 == 0: print(i//100000)

    # 记录当天的资产价值


# 转换为Pandas Series以便于绘图
equity_series_none = pd.Series(equity_curves_none, index = df.index)
equity_series_bound = pd.Series(equity_curves_bound, index = df.index)
equity_series_shift = pd.Series(equity_curves_shift, index = df.index)
equity_series_filtd = pd.Series(equity_curves_filtd, index = df.index)

# 绘制资产曲线
plt.figure(figsize=(10, 6))
plt.plot(equity_series_none, label='Not filtered', color='red', linewidth=1)
plt.plot(equity_series_shift, label='filter horizontally', color='orange', linewidth=1)
plt.plot(equity_series_bound, label='filter vertically', color='yellow', linewidth=1)
plt.plot(equity_series_filtd, label='fully filtered', color='green', linewidth=1)
import matplotlib.dates as mdates 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 
plt.title('Backtest Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

for equity_series in [equity_series_none, equity_series_bound, equity_series_filtd, equity_series_shift]:
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
    print(f"Full Return: {round((equity_series[-1]/initial_cash - 1)*100, 3)}%")
    print(f"Daily Return: {daily_return * 100:.4f}%")
    print(f"Annualized Return: {annual_return * 100:.4f}%")
    print("Daily Sharpe Ratio:", sharpe_ratio_daily)

    # 计算年化夏普比率
    sharpe_ratio_yearly = sharpe_ratio_daily * (365)**0.5  # 252是交易日的数量
    print("Yearly Sharpe Ratio:", round(sharpe_ratio_yearly, 3))
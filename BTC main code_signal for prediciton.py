"""
The code is used for predicting the returns of the btc, by applying the signal called "score".
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('start!')

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(__file__))

# 导入数据
df = pd.read_csv("BTCUSDT(30d).csv", parse_dates=['Date'], index_col='Date')#.head(20000)

# 日期正序
close_price = df['Close']

# 格式转化
df = df[pd.to_datetime(df.index, errors='coerce').notna()]
df.index = pd.to_datetime(df.index)
df = df.asfreq('s')
close_price = pd.to_numeric(close_price.astype(str).str.replace(',', ''), errors='coerce').dropna()

# 可调节参数
mem = 25 # 用户的峰值记忆力
srt = 2 # 用户的短期感知
pay_ratio = 1 # 支付比例
threshold = 1e-9 # 阈值
bound = 12e-9
fee = 0.00005 # 手续费
delay = 1 # 延迟

# 计算
log_returns = np.log(close_price / close_price.shift(1)).dropna().asfreq('s')
df['bargain_price'] = df['Close'].shift(-delay)
df['log_returns'] = np.log(close_price / close_price.shift(1)).dropna().asfreq('s')
df['log_ret_prev'] = df['log_returns'].shift(1)
df['ret_mean'] = df['log_ret_prev'].rolling(window=srt).mean()
df['peak'] = df['ret_mean'].rolling(window=mem).max()
df['valley'] = df['ret_mean'].rolling(window=mem).min()
df['score'] = np.where(df['log_ret_prev'] < 0,
                       df['valley'] * abs(df['log_ret_prev']*3.15),
                       df['peak'] * abs(df['log_ret_prev']))

df['main_scr'] = df['score'].rolling(window=10).apply(lambda x: x.loc[x.abs().idxmax()], raw=False) / 3
df['score'] = np.where(np.sign(df['main_scr'] + df['score']) != np.sign(df['score']), 0, df['score'])

# 对齐数据
df = df[['score','bargain_price']].dropna()

# 初始化策略变量
initial_cash = 1000000  # 初始资金（可以调整）
position = 0  # 初始仓位（0代表不持有）
cash = equity = initial_cash  # 初始现金
equity_curves = []  # 记录每日的资产曲线
position_shift = []  # 记录每日的仓位变化
score_distribute = [] # 记录信号变化
score_effective = [] # 记录有效信号
predicted_returns = []  # 记录每日的预测收益

# 滚动预测并实施交易策略
for i in range(len(df['score'])):

    # 交易策略：如果预测的收益率为正，则买入；为负则卖出
    stats = df['score'].iloc[i]
    price = df['bargain_price'].iloc[i]
    if abs(stats) > threshold and abs(stats) < bound:
        score_effective.append(stats)
        if stats > 0:  # 如果预测为正且没有持仓
            buy = min(cash, equity * pay_ratio)
            position += ( buy / price / (1+fee) ) # 用现金买入比特币
            cash -= buy  
        elif stats < 0:  # 如果预测为负且有持仓
            sold = min(position, equity / price * pay_ratio)
            cash += ( sold * price / (1+fee) ) # 卖出比特币
            position -= sold
    else:
        score_effective.append(0)

    # 记录当天的资产价值
    equity = cash + position * price
    equity_curves.append(equity)
    position_shift.append(position*price/equity)
    score_distribute.append(stats)


# 转换为Pandas Series以便于绘图
equity_series = pd.Series(equity_curves, index = df.index)

import matplotlib.dates as mdates
# 绘制资产曲线
show_return = 1
if show_return:
    plt.figure(figsize=(10, 6))
    plt.plot(equity_series, label='Equity Curve', color='green', linewidth=1)    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.title('Backtest Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()

# 绘制score曲线
show_score = 1
if show_score:
    power_series = pd.Series(score_distribute, index = df.index)
    plt.figure(figsize=(10, 6))
    plt.plot(power_series.index, power_series, label='Score', color='red', linewidth=1)   
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 
    plt.title('Backtest Score Curve')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.legend()

# 绘制score曲线
show_ef_score = 1
if show_ef_score:
    score_ef_series = pd.Series(score_effective, index = df.index)
    plt.figure(figsize=(10, 6))
    plt.plot(score_ef_series.index, score_ef_series, label='Effective Score', color='purple', linewidth=1)   
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 
    plt.title('Backtest Effective Score Curve')
    plt.xlabel('Time')
    plt.ylabel('Effective Score')
    plt.legend()

show_position = 0
if show_position:
    position_series = pd.Series(position_shift)
    plt.figure(figsize=(10, 6))
    plt.plot(position_series.index, position_series, label='position', color='blue', linewidth=0.1)
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.title('Backtest position Curve')
    plt.xlabel('position')
    plt.ylabel('tick')
    plt.legend()

show_btc = 1
if show_btc:
    plt.figure(figsize=(10, 6))
    plt.plot(close_price.index, close_price, label='close_price', color='blue', linewidth=1)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.title('close_price Curve')
    plt.xlabel('close_price')
    plt.ylabel('tick')
    plt.legend()
    
plt.show()

risk_free_rate = 0

# 将数据聚合到日频
daily_data = equity_series.resample('D').last()  # 使用最后的每一天数据，您也可以根据需要选择'ohlc'等聚合方式

# 计算每日的收益率
daily_returns = daily_data.pct_change().dropna()
daily_return = (1 + daily_returns).prod() ** (1 / len(daily_returns)) - 1
annual_return = (1 + daily_return) ** 365 - 1

# 计算每日的夏普比率
sharpe_ratio_daily = (daily_returns.mean() - risk_free_rate) / daily_returns.std()

# 输出每日回报和夏普比率
print(f"Full Return: {round((equity_series[-1]/initial_cash - 1)*100, 3)}%")
print(f"Daily Return: {daily_return * 100:.4f}%")
print(f"Annualized Return: {annual_return * 100:.4f}%")
print("Daily Sharpe Ratio:", sharpe_ratio_daily)

# 计算年化夏普比率
sharpe_ratio_yearly = sharpe_ratio_daily * (365)**0.5  # 252是交易日的数量
print("Yearly Sharpe Ratio:", round(sharpe_ratio_yearly, 3))
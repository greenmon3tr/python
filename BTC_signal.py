import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('start!')

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(__file__))

# 导入数据
df = pd.read_csv("BTCUSDT(30d).csv", parse_dates=['Date'], index_col='Date')#.head(200000)

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
ave = 3.15 # 用户的损失厌恶系数
pay_ratio = 1 # 支付比例
threshold = 1e-9 # 阈值
bound = 12e-9 # 过滤上限
fee = 0.00005 # 手续费
delay = 1 # 网络延迟
initial_cash = 1000000  # 初始资金（可以调整）

# 计算
df['bargain_price'] = df['Close'].shift(-delay)
df['log_returns'] = np.log(close_price / close_price.shift(1)).dropna().asfreq('s')
df['log_ret_prev'] = df['log_returns'].shift(1)
df['ret_mean'] = df['log_ret_prev'].rolling(window=srt).mean()
df['peak'] = df['ret_mean'].rolling(window=mem).max()
df['valley'] = df['ret_mean'].rolling(window=mem).min()
df['score'] = np.where(df['log_ret_prev'] < 0,
                       df['valley'] * abs(df['log_ret_prev']*ave),
                       df['peak'] * abs(df['log_ret_prev']))

df['main_scr'] = df['score'].rolling(window=10).apply(lambda x: x.loc[x.abs().idxmax()], raw=False) / 3
df['score'] = np.where(np.sign(df['main_scr'] + df['score']) != np.sign(df['score']), 0, df['score'])

# 对齐数据
df = df[['score','bargain_price']].dropna()

# 初始化变量
position = 0  # 初始仓位（0代表不持有）
cash = equity = initial_cash  # 初始现金(无杠杆)
equity_curves = []  # 记录每日的资产曲线

# 滚动交易策略
for i in range(len(df['score'])):

    # 交易策略：如果预测信号为正，则买入；为负则卖出
    signal = df['score'].iloc[i]
    price = df['bargain_price'].iloc[i]
    if abs(signal) > threshold and abs(signal) < bound:
        if signal > 0:  # 如果预测为正
            buy = min(cash, equity * pay_ratio)
            position += ( buy / price / (1+fee) ) 
            cash -= buy  
        elif signal < 0:  # 如果预测为负
            sold = min(position, equity / price * pay_ratio)
            cash += ( sold * price / (1+fee) ) 
            position -= sold

    # 记录当天的资产价值
    equity = cash + position * price
    equity_curves.append(equity)

# 资产—时间 series
equity_series = pd.Series(equity_curves)
equity_series.index = df.index

# 可视化：资产曲线
import matplotlib.dates as mdates
show_return = 1
if show_return:
    plt.figure(figsize=(10, 6))
    plt.plot(equity_series, label='Equity Curve', color='blue', linewidth=1)    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.title('Backtest Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.legend()
plt.show()

risk_free_rate = 0

# 输出：收益率 & 夏普比率
daily_data = equity_series.resample('D').last() # 将数据聚合到日频

# 计算日收益率、年化收益率
daily_returns = daily_data.pct_change().dropna()
daily_return = (1 + daily_returns).prod() ** (1 / len(daily_returns)) - 1
annual_return = (1 + daily_return) ** 365 - 1
print(f"Full Return: {round((equity_series[-1]/initial_cash - 1)*100, 3)}%") # 数据集中的最终收益率
print(f"Daily Return: {daily_return * 100:.4f}%")
print(f"Annualized Return: {annual_return * 100:.4f}%")

# 计算年化夏普比率
sharpe_ratio_daily = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
sharpe_ratio_yearly = sharpe_ratio_daily * (365)**0.5  # 365是交易日的数量
print("Yearly Sharpe Ratio:", round(sharpe_ratio_yearly, 3))
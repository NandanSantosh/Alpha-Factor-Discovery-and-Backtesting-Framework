import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model1=LinearRegression()
model=RandomForestRegressor(n_estimators=200,max_depth=5,random_state=42)
stocks=["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","LT.NS","ITC.NS","BHARTIARTL.NS","AXISBANK.NS"]

data=[]
for i in stocks:
    stock=yf.download(i,start="2020-01-01",end="2025-01-01")
    stock.columns=stock.columns.get_level_values(0)
    stock["Return"]=stock["Close"].pct_change()
    stock["MA20"]=stock["Close"].rolling(20).mean()
    stock["MA_Ratio"]=stock["Close"]/stock["MA20"]
    stock["Volatility"]=stock["Return"].rolling(20).std()
    stock["Momentum20"]=(stock["Close"]/stock["Close"].shift(20))-1
    #new features since corr coeff is not positive
    stock["Momentum5"]=((stock["Close"])/stock["Close"].shift(5))-1
    stock["Momentum10"]=(stock["Close"]/stock["Close"].shift(10))-1
    stock["VolumeMA20"]=stock["Volume"].rolling(20).mean()
    stock["VolumeRatio"]=stock["Volume"]/stock["VolumeMA20"]
    stock["MA5"]=stock["Close"].rolling(5).mean()
    stock["MA5_MA20_Ratio"]=stock["MA5"]/stock["MA20"]
    stock["Volatility60"]=stock["Return"].rolling(60).std()
    stock["VolatilityRatio"]=stock["Volatility"]/stock["Volatility60"]
    delta=stock["Close"].diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.rolling(14).mean()
    avg_loss=loss.rolling(14).mean()
    rsi=avg_gain/avg_loss
    stock["RSI"]=100*(rsi/(rsi+1))
    stock["Target"]=(stock["Close"].shift(-5)/stock["Close"])-1
    stock=stock.dropna()
    stock["i"]=i
    data.append(stock)

all_data=pd.concat(data)
# print(all_data.shape)
# print(all_data["i"].value_counts())

features=["MA_Ratio","Momentum5","Momentum10","Momentum20","Volatility","VolatilityRatio","VolumeRatio","RSI","MA5_MA20_Ratio"]

x=all_data[features]
y=all_data["Target"]

train=all_data[all_data.index<"2024-01-01"]
test=all_data[all_data.index>="2024-01-01"]

x_train=train[features]
y_train=train["Target"]
x_test=test[features]
y_test=test["Target"]

model.fit(x_train,y_train)
pred=model.predict(x_test)

#for backtesting from here is entire backtesting
test=test.copy()
test["Prediction"]=pred

portfolio_returns=[]
#this is for long only i.e only taking the top 3 companies
# for date,group in test.groupby(test.index):
#     group=group.sort_values("Prediction",ascending=False)
#     top=group.head(3)
#     #daily return is the actual target return
#     daily_return=top["Target"].mean()
#     portfolio_returns.append(daily_return)


# portfolio_returns=pd.Series(portfolio_returns)

#now we will see both long-short portfolio
for date,group in test.groupby(test.index):
    group=group.sort_values("Prediction",ascending=False)
    top=group.head(3)
    bottom=group.tail(3)
    long_return=top["Target"].mean()
    short_return=bottom["Target"].mean()
    portfolio_return=long_return-short_return
    portfolio_returns.append(portfolio_return)

portfolio_returns=pd.Series(portfolio_returns)
#cumulative is for looking at the graph to give me more clear idea if graph is increasing then returns are positive always
cumulative=(1+portfolio_returns).cumprod()

#benchmark i.e comparing my model against
benchmark_returns=[]
for date,group in test.groupby(test.index):
    daily_return=group["Target"].mean()
    benchmark_returns.append(daily_return)
benchmark_returns=pd.Series(benchmark_returns)
benchmark_cumulative=(1+benchmark_returns).cumprod()
print(benchmark_cumulative.iloc[-1])
plt.figure(figsize=(10,5))

plt.plot(cumulative.values,label="Strategy")
plt.plot(benchmark_cumulative.values,label="Equal Weight")
plt.legend()
plt.title("Strategy vs Benchmark")
plt.show()


# print(benchmark_returns.head())
# print(benchmark_returns.mean())
# print(benchmark_cumulative.tail())
# print(portfolio_returns.mean())
# print((portfolio_returns>0).mean())
# print(cumulative.iloc[-1])
# plt.plot(cumulative)
# plt.title("Top 3 portfolio")
# plt.show()
# # print("MSE:",mean_squared_error(y_test,pred))
# print("Correlation:",np.corrcoef(y_test,pred)[0,1])
# print(model.feature_importances_)
# print(model.coef_)
# print(model.intercept_)
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

model=LinearRegression()
stock =yf.download("RELIANCE.NS",start="2020-01-01",end="2025-01-01")

stock.columns=stock.columns.get_level_values(0)

stock["Return"]=stock["Close"].pct_change()
stock["MA5"]=stock["Close"].rolling(5).mean()
stock["MA20"]=stock["Close"].rolling(20).mean()

stock["MA_Ratio"]=(stock["Close"]/stock["MA20"])
stock["Volatility"]=stock["Return"].rolling(20).std()
stock["Momentum20"]=(stock["Close"]/stock["Close"].shift(20))-1
delta=stock["Close"].diff()
gain=delta.clip(lower=0)
loss=-delta.clip(upper=0)
avg_gain=gain.rolling(14).mean()
avg_loss=loss.rolling(14).mean()
rs=avg_gain/avg_loss
stock["RSI"]=100-(100/(1+rs))
stock["Target"]=stock["Return"].shift(-1)

stock=stock.dropna();#this is for removing nan's since we cannot train ml with nan's -not a numbers

features=["MA_Ratio","Volatility","Momentum20","RSI"]

x=stock[features]

y=stock["Target"]

split=int(len(stock)*0.8)

x_train=x[:split]
x_test=x[split:]

y_train=y[:split]
y_test=y[split:]

model.fit(x_train,y_train)
pred=model.predict(x_test)

mse=mean_squared_error(y_test,pred)

corr=np.corrcoef(y_test,pred)[0,1]
last_features=x.iloc[[-1]]
predicted_return=model.predict(last_features)

print(predicted_return)
# print(stock.head())
# print(stock.columns)
# print(stock.shape)
# print(stock[["Close","Return"]].head(10))
# print(stock[["Close","MA5","MA20"]].head(25))
# print(stock[["Close","MA20","MA_Ratio"]].tail())
# print(stock[["Return","Volatility"]].tail())
# print(stock[["Close","MA20","MA_Ratio","Volatility","Momentum20"]].tail(10))
# print(stock[["Close","RSI"]].tail(20))
# print(stock[["Return","Target"]].head(10))

# print(stock[["Return","Target"]].head(10))
# print(x.head())
# print(y.head())
# print(x.shape)

# print(x_train.shape)
# print(x_test.shape)

# for i in range(10):
#     print("Actual:",y_test.iloc[i],"predicted:",pred[i])

# print(mse)
# print(corr)

# print(model.coef_)
# print(model.intercept_)

import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 获取股票数据
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# 计算技术分析指标
def calculate_technical_indicators(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    return stock_data

# 计算相对强弱指标（RSI）
def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 构建机器学习模型
def build_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 测试模型
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# 主函数
def main():
    # 获取股票数据
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-05-24'
    stock_data = get_stock_data(symbol, start_date, end_date)

    # 计算技术分析指标
    stock_data = calculate_technical_indicators(stock_data)

    # 特征工程
    features = ['SMA_50', 'SMA_200', 'RSI']
    stock_data = stock_data.dropna(subset=features)  # 移除包含 NaN 值的行
    X = stock_data[features]
    y = stock_data['Close']

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建并测试机器学习模型
    model = build_model(X_train, y_train)
    mse = test_model(model, X_test, y_test)
    print('Mean Squared Error:', mse)

    # 可视化股价走势和指标
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.plot(stock_data['SMA_50'], label='SMA 50')
    plt.plot(stock_data['SMA_200'], label='SMA 200')
    plt.legend()
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    main()

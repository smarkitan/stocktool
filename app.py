# app.py
from flask import Flask, jsonify, render_template, request
import yfinance as yf
from flask_cors import CORS
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
CORS(app, resources={r"/api/*": {"origins": "https://stefanstocktool.netlify.app"}})

### API Routes ###

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Fetch latest stock data for a given symbol."""
    app.logger.info(f"Fetching stock data for symbol: {symbol}")
    try:
        df = yf.download(symbol, period="1d", interval="1d")
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        latest_data = df.iloc[-1]
        ticker = yf.Ticker(symbol)
        stock_info = ticker.info

        response_data = {
            "companyName": stock_info.get('longName', stock_info.get('shortName', symbol)),
            "lastClosePrice": latest_data['Close'],
            "lastCloseDate": latest_data.name.isoformat(),
            "openPrice": latest_data['Open'],
            "highPrice": latest_data['High'],
            "lowPrice": latest_data['Low'],
            "volume": latest_data['Volume'],
            "exchangeInfo": "NasdaqGS - Nasdaq Real Time Price • USD",
            "compareLink": f"/compare/{symbol}",
            **{key: stock_info.get(key, 'N/A') for key in [
                'regularMarketPreviousClose', 'marketCap', 'regularMarketOpen', 
                'beta', 'bid', 'bidSize', 'trailingPE', 'ask', 'askSize',
                'trailingEps', 'regularMarketDayLow', 'regularMarketDayHigh',
                'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'dividendRate', 
                'dividendYield', 'regularMarketVolume', 'exDividendDate', 
                'averageVolume', 'targetMeanPrice', 'enterpriseValue', 
                'priceToBook', 'priceToSalesTrailing12Months', 
                'enterpriseToEbitda', 'operatingMargins', 'grossMargins', 
                'profitMargins', 'earningsGrowth', 'sector', 
                'industry', 'totalRevenue', 'revenueGrowth', 
                'operatingCashflow'
            ]}
        }
        app.logger.info(f"Data fetched successfully for symbol: {symbol}")
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error fetching stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/news', methods=['GET'])
def get_stock_news(symbol):
    """Fetch news related to a given stock symbol."""
    app.logger.info(f"Fetching stock news for symbol: {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        news_data = ticker.news

        if not news_data:
            return jsonify({"error": "No news found"}), 404

        news_items = [{
            "title": item.get('title'),
            "link": item.get('link'),
            "publisher": item.get('publisher'),
            "publishedDate": item.get('providerPublishTime')
        } for item in news_data]

        app.logger.info(f"Stock news fetched successfully for symbol: {symbol}")
        return jsonify(news_items)
    except Exception as e:
        app.logger.error(f"Error fetching stock news for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/intraday', methods=['GET'])
def get_intraday_stock_data(symbol):
    """Fetch intraday stock data for a given symbol."""
    app.logger.info(f"Fetching intraday stock data for symbol: {symbol}")
    try:
        df = yf.download(symbol, period="1d", interval="1m")
        if df.empty:
            return jsonify({"error": "No intraday data found"}), 404

        intraday_data = {
            "datetime": df.index.strftime('%b %d, %I:%M %p').tolist(),
            "close": df['Close'].tolist(),
            "open": df['Open'].tolist(),
            "high": df['High'].tolist(),
            "low": df['Low'].tolist(),
            "volume": df['Volume'].tolist()
        }

        app.logger.info(f"Intraday stock data fetched successfully for symbol: {symbol}")
        return jsonify(intraday_data)
    except Exception as e:
        app.logger.error(f"Error fetching intraday data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/historical', methods=['GET'])
def get_stock_historical_data(symbol):
    """Fetch historical stock data for a given symbol with an optional period."""
    period = request.args.get('period', '1d')
    app.logger.info(f"Fetching historical stock data for symbol: {symbol} with period: {period}")
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return jsonify({"error": "No historical data found"}), 404
        
        data = {
            "datetime": hist.index.strftime('%Y-%m-%d').tolist(),
            "close": hist['Close'].tolist(),
            "open": hist['Open'].tolist(),
            "high": hist['High'].tolist(),
            "low": hist['Low'].tolist(),
            "volume": hist['Volume'].tolist(),
        }
        app.logger.info(f"Historical stock data fetched successfully for symbol: {symbol}")
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/test_stock_data/<symbol>', methods=['GET'])
def test_stock_data_route(symbol):
    """Fetch test stock data for a given symbol within a specified date range."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    app.logger.info(f"Fetching test stock data for symbol: {symbol} from {start_date} to {end_date}")
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return jsonify({"error": f"No data found for {symbol} from {start_date} to {end_date}"}), 404
        
        latest_data = stock.history(period='1d').iloc[-1] if not stock.history(period='1d').empty else None

        last_close_price = latest_data['Close'] if latest_data is not None else 'N/A'
        last_close_date = latest_data.name.isoformat() if latest_data is not None else 'N/A'

        last_dividend_value = stock.info.get('dividendRate', 'N/A')
        last_dividend_date = stock.info.get('exDividendDate', 'N/A')

        data = {
            "datetime": hist.index.strftime('%Y-%m-%d').tolist(),
            "close": hist['Close'].tolist(),
            "open": hist['Open'].tolist(),
            "high": hist['High'].tolist(),
            "low": hist['Low'].tolist(),
            "volume": hist['Volume'].tolist(),
            "lastDividendValue": last_dividend_value,
            "lastDividendDate": last_dividend_date,
            "lastClosePrice": last_close_price,
            "lastCloseDate": last_close_date
        }

        app.logger.info(f"Test stock data fetched successfully for symbol: {symbol}")
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching test stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

### Machine Learning Prediction Route ###

@app.route('/api/predict_stock/<symbol>/<int:num_years>', methods=['GET'])
def predict_stock_route(symbol, num_years):
    """Predict future stock prices using an LSTM model."""
    app.logger.info(f"Predicting stock prices for symbol: {symbol} over the next {num_years} years.")
    try:
        period_past = f"{num_years}y"
        period_future_days = 360 * num_years

        df = yf.download(symbol, period=period_past)
        if df.empty:
            return jsonify({"error": "No data found for the given symbol and period."}), 404

        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        look_back = 60
        x_train, y_train = [], []

        for i in range(look_back, len(scaled_data)):
            x_train.append(scaled_data[i - look_back:i, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        x_test = scaled_data[-look_back:].reshape(1, -1, 1)

        predicted_prices = []
        for _ in range(period_future_days):
            predicted_price = model.predict(x_test)
            predicted_prices.append(predicted_price[0][0])
            x_test = np.append(x_test[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

        response = {
            "predictedPrices": predicted_prices.tolist(),
            "message": f"Predicted prices for the next {period_future_days} days."
        }
        app.logger.info(f"Predicted stock prices successfully for symbol: {symbol}")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error predicting stock prices for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

### Main Route ###

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

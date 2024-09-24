# app.py
from flask import Flask, jsonify, render_template, request
import yfinance as yf
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
CORS(app, resources={rapi {origins httpsstefanstocktool.netlify.app}})  # Adjust the origin as needed

### Existing Routes ###

@app.route('apistocksymbol')
def get_stock_data(symbol)
    app.logger.info(fFetching stock data for symbol {symbol})
    try
        df = yf.download(symbol, period=1d, interval=1d)
        if df.empty
            app.logger.warning(fNo data found for symbol {symbol})
            return jsonify({error No data found}), 404

        latest_data = df.iloc[-1]
        last_close_price = latest_data['Close']
        last_close_date = latest_data.name.isoformat()
        open_price = latest_data['Open']
        high_price = latest_data['High']
        low_price = latest_data['Low']
        volume = latest_data['Volume']

        ticker = yf.Ticker(symbol)
        stock_info = ticker.info

        company_name = stock_info.get('longName', stock_info.get('shortName', symbol))
        exchange_info = NasdaqGS - Nasdaq Real Time Price • USD
        compare_link = fcompare{symbol}

        app.logger.info(fData fetched successfully for symbol {symbol})
        return jsonify({
            companyName company_name,
            lastClosePrice last_close_price,
            lastCloseDate last_close_date,
            openPrice open_price,
            highPrice high_price,
            lowPrice low_price,
            volume volume,
            exchangeInfo exchange_info,
            compareLink compare_link,
            previousClose stock_info.get('regularMarketPreviousClose', 'NA'),
            marketCap stock_info.get('marketCap', 'NA'),
            open stock_info.get('regularMarketOpen', 'NA'),
            beta stock_info.get('beta', 'NA'),
            bid stock_info.get('bid', 'NA'),
            bidSize stock_info.get('bidSize', 'NA'),
            trailingPE stock_info.get('trailingPE', 'NA'),
            ask stock_info.get('ask', 'NA'),
            askSize stock_info.get('askSize', 'NA'),
            trailingEps stock_info.get('trailingEps', 'NA'),
            regularMarketDayLow stock_info.get('regularMarketDayLow', 'NA'),
            regularMarketDayHigh stock_info.get('regularMarketDayHigh', 'NA'),
            fiftyTwoWeekLow stock_info.get('fiftyTwoWeekLow', 'NA'),
            fiftyTwoWeekHigh stock_info.get('fiftyTwoWeekHigh', 'NA'),
            dividendRate stock_info.get('dividendRate', 'NA'),
            dividendYield stock_info.get('dividendYield', 'NA'),
            regularMarketVolume stock_info.get('regularMarketVolume', 'NA'),
            exDividendDate stock_info.get('exDividendDate', 'NA'),
            averageVolume stock_info.get('averageVolume', 'NA'),
            targetMeanPrice stock_info.get('targetMeanPrice', 'NA'),
            enterpriseValue stock_info.get('enterpriseValue', 'NA'),
            priceToBook stock_info.get('priceToBook', 'NA'),
            priceToSalesTrailing12Months stock_info.get('priceToSalesTrailing12Months', 'NA'),
            enterpriseToEbitda stock_info.get('enterpriseToEbitda', 'NA'),
            operatingMargins stock_info.get('operatingMargins', 'NA'),
            grossMargins stock_info.get('grossMargins', 'NA'),
            profitMargins stock_info.get('profitMargins', 'NA'),
            earningsGrowth stock_info.get('earningsGrowth', 'NA'),
            sector stock_info.get('sector', 'NA'),
            industry stock_info.get('industry', 'NA'),
            totalRevenue stock_info.get('totalRevenue', 'NA'),
            revenueGrowth stock_info.get('revenueGrowth', 'NA'),
            operatingCashflow stock_info.get('operatingCashflow', 'NA')
        })
    except Exception as e
        app.logger.error(fError fetching stock data for {symbol} {e}, exc_info=True)
        return jsonify({error str(e)}), 500

@app.route('apistocksymbolnews')
def get_stock_news(symbol)
    app.logger.info(fFetching stock news for symbol {symbol})
    try
        ticker = yf.Ticker(symbol)
        news_data = ticker.news

        if not news_data
            app.logger.warning(fNo news found for symbol {symbol})
            return jsonify({error No news found}), 404

        news_items = [{
            title item.get('title'),
            link item.get('link'),
            publisher item.get('publisher'),
            publishedDate item.get('providerPublishTime')
        } for item in news_data]

        app.logger.info(fStock news fetched successfully for symbol {symbol})
        return jsonify(news_items)
    except Exception as e
        app.logger.error(fError fetching stock news for {symbol} {e}, exc_info=True)
        return jsonify({error str(e)}), 500

@app.route('apistocksymbolintraday')
def get_intraday_stock_data(symbol)
    app.logger.info(fFetching intraday stock data for symbol {symbol})
    try
        df = yf.download(symbol, period=1d, interval=1m)
        if df.empty
            app.logger.warning(fNo intraday data found for symbol {symbol})
            return jsonify({error No intraday data found}), 404

        intraday_data = {
            datetime df.index.strftime('%b %d, %I%M %p').tolist(),
            close df['Close'].tolist(),
            open df['Open'].tolist(),
            high df['High'].tolist(),
            low df['Low'].tolist(),
            volume df['Volume'].tolist()
        }

        app.logger.info(fIntraday stock data fetched successfully for symbol {symbol})
        return jsonify(intraday_data)
    except Exception as e
        app.logger.error(fError fetching intraday data for {symbol} {e}, exc_info=True)
        return jsonify({error str(e)}), 500

@app.route('apistocksymbolhistorical', methods=['GET'])
def get_stock_historical_data(symbol)
    period = request.args.get('period', '1d')  # Retrieve period parameter from query string
    app.logger.info(fFetching historical stock data for symbol {symbol} with period {period})
    try
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty
            app.logger.warning(fNo historical data found for symbol {symbol} with period {period})
            return jsonify({error No historical data found}), 404
        
        data = {
            datetime hist.index.strftime('%Y-%m-%d').tolist(),
            close hist['Close'].tolist(),
            open hist['Open'].tolist(),
            high hist['High'].tolist(),
            low hist['Low'].tolist(),
            volume hist['Volume'].tolist(),
        }
        app.logger.info(fHistorical stock data fetched successfully for symbol {symbol})
        return jsonify(data)
    except Exception as e
        app.logger.error(fError fetching historical data for {symbol} {e}, exc_info=True)
        return jsonify({error str(e)}), 500

@app.route('apitest_stock_datasymbol', methods=['GET'])
def test_stock_data_route(symbol)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    app.logger.info(fFetching test stock data for symbol {symbol} from {start_date} to {end_date})
    try
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty
            app.logger.warning(fNo data found for symbol {symbol} from {start_date} to {end_date})
            return jsonify({error fNo data found for {symbol} from {start_date} to {end_date}}), 404
        
        # Obtain the most recent data (i.e., the latest entry in the historical data)
        most_recent_hist = stock.history(period='1d')  # Get the most recent data
        latest_data = most_recent_hist.iloc[-1] if not most_recent_hist.empty else None

        if latest_data is not None
            last_close_price = latest_data['Close']
            last_close_date = latest_data.name.isoformat()
        else
            last_close_price = 'NA'
            last_close_date = 'NA'

        # Get last dividend information
        info = stock.info
        last_dividend_value = info.get('dividendRate', 'NA')
        last_dividend_date = info.get('exDividendDate', 'NA')

        # Prepare response data
        data = {
            datetime hist.index.strftime('%Y-%m-%d').tolist(),
            close hist['Close'].tolist(),
            open hist['Open'].tolist(),
            high hist['High'].tolist(),
            low hist['Low'].tolist(),
            volume hist['Volume'].tolist(),
            lastDividendValue last_dividend_value,
            lastDividendDate last_dividend_date,
            lastClosePrice last_close_price,
            lastCloseDate last_close_date
        }

        app.logger.info(fTest stock data fetched successfully for symbol {symbol})
        return jsonify(data)

    except Exception as e
        app.logger.error(fError fetching test stock data for {symbol} {e}, exc_info=True)
        return jsonify({error str(e)}), 500

### New Machine Learning Prediction Route ###

def predict_stock(symbol, num_years)
    
    Predict future stock prices using an LSTM model and calculate investment returns.

    Parameters
        symbol (str) Stock ticker symbol.
        num_years (int) Number of years for prediction.

    Returns
        dict Contains predictions and investment data or an error message.
    
    try
        period_past = f{num_years}y
        period_future_days = 360  num_years  # Approximate number of trading days per year

        # Download historical data
        df = yf.download(symbol, period=period_past)

        if df.empty
            return {error No data found for the given symbol and period.}

        # Use 'Close' prices for prediction
        data = df['Close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Define time window
        look_back = 60
        x_train, y_train = [], []

        for i in range(look_back, len(scaled_data))
            x_train.append(scaled_data[i - look_backi, 0])
            y_train.append(scaled_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Prepare data for prediction
        test_data = scaled_data[-look_back]
        test_data = np.reshape(test_data, (1, look_back, 1))

        predictions = []
        for _ in range(period_future_days)
            predicted_price = model.predict(test_data, verbose=0)
            predictions.append(predicted_price[0, 0])
            test_data = np.append(test_data[, 1, ], [[predicted_price]], axis=1)

        # Denormalize predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

        # Investment Calculations
        today = datetime.today()
        past_date = today - timedelta(days=365  num_years)
        df_past = yf.download(symbol, start=past_date.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

        if df_past.empty
            return {error Insufficient historical data for investment calculations.}

        price_past = df_past['Close'].iloc[0]
        price_today = df_past['Close'].iloc[-1]

        # Investment 1 1000 Euros invested num_years ago
        invest_initial = 1000
        num_shares = invest_initial  price_past
        value_today = num_shares  price_today

        # Investment 2 1000 Euros invested today
        invest_future_initial = 1000
        num_shares_future = invest_future_initial  price_today
        predicted_price_future = predictions[-1]
        value_future = num_shares_future  predicted_price_future

        investment_data = {
            investment_initial {
                description f1000 Euros invested {num_years} years ago,
                price_past round(price_past, 2),
                num_shares round(num_shares, 4),
                value_today round(value_today, 2)
            },
            investment_future {
                description 1000 Euros invested today,
                price_today round(price_today, 2),
                num_shares_future round(num_shares_future, 4),
                predicted_price_future round(predicted_price_future, 2),
                value_future round(value_future, 2)
            }
        }

        return {
            predictions predictions,
            investment_data investment_data
        }

    except Exception as e
        return {error str(e)}

@app.route('apipredict', methods=['POST'])
def api_predict()
    
    Endpoint to predict future stock prices and calculate investment returns.

    Expects a JSON payload with
        - symbol (str) Stock ticker symbol.
        - num_years (int) Number of years for prediction.

    Returns
        JSON Predictions and investment data or an error message.
    
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    num_years = data.get('num_years', 10)

    if not symbol
        return jsonify({error Symbol is required.}), 400

    try
        num_years = int(num_years)
        if num_years = 0
            return jsonify({error Number of years must be a positive integer.}), 400
    except ValueError
        return jsonify({error Number of years must be an integer.}), 400

    result = predict_stock(symbol, num_years)

    if error in result
        return jsonify(result), 400

    return jsonify(result)

### Existing Root Route ###

@app.route('')
def index()
    return render_template('index.html')

### Running the Flask Application ###

if __name__ == '__main__'
    app.run(host='0.0.0.0', port=5000)

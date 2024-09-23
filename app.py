from flask import Flask, jsonify, render_template, request
import yfinance as yf
from flask_cors import CORS
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import json

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://stefanstocktool.netlify.app"}})  # Allow specific origins

warnings.filterwarnings('ignore')  # Suprimă avertismentele pentru claritate

# ===========================
# Funcții pentru Modelul ML
# ===========================

def get_monthly_data(symbol, period="10y"):
    df = yf.download(symbol, period=period, interval="1mo")
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    return df

def train_test_split_ml(df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test

def train_prophet_model(train):
    model = Prophet()
    model.fit(train)
    return model

def predict_prophet_model(model, periods):
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast

def prepare_lstm_data(train, test, scaler, look_back=12):
    # Combine train and test for scaling
    combined = pd.concat([train, test], axis=0)
    scaled = scaler.fit_transform(combined[['y']])

    # Split back
    scaled_train = scaled[:len(train)]
    scaled_test = scaled[len(train):]

    # Create sequences
    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train, look_back)
    X_test, y_test = create_sequences(scaled_test, look_back)

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test, scaled, combined

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_and_improve_model(test, predictions, model_name):
    mape = mean_absolute_percentage_error(test['y'], predictions) * 100
    return mape

def train_and_predict(symbol):
    # Colectare date
    df = get_monthly_data(symbol, '10y')
    train, test = train_test_split_ml(df, test_size=0.2)

    # Dicționar pentru predicții
    predictions_dict = {}

    # Model Prophet
    try:
        prophet_model = train_prophet_model(train)
        prophet_forecast = predict_prophet_model(prophet_model, periods=len(test))
        prophet_pred = prophet_forecast['yhat'][-len(test):].values
        prophet_mape = evaluate_and_improve_model(test, prophet_pred, 'Prophet')
        predictions_dict['Prophet'] = {
            "predictions": prophet_pred.tolist(),
            "mape": prophet_mape
        }
    except Exception as e:
        print(f"Prophet nu a putut fi antrenat: {e}")

    # Model LSTM
    try:
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train, y_train, X_test, y_test, scaled, combined = prepare_lstm_data(train, test, scaler)
        lstm_model = build_lstm_model((X_train.shape[1], 1))
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        lstm_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, callbacks=[early_stop])

        # Predicții
        lstm_pred_scaled = lstm_model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
        lstm_mape = evaluate_and_improve_model(test, lstm_pred.flatten(), 'LSTM')
        predictions_dict['LSTM'] = {
            "predictions": lstm_pred.flatten().tolist(),
            "mape": lstm_mape
        }
    except Exception as e:
        print(f"LSTM nu a putut fi antrenat: {e}")

    # Determină cel mai bun model
    best_model = None
    best_mape = 100

    for model_name, data in predictions_dict.items():
        mape = data["mape"]
        if mape < best_mape:
            best_mape = mape
            best_model = model_name

    # Îmbunătățirea modelului dacă MAPE > 20%
    if best_mape > 20 and best_model is not None:
        print(f"Încercăm să îmbunătățim modelul {best_model} cu MAPE: {best_mape:.2f}%")
        if best_model == 'Prophet':
            try:
                prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=False)
                prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                prophet_model.fit(train)
                prophet_forecast = predict_prophet_model(prophet_model, periods=len(test))
                prophet_pred = prophet_forecast['yhat'][-len(test):].values
                prophet_mape = evaluate_and_improve_model(test, prophet_pred, 'Prophet')
                predictions_dict['Prophet'] = {
                    "predictions": prophet_pred.tolist(),
                    "mape": prophet_mape
                }
                if prophet_mape < best_mape:
                    best_mape = prophet_mape
                    best_model = 'Prophet'
            except Exception as e:
                print(f"Prophet îmbunătățit nu a putut fi antrenat: {e}")

        elif best_model == 'LSTM':
            try:
                # Re-antrenăm LSTM cu mai multe epoci
                lstm_model = build_lstm_model((X_train.shape[1],1))
                lstm_model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, callbacks=[early_stop])
                lstm_pred_scaled = lstm_model.predict(X_test)
                lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
                lstm_mape = evaluate_and_improve_model(test, lstm_pred.flatten(), 'LSTM')
                predictions_dict['LSTM'] = {
                    "predictions": lstm_pred.flatten().tolist(),
                    "mape": lstm_mape
                }
                if lstm_mape < best_mape:
                    best_mape = lstm_mape
                    best_model = 'LSTM'
            except Exception as e:
                print(f"LSTM îmbunătățit nu a putut fi antrenat: {e}")

    # Selectează cel mai bun model cu MAPE <= 20%
    selected_model = None
    if best_mape <= 20:
        selected_model = best_model
    else:
        print(f"Nu am reușit să atingem o MAPE de 20%. Cel mai bun model are o MAPE de {best_mape:.2f}%.")

    # Generarea forecast-ului pentru următorii 10 ani
    forecast_dates = []
    forecast_prices = []

    if selected_model:
        if selected_model == 'Prophet':
            try:
                total_periods = 120  # 10 ani * 12 luni
                forecast_future = predict_prophet_model(prophet_model, periods=total_periods)
                forecast_dates = forecast_future['ds'].dt.strftime('%Y-%m').tolist()[-total_periods:]
                forecast_prices = forecast_future['yhat'].tolist()[-total_periods:]
            except Exception as e:
                print(f"Prophet nu a putut genera forecast-ul: {e}")
        elif selected_model == 'LSTM':
            try:
                # Previne dependențele pe termen lung prin utilizarea ultimelor date din train+test
                last_sequence = scaled[-12:]  # 12 luni pentru look_back=12
                lstm_forecast = []
                for _ in range(120):
                    last_sequence_reshaped = last_sequence.reshape((1, 12, 1))
                    next_pred_scaled = lstm_model.predict(last_sequence_reshaped)
                    next_pred = scaler.inverse_transform(next_pred_scaled)[0,0]
                    lstm_forecast.append(next_pred)
                    # Actualizează secvența
                    next_scaled = scaler.transform(np.array([[next_pred]]))
                    last_sequence = np.append(last_sequence[1:], next_scaled, axis=0)
                forecast_prices = lstm_forecast
                last_date = df['ds'].max()
                forecast_dates = [(last_date + pd.DateOffset(months=i+1)).strftime('%Y-%m') for i in range(120)]
            except Exception as e:
                print(f"LSTM nu a putut genera forecast-ul: {e}")

    # Pregătirea datelor istorice
    historical_dates = df['ds'].dt.strftime('%Y-%m').tolist()
    historical_prices = df['y'].tolist()

    # Pregătirea datelor de predicție
    test_dates = test['ds'].dt.strftime('%Y-%m').tolist()
    test_actual = test['y'].tolist()
    test_predicted = predictions_dict.get(selected_model, {}).get("predictions", [])

    # Pregătirea datelor de forecast
    forecast_data = {
        "dates": forecast_dates,
        "prices": forecast_prices
    }

    # Pregătirea răspunsului
    response = {
        "historical": {
            "dates": historical_dates,
            "prices": historical_prices
        },
        "test": {
            "dates": test_dates,
            "actual": test_actual,
            "predicted": test_predicted
        },
        "forecast": forecast_data,
        "best_model": selected_model,
        "best_mape": best_mape
    }

    return response

# ===========================
# Endpoint-uri Existente
# ===========================

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    app.logger.info(f"Fetching stock data for symbol: {symbol}")
    try:
        df = yf.download(symbol, period="1d", interval="1d")
        if df.empty:
            app.logger.warning(f"No data found for symbol: {symbol}")
            return jsonify({"error": "No data found"}), 404

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
        exchange_info = "NasdaqGS - Nasdaq Real Time Price • USD"
        compare_link = f"/compare/{symbol}"

        app.logger.info(f"Data fetched successfully for symbol: {symbol}")
        return jsonify({
            "companyName": company_name,
            "lastClosePrice": last_close_price,
            "lastCloseDate": last_close_date,
            "openPrice": open_price,
            "highPrice": high_price,
            "lowPrice": low_price,
            "volume": volume,
            "exchangeInfo": exchange_info,
            "compareLink": compare_link,
            "previousClose": stock_info.get('regularMarketPreviousClose', 'N/A'),
            "marketCap": stock_info.get('marketCap', 'N/A'),
            "open": stock_info.get('regularMarketOpen', 'N/A'),
            "beta": stock_info.get('beta', 'N/A'),
            "bid": stock_info.get('bid', 'N/A'),
            "bidSize": stock_info.get('bidSize', 'N/A'),
            "trailingPE": stock_info.get('trailingPE', 'N/A'),
            "ask": stock_info.get('ask', 'N/A'),
            "askSize": stock_info.get('askSize', 'N/A'),
            "trailingEps": stock_info.get('trailingEps', 'N/A'),
            "regularMarketDayLow": stock_info.get('regularMarketDayLow', 'N/A'),
            "regularMarketDayHigh": stock_info.get('regularMarketDayHigh', 'N/A'),
            "fiftyTwoWeekLow": stock_info.get('fiftyTwoWeekLow', 'N/A'),
            "fiftyTwoWeekHigh": stock_info.get('fiftyTwoWeekHigh', 'N/A'),
            "dividendRate": stock_info.get('dividendRate', 'N/A'),
            "dividendYield": stock_info.get('dividendYield', 'N/A'),
            "regularMarketVolume": stock_info.get('regularMarketVolume', 'N/A'),
            "exDividendDate": stock_info.get('exDividendDate', 'N/A'),
            "averageVolume": stock_info.get('averageVolume', 'N/A'),
            "targetMeanPrice": stock_info.get('targetMeanPrice', 'N/A'),
            "enterpriseValue": stock_info.get('enterpriseValue', 'N/A'),
            "priceToBook": stock_info.get('priceToBook', 'N/A'),
            "priceToSalesTrailing12Months": stock_info.get('priceToSalesTrailing12Months', 'N/A'),
            "enterpriseToEbitda": stock_info.get('enterpriseToEbitda', 'N/A'),
            "operatingMargins": stock_info.get('operatingMargins', 'N/A'),
            "grossMargins": stock_info.get('grossMargins', 'N/A'),
            "profitMargins": stock_info.get('profitMargins', 'N/A'),
            "earningsGrowth": stock_info.get('earningsGrowth', 'N/A'),
            "sector": stock_info.get('sector', 'N/A'),
            "industry": stock_info.get('industry', 'N/A'),
            "totalRevenue": stock_info.get('totalRevenue', 'N/A'),
            "revenueGrowth": stock_info.get('revenueGrowth', 'N/A'),
            "operatingCashflow": stock_info.get('operatingCashflow', 'N/A')
        })
    except Exception as e:
        app.logger.error(f"Error fetching stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/news')
def get_stock_news(symbol):
    app.logger.info(f"Fetching stock news for symbol: {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        news_data = ticker.news

        if not news_data:
            app.logger.warning(f"No news found for symbol: {symbol}")
            return jsonify({"error": "No news found"}), 404

        news_items = [{
            "title": item.get('title'),
            "link": item.get('link'),
            "publisher": item.get('publisher'),
            "publishedDate": datetime.fromtimestamp(item.get('providerPublishTime')).isoformat() if item.get('providerPublishTime') else 'N/A'
        } for item in news_data]

        app.logger.info(f"Stock news fetched successfully for symbol: {symbol}")
        return jsonify(news_items)
    except Exception as e:
        app.logger.error(f"Error fetching stock news for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/intraday')
def get_intraday_stock_data(symbol):
    app.logger.info(f"Fetching intraday stock data for symbol: {symbol}")
    try:
        df = yf.download(symbol, period="1d", interval="1m")
        if df.empty:
            app.logger.warning(f"No intraday data found for symbol: {symbol}")
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
    period = request.args.get('period', '1d')  # Retrieve "period" parameter from query string
    app.logger.info(f"Fetching historical stock data for symbol: {symbol} with period: {period}")
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            app.logger.warning(f"No historical data found for symbol: {symbol} with period: {period}")
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
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    app.logger.info(f"Fetching test stock data for symbol: {symbol} from {start_date} to {end_date}")
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            app.logger.warning(f"No data found for symbol: {symbol} from {start_date} to {end_date}")
            return jsonify({"error": f"No data found for {symbol} from {start_date} to {end_date}"}), 404

        # Obtain the most recent data (i.e., the latest entry in the historical data)
        most_recent_hist = stock.history(period='1d')  # Get the most recent data
        latest_data = most_recent_hist.iloc[-1] if not most_recent_hist.empty else None

        if latest_data is not None:
            last_close_price = latest_data['Close']
            last_close_date = latest_data.name.isoformat()
        else:
            last_close_price = 'N/A'
            last_close_date = 'N/A'

        # Get last dividend information
        info = stock.info
        last_dividend_value = info.get('dividendRate', 'N/A')
        last_dividend_date = datetime.fromtimestamp(info['exDividendDate']).isoformat() if info.get('exDividendDate') else 'N/A'

        # Prepare response data
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

@app.route('/api/simulate-trading-strategy', methods=['POST'])
def simulate_trading_strategy():
    data = request.get_json()
    tickers = data.get('tickers', ['AAPL'])  # Dacă nu este furnizat niciun simbol, implicit este 'AAPL'

    app.logger.info(f"Simulating trading strategy for tickers: {tickers}")

    try:
        # Pasul 2: Preluarea datelor
        data = yf.download(tickers, start='2000-01-01', end='2024-09-19')
        close = data['Close'].dropna()

        if close.empty:
            app.logger.warning(f"No data downloaded for tickers: {tickers}")
            return jsonify({"error": "No data downloaded"}), 404

        close = close.to_frame()
        close.columns = tickers

        # Prelucrarea și ingineria caracteristicilor
        ticker = tickers[0]
        close[f'{ticker}_Return'] = close[ticker].pct_change() * 100
        close[f'{ticker}_Open'] = close[ticker].shift(1)
        close[f'{ticker}_High'] = close[[ticker, f'{ticker}_Open']].max(axis=1)
        close[f'{ticker}_Low'] = close[[ticker, f'{ticker}_Open']].min(axis=1)
        close[f'{ticker}_High_Low_Range'] = (close[f'{ticker}_High'] - close[f'{ticker}_Low']) / close[f'{ticker}_Open'] * 100
        close[f'{ticker}_Open_Close_Range'] = (close[ticker] - close[f'{ticker}_Open']) / close[f'{ticker}_Open'] * 100

        close[f'{ticker}_Trend'] = close[ticker].rolling(window=5).mean()
        close[f'{ticker}_Volatility'] = close[f'{ticker}_Return'].rolling(window=5).std()
        close.dropna(inplace=True)

        # Crearea variabilei țintă
        close['Target'] = close[f'{ticker}_Open_Close_Range'].shift(-1)
        close['Target_Label'] = close['Target'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        close.dropna(inplace=True)

        # Împărțirea datelor
        features = close[[f'{ticker}_Return', f'{ticker}_High_Low_Range', f'{ticker}_Open_Close_Range', f'{ticker}_Trend', f'{ticker}_Volatility']]
        labels = close['Target_Label']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, shuffle=False)

        # Normalizare și antrenarea modelului
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Evaluarea modelului
        y_pred = clf.predict(scaler.transform(X_test))
        test_results = X_test.copy()
        test_results['Actual'] = y_test
        test_results['Predicted'] = y_pred

        # Simularea strategiilor de return
        test_results['Predicted_Shifted'] = test_results['Predicted'].shift(1).fillna(0)
        test_results['Strategy_Return'] = test_results['Predicted_Shifted'] * (close[f'{ticker}_Open_Close_Range'].iloc[X_train.shape[0]:].values)
        test_results['Cumulative_Strategy_Return'] = test_results['Strategy_Return'].cumsum()
        test_results['Cumulative_AAPL_Return'] = close[f'{ticker}_Open_Close_Range'].iloc[X_train.shape[0]:].cumsum()

        # Pregătirea datelor pentru răspuns
        response_data = {
            "cumulativeStrategyReturn": test_results['Cumulative_Strategy_Return'].tolist(),
            "cumulativeAAPLReturn": test_results['Cumulative_AAPL_Return'].tolist(),
            "predictions": test_results[['Predicted', 'Actual', 'Strategy_Return']].tail().to_dict(orient='records')
        }

        app.logger.info("Trading strategy simulation successful.")
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error simulating trading strategy: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/machine-learning-calculation', methods=['POST'])
def machine_learning_calculation():
    data = request.get_json()
    tickers = data.get('tickers', ['AAPL'])  # Utilizează AAPL ca simbol implicit

    app.logger.info(f"Starting machine learning calculation for tickers: {tickers}")

    try:
        # Obține datele pentru tickere
        data = yf.download(tickers, start='2000-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        close = data['Close'].dropna()

        if close.empty:
            app.logger.warning(f"No data found for tickers: {tickers}")
            return jsonify({"error": "No data found"}), 404

        close = close.to_frame()
        close.columns = tickers

        # Prelucrarea datelor
        ticker = tickers[0]
        close[f'{ticker}_Return'] = close[ticker].pct_change() * 100

        # Crearea caracteristicilor suplimentare
        close[f'{ticker}_Open'] = close[ticker].shift(1)
        close[f'{ticker}_High'] = close[[ticker, f'{ticker}_Open']].max(axis=1)
        close[f'{ticker}_Low'] = close[[ticker, f'{ticker}_Open']].min(axis=1)
        close[f'{ticker}_High_Low_Range'] = (close[f'{ticker}_High'] - close[f'{ticker}_Low']) / close[f'{ticker}_Open'] * 100
        close[f'{ticker}_Open_Close_Range'] = (close[ticker] - close[f'{ticker}_Open']) / close[f'{ticker}_Open'] * 100
        close[f'{ticker}_Trend'] = close[ticker].rolling(window=5).mean()  # 5-day moving average
        close[f'{ticker}_Volatility'] = close[f'{ticker}_Return'].rolling(window=5).std()  # 5-day volatility

        # Dropping rows with NaN values
        close.dropna(inplace=True)

        # Crearea variabilei țintă
        close['Target'] = close[f'{ticker}_Open_Close_Range'].shift(-1)

        # Crearea etichetelor pentru țintă
        close['Target_Label'] = close['Target'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        close.dropna(inplace=True)

        # Împărțirea datelor
        features = close[[f'{ticker}_Return', f'{ticker}_High_Low_Range', f'{ticker}_Open_Close_Range', f'{ticker}_Trend', f'{ticker}_Volatility']]
        labels = close['Target_Label']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, shuffle=False)

        # Normalizarea și antrenarea modelului
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Evaluarea modelului
        y_pred = clf.predict(scaler.transform(X_test))
        predictions = [int(pred) for pred in y_pred]  # Conversie la int

        # Pregătirea datelor pentru răspuns
        response_data = {
            "predictions": predictions,
            "actual": [int(actual) for actual in y_test.tolist()]  # Conversie la int
        }

        app.logger.info("Machine learning calculation successful.")
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error during machine learning calculation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===========================
# Noua Funcționalitate: Predicții ML
# ===========================

@app.route('/api/predict_stock/<symbol>', methods=['GET'])
def predict_stock(symbol):
    app.logger.info(f"Starting prediction for symbol: {symbol}")
    try:
        prediction_result = train_and_predict(symbol)
        app.logger.info(f"Prediction for {symbol} completed successfully.")
        return jsonify(prediction_result)
    except Exception as e:
        app.logger.error(f"Error during prediction for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ===========================
# Endpoint-ul Principal
# ===========================

@app.route('/')
def index():
    return render_template('index.html')

# ===========================
# Rularea Aplicației
# ===========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

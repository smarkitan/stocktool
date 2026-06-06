from flask import Flask, jsonify, render_template, request 
import yfinance as yf
from flask_cors import CORS
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import json
import requests
import urllib.parse
import urllib.request

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://stefanstocktool.netlify.app"}})
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}

session = requests.Session()
session.headers.update(HEADERS)

def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_unix(date_string):
    if not date_string:
        return None
    return int(datetime.fromisoformat(date_string).timestamp())


def _format_iso(ts):
    return datetime.utcfromtimestamp(int(ts)).isoformat()


def yahoo_chart(symbol, *, range_value=None, interval="1d", start_date=None, end_date=None):
    """Fetch public quote/history data from Yahoo Chart API without yfinance crumb negotiation.

    Render/yfinance is currently rate-limited by Yahoo (429). The public chart endpoint
    still returns data reliably with browser headers and crumb=none.
    """
    params = {"interval": interval, "events": "history", "crumb": "none"}
    if range_value:
        params["range"] = range_value
    else:
        period1 = _to_unix(start_date)
        period2 = _to_unix(end_date) or int(datetime.utcnow().timestamp())
        if period1 is None:
            raise ValueError("start_date is required")
        params["period1"] = str(period1)
        params["period2"] = str(period2)

    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(symbol.upper()) + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={**HEADERS, "Accept": "application/json,text/plain,*/*"})
    with urllib.request.urlopen(req, timeout=20) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    chart = data.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("No Yahoo chart data found")
    return results[0]


def chart_to_rows(result, date_format="iso"):
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    rows = []
    for idx, ts in enumerate(timestamps):
        close = _safe_float(closes[idx] if idx < len(closes) else None)
        if close is None:
            continue
        if date_format == "intraday":
            dt = datetime.fromtimestamp(ts).strftime('%b %d, %I:%M %p')
        elif date_format == "date":
            dt = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
        else:
            dt = _format_iso(ts)
        rows.append({
            "datetime": dt,
            "close": close,
            "open": _safe_float(opens[idx] if idx < len(opens) else None, close),
            "high": _safe_float(highs[idx] if idx < len(highs) else None, close),
            "low": _safe_float(lows[idx] if idx < len(lows) else None, close),
            "volume": _safe_int(volumes[idx] if idx < len(volumes) else None, 0),
        })
    return rows


def rows_payload(rows):
    return {
        "datetime": [row["datetime"] for row in rows],
        "close": [row["close"] for row in rows],
        "open": [row["open"] for row in rows],
        "high": [row["high"] for row in rows],
        "low": [row["low"] for row in rows],
        "volume": [row["volume"] for row in rows],
    }


def build_quote_payload(symbol):
    result = yahoo_chart(symbol, range_value="5d", interval="1d")
    meta = result.get("meta") or {}
    rows = chart_to_rows(result)
    if not rows:
        raise RuntimeError("No data found")
    latest = rows[-1]
    company_name = meta.get("longName") or meta.get("shortName") or symbol.upper()
    exchange_name = meta.get("fullExchangeName") or meta.get("exchangeName") or "N/A"
    currency = meta.get("currency") or "USD"
    price = _safe_float(meta.get("regularMarketPrice"), latest["close"])
    symbol_key = symbol.upper()
    return {
        "companyName": company_name,
        # The existing React frontend expects the detailed /api/stock payload to
        # mirror yfinance's old multi-index shape: price fields are objects keyed
        # by symbol (for example {"AAPL": 307.34}). Keep that contract intact.
        "lastClosePrice": {symbol_key: price},
        "lastCloseDate": _format_iso(meta.get("regularMarketTime")) if meta.get("regularMarketTime") else latest["datetime"],
        "openPrice": {symbol_key: latest["open"]},
        "highPrice": {symbol_key: latest["high"]},
        "lowPrice": {symbol_key: latest["low"]},
        "volume": {symbol_key: latest["volume"]},
        "exchangeInfo": f"{exchange_name} • {currency}",
        "compareLink": f"/compare/{symbol.upper()}",
        "previousClose": _safe_float(meta.get("chartPreviousClose"), "N/A"),
        "marketCap": meta.get("marketCap", "N/A"),
        "open": latest["open"],
        "beta": "N/A",
        "bid": "N/A",
        "bidSize": "N/A",
        "trailingPE": "N/A",
        "ask": "N/A",
        "askSize": "N/A",
        "trailingEps": "N/A",
        "regularMarketDayLow": _safe_float(meta.get("regularMarketDayLow"), latest["low"]),
        "regularMarketDayHigh": _safe_float(meta.get("regularMarketDayHigh"), latest["high"]),
        "fiftyTwoWeekLow": meta.get("fiftyTwoWeekLow", "N/A"),
        "fiftyTwoWeekHigh": meta.get("fiftyTwoWeekHigh", "N/A"),
        "dividendRate": "N/A",
        "dividendYield": "N/A",
        "regularMarketVolume": meta.get("regularMarketVolume", latest["volume"]),
        "exDividendDate": "N/A",
        "averageVolume": "N/A",
        "targetMeanPrice": "N/A",
        "enterpriseValue": "N/A",
        "priceToBook": "N/A",
        "priceToSalesTrailing12Months": "N/A",
        "enterpriseToEbitda": "N/A",
        "operatingMargins": "N/A",
        "grossMargins": "N/A",
        "profitMargins": "N/A",
        "earningsGrowth": "N/A",
        "sector": "N/A",
        "industry": "N/A",
        "totalRevenue": "N/A",
        "revenueGrowth": "N/A",
        "operatingCashflow": "N/A",
    }


@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    app.logger.info(f"Fetching stock data for symbol: {symbol}")
    try:
        return jsonify(build_quote_payload(symbol))
    except Exception as e:
        app.logger.error(f"Error fetching stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/news')
def get_stock_news(symbol):
    app.logger.info(f"Fetching stock news for symbol: {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        raw_news_data = ticker.news

        if not raw_news_data:
            app.logger.warning(f"No news found for symbol: {symbol}")
            return jsonify([])

        # Funcție pentru a converti data ISO în timestamp UNIX
        def convert_to_timestamp(iso_str):
            try:
                return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())
            except:
                return None

        # Extragem corect datele din structura nested
        news_items = []
        for item in raw_news_data:
            content = item.get("content", {})
            news_items.append({
                "title": content.get("title"),
                "link": content.get("canonicalUrl", {}).get("url"),
                "publisher": content.get("provider", {}).get("displayName"),
                "publishedDate": convert_to_timestamp(content.get("pubDate"))
            })

        app.logger.info(f"Stock news fetched successfully for symbol: {symbol}")
        return jsonify(news_items)

    except Exception as e:
        app.logger.error(f"Error fetching stock news for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/intraday')
def get_intraday_stock_data(symbol):
    app.logger.info(f"Fetching intraday stock data for symbol: {symbol}")
    try:
        result = yahoo_chart(symbol, range_value="1d", interval="1m")
        rows = chart_to_rows(result, date_format="intraday")
        if not rows:
            return jsonify({"error": "No intraday data found"}), 404
        return jsonify(rows_payload(rows))
    except Exception as e:
        app.logger.error(f"Error fetching intraday data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/intraday-short', methods=['GET'])
def get_intraday_short_data(symbol):
    app.logger.info(f"Fetching short intraday stock data for symbol: {symbol}")
    try:
        result = yahoo_chart(symbol, range_value="1d", interval="1m")
        rows = chart_to_rows(result, date_format="intraday")
        if not rows:
            return jsonify({"error": "No intraday data found"}), 404
        payload = rows_payload(rows)
        return jsonify({"datetime": payload["datetime"], "close": payload["close"], "open": payload["open"]})
    except Exception as e:
        app.logger.error(f"Error fetching short intraday data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/historical', methods=['GET'])
def get_stock_historical_data(symbol):
    period = request.args.get('period', '1mo')
    app.logger.info(f"Fetching historical stock data for symbol: {symbol} with period: {period}")
    try:
        result = yahoo_chart(symbol, range_value=period, interval="1d")
        rows = chart_to_rows(result, date_format="date")
        if not rows:
            return jsonify({"error": "No historical data found"}), 404
        return jsonify(rows_payload(rows))
    except Exception as e:
        app.logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/test_stock_data/<symbol>', methods=['GET'])
def test_stock_data_route(symbol):
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    app.logger.info(f"Fetching test stock data for symbol: {symbol} from {start_date} to {end_date}")
    try:
        result = yahoo_chart(symbol, start_date=start_date, end_date=end_date, interval="1d")
        rows = chart_to_rows(result, date_format="date")
        if not rows:
            return jsonify({"error": f"No data found for {symbol} from {start_date} to {end_date}"}), 404
        latest = build_quote_payload(symbol)
        data = rows_payload(rows)
        data.update({
            "lastDividendValue": "N/A",
            "lastDividendDate": "N/A",
            "lastClosePrice": latest["lastClosePrice"].get(symbol.upper()) if isinstance(latest["lastClosePrice"], dict) else latest["lastClosePrice"],
            "lastCloseDate": latest["lastCloseDate"],
        })
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching test stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route('/api/stock/simple/<symbol>', methods=['GET'])
def get_simple_stock_data(symbol):
    try:
        data = build_quote_payload(symbol)
        symbol_key = symbol.upper()
        price = data["lastClosePrice"].get(symbol_key) if isinstance(data["lastClosePrice"], dict) else data["lastClosePrice"]
        return jsonify({
            "company": data["companyName"],
            "companyName": data["companyName"],
            "symbol": symbol_key,
            "lastClosePrice": price,
            "previousClose": data["previousClose"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



########## New Machine Learning Prediction Route ########



@app.route('/api/stock/<symbol>/predict', methods=['GET'])
def predict_stock(symbol):
    # Retrieve query parameters with default values
    amount = float(request.args.get('amount', 1000))
    currency = request.args.get('currency', 'USD').upper()
    years = int(request.args.get('years', 5))

    try:
        # Fetch historical stock data with maximum available period
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='max')
        if hist.empty:
            app.logger.warning(f"No historical data found for symbol: {symbol}")
            return jsonify({"error": "No historical data found"}), 404

        hist = hist.reset_index()
        hist['Date'] = pd.to_datetime(hist['Date'])

        # Get the latest available date and closing price
        latest_date = hist['Date'].max()
        latest_price = hist.loc[hist['Date'] == latest_date, 'Close'].values[0]

        # Calculate the date 'years' ago
        past_date = latest_date - pd.DateOffset(years=years)
        past_data = hist[hist['Date'] <= past_date]

        if past_data.empty:
            app.logger.warning(f"Not enough historical data to get price {years} years ago for symbol: {symbol}")
            price_past = 'N/A'
        else:
            closest_past_date = past_data['Date'].max()
            price_past = hist.loc[hist['Date'] == closest_past_date, 'Close'].values[0]

        # Prepare data for the Linear Regression model
        hist_sorted = hist.sort_values('Date')
        hist_sorted['Date_ordinal'] = hist_sorted['Date'].map(datetime.toordinal)
        X = hist_sorted['Date_ordinal'].values.reshape(-1, 1)
        y = hist_sorted['Close'].values

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future prices for each year
        future_dates = []
        future_prices = []
        for i in range(1, years + 1):
            year_future_date = latest_date + pd.DateOffset(years=i)
            year_future_date_ordinal = year_future_date.toordinal()
            price_pred = model.predict(np.array([[year_future_date_ordinal]]))[0]
            price_pred = max(0, price_pred)  # Ensure non-negative prices

            # Currency conversion if needed
            if currency == 'EUR':
                exchange_symbol = 'USDEUR=X'
                exchange_hist = yf.Ticker(exchange_symbol).history(period='1d')
                if exchange_hist.empty:
                    app.logger.warning(f"No exchange rate data found for symbol: {exchange_symbol}")
                    return jsonify({"error": "No exchange rate data found"}), 404
                exchange_rate = exchange_hist['Close'].iloc[-1]
                price_pred *= exchange_rate

            future_dates.append(year_future_date.strftime('%Y-%m-%d'))
            future_prices.append(round(price_pred, 2))

        # Calculate investment details
        num_shares = amount / latest_price
        value_today = num_shares * latest_price
        value_future = num_shares * future_prices[-1]

        # Currency conversion for current prices if needed
        if currency == 'EUR':
            exchange_symbol = 'USDEUR=X'
            exchange_hist = yf.Ticker(exchange_symbol).history(period='1d')
            if not exchange_hist.empty:
                exchange_rate = exchange_hist['Close'].iloc[-1]
                latest_price *= exchange_rate
                value_today *= exchange_rate

        investment = {
            "description": f"Investment Analysis for {symbol.upper()}",
            "price_past": round(price_past, 2) if price_past != 'N/A' else 'N/A',
            "num_shares": round(num_shares, 4),
            "value_today": round(value_today, 2),
            "price_today": round(latest_price, 2),
            "predicted_price_future": future_prices[-1],
            "value_future": round(value_future, 2),
            "num_years": years
        }

        # Prepare data for the chart
        historical_dates = hist_sorted['Date'].dt.strftime('%Y-%m-%d').tolist()
        historical_prices = hist_sorted['Close'].tolist()

        response = {
            "investment": investment,
            "historical": {
                "dates": historical_dates,
                "prices": historical_prices
            },
            "predicted": {
                "dates": future_dates,
                "prices": future_prices
            }
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error predicting stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ... [Machine Learning end] ...

# ....[ML hard cu grafice de prediceie]....#

@app.route('/api/simulate-trading-strategy', methods=['POST'])
def simulate_trading_strategy():
    data = request.get_json()
    tickers = data.get('tickers', ['AAPL'])  # Dacă nu este furnizat niciun simbol, implicit este 'AAPL'

    app.logger.info(f"Simulating trading strategy for tickers: {tickers}")

    try:
        # Pasul 2: Preluarea datelor
        data = yf.download(tickers, start='2000-01-01', end=datetime.today().strftime('%Y-%m-%d'))
        close = data['Close'].dropna()

        if close.empty:
            app.logger.warning(f"No data downloaded for tickers: {tickers}")
            return jsonify({"error": "No data downloaded"}), 404

        # Asigură-te că `close` este un DataFrame
        if isinstance(close, pd.Series):
            close = close.to_frame()
        
        close.columns = tickers  # Ajustează numele coloanelor

        # Convertim indexul `DatetimeIndex` în string
        close.index = close.index.strftime('%Y-%m-%d')

        # Returnăm datele într-un format JSON serializabil
        return jsonify({"message": "Data processed successfully", "data": close.to_dict()}), 200

    except Exception as e:
        app.logger.error(f"Error simulating trading strategy: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to simulate trading strategy: {str(e)}"}), 500



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

# ...............end ML hard ........#



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

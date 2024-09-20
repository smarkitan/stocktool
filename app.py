from flask import Flask, jsonify, render_template, request
import yfinance as yf
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://stefanstocktool.netlify.app"}})  # Allow all origins to access /api/* routes

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
            "publishedDate": item.get('providerPublishTime')
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
        last_dividend_date = info.get('exDividendDate', 'N/A')

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
    tickers = data.get('tickers', ['AAPL'])
    
    app.logger.info(f"Simulating trading strategy for tickers: {tickers}")

    try:
        # Step 2: Data Retrieval
        data = yf.download(tickers, start='2000-01-01', end='2024-09-19')
        close = data['Close'].dropna()
        
        if close.empty:
            app.logger.warning(f"No data downloaded for tickers: {tickers}")
            return jsonify({"error": "No data downloaded"}), 404
        
        close = close.to_frame()
        close.columns = tickers

        # Step 3: Data Preprocessing and Feature Engineering
        close['AAPL_Return'] = close['AAPL'].pct_change() * 100
        close['AAPL_Open'] = close['AAPL'].shift(1)
        close['AAPL_High'] = close[['AAPL', 'AAPL_Open']].max(axis=1)
        close['AAPL_Low'] = close[['AAPL', 'AAPL_Open']].min(axis=1)
        close['AAPL_High_Low_Range'] = (close['AAPL_High'] - close['AAPL_Low']) / close['AAPL_Open'] * 100
        close['AAPL_Open_Close_Range'] = (close['AAPL'] - close['AAPL_Open']) / close['AAPL_Open'] * 100
        
        close['AAPL_Trend'] = close['AAPL'].rolling(window=5).mean()
        close['AAPL_Volatility'] = close['AAPL_Return'].rolling(window=5).std()
        close.dropna(inplace=True)

        # Creating Target Variable
        close['Target'] = close['AAPL_Open_Close_Range'].shift(-1)
        close['Target_Label'] = close['Target'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        close.dropna(inplace=True)

        # Splitting the Data
        features = close[['AAPL_Return', 'AAPL_High_Low_Range', 'AAPL_Open_Close_Range', 'AAPL_Trend', 'AAPL_Volatility']]
        labels = close['Target_Label']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10, shuffle=False)

        # Normalization and Model Training
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)

        # Model Evaluation
        y_pred = clf.predict(scaler.transform(X_test))
        test_results = X_test.copy()
        test_results['Actual'] = y_test
        test_results['Predicted'] = y_pred

        # Simulate Strategy Returns
        test_results['Predicted_Shifted'] = test_results['Predicted'].shift(1).fillna(0)
        test_results['Strategy_Return'] = test_results['Predicted_Shifted'] * (close['AAPL_Open_Close_Range'].iloc[X_train.shape[0]:].values)
        test_results['Cumulative_Strategy_Return'] = test_results['Strategy_Return'].cumsum()
        test_results['Cumulative_AAPL_Return'] = close['AAPL_Open_Close_Range'].iloc[X_train.shape[0]:].cumsum()

        # Prepare response data
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



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

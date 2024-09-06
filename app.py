from flask import Flask, jsonify, render_template, request
import yfinance as yf
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins to access /api/* routes

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    try:
        df = yf.download(symbol, period="1d", interval="1d")
        if df.empty:
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

        return jsonify(news_items)
    except Exception as e:
        app.logger.error(f"Error fetching stock news: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/<symbol>/intraday')
def get_intraday_stock_data(symbol):
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

        return jsonify(intraday_data)

    except Exception as e:
        app.logger.error(f"Error fetching intraday data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock-data-by-date', methods=['GET'])
def fetch_stock_data_by_date():
    symbol = request.args.get('symbol')
    date = request.args.get('date')

    if not symbol or not date:
        return jsonify({'error': 'Symbol and date are required'}), 400

    try:
        # Convert date to datetime object
        start_date = datetime.strptime(date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=1)

        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return jsonify({'error': f'No data found for {symbol} from {date} to {end_date.date()}'}), 404

        # Example of defining start and end
        start = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')

        # Process stock data
        data = {
            'datetime': hist.index.strftime('%Y-%m-%d').tolist(),
            'close': hist['Close'].tolist(),
            'open': hist['Open'].tolist(),
            'high': hist['High'].tolist(),
            'low': hist['Low'].tolist()
        }

        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching stock data for {symbol} on {date}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/stock/<symbol>/historical', methods=['GET'])
def get_stock_historical_data(symbol):
    period = request.args.get('period', '1d')  # Retrieve "period" parameter from query string
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
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

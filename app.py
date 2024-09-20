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

        response_data = {
            "historicalData": hist.reset_index().to_dict(orient='records'),
            "lastClosePrice": last_close_price,
            "lastCloseDate": last_close_date,
            "lastDividendValue": last_dividend_value,
            "lastDividendDate": last_dividend_date
        }

        app.logger.info(f"Test stock data fetched successfully for symbol: {symbol}")
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error fetching test stock data for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_investment():
    data = request.get_json()
    symbol = data.get('symbol')
    investment_amount = data.get('investmentAmount')
    purchase_date = data.get('purchaseDate')
    end_date = data.get('endDate')

    app.logger.info(f"Simulating investment for symbol: {symbol}, amount: {investment_amount}, from {purchase_date} to {end_date}")

    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=purchase_date, end=end_date)

        if hist.empty:
            app.logger.warning(f"No historical data found for symbol: {symbol} from {purchase_date} to {end_date}")
            return jsonify({"error": "No historical data found"}), 404
        
        # Calculate the number of shares purchased
        purchase_price = hist['Close'].iloc[0]
        num_of_shares = investment_amount / purchase_price

        # Calculate current value
        latest_data = stock.history(period='1d').iloc[-1]
        current_price = latest_data['Close']
        current_value = num_of_shares * current_price

        # Calculate profit and return percentage
        profit = current_value - investment_amount
        return_percentage = (profit / investment_amount) * 100 if investment_amount > 0 else 0

        # Prepare response data
        response_data = {
            "numOfShares": num_of_shares,
            "purchasePrice": purchase_price,
            "currentPrice": current_price,
            "currentValue": current_value,
            "profit": profit,
            "returnPercentage": return_percentage
        }

        app.logger.info(f"Simulation successful for symbol: {symbol}")
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error simulating investment for {symbol}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

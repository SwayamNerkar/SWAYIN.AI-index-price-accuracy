import requests
import pandas as pd
import yfinance as yf

API_KEY = "YOUR API KEY"

def load_live_data(symbol, interval="1m"):
    """
    Fallback yfinance live loader if AlphaVantage limit is reached or market is closed
    """
    ticker = yf.Ticker(symbol)
    
    # yfinance max intraday periods: 1m (7 days), others (60 days)
    max_period = "7d" if interval == "1m" else "60d"
    
    # Try fetching today first
    data = ticker.history(period="1d", interval=interval)
    
    # If today is weekend/closed OR doesn't have enough rows for AI (65), fetch historical max
    if len(data) < 65:
        data = ticker.history(period=max_period, interval=interval)
        
    data.dropna(inplace=True)
    return data

def load_data(symbol="IBM", interval="1min"):
    url = f"https://www.alphavantage.co/query"
    
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": "compact"
    }

    r = requests.get(url, params=params)
    data = r.json()

    key = f"Time Series ({interval})"
    
    if key not in data:
        # Fallback to yfinance if AlphaVantage is unavailable or limit reached
        yf_interval = interval.replace("min", "m") if "min" in interval else "1m"
        return load_live_data(symbol, yf_interval)

    df = pd.DataFrame.from_dict(data[key], orient='index')

    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })

    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df

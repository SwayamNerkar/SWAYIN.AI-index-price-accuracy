from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_indicators(df):
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    
    # MACD
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    return df
    
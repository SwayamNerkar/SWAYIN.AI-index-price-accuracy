import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

scaler = MinMaxScaler()

def prepare_data(df):
    data = df[['Close']].values
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaled

def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(60,1)))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model

def predict(model, scaled_data):
    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1,60,1)

    pred = model.predict(last_60)
    return scaler.inverse_transform(pred)[0][0]
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# Yeni dosyayı oku
df = pd.read_csv('Final_Structured_Data.csv')
features = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']
data_clean = df[features].dropna().reset_index(drop=True)

# Normalizasyon
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_clean)

# LSTM formatına hazırla
timesteps = 10
X_lstm = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])

# Modeli kur
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(timesteps, X_lstm.shape[2])),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(timesteps),
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_lstm.shape[2]))
])
model.compile(optimizer='adam', loss='mse')

# Eğitimi başlat
model.fit(X_lstm, X_lstm, epochs=30, batch_size=32, validation_split=0.1, verbose=2)

# Eğitilen ağırlıkları kaydet
model.save_weights('trained_lstm_model.weights.h5')
print("✅ Eğitim tamamlandı ve model ağırlıkları kaydedildi: trained_lstm_model.weights.h5")

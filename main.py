import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# veri yükleme
df = pd.read_csv("/Users/mirhanmetin/Downloads/final_corrected_clean_normal_training_data.csv")
features = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']
data_clean = df[features].dropna().reset_index(drop=True)

# normalization
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_clean)

# lstm için zaman tanıma
timesteps = 10
X_lstm = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])

# lstm
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(timesteps, X_lstm.shape[2])),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(timesteps),
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_lstm.shape[2]))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_lstm, X_lstm, epochs=30, batch_size=32, validation_split=0.1, verbose=2)

# mse hesaplanması
X_pred = model.predict(X_lstm)
mse = np.mean(np.power(X_lstm - X_pred, 2), axis=(1, 2))
threshold = np.percentile(mse, 95)
print("MSE Eşik Değeri (Threshold):", threshold)
print("MSE Değeri:", mse[195])  # 196. veri noktası için MSE'yi kontrol et
lstm_anomalies = mse > threshold

# isolation forest ve svm
X_ml = pd.DataFrame(scaled_data[:-timesteps], columns=features)
iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_preds = iso_model.fit_predict(X_ml)
svm_model = OneClassSVM(kernel='rbf', nu=0.01, gamma='auto')
svm_preds = svm_model.fit_predict(X_ml)

# sonuçlar
results = X_ml.copy()
results['LSTM_MSE'] = mse
results['LSTM_Anomaly'] = lstm_anomalies
results['ISO_Anomaly'] = iso_preds == -1
results['SVM_Anomaly'] = svm_preds == -1
# lstm + svm veya lstm + isolation forest şeklinde Final_Anomaly kararı
results['Final_Anomaly'] = ((results['LSTM_Anomaly'] & results['SVM_Anomaly']) |
                            (results['LSTM_Anomaly'] & results['ISO_Anomaly']))

# mantıksal kısıtlamalar/uyarılar
def logical_check(row):
    alerts = []
    left = row['left_foot_pressure']
    right = row['right_foot_pressure']
    core = row['core_stability']
    mse = row['LSTM_MSE']

    if left + right > 1.2:
        alerts.append("Toplam basınç yüksek!")
    # dengesizlik uyarı sistemi
    diff = abs(left - right)
    if diff > 0.4:
        alerts.append("Ayaklar arası ciddi dengesizlik!")
    elif diff > 0.2:
        alerts.append("Ayaklar arası dengesizlik var!")

    # basınç uyarı sistemi
    if left < 0.3:
        alerts.append("Sol ayak az basıyor!")
    if right < 0.3:
        alerts.append("Sağ ayak az basıyor!")

    if core < 0.4:
        alerts.append("Stabilite düşük!")
    if mse > threshold:
        alerts.append("Öğrenilmemiş (mse)")
    if not alerts:
        alerts.append("Fiziksel parametreler normal")
    return " | ".join(alerts)

results['Logic_Alert'] = results.apply(logical_check, axis=1)

# grafik
plt.ion()
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_title("Anomaly Detection", fontsize=14)
ax.set_xlabel("Left Foot Pressure")
ax.set_ylabel("Right Foot Pressure")
ax.grid(True)

for i in range(len(results)):
    row = results.iloc[i]
    x = row['left_foot_pressure']
    y = row['right_foot_pressure']

    if row['Final_Anomaly']:
        color = 'red'
    elif "UYARI!" in row['Logic_Alert']:
        color = 'orange'
    else:
        color = 'green'

    ax.scatter(x, y, color=color, edgecolors='k', s=80)
    ax.annotate(str(i + 1), (x + 0.005, y + 0.005), fontsize=7)

    print(f"\n{i + 1}. DATA POİNT")
    print(f"Left: {x:.2f}, Right: {y:.2f}, Core: {row['core_stability']:.2f}")
    print(f"LSTM Anomaly (True/False): {lstm_anomalies[i]}")
    print(f"ISO Anomaly (True/False): {row['ISO_Anomaly']}")
    print(f"SVM Anomaly (True/False): {row['SVM_Anomaly']}")
    print(f"Final Anomaly: {row['Final_Anomaly']}, MSE: {row['LSTM_MSE']:.4f}")
    print(f"Logic Check: {row['Logic_Alert']}")
    plt.pause(0.05)

plt.ioff()
plt.show()
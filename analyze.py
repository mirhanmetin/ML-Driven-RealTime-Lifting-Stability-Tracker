import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lstm_model import build_lstm_autoencoder
from anomaly_detection import run_isolation_forest, run_oneclass_svm
from logic_rules import logical_check
import matplotlib.pyplot as plt
import os

def run_analysis():
    filepath = "./final_corrected_clean_normal_training_data.csv"
    features = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']

    df = pd.read_csv(filepath)
    data_clean = df[features].dropna().reset_index(drop=True)

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_clean)

    timesteps = 10
    if len(scaled_data) < timesteps:
        raise ValueError("Not enough data for LSTM processing.")

    # Create time series
    X_lstm = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])

    # Build and train LSTM Autoencoder
    model = build_lstm_autoencoder(timesteps, X_lstm.shape[2])
    model.fit(X_lstm, X_lstm, epochs=1, batch_size=32, validation_split=0.1, verbose=0)

    # Predict & calculate MSE
    X_pred = model.predict(X_lstm)
    mse = np.mean(np.power(X_lstm - X_pred, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)
    lstm_anomalies = mse > threshold

    # Classical methods
    X_ml = pd.DataFrame(scaled_data[:-timesteps], columns=features)
    iso_anomalies = run_isolation_forest(X_ml)
    svm_anomalies = run_oneclass_svm(X_ml)

    # Anomaly decision
    final_anomalies = ((lstm_anomalies & svm_anomalies) | (lstm_anomalies & iso_anomalies))
    anomaly_detected = final_anomalies.any()

    # Scores
    stability_score = float(1 - np.mean(np.abs(X_ml['left_foot_pressure'] - X_ml['right_foot_pressure'])))
    balance_std = np.std(X_ml['left_foot_pressure'] - X_ml['right_foot_pressure'])
    balance_score = float(max(0, 1 - balance_std))

    # Suggestions
    suggestions = []
    if anomaly_detected:
        suggestions.append({
            'type': 'anomaly',
            'message': 'Anomaly detected!',
            'details': ['Check foot placement', 'Improve core stability']
        })
    if stability_score < 0.7:
        suggestions.append({
            'type': 'stability',
            'message': 'Low stability detected.',
            'details': ['Strengthen your core', 'Practice balance drills']
        })
    if balance_score < 0.7:
        suggestions.append({
            'type': 'balance',
            'message': 'Balance issue detected.',
            'details': ['Work on symmetry', 'Focus on even weight distribution']
        })
    if not suggestions:
        suggestions.append({
            'type': 'form',
            'message': 'Good form!',
            'details': ['Keep it up!']
        })

    # 📊 Plot oluştur
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Anomaly Detection", fontsize=14)
    ax.set_xlabel("Left Foot Pressure")
    ax.set_ylabel("Right Foot Pressure")
    ax.grid(True)

    for i in range(len(X_ml)):
        x = X_ml.iloc[i]['left_foot_pressure']
        y = X_ml.iloc[i]['right_foot_pressure']
        color = 'red' if final_anomalies[i] else 'green'
        ax.scatter(x, y, color=color, edgecolors='k', s=80)

    os.makedirs('static', exist_ok=True)
    plot_path = 'static/anomaly_plot.png'
    plt.savefig(plot_path)
    plt.close(fig)

    return {
        'classification_confidence': 0.85,
        'stability_score': stability_score,
        'balance_score': balance_score,
        'anomaly_detected': bool(anomaly_detected),
        'plot_url': f'/{plot_path}',
        'suggestions': suggestions
    }

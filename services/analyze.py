from models import db, SensorData, PerformanceMetrics, Feedback
import uuid
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from services.lstm_model import build_lstm_autoencoder
from services.anomaly_detection import run_isolation_forest, run_oneclass_svm
from services.logic_rules import logical_check
from utils.socket_logger import SocketIOCallback


def run_analysis_realtime(session_id, df, socketio):
    features = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']
    data_clean = df[features].dropna().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_clean)

    timesteps = 10
    if len(scaled_data) < timesteps:
        socketio.emit('analysis_error', {'message': 'Not enough data for LSTM analysis.'}, room=session_id)
        return

    X_lstm = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])

    # Build and fit model directly here
    model = build_lstm_autoencoder(timesteps, X_lstm.shape[2])
    model.fit(
    X_lstm, X_lstm,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=0,
    callbacks=[SocketIOCallback(socketio, session_id)]
)


    X_pred = model.predict(X_lstm)
    mse = np.mean(np.power(X_lstm - X_pred, 2), axis=(1, 2))
    threshold = np.percentile(mse, 95)
    lstm_anomalies = mse > threshold

    X_ml = pd.DataFrame(scaled_data[:-timesteps], columns=features)
    iso_anomalies = run_isolation_forest(X_ml)
    svm_anomalies = run_oneclass_svm(X_ml)

    results = X_ml.copy()
    results['LSTM_MSE'] = mse
    results['LSTM_Anomaly'] = lstm_anomalies
    results['ISO_Anomaly'] = iso_anomalies
    results['SVM_Anomaly'] = svm_anomalies
    results['Final_Anomaly'] = ((results['LSTM_Anomaly'] & results['SVM_Anomaly']) |
                                (results['LSTM_Anomaly'] & results['ISO_Anomaly']))

    results['Logic_Alert'] = results.apply(lambda row: logical_check(row, threshold), axis=1)

    for i, row in results.iterrows():
        socketio.emit('datapoint_feedback', {
            'index': i + 1,
            'left': float(row['left_foot_pressure']),
            'right': float(row['right_foot_pressure']),
            'core': float(row['core_stability']),
            'mse': float(row['LSTM_MSE']),
            'lstm_anomaly': bool(row['LSTM_Anomaly']),
            'iso_anomaly': bool(row['ISO_Anomaly']),
            'svm_anomaly': bool(row['SVM_Anomaly']),
            'final_anomaly': bool(row['Final_Anomaly']),
            'logic_alert': row['Logic_Alert'],
        }, room=session_id)
        time.sleep(0.05)

    socketio.emit('analysis_complete', {'message': 'Analysis complete!'}, room=session_id)

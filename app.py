from flask import Flask, request, jsonify, render_template
import logging
import traceback
import os

from lstm_model import build_lstm_autoencoder
from anomaly_detection import run_isolation_forest, run_oneclass_svm
from logic_rules import logical_check
from analyze import run_analysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Log yapılandırması
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ML Sistemi başlat
logger.info("Initializing anomaly detection system...")

try:
    scaler = MinMaxScaler()
    timesteps = 10
    n_features = 3
    logger.debug("Building LSTM Autoencoder model...")
    lstm_model = build_lstm_autoencoder(timesteps, n_features)
    logger.info("LSTM Autoencoder initialized successfully")

    if os.path.exists('trained_lstm_model.h5'):
        logger.info("Loading trained LSTM model weights...")
        lstm_model.load_weights('trained_lstm_model.h5')
        logger.info("LSTM model weights loaded successfully")
    else:
        logger.warning("No pre-trained LSTM model found. Model is untrained.")

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/')
def index():
    logger.info("Rendering static index page")
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Burada hareket türünü alıyoruz (şimdilik kullanmıyoruz ama ileride lazım olabilir)
        data = request.get_json()
        movement_type = data.get('movement_type')

        # Analizi başlatıyoruz
        results = run_analysis()

        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     try:
#         logger.info("Received analysis request")

#         data = request.get_json()
#         filepath = data.get('filepath')
        
#         if not filepath or not os.path.exists(filepath):
#             return jsonify({'status': 'error', 'message': 'Invalid or missing filepath'}), 400
        
#         logger.info(f"Loading data from file: {filepath}")
#         df = pd.read_csv(filepath)

#         expected_columns = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']
#         if not all(col in df.columns for col in expected_columns):
#             return jsonify({'status': 'error', 'message': f"CSV must contain columns: {expected_columns}"}), 400

#         data_clean = df[expected_columns].dropna().reset_index(drop=True)
#         scaled_data = scaler.fit_transform(data_clean)

#         if len(scaled_data) < timesteps:
#             return jsonify({'status': 'error', 'message': f'Not enough data points (need at least {timesteps})'}), 400

#         X_lstm = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])
        
#         logger.debug("Running LSTM prediction...")
#         X_pred = lstm_model.predict(X_lstm)
#         mse = np.mean(np.power(X_lstm - X_pred, 2), axis=(1, 2))
#         threshold = np.percentile(mse, 95)
#         lstm_anomalies = mse > threshold

#         logger.debug("Running Isolation Forest & SVM...")
#         X_ml = pd.DataFrame(scaled_data[:-timesteps], columns=expected_columns)
#         iso_anomalies = run_isolation_forest(X_ml)
#         svm_anomalies = run_oneclass_svm(X_ml)

#         results = []
#         for i in range(len(X_ml)):
#             row = {
#                 'left_foot_pressure': float(X_ml.iloc[i]['left_foot_pressure']),
#                 'right_foot_pressure': float(X_ml.iloc[i]['right_foot_pressure']),
#                 'core_stability': float(X_ml.iloc[i]['core_stability']),
#                 'LSTM_MSE': float(mse[i]),
#                 'LSTM_Anomaly': bool(lstm_anomalies[i]),
#                 'ISO_Anomaly': bool(iso_anomalies[i]),
#                 'SVM_Anomaly': bool(svm_anomalies[i]),
#                 'Final_Anomaly': bool(
#                     (lstm_anomalies[i] and svm_anomalies[i]) or
#                     (lstm_anomalies[i] and iso_anomalies[i])
#                 ),
#                 'Logic_Alert': logical_check({
#                     'left_foot_pressure': X_ml.iloc[i]['left_foot_pressure'],
#                     'right_foot_pressure': X_ml.iloc[i]['right_foot_pressure'],
#                     'core_stability': X_ml.iloc[i]['core_stability'],
#                     'LSTM_MSE': mse[i]
#                 }, threshold)
#             }
#             results.append(row)

#         # 📊 Plot oluştur ve kaydet
#         logger.info("Generating anomaly detection plot...")
#         fig, ax = plt.subplots(figsize=(12, 7))
#         ax.set_title("Anomaly Detection", fontsize=14)
#         ax.set_xlabel("Left Foot Pressure")
#         ax.set_ylabel("Right Foot Pressure")
#         ax.grid(True)

#         for i, row in enumerate(results):
#             x = row['left_foot_pressure']
#             y = row['right_foot_pressure']

#             if row['Final_Anomaly']:
#                 color = 'red'
#             elif "UYARI!" in row['Logic_Alert']:
#                 color = 'orange'
#             else:
#                 color = 'green'

#             ax.scatter(x, y, color=color, edgecolors='k', s=80)
#             ax.annotate(str(i + 1), (x + 0.005, y + 0.005), fontsize=7)

#         os.makedirs('static', exist_ok=True)
#         plot_path = 'static/anomaly_plot.png'
#         plt.savefig(plot_path)
#         plt.close(fig)

#         logger.info(f"Plot saved to {plot_path}")

#         return jsonify({
#             'status': 'success',
#             'results': results,
#             'plot_url': f'/{plot_path}'
#         })

#     except Exception as e:
#         logger.error(f"Error during analysis: {str(e)}")
#         logger.error(traceback.format_exc())
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 500

if __name__ == '__main__':
    logger.info("Starting Flask anomaly detection API (file-based with plot)...")
    app.run(debug=True, port=5000)
    logger.info("Flask API started successfully")
    logger.info("Flask API is running on port 5000")
    logger.info("Flask API is ready to accept requests")
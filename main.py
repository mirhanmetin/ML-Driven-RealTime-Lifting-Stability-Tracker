from utils.data_preprocessing import load_and_clean_data
from lstm_model import build_lstm_autoencoder
from anomaly_detection import run_isolation_forest, run_oneclass_svm
from logic_rules import logical_check
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# 1- Load data and clean
filepath = "./final_corrected_clean_normal_training_data.csv"
features = ['left_foot_pressure', 'right_foot_pressure', 'core_stability']
data_clean = load_and_clean_data(filepath, features)

# 2️- Normalization
# @transform = Scales the data to a range of 0-1 using the min and max values ​​calculated in the fit step.
# @fit = Calculates the min and max values ​​of the data. The fit method is called on the training data only.
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_clean) 

# Data preprocessing for LSTM
# @timesteps = Number of previous time steps to consider for each sample.
# @scaled_data = Normalized data ready for LSTM input.
# @reshape = Reshapes the data into a 3D array with shape (samples, timesteps, features).
# @X_lstm = 3D array of shape (samples, timesteps, features) for LSTM input.
timesteps = 10
X_lstm = np.array([scaled_data[i:i + timesteps] for i in range(len(scaled_data) - timesteps)])

# Create LSTM model and train

# @build_lstm_autoencoder = Function to build the LSTM Autoencoder model.
# @model = LSTM Autoencoder model for time series anomaly detection.

# @fit = Trains the model on the input data (X_lstm) for 30 epochs with a batch size of 32 and a validation split of 0.1.
# @epochs = Number of epochs to train the model.
# @batch_size = Number of samples per gradient update.
# @validation_split = Fraction of the training data to be used as validation data.
# @verbose = Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
model = build_lstm_autoencoder(timesteps, X_lstm.shape[2])
model.fit(X_lstm, X_lstm, epochs=30, batch_size=32, validation_split=0.1, verbose=2)

# The trained Autoencoder model now makes predictions for each sequence (block of 10) in the dataset.
# So MSE is for us -> how much deviation from “usual” = anomaly signal.
# @X_pred = Model predictions for the input data (X_lstm).
# @mse = Mean Squared Error between the input data and the model predictions.
# @threshold = 95th percentile of the MSE values, used to determine anomalies.
X_pred = model.predict(X_lstm)
mse = np.mean(np.power(X_lstm - X_pred, 2), axis=(1, 2))
threshold = np.percentile(mse, 95)

# 6️⃣ Isolation Forest & SVM çalıştır
# @X_ml = DataFrame of the scaled data without the last timesteps, used for Isolation Forest and SVM.
# @pd.DataFrame = Creates a DataFrame from the scaled data without the last timesteps.
# @iso_anomalies = Anomaly scores from Isolation Forest.
# @svm_anomalies = Anomaly scores from One-Class SVM.
# @run_isolation_forest = Function to run Isolation Forest for anomaly detection.
# @run_oneclass_svm = Function to run One-Class SVM for anomaly detection.
X_ml = pd.DataFrame(scaled_data[:-timesteps], columns=features)
iso_anomalies = run_isolation_forest(X_ml)
svm_anomalies = run_oneclass_svm(X_ml)

# 7️⃣ Sonuç DataFrame'i oluştur
results = X_ml.copy()
results['LSTM_MSE'] = mse
results['LSTM_Anomaly'] = mse > threshold
results['ISO_Anomaly'] = iso_anomalies
results['SVM_Anomaly'] = svm_anomalies
results['Final_Anomaly'] = ((results['LSTM_Anomaly'] & results['SVM_Anomaly']) |
                            (results['LSTM_Anomaly'] & results['ISO_Anomaly']))

# 8️⃣ Mantıksal kontroller
results['Logic_Alert'] = results.apply(lambda row: logical_check(row, threshold), axis=1)

# 9️⃣ Görselleştir
plot_anomalies(results)

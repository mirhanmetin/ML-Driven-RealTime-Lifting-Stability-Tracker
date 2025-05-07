from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# @iso_anomalies = Anomaly scores from Isolation Forest.
# @run_isolation_forest = Function to run Isolation Forest for anomaly detection.
# @X = Input data for anomaly detection.
# @contamination = Proportion of outliers in the data.
# @preds = Anomaly predictions from the model.
# @return = Boolean array indicating whether each sample is an anomaly (true) or not (false).
def run_isolation_forest(X, contamination=0.01):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    return preds == -1

# @svm_anomalies = Anomaly scores from One-Class SVM.
# @run_oneclass_svm = Function to run One-Class SVM for anomaly detection.
# @X = Input data for anomaly detection.
# @nu = An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
# @preds = Anomaly predictions from the model.
# @return = Boolean array indicating whether each sample is an anomaly (true) or not (false).
def run_oneclass_svm(X, nu=0.01):
    model = OneClassSVM(kernel='rbf', nu=nu, gamma='auto')
    preds = model.fit_predict(X)
    return preds == -1
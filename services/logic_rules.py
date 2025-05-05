
# Logic rules for the application.
# It includes the function to check the logical conditions based on the input data.

# @logical_check = Function to check the logical conditions based on the input data.
# @row = A row of data containing the features to be checked.
# @threshold = Threshold value for the MSE to determine anomalies.
# @return = A string containing the alerts generated based on the logical conditions.
def logical_check(row, threshold):
    alerts = [] # List of alerts generated based on the logical conditions
    left = row['left_foot_pressure'] # Left foot pressure value.
    right = row['right_foot_pressure'] # Right foot pressure value
    core = row['core_stability'] # Core stability value
    mse = row['LSTM_MSE'] # Mean Squared Error value

    if left + right > 1.2:
        alerts.append("Toplam basınç yüksek!")
    diff = abs(left - right)
    if diff > 0.4:
        alerts.append("Ayaklar arası ciddi dengesizlik!")
    elif diff > 0.2:
        alerts.append("Ayaklar arası dengesizlik var!")
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

import pandas as pd
import numpy as np

# Eski CSV'yi oku
df_original = pd.read_csv('final_corrected_clean_normal_training_data.csv')
print(f"Orijinal veri boyutu: {df_original.shape}")

# Yeni simüle edilmiş veri üret (10.000 satır)
n_new = 10000
new_data = {
    'left_foot_pressure': np.random.normal(50, 5, n_new),   # ortalama 50, std 5
    'right_foot_pressure': np.random.normal(50, 5, n_new),
    'core_stability': np.random.normal(0.85, 0.05, n_new)  # ortalama 0.85, std 0.05
}
df_new = pd.DataFrame(new_data)

# Birleştir
df_combined = pd.concat([df_original, df_new], ignore_index=True)
print(f"Yeni veri boyutu: {df_combined.shape}")

# Yeni dosyaya kaydet
df_combined.to_csv('extended_training_data.csv', index=False)
print("✅ Genişletilmiş veri dosyası kaydedildi: extended_training_data.csv")

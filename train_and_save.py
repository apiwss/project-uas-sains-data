"""
Script ini dijalankan SEKALI di Google Colab untuk:
1. Melatih model dari Car_sales.csv
2. Menyimpan model (car_price_model.pkl) dan stats (model_stats.json)
3. File-file tersebut kemudian di-upload ke repository GitHub bersama app.py

Jalankan di Colab:
  !python train_and_save.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle, json, warnings
warnings.filterwarnings('ignore')

# ── Load & Clean ──────────────────────────────────────────────────────────────
df = pd.read_csv('Car_sales.csv')
df_clean = df.dropna(subset=['Price_in_thousands'])

numeric_cols = ['Engine_size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
                'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency',
                'Power_perf_factor', '__year_resale_value']
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# ── Features ──────────────────────────────────────────────────────────────────
features = ['Engine_size', 'Horsepower', 'Wheelbase', 'Width', 'Length',
            'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency', 'Power_perf_factor']
target = 'Price_in_thousands'

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train ─────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"✅ Model trained  |  RMSE: {rmse:.4f}  |  R²: {r2:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ car_price_model.pkl saved")

# ── Save stats ────────────────────────────────────────────────────────────────
df_clean['Full_Name'] = df_clean['Manufacturer'] + ' ' + df_clean['Model']
top10 = df_clean.nlargest(10, 'Sales_in_thousands')

stats = {
    'rmse': round(float(rmse), 4),
    'r2':   round(float(r2),   4),
    'feature_mins':  {f: float(df_clean[f].min())  for f in features},
    'feature_maxs':  {f: float(df_clean[f].max())  for f in features},
    'feature_means': {f: float(df_clean[f].mean()) for f in features},
    'coef':          dict(zip(features, [round(float(c), 4) for c in model.coef_])),
    'intercept':     round(float(model.intercept_), 4),
    'top10': top10[['Full_Name', 'Sales_in_thousands', 'Price_in_thousands',
                    'Engine_size', 'Horsepower', 'Fuel_efficiency']].round(3).to_dict('records'),
}

with open('model_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print("✅ model_stats.json saved")
print("\nFile siap di-upload ke GitHub repository bersama app.py dan requirements.txt")

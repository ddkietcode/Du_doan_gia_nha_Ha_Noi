import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib # Đổi từ pickle sang joblib
import re
import json

print("1. Đang đọc dữ liệu...")
df = pd.read_csv('data/VN_housing_dataset.csv')
total_rows_initial = len(df)

print("2. Đang làm sạch dữ liệu chuyên sâu...")

def extract_numeric(series):
    s = series.astype(str).str.replace(',', '.', regex=False)
    s = s.str.extract(r'(\d+\.?\d*)', expand=False)
    return pd.to_numeric(s, errors='coerce')

df['Diện tích'] = extract_numeric(df['Diện tích'])
df['Số phòng ngủ'] = df['Số phòng ngủ'].astype(str).str.replace('nhiều hơn 10', '10', case=False)
df['Số phòng ngủ'] = extract_numeric(df['Số phòng ngủ'])
df['Giá_triệu_m2'] = extract_numeric(df['Giá/m2'])
df['Giá_Tỷ_VNĐ'] = (df['Giá_triệu_m2'] * df['Diện tích']) / 1000

features = ['Quận', 'Loại hình nhà ở', 'Diện tích', 'Số phòng ngủ']
target = 'Giá_Tỷ_VNĐ'
df_clean = df[features + [target]].dropna()

df_clean = df_clean[
    (df_clean['Diện tích'] >= 10) & (df_clean['Diện tích'] <= 1000) &
    (df_clean['Giá_Tỷ_VNĐ'] >= 0.5) & (df_clean['Giá_Tỷ_VNĐ'] <= 500)
]

print(f"-> Đã lọc sạch! Số lượng dòng hợp lệ còn lại: {len(df_clean)} / {total_rows_initial}")

df_clean.to_csv('data/cleaned_data.csv', index=False)

print("3. Đang thiết lập và huấn luyện mô hình Random Forest...")
X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Quận', 'Loại hình nhà ở']
numeric_features = ['Diện tích', 'Số phòng ngủ']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Thêm max_depth=25 để giới hạn độ sâu của cây, giúp file nhẹ hơn rất nhiều
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print("=======================================")
print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH:")
print(f"- R2 Score (Độ chính xác): {r2_score(y_test, y_pred):.2f}")
print(f"- RMSE (Sai số trung bình): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} Tỷ VNĐ")
print("=======================================")

print("4. Đang lưu mô hình dạng nén (joblib)...")
# Lưu file dạng .gz và sử dụng mức nén compress=3
joblib.dump(model_pipeline, 'models/rf_model.pkl.gz', compress=3)

print("Hoàn tất! File cleaned_data.csv và rf_model.pkl.gz đã được cập nhật thành công.")

metrics = {
    "R2_Score": round(r2_score(y_test, y_pred), 2),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
}

with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f)
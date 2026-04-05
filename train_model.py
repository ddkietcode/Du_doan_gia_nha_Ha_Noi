import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib 
import re
import json

print("1. Đang đọc dữ liệu...")
df = pd.read_csv('data/VN_housing_dataset.csv')
total_rows_initial = len(df)

print("2. Đang làm sạch dữ liệu chuyên sâu...")

# Hàm trích xuất số liệu cơ bản
def extract_numeric(series):
    s = series.astype(str).str.replace(',', '.', regex=False)
    s = s.str.extract(r'(\d+\.?\d*)', expand=False)
    return pd.to_numeric(s, errors='coerce')

# --- CẢI TIẾN 1: Hàm tính Giá nhà linh hoạt xử lý mọi đơn vị ---
def parse_price(row):
    price_str = str(row['Giá/m2']).lower().replace(',', '.')
    area = row['Diện tích']
    
    if pd.isna(area) or area <= 0:
        return np.nan
        
    # Lấy con số đầu tiên trong chuỗi
    match = re.search(r'(\d+\.?\d*)', price_str)
    if not match:
        return np.nan
        
    val = float(match.group(1))
    
    # Xác định đơn vị và tính quy ra Tổng giá (Tỷ VNĐ)
    if 'tỷ' in price_str:
        if 'm' in price_str: # ví dụ: "1.5 tỷ/m²"
            return val * area
        else: # ví dụ: "5 tỷ" (tổng giá đã cho sẵn)
            return val
    elif 'triệu' in price_str:
        if 'm' in price_str: # ví dụ: "86,96 triệu/m²"
            return (val * area) / 1000
        else: # ví dụ: "3000 triệu"
            return val / 1000
    else: 
        # Trường hợp phổ biến từ crawler: "247.787 đ/m²" (chính là 247.787 triệu/m²)
        return (val * area) / 1000

# Áp dụng làm sạch
# ... (Giữ nguyên hàm extract_numeric và parse_price ở trên) ...

df['Diện tích'] = extract_numeric(df['Diện tích'])
df['Số phòng ngủ'] = df['Số phòng ngủ'].astype(str).str.replace('nhiều hơn 10', '10', case=False)
df['Số phòng ngủ'] = extract_numeric(df['Số phòng ngủ'])

# 1. Tính toán giá trị (Tỷ VNĐ) theo hàm mới
df['Giá_Tỷ_VNĐ'] = df.apply(parse_price, axis=1)

# 2. CẢI TIẾN MỚI: Tính ngược lại Đơn giá (Triệu/m2) để làm bộ lọc chống tin đăng ảo
df['Đơn_giá_triệu_m2'] = (df['Giá_Tỷ_VNĐ'] * 1000) / df['Diện tích']

df['Giấy tờ pháp lý'] = df['Giấy tờ pháp lý'].fillna('Không xác định')

features = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý', 'Diện tích', 'Số phòng ngủ']
target = 'Giá_Tỷ_VNĐ'

# Trở lại việc dropna cho 'Số phòng ngủ' (thay vì fillna) vì đây là feature cực kỳ quyết định giá
df_clean = df.dropna(subset=['Quận', 'Loại hình nhà ở', 'Số phòng ngủ', target])

# 3. BỘ LỌC OUTLIERS NGHIÊM NGẶT THEO THỰC TẾ
# Các giới hạn này giúp loại bỏ biệt thự siêu sang/tin ảo để mô hình dự đoán tốt nhà ở phổ thông
df_clean = df_clean[
    (df_clean['Diện tích'] >= 15) & (df_clean['Diện tích'] <= 250) &            # Bỏ nhà > 250m2 (thường là đất nền/dự án)
    (df_clean['Giá_Tỷ_VNĐ'] >= 0.5) & (df_clean['Giá_Tỷ_VNĐ'] <= 40) &          # Cấp trần giá nhà ở mức 40 Tỷ (thay vì 500 Tỷ)
    (df_clean['Đơn_giá_triệu_m2'] >= 10) & (df_clean['Đơn_giá_triệu_m2'] <= 350) & # Lọc đơn giá từ 10 - 350 triệu/m2
    (df_clean['Số phòng ngủ'] > 0) & (df_clean['Số phòng ngủ'] <= 10)
]

print(f"-> Đã lọc sạch! Số lượng dòng hợp lệ còn lại: {len(df_clean)} / {total_rows_initial}")

# ... (Tiếp tục huấn luyện mô hình Random Forest như cũ) ...
df_clean.to_csv('data/cleaned_data.csv', index=False)

print("3. Đang thiết lập và huấn luyện mô hình Random Forest...")
X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cập nhật biến phân loại để thêm Giấy tờ pháp lý
categorical_features = ['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý']
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
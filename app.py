import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
import json 

st.set_page_config(page_title="Dự đoán Giá Nhà Hà Nội", layout="wide")

# --- CACHE DỮ LIỆU VÀ MÔ HÌNH ---
@st.cache_data
def load_raw_data():
    return pd.read_csv("data/VN_housing_dataset.csv")

@st.cache_data
def load_cleaned_data():
    try:
        return pd.read_csv("data/cleaned_data.csv")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.pkl.gz")

df_raw = load_raw_data()
df_clean = load_cleaned_data()
model = load_model()

# Lấy chỉ số RMSE để làm độ tin cậy cho dự đoán
try:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
        rmse_val = float(metrics.get("RMSE", 2.0)) # Mặc định 2.0 nếu không có
except:
    rmse_val = 2.0

# --- THANH ĐIỀU HƯỚNG ---
st.sidebar.title("Danh mục")
page = st.sidebar.radio("Chọn trang:", 
                        ["Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)", 
                         "Trang 2: Triển khai mô hình", 
                         "Trang 3: Đánh giá & Hiệu năng (Evaluation)"])

# --- TRANG 1: GIỚI THIỆU & EDA ---
if page == "Trang 1: Giới thiệu & Khám phá dữ liệu (EDA)":
    st.title("Phân tích Dữ liệu Nhà ở Hà Nội")
    st.markdown("**Tên đề tài:** Dự đoán giá nhà ở Hà Nội bằng Random Forest")
    st.markdown("**Họ tên SV:** Đoàn Đức Kiệt - **MSSV:** [Điền MSSV của bạn]")
    st.markdown("**Giá trị thực tiễn:** Ứng dụng giúp người mua, người bán và các nhà đầu tư bất động sản có cái nhìn khách quan về mức giá thị trường, từ đó đưa ra các quyết định giao dịch hợp lý dựa trên đặc trưng của căn nhà.")
    
    st.subheader("1. Dữ liệu thô (Raw Data)")
    st.dataframe(df_raw.head(15))
    
    st.subheader("2. Biểu đồ phân tích dữ liệu")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Biểu đồ 1: Phân bố số lượng nhà theo Quận/Huyện**")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(y='Quận', data=df_raw, order=df_raw['Quận'].value_counts().index[:10], ax=ax1, palette="mako")
        ax1.set_xlabel("Số lượng tin đăng")
        ax1.set_ylabel("Quận/Huyện")
        st.pyplot(fig1)
        
    with col2:
        st.write("**Biểu đồ 2: Tỷ lệ phân bố Loại hình nhà ở**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        loai_nha_counts = df_raw['Loại hình nhà ở'].value_counts()
        ax2.pie(loai_nha_counts, labels=loai_nha_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
        ax2.axis('equal') 
        st.pyplot(fig2)
        
    st.subheader("3. Nhận xét về tập dữ liệu")
    st.info("""
    - **Độ lệch của dữ liệu (Data Skewness):** Dữ liệu bị lệch khá nhiều (imbalanced) về mặt địa lý và loại hình. Số lượng tin đăng tập trung dày đặc ở các quận Đống Đa, Cầu Giấy, Hà Đông. Các loại hình như "Nhà ngõ, hẻm" và "Nhà mặt phố" chiếm hơn 80% tổng bộ dữ liệu.
    - **Các đặc trưng quan trọng:** Thông qua phân tích, `Diện tích` và `Quận` (vị trí) là hai đặc trưng có tính quyết định mạnh mẽ nhất đến tổng giá trị của một căn nhà.
    - **Vấn đề nhiễu:** Dữ liệu thô có chứa nhiều text trong các cột số (như 'm²', 'phòng') và có những khoảng giá trị ngoại lai (outliers) phi lý. Do đó, toàn bộ đã được đưa qua Pipeline làm sạch trước khi huấn luyện.
    """)

# --- TRANG 2: DỰ ĐOÁN GIÁ NHÀ ---
elif page == "Trang 2: Triển khai mô hình":
    st.title("Dự báo Giá Bất Động Sản")
    st.write("Sử dụng các thanh trượt và menu thả xuống để nhập thông số căn nhà.")
    
    if df_clean is not None:
        quan_list = sorted(df_clean['Quận'].unique())
        loai_nha_list = sorted(df_clean['Loại hình nhà ở'].unique())
    else:
        quan_list = df_raw['Quận'].dropna().unique()
        loai_nha_list = df_raw['Loại hình nhà ở'].dropna().unique()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            quan = st.selectbox("Chọn Quận/Huyện", quan_list)
            dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=50.0, step=1.0)
            
        with col2:
            loai_nha = st.selectbox("Chọn Loại hình nhà ở", loai_nha_list)
            so_phong = st.number_input("Số phòng ngủ", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
            
        submit_button = st.form_submit_button("Tiến hành Dự đoán")
        
    if submit_button:
        # Tiền xử lý input giống hệt lúc huấn luyện thông qua Pipeline
        input_data = pd.DataFrame({
            'Quận': [quan],
            'Loại hình nhà ở': [loai_nha],
            'Diện tích': [dien_tich],
            'Số phòng ngủ': [so_phong]
        })
        
        prediction = model.predict(input_data)[0]
        
        # Hiển thị kết quả và độ tin cậy
        st.success(f"🏠 Giá nhà dự báo là: **{prediction:.2f} tỷ VNĐ**")
        
        # Tính toán khoảng tin cậy dựa trên RMSE
        lower_bound = max(0, prediction - rmse_val)
        upper_bound = prediction + rmse_val
        
        st.info(f"**Độ tin cậy của dự đoán:** Căn cứ vào độ lỗi của mô hình, giá trị thực tế của căn nhà có xác suất cao nằm trong khoảng từ **{lower_bound:.2f} tỷ VNĐ** đến **{upper_bound:.2f} tỷ VNĐ**.")
        st.progress(min(prediction/100, 1.0))

# --- TRANG 3: ĐÁNH GIÁ HIỆU NĂNG ---
elif page == "Trang 3: Đánh giá & Hiệu năng (Evaluation)":
    st.title("Đánh giá Mô hình")
    
    st.write("Các chỉ số đo lường hiệu năng của thuật toán Random Forest Regressor trên tập kiểm thử:")
    
    try:
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
            r2 = str(metrics["R2_Score"])
            rmse = str(metrics["RMSE"]) + " Tỷ VNĐ"
    except:
        r2 = "N/A"
        rmse = "N/A"

    col1, col2 = st.columns(2)
    col1.metric("R2 Score (Hệ số xác định)", r2)
    col2.metric("RMSE (Sai số toàn phương trung bình)", rmse)
    
    st.subheader("1. Biểu đồ kỹ thuật: Giá trị Thực tế vs Dự đoán")
    if df_clean is not None:
        sample = df_clean.sample(min(300, len(df_clean)), random_state=42)
        X_sample = sample[['Quận', 'Loại hình nhà ở', 'Diện tích', 'Số phòng ngủ']]
        y_actual = sample['Giá_Tỷ_VNĐ']
        y_pred = model.predict(X_sample)
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(y_actual, y_pred, alpha=0.6, color='teal', edgecolors='w', s=70)
        
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Đường dự đoán hoàn hảo (y=x)")
        
        ax3.set_title("So sánh mức giá dự đoán và giá thực tế")
        ax3.set_xlabel("Giá trị Thực tế (Tỷ VNĐ)")
        ax3.set_ylabel("Giá trị Dự đoán (Tỷ VNĐ)")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Không tìm thấy dữ liệu để vẽ biểu đồ kỹ thuật.")

    st.subheader("2. Phân tích sai số (Error Analysis)")
    st.error("""
    **Mô hình thường dự đoán sai ở đâu?**
    - **Phân khúc bất động sản siêu cao cấp:** Biểu đồ cho thấy đối với những căn nhà có giá trị rất lớn (> 30 tỷ), thuật toán thường dự đoán thấp hơn giá trị thực tế (Underfitting tại các điểm này). Lý do là do mẫu dữ liệu ở phân khúc này quá ít (imbalanced data).
    - **Nhà trong ngõ sâu:** Mô hình có thể định giá quá cao cho những căn nhà diện tích rộng nhưng lại nằm sâu trong ngõ hẻm ô tô không vào được. 
    
    **Hướng cải thiện:**
    1. Bổ sung thêm các đặc trưng (features) quan trọng chưa có trong tập dữ liệu: *Độ rộng ngõ trước nhà, Khoảng cách ra đường lớn, Chất lượng nội thất, Hướng nhà*.
    2. Cào thêm dữ liệu về các căn biệt thự/nhà mặt phố cao cấp để làm cân bằng phổ dữ liệu huấn luyện.
    3. Thử nghiệm các kiến trúc mô hình Gradient Boosting (XGBoost, LightGBM) để tối ưu hóa sai số dư (residuals).
    """)
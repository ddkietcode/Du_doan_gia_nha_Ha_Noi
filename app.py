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
    st.markdown("**Tên đề tài:** Dự đoán giá nhà ở Hà Nội bằng Random Forest nhằm hỗ trợ người mua nhà và nhà đầu tư ra quyết định")
    st.markdown("**Họ tên SV:** Đoàn Đức Kiệt - **MSSV:** 21T1020463")
    st.markdown("**Giá trị thực tiễn:** Ước lượng giá bất động sản dựa trên các đặc trưng như quận, loại hình nhà ở, diện tích, số phòng ngủ, giấy tờ pháp lý nhằm hỗ trợ người mua và nhà đầu tư ra quyết định.")
    
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
    - **Độ lệch của dữ liệu (Data Skewness):** Dữ liệu bị lệch khá mạnh (imbalanced) về mặt địa lý và loại hình. Số lượng tin đăng tập trung dày đặc ở các quận nội thành (như Đống Đa, Thanh Xuân, Hoàng Mai). Trong khi đó, "Nhà ngõ, hẻm" và "Nhà mặt phố" là hai loại hình chiếm áp đảo tới hơn 80% tổng bộ dữ liệu.
    - **Trọng số đặc trưng (Feature Importance):** Mô hình được huấn luyện dựa trên 5 biến số cốt lõi. Trong đó, Diện tích và Quận (vị trí) là hai đặc trưng mang trọng số quyết định mạnh mẽ nhất đến tổng giá trị tài sản. Các yếu tố còn lại như Giấy tờ pháp lý, Loại hình nhà ở và Số phòng ngủ đóng vai trò tinh chỉnh để tăng độ chính xác cho từng phân khúc cụ thể.
    - **Xử lý nhiễu (Data Cleaning):** Dữ liệu thô ban đầu chứa nhiều văn bản hỗn hợp trong các cột số (như 'm²', 'phòng') và có những khoảng giá trị ngoại lai (outliers) phi lý (tin đăng ảo, giá sai lệch). Do đó, toàn bộ dữ liệu đã được tự động đưa qua Pipeline để bóc tách số liệu và chuẩn hóa trước khi đưa vào huấn luyện.    
    """)

# --- TRANG 2: DỰ ĐOÁN GIÁ NHÀ ---
elif page == "Trang 2: Triển khai mô hình":
    st.title("Dự báo Giá Bất Động Sản")
    st.write("Sử dụng các thanh trượt và menu thả xuống để nhập thông số căn nhà.")
    
    # --- CẬP NHẬT MỚI: Bổ sung danh sách Giấy tờ pháp lý ---
    if df_clean is not None:
        quan_list = sorted(df_clean['Quận'].unique())
        loai_nha_list = sorted(df_clean['Loại hình nhà ở'].unique())
        phap_ly_list = sorted(df_clean['Giấy tờ pháp lý'].astype(str).unique())
    else:
        quan_list = df_raw['Quận'].dropna().unique()
        loai_nha_list = df_raw['Loại hình nhà ở'].dropna().unique()
        phap_ly_list = df_raw['Giấy tờ pháp lý'].fillna('Không xác định').unique()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            quan = st.selectbox("Chọn Quận/Huyện", quan_list)
            loai_nha = st.selectbox("Chọn Loại hình nhà ở", loai_nha_list)
            # --- CẬP NHẬT MỚI: Thêm selectbox cho Giấy tờ pháp lý ---
            phap_ly = st.selectbox("Giấy tờ pháp lý", phap_ly_list)
            
        with col2:
            dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=50.0, step=1.0)
            so_phong = st.number_input("Số phòng ngủ", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
            
        submit_button = st.form_submit_button("Tiến hành Dự đoán")
        
    if submit_button:
        # --- CẬP NHẬT MỚI: Thêm cột 'Giấy tờ pháp lý' vào input_data ---
        input_data = pd.DataFrame({
            'Quận': [quan],
            'Loại hình nhà ở': [loai_nha],
            'Giấy tờ pháp lý': [phap_ly],
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
        # --- CẬP NHẬT MỚI: Thêm 'Giấy tờ pháp lý' vào mảng features để dự đoán biểu đồ ---
        X_sample = sample[['Quận', 'Loại hình nhà ở', 'Giấy tờ pháp lý', 'Diện tích', 'Số phòng ngủ']]
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
        
    st.markdown("""
        **Ý nghĩa biểu đồ:**
        - **Trục X / Trục Y:** Biểu thị giá trị Thực tế / Dự đoán (Tỷ VNĐ).
        - **Đường đứt nét màu đỏ (y=x):** Đường dự đoán hoàn hảo. Các điểm nằm trên là đoán cao hơn (Over-predict), nằm dưới là đoán thấp hơn (Under-predict) thực tế.
        """)

    col1, col2 = st.columns(2)
        
    with col1:
        st.success("""
            **🟢 Phân khúc dưới 10 tỷ VNĐ:**
            - **Nhận xét:** Dữ liệu tập trung dày đặc (1 - 8 tỷ) và bám sát đường đỏ.
            - **Đánh giá:** Mô hình hoạt động **rất tốt và đáng tin cậy**, độ chính xác cao đối với nhóm tài sản phổ thông.
            """)
            
        st.error("""
            **🔴 Điểm bất thường (Outliers):**
            - **Đoán quá cao:** Ví dụ nhà thực tế **5.5 tỷ** nhưng mô hình dự đoán gần **19 tỷ**; hoặc 14 tỷ đoán thành 25 tỷ.
            - **Đoán quá thấp:** Nhà thực tế **13 tỷ** nhưng chỉ đoán **4.5 tỷ**; hoặc 17 tỷ đoán thành 8 tỷ.
            """)

    with col2:
            st.warning("""
            **🟠 Phân khúc trên 10 tỷ VNĐ:**
            - **Nhận xét:** Dữ liệu phân tán rộng thành hình phễu (hiện tượng Heteroscedasticity).
            - **Đánh giá:** Mô hình **thiếu độ chính xác** và gặp khó khăn khi định giá các tài sản có giá trị lớn.
            """)
            
            st.info("""
            **💡 Đề xuất cải thiện cho Model:**
            1. **Thu thập thêm dữ liệu** cho nhóm nhà > 10 tỷ để xử lý Data Imbalance.
            2. **Thêm Đặc trưng (Features):** Bổ sung biến về độ rộng ngõ, mặt tiền, phong thủy...
            3. **Log Transformation:** Biến đổi Logarit cho biến Giá để giảm hiện tượng phương sai thay đổi.
            4. **Lọc Outliers:** Rà soát lại dữ liệu gốc của các điểm dự đoán sai lệch nghiêm trọng.
            """)
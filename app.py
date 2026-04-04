import streamlit as st
import pandas as pd
import joblib # Đổi từ pickle sang joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json # Thêm json để tự động đọc thông số đánh giá

st.set_page_config(page_title="Dự đoán Giá Nhà Hà Nội", layout="wide")

# --- CACHE DỮ LIỆU VÀ MÔ HÌNH ---
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")

@st.cache_resource
def load_model():
    # Load file mô hình đã nén
    return joblib.load("models/rf_model.pkl.gz")

df = load_data()
model = load_model()

# --- THANH ĐIỀU HƯỚNG ---
st.sidebar.title("Danh mục")
page = st.sidebar.radio("Chọn chức năng:", 
                        ["Trang 1: Giới thiệu & EDA", 
                         "Trang 2: Dự đoán Giá Nhà", 
                         "Trang 3: Đánh giá Hiệu năng"])

# --- TRANG 1 ---
if page == "Trang 1: Giới thiệu & EDA":
    st.title("Phân tích Dữ liệu Nhà ở Hà Nội")
    st.markdown("**Tên đề tài:** Dự đoán giá nhà ở Hà Nội bằng Random Forest")
    st.markdown("**Họ tên SV:** [Đoàn Đức Kiệt] - **MSSV:** [21T1020463]")
    
    st.subheader("1. Dữ liệu đã làm sạch")
    st.dataframe(df.head(10))
    
    st.subheader("2. Khám phá dữ liệu (EDA)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top 10 Quận có nhiều dữ liệu bán nhà nhất**")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.countplot(y='Quận', data=df, order=df['Quận'].value_counts().index[:10], ax=ax1, palette="viridis")
        st.pyplot(fig1)
        
    with col2:
        st.write("**Phân bố giá nhà (Tỷ VNĐ)**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(df[df['Giá_Tỷ_VNĐ'] < 50]['Giá_Tỷ_VNĐ'], bins=30, kde=True, ax=ax2, color="salmon")
        ax2.set_xlabel("Giá (Tỷ VNĐ)")
        st.pyplot(fig2)

# --- TRANG 2 ---
elif page == "Trang 2: Dự đoán Giá Nhà":
    st.title("Dự đoán Giá Nhà Hà Nội")
    st.write("Vui lòng nhập các thông tin của căn nhà để nhận dự báo giá.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            quan = st.selectbox("Quận/Huyện", df['Quận'].unique())
            dien_tich = st.number_input("Diện tích (m²)", min_value=10.0, max_value=2000.0, value=50.0, step=1.0)
            
        with col2:
            loai_nha = st.selectbox("Loại hình nhà ở", df['Loại hình nhà ở'].unique())
            so_phong = st.number_input("Số phòng ngủ", min_value=1.0, max_value=20.0, value=2.0, step=1.0)
            
        submit_button = st.form_submit_button("Dự đoán Giá")
        
    if submit_button:
        input_data = pd.DataFrame({
            'Quận': [quan],
            'Loại hình nhà ở': [loai_nha],
            'Diện tích': [dien_tich],
            'Số phòng ngủ': [so_phong]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"💰 Giá nhà dự báo khoảng: **{prediction:.2f} Tỷ VNĐ**")
        st.progress(min(prediction/100, 1.0))

# --- TRANG 3 ---
elif page == "Trang 3: Đánh giá Hiệu năng":
    st.title("Đánh giá Mô hình Random Forest")
    
    st.write("Sau khi huấn luyện trên tập dữ liệu, mô hình đạt được các chỉ số sau (trên tập Test 20%):")
    
    # Tự động đọc file json để lấy thông số hiển thị
    try:
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
            r2 = str(metrics["R2_Score"])
            rmse = str(metrics["RMSE"]) + " Tỷ VNĐ"
    except:
        r2 = "N/A"
        rmse = "N/A"

    col1, col2 = st.columns(2)
    col1.metric("R2 Score (Độ R-bình phương)", r2) 
    col2.metric("RMSE (Sai số toàn phương trung bình)", rmse)
    
    st.info("""
    **Nhận xét mô hình:**
    - **Ưu điểm:** Mô hình Random Forest bắt được các mối quan hệ phi tuyến tính tốt, đặc biệt giữa vị trí (Quận) và Giá.
    - **Hạn chế (Sai số):** Sai số vẫn còn ở mức vài tỷ do tập dữ liệu bị nhiễu (giá ảo) và thiếu một số đặc trưng cực kỳ quan trọng như: Độ rộng ngõ, Khoảng cách tới đường lớn, hay Nội thất.
    """)
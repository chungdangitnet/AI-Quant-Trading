# dashboard/app.py
import streamlit as st
import pandas as pd
import os
import glob
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# 1. Cấu hình trang Web
st.set_page_config(page_title="AI Quant Trading Dashboard", page_icon="📈", layout="wide")

# Xác định đường dẫn tuyệt đối của thư mục gốc (STOCK/)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Tiêu đề Dashboard
st.title("📈 AI Quantitative Trading Dashboard")
st.markdown("Hệ thống tự động quét tín hiệu giao dịch và dự báo xu hướng cổ phiếu bằng Machine Learning.")
st.markdown("---")

# 3. Hàm tìm và đọc báo cáo mới nhất từ thư mục reports/ (Sử dụng đường dẫn tuyệt đối)
@st.cache_data
def load_latest_report():
    reports_pattern = os.path.join(parent_dir, "reports", "daily_scan_*.csv")
    report_files = glob.glob(reports_pattern)
    if not report_files:
        return None
    # Lấy file mới nhất dựa trên thời gian tạo
    latest_file = max(report_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    return df, latest_file

# Lấy dữ liệu báo cáo
result = load_latest_report()

# 4. Hiển thị Bảng Tín hiệu AI
st.header("🎯 Bảng Tín hiệu AI Khuyến nghị Hôm nay")

if result is not None:
    df_report, file_path = result
    report_date = os.path.basename(file_path).replace("daily_scan_", "").replace(".csv", "")
    
    st.info(f"📅 Ngày cập nhật dữ liệu: **{report_date.replace('_', '/')}**")
    
    # Hiển thị DataFrame và tô màu tự động cho tín hiệu BUY/SELL
    def highlight_signals(val):
        if 'BUY' in str(val):
            return 'background-color: #d4edda; color: green; font-weight: bold;'
        elif 'SELL' in str(val) or 'HOLD' in str(val):
            return 'background-color: #f8d7da; color: red; font-weight: bold;'
        return ''

    # Tương thích với các phiên bản Pandas mới
    if hasattr(df_report.style, 'map'):
        styled_df = df_report.style.map(highlight_signals, subset=['Signal'])
    else:
        styled_df = df_report.style.applymap(highlight_signals, subset=['Signal'])

    st.dataframe(styled_df, width='stretch', height=400)
else:
    st.warning("⚠️ Chưa tìm thấy báo cáo nào! Hãy chạy file `python deployment/daily_scanner.py` trước.")

st.markdown("---")

# 5. Khu vực vẽ biểu đồ nến tương tác (Interactive Candlestick)
st.header("📊 Phân tích Biểu đồ Kỹ thuật")

selected_ticker = st.text_input("🔍 Nhập mã cổ phiếu (VD: AAPL, MSFT, TSLA):", value="AAPL")

if selected_ticker:
    try:
        df_chart = yf.download(selected_ticker.strip().upper(), period="6mo", interval="1d", progress=False)
        if isinstance(df_chart.columns, pd.MultiIndex):
            df_chart.columns = df_chart.columns.get_level_values(0)
            
        if not df_chart.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=df_chart.index,
                open=df_chart['Open'],
                high=df_chart['High'],
                low=df_chart['Low'],
                close=df_chart['Close'],
                name="Nến giá"
            )])
            
            fig.update_layout(
                title=f"Biểu đồ giá {selected_ticker.upper()} (6 tháng qua)",
                yaxis_title="Giá (USD)",
                xaxis_title="Thời gian",
                template="plotly_dark",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Không tìm thấy dữ liệu cho mã này.")
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
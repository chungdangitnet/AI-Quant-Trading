# models/train_model.py
import sys
import os
import pandas as pd
import yfinance as yf
import joblib
from xgboost import XGBClassifier

# Thêm đường dẫn thư mục gốc vào system path để nhận diện các module (utils, config)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.feature_engineer import TechnicalFeatures

def train_quant_model():
    print("🚀 Bắt đầu quá trình huấn luyện lại mô hình AI (XGBoost)...")

    # 1. Tải dữ liệu lịch sử mẫu (Sử dụng cổ phiếu đại diện như AAPL hoặc gộp nhiều mã)
    # Lấy dữ liệu 2 năm để đảm bảo đủ dữ liệu sau khi drop NaNs
    print("📥 Đang tải dữ liệu lịch sử và tích hợp vĩ mô...")
    df = yf.download("AAPL", period="2y", interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        print("❌ Không tải được dữ liệu lịch sử.")
        return

    # 2. Tạo đặc trưng kỹ thuật và vĩ mô thông qua TechnicalFeatures
    te = TechnicalFeatures(df)
    df_features = te.generate_all_features()

    # 3. Tạo nhãn mục tiêu (Target) cho bài toán phân loại:
    # Nếu giá ngày mai (Close shift -1) cao hơn giá hôm nay -> Tăng (1), ngược lại -> Giảm/Đi ngang (0)
    df_features['Target'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    
    # Loại bỏ dòng cuối cùng vì shift(-1) sẽ sinh ra giá trị NaN ở nhãn
    df_features.dropna(inplace=True)

    # 4. Phân chia tập dữ liệu đầu vào (X) và nhãn (y)
    X = df_features.drop(columns=['Target'])
    y = df_features['Target']

    print(f"📊 Tổng số lượng mẫu dữ liệu huấn luyện: {len(X)}")
    print(f"📊 Tổng số lượng features (đầu vào): {X.shape[1]}")

    # 5. Khởi tạo và huấn luyện mô hình XGBoost
    print("🤖 Đang huấn luyện mô hình XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    model.fit(X, y)

    # 6. Lưu file mô hình vào thư mục models/
    os.makedirs(current_dir, exist_ok=True)
    model_path = os.path.join(current_dir, "xgboost_model.pkl")
    
    joblib.dump(model, model_path)
    print(f"✅ Huấn luyện thành công! Mô hình mới đã được lưu tại: {model_path}")

if __name__ == "__main__":
    train_quant_model()
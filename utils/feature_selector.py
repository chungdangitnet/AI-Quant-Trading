# utils/feature_selector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureSelector")

class FeatureSelector:
    """Mô-đun tạo nhãn dự đoán và lựa chọn đặc trưng quan trọng cho Machine Learning."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_target(self, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        """Tạo nhãn phân loại hướng đi của giá trong tương lai."""
        logger.info(f"Tạo nhãn Target với khoảng thời gian dự phóng (horizon) = {horizon} phiên...")
        
        # Tính lợi nhuận tương lai
        self.df['Future_Return'] = self.df['Close'].shift(-horizon) / self.df['Close'] - 1
        
        # Gán nhãn: 1 nếu tăng, 0 nếu giảm/đi ngang
        self.df['Target'] = (self.df['Future_Return'] > threshold).astype(int)
        
        # Xóa các dòng cuối bị NaN do hàm shift(-horizon)
        self.df.dropna(inplace=True)
        return self.df

    def select_features_by_rf(self, top_n: int = 10) -> tuple:
        """Xếp hạng và lọc ra top đặc trưng quan trọng nhất bằng Random Forest."""
        logger.info("Đang tính toán mức độ quan trọng của đặc trưng bằng Random Forest...")
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Future_Return', 'Target']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols]
        y = self.df['Target']
        
        # Huấn luyện nhanh Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Trích xuất độ quan trọng
        importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        top_features = importances.head(top_n).index.tolist()
        
        logger.info(f"Top {top_n} features được chọn: {top_features}")
        return top_features, importances
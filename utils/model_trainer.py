# utils/model_trainer.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelTrainer")

class ModelTrainer:
    """Mô-đun huấn luyện và đánh giá mô hình Machine Learning tài chính."""

    def __init__(self, df: pd.DataFrame, target_col: str = 'Target'):
        self.df = df.copy()
        self.target_col = target_col

    def time_series_split(self, test_size: float = 0.2):
        """Chia dữ liệu theo chuẩn chuỗi thời gian chống rò rỉ dữ liệu (Data Leakage)."""
        split_idx = int(len(self.df) * (1 - test_size))
        
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]
        
        feature_cols = [col for col in self.df.columns if col != self.target_col and col != 'Close']
        
        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[self.target_col]
        
        logger.info(f"Kích thước tập Train: {X_train.shape}, Tập Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_xgboost(self, X_train, y_train, X_test, y_test) -> tuple:
        """Huấn luyện mô hình XGBoost Classifier và trả về kết quả đánh giá."""
        logger.info("Bắt đầu huấn luyện mô hình XGBoost...")
        
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.03, 
            max_depth=4, 
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        
        # Dự đoán trên tập Test
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Tính toán các chỉ số đánh giá
        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1-Score": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probs)
        }
        
        logger.info("Hoàn tất huấn luyện mô hình.")
        return model, metrics
    def train_xgboost(self, X_train, y_train, X_test, y_test, **kwargs):
        """
        Huấn luyện mô hình XGBoost.
        Hỗ trợ truyền các tham số siêu tham số (Hyperparameters) thông qua **kwargs
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import xgboost as xgb
        
        # Khởi tạo mô hình, truyền toàn bộ cấu hình từ **kwargs vào XGBClassifier
        model = xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            **kwargs  # BÍ QUYẾT LÀ Ở ĐÂY: Chấp nhận mọi tham số truyền vào
        )
        
        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        
        # Dự đoán
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Đánh giá Metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
        return model, metrics
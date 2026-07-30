import os
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataPreprocessor")


class DataPreprocessor:
    """Mô-đun tiền xử lý dữ liệu chuỗi thời gian tài chính chuẩn Quantitative Trading."""

    def __init__(self):
        self.scaler = None

    def clean_time_series(self, df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
        """
        Làm sạch căn bản: Loại bỏ trùng lặp, sắp xếp thời gian và điền khuyết an toàn (No Lookahead).
        """
        df = df.copy()
        
        # 1. Loại bỏ các dòng trùng lặp Index
        df = df[~df.index.duplicated(keep="first")]
        
        # 2. Đảm bảo Index đã được sắp xếp theo thời gian tăng dần
        df = df.sort_index()

        # 3. Tạo khung thời gian liên tục (Reindex) để phát hiện ngày bị thiếu
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_idx)

        # 4. Điền giá trị khuyết thiếu bằng Forward Fill (Lấy giá hôm qua điền cho hôm nay)
        # Giá Close/Adj Close điền ffill, sau đó ffill tiếp cho Open, High, Low
        cols_ohlc = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
        df[cols_ohlc] = df[cols_ohlc].ffill()
        
        # Nếu vẫn còn NaN ở những dòng đầu tiên (do chưa có dữ liệu quá khứ), dùng bfill chỉ cho đoạn đầu
        df[cols_ohlc] = df[cols_ohlc].bfill()

        # Volume bị thiếu trong ngày nghỉ thì điền bằng 0
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].fillna(0)

        df.index.name = "Date"
        logger.info(f"Đã làm sạch dữ liệu. Kích thước sau xử lý: {df.shape}")
        return df

    def handle_outliers_iqr(self, df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
        """
        Xử lý Outliers bằng phương pháp Clipping IQR (Winsorization) thay vì xóa bỏ.
        """
        df_clipped = df.copy()
        for col in columns:
            if col not in df_clipped.columns:
                continue
            
            Q1 = df_clipped[col].quantile(0.25)
            Q3 = df_clipped[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Đếm số lượng outliers
            outliers_count = ((df_clipped[col] < lower_bound) | (df_clipped[col] > upper_bound)).sum()
            
            # Clipping (giới hạn giá trị nằm trong biên)
            df_clipped[col] = np.clip(df_clipped[col], lower_bound, upper_bound)
            logger.info(f"Cột '{col}': Phát hiện & Clip {outliers_count} giá trị ngoại lệ.")

        return df_clipped

    def resample_ohlcv(self, df: pd.DataFrame, rule: str = "W") -> pd.DataFrame:
        """
        Gộp chuỗi thời gian (Resampling) sang khung thời gian lớn hơn (ví dụ 'W': Tuần, 'M': Tháng).
        Quy tắc gộp OHLCV chuẩn:
        - Open: Lấy giá Open đầu kỳ
        - High: Lấy giá High cao nhất kỳ
        - Low: Lấy giá Low thấp nhất kỳ
        - Close: Lấy giá Close cuối kỳ
        - Volume: Tổng Volume trong kỳ
        """
        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }
        if "Adj Close" in df.columns:
            agg_dict["Adj Close"] = "last"

        df_resampled = df.resample(rule).agg(agg_dict).dropna()
        logger.info(f"Đã Resample dữ liệu sang khung '{rule}'. Kích thước mới: {df_resampled.shape}")
        return df_resampled

    def fit_transform_scale(self, df: pd.DataFrame, columns: list, method: str = "standard") -> pd.DataFrame:
        """
        Chuẩn hóa thang đo (Scaling) các cột tính năng.
        method: 'standard' | 'minmax' | 'robust'
        """
        df_scaled = df.copy()
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method phải là 'standard', 'minmax' hoặc 'robust'")

        df_scaled[columns] = self.scaler.fit_transform(df[columns])
        logger.info(f"Đã Scale các cột {columns} bằng phương pháp '{method}'.")
        return df_scaled
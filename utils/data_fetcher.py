import os
import sqlite3
import logging
from typing import Optional, Union, Dict, Any
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests

# Cấu hình Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataFetcher")


class StockDataFetcher:
    """Mô-đun thu thập dữ liệu đa nguồn chuyên nghiệp cho Quantitative Trading."""

    def __init__(self, db_path: str = "data/raw/stock_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs("data/raw/parquet", exist_ok=True)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa tên cột và kiểu dữ liệu về dạng tiêu chuẩn: Open, High, Low, Close, Volume."""
        if df.empty:
            return df
        
        # Flatten MultiIndex Columns nếu có (thường gặp ở yfinance phiên bản mới)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Chuyển tên cột về dạng chuẩn hóa
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", 
            "close": "Close", "adj close": "Adj Close", 
            "vol": "Volume", "volume": "Volume"
        }, inplace=True)

        # Giữ lại các cột chuẩn
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Dữ liệu thiếu cột bắt buộc: {col}")

        # Đảm bảo Index là DatetimeIndex và gán tên là 'Date'
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df = df.sort_index()
        
        # Ép kiểu dữ liệu numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        keep_cols = required_cols + (["Adj Close"] if "Adj Close" in df.columns else [])
        res_df = df[keep_cols].copy()
        res_df.index.name = "Date"
        return res_df

    def fetch_yahoo(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Thu thập dữ liệu từ Yahoo Finance (Tự động hỗ trợ mã VN)."""
        # Danh sách mã chỉ số thế giới/Mỹ phổ biến giữ nguyên, còn lại các mã 3 ký tự viết hoa tự động kiểm tra
        if not ticker.startswith("^") and not ticker.endswith(".VN") and len(ticker) == 3 and ticker.isupper():
            # Tự động chuyển đổi nếu là mã cổ phiếu VN 3 chữ cái
            logger.info(f"Phát hiện mã 3 ký tự '{ticker}', tự động chuyển đổi thành '{ticker}.VN'")
            ticker = f"{ticker}.VN"

        logger.info(f"Đang tải dữ liệu Yahoo Finance cho {ticker} từ {start_date} đến {end_date}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                logger.warning(f"Không có dữ liệu trả về cho ticker {ticker}")
                return pd.DataFrame()
            
            df = self._standardize_dataframe(df)
            logger.info(f"Tải thành công {len(df)} bản ghi cho {ticker}")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu Yahoo Finance ({ticker}): {e}")
            raise
        
    def fetch_binance(self, symbol: str, interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """Thu thập dữ liệu Crypto từ Binance Public REST API."""
        logger.info(f"Đang tải dữ liệu Binance cho {symbol} (Khung: {interval})...")
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            cols = ["Open_Time", "Open", "High", "Low", "Close", "Volume", 
                    "Close_Time", "Quote_Asset_Volume", "Number_of_Trades", 
                    "Taker_Buy_Base", "Taker_Buy_Quote", "Ignore"]
            
            df = pd.DataFrame(data, columns=cols)
            df["Date"] = pd.to_datetime(df["Open_Time"], unit="ms")
            df.set_index("Date", inplace=True)
            
            df = self._standardize_dataframe(df)
            logger.info(f"Tải thành công {len(df)} bản ghi Crypto cho {symbol}")
            return df
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu Binance ({symbol}): {e}")
            raise

    def fetch_vnindex_sample(self, ticker: str = "VNINDEX") -> pd.DataFrame:
        """Thu thập dữ liệu chỉ số Việt Nam."""
        logger.info(f"Đang tải dữ liệu VNINDEX/Cổ phiếu VN: {ticker}...")
        
        if ticker == "VNINDEX":
            df = self.fetch_yahoo("^VNINDEX", start_date="2015-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))
            if df.empty:
                logger.warning("Không thể lấy dữ liệu ^VNINDEX từ Yahoo, tự động chuyển sang mô phỏng ETF E1VFVN30.VN")
                df = self.fetch_yahoo("E1VFVN30.VN", start_date="2015-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))
            return df
        else:
            yf_ticker = f"{ticker}.VN" if not ticker.endswith(".VN") else ticker
            return self.fetch_yahoo(yf_ticker, start_date="2015-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))

    def save_to_parquet(self, df: pd.DataFrame, ticker: str):
        """Lưu DataFrame xuống định dạng Apache Parquet."""
        if df.empty:
            logger.warning(f"DataFrame rỗng, không thể lưu Parquet cho {ticker}")
            return
        file_path = f"data/raw/parquet/{ticker}.parquet"
        df.to_parquet(file_path, engine="pyarrow", compression="snappy")
        logger.info(f"Đã lưu dữ liệu {ticker} vào Parquet: {file_path}")

    def save_to_sqlite(self, df: pd.DataFrame, table_name: str):
        """Lưu DataFrame vào CSDL SQLite và ép lưu cột Index tên là Date."""
        if df.empty:
            logger.warning(f"DataFrame rỗng, không thể lưu SQLite cho bảng {table_name}")
            return
            
        df_to_save = df.copy()
        df_to_save.index.name = "Date"
        
        with sqlite3.connect(self.db_path) as conn:
            df_to_save.to_sql(table_name, conn, if_exists="replace", index=True, index_label="Date")
        logger.info(f"Đã lưu dữ liệu {table_name} vào SQLite DB: {self.db_path}")

    def load_from_sqlite(self, table_name: str) -> pd.DataFrame:
        """Đọc dữ liệu từ SQLite DB một cách linh hoạt và an toàn."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        if df.empty:
            return df
            
        # Tìm cột chứa ngày tháng (Date hoặc index)
        date_col = next((c for c in ["Date", "date", "index", "Datetime"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.index.name = "Date"
            
        return df
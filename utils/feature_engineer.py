# utils/feature_engineer.py
import pandas as pd
import ta
import numpy as np
import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureEngineer")

class TechnicalFeatures:
    """Mô-đun tự động tạo hàng loạt chỉ báo kỹ thuật và bổ sung dữ liệu vĩ mô toàn cầu."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        self.df.sort_index(inplace=True)

    def _fetch_macro_data(self):
        """Hàm phụ trợ tải dữ liệu vĩ mô (DXY và S&P 500) để tiêm vào tập dữ liệu"""
        try:
            macro_tickers = {"DXY": "DX-Y.NYB", "SP500": "^GSPC"}
            df_macro = pd.DataFrame()

            for name, ticker in macro_tickers.items():
                temp_df = yf.download(ticker, period="1y", interval="1d", progress=False)
                if not temp_df.empty:
                    if isinstance(temp_df.columns, pd.MultiIndex):
                        temp_df.columns = temp_df.columns.get_level_values(0)
                    df_macro[name] = temp_df['Close']

            # Xử lý lấp đầy dữ liệu trống (nếu có lệch ngày giao dịch)
            df_macro = df_macro.ffill().dropna()
            return df_macro
        except Exception as e:
            logger.warning(f"Không thể tải dữ liệu vĩ mô, bỏ qua bước này: {e}")
            return None

    def generate_all_features(self) -> pd.DataFrame:
        logger.info("Bắt đầu tính toán Technical Indicators & Macro Features...")
        initial_len = len(self.df)

        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        volume = self.df['Volume']

        # 1. Chỉ báo Xu hướng (Trend)
        self.df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        self.df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        self.df['EMA_20'] = ta.trend.ema_indicator(close, window=20)
        self.df['MACD'] = ta.trend.macd(close)
        self.df['MACD_Signal'] = ta.trend.macd_signal(close)
        self.df['ADX'] = ta.trend.adx(high, low, close, window=14)

        # 2. Chỉ báo Động lượng (Momentum)
        self.df['RSI_14'] = ta.momentum.rsi(close, window=14)
        self.df['Stoch_k'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
        self.df['ROC'] = ta.momentum.roc(close, window=10)

        # 3. Chỉ báo Biến động (Volatility)
        self.df['BBU_20_2.0'] = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
        self.df['BBL_20_2.0'] = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
        self.df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)

        # 4. Chỉ báo Khối lượng (Volume)
        self.df['OBV'] = ta.volume.on_balance_volume(close, volume)
        self.df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)

        # 5. Tích hợp Dữ liệu Vĩ mô (Macroeconomic Features)
        df_macro = self._fetch_macro_data()
        if df_macro is not None:
            # Join dữ liệu vĩ mô theo chỉ mục thời gian (Index)
            self.df = self.df.join(df_macro, how='left')
            # Lấp đầy các giá trị vĩ mô còn thiếu bằng phương pháp forward fill
            self.df['DXY'] = self.df['DXY'].ffill().bfill()
            self.df['SP500'] = self.df['SP500'].ffill().bfill()

        # Xóa các dòng NaN bị thiếu ở đầu do độ trễ của các chỉ báo kỹ thuật
        self.df.dropna(inplace=True)
        final_len = len(self.df)

        logger.info(f"Hoàn tất. Tổng số features (bao gồm vĩ mô): {self.df.shape[1]}")
        logger.info(f"Đã loại bỏ {initial_len - final_len} dòng NaN ở đầu chuỗi dữ liệu.")

        return self.df
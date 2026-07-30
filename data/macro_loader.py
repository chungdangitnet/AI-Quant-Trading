# data/macro_loader.py
import yfinance as yf
import pandas as pd

def get_macro_indicators():
    """
    Hàm tải dữ liệu vĩ mô: DXY (Chỉ số sức mạnh USD) và S&P 500 (Đại diện thị trường chung)
    """
    try:
        # Tải dữ liệu DXY và S&P 500 trong 100 ngày gần nhất
        macro_tickers = {"DXY": "DX-Y.NYB", "SP500": "^GSPC"}
        df_macro = pd.DataFrame()

        for name, ticker in macro_tickers.items():
            temp_df = yf.download(ticker, period="100d", interval="1d", progress=False)
            if not temp_df.empty:
                if isinstance(temp_df.columns, pd.MultiIndex):
                    temp_df.columns = temp_df.columns.get_level_values(0)
                df_macro[name] = temp_df['Close']

        # Xử lý lấp đầy dữ liệu bị thiếu (nếu có ngày nghỉ lễ lệch nhau)
        df_macro = df_macro.ffill().dropna()
        return df_macro
    except Exception as e:
        print(f"⚠️ Không thể tải dữ liệu vĩ mô: {e}")
        return None
# data/vn_loader.py
import pandas as pd
from datetime import datetime, timedelta
from vnstock.api.quote import Quote

def get_vn_stock_data(symbol: str, days: int = 100):
    """
    Hàm lấy dữ liệu lịch sử chứng khoán Việt Nam sử dụng vnstock API mới (Quote)
    """
    try:
        # Tính khoảng thời gian lấy dữ liệu
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Khởi tạo đối tượng Quote theo chuẩn API mới của vnstock (dùng nguồn 'VCI')
        q = Quote(symbol=symbol, source='VCI')
        df = q.history(start=start_date, end=end_date, interval='1D')
        
        if df is None or df.empty:
            return None
            
        # Chuẩn hóa tên cột thời gian
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'Date'})
        elif 'tradingDate' in df.columns:
            df = df.rename(columns={'tradingDate': 'Date'})
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # Chuẩn hóa tên các cột giá trị viết hoa để khớp với TechnicalFeatures
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Lọc ra các cột cần thiết và ép kiểu dữ liệu số
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df[required_cols].dropna()
        
    except Exception as e:
        print(f"⚠️ Lỗi khi tải dữ liệu vnstock cho mã {symbol}: {e}")
        return None
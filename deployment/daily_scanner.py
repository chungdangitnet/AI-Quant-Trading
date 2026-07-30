# deployment/daily_scanner.py
import sys
import os
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config.tickers import WATCHLIST
from utils.feature_engineer import TechnicalFeatures
from data.vn_loader import get_vn_stock_data  # Import module vnstock vừa tạo

def run_daily_scan():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BẮT ĐẦU QUÉT DANH MỤC LAZY (MỸ + VIỆT NAM)...\\n")
    
    model_path = os.path.join(parent_dir, "models", "xgboost_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(parent_dir, "data", "processed", "xgboost_model.pkl")
        
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model tại: {model_path}")
        return

    model = joblib.load(model_path)
    expected_features = model.feature_names_in_

    results = []

    for ticker in WATCHLIST:
        try:
            # PHÂN NHÁNH LẤY DỮ LIỆU THÔNG MINH
            if ".VN" in ticker:
                clean_symbol = ticker.replace(".VN", "")
                df = get_vn_stock_data(clean_symbol, days=100)
            else:
                df = yf.download(ticker, period="100d", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

            if df is None or len(df) < 50:
                continue

            # Tính toán chỉ báo kỹ thuật
            te = TechnicalFeatures(df)
            df_features = te.generate_all_features()

            latest_data = df_features.iloc[[-1]]
            X_live = latest_data[expected_features]

            pred = model.predict(X_live)[0]
            prob = model.predict_proba(X_live)[0][1] * 100 

            currency = "VND" if ".VN" in ticker else "USD"

            results.append({
                "Ticker": ticker,
                "Price": round(latest_data['Close'].values[0], 2),
                "Currency": currency,
                "Signal": "BUY 🟢" if pred == 1 else "SELL/HOLD 🔴",
                "Confidence (%)": round(prob, 2)
            })

        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý {ticker}: {e}")

    if not results:
        print("⚠️ Không có dữ liệu nào được quét thành công.")
        return

    scan_df = pd.DataFrame(results)
    scan_df.sort_values(by="Confidence (%)", ascending=False, inplace=True)
    
    reports_dir = os.path.join(parent_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    today_str = datetime.now().strftime('%Y_%m_%d')
    report_path = os.path.join(reports_dir, f"daily_scan_{today_str}.csv")
    scan_df.to_csv(report_path, index=False)
    
    print("="*65)
    print(" TOP CƠ HỘI MUA TIỀN NĂNG NHẤT HÔM NAY (MỸ + VN)")
    print("="*65)
    print(scan_df.head(15).to_string(index=False))
    print("="*65)
    print(f"\n✅ Báo cáo chi tiết đã lưu tại: {report_path}")

if __name__ == "__main__":
    run_daily_scan()
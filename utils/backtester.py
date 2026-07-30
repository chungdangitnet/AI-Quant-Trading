# utils/backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Backtester")


class VectorizedBacktester:
    """Mô-đun kiểm định chiến lược giao dịch tự động chống rò rỉ dữ liệu."""

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        """
        initial_capital: Vốn ban đầu ($)
        fee_rate: Phí giao dịch (0.001 = 0.1% mỗi lượt giao dịch)
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

    def run_backtest(self, predictions: np.ndarray) -> pd.DataFrame:
        """Thực thi mô phỏng giao dịch dựa trên dự đoán từ Model AI."""
        logger.info("Bắt đầu mô phỏng Backtest...")

        self.df['Signal'] = predictions
        
        # Lợi nhuận hàng ngày của cổ phiếu
        self.df['Asset_Return'] = self.df['Close'].pct_change().fillna(0)

        # Vị thế nắm giữ: Dùng tín hiệu hôm nay (t-1) áp dụng cho lợi nhuận ngày mai (t)
        self.df['Position'] = self.df['Signal'].shift(1).fillna(0)

        # Phát hiện thời điểm vào/ra lệnh để tính phí giao dịch
        self.df['Trade_Occurred'] = (self.df['Position'].diff().abs() > 0).astype(int)
        self.df['Transaction_Cost'] = self.df['Trade_Occurred'] * self.fee_rate

        # Lợi nhuận ròng của chiến lược
        self.df['Strategy_Return'] = (self.df['Asset_Return'] * self.df['Position']) - self.df['Transaction_Cost']

        # Tính toán đường cong vốn (Equity Curve)
        self.df['Equity_AI'] = self.initial_capital * (1 + self.df['Strategy_Return']).cumprod()
        self.df['Equity_BuyHold'] = self.initial_capital * (1 + self.df['Asset_Return']).cumprod()

        return self.df

    def evaluate_performance(self) -> dict:
        """Tính toán các chỉ số đo lường hiệu suất tài chính."""
        df = self.df

        # Lợi nhuận tổng
        total_ret_ai = (df['Equity_AI'].iloc[-1] / self.initial_capital) - 1
        total_ret_bh = (df['Equity_BuyHold'].iloc[-1] / self.initial_capital) - 1

        # Sharpe Ratio (Giả định Lãi suất phi rủi ro = 0)
        daily_ret = df['Strategy_Return']
        sharpe_ratio = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0

        # Maximum Drawdown (MDD)
        peak_ai = df['Equity_AI'].cummax()
        drawdown_ai = (peak_ai - df['Equity_AI']) / peak_ai
        max_drawdown_ai = drawdown_ai.max()

        peak_bh = df['Equity_BuyHold'].cummax()
        drawdown_bh = (peak_bh - df['Equity_BuyHold']) / peak_bh
        max_drawdown_bh = drawdown_bh.max()

        # Win Rate (% số ngày có lợi nhuận dương khi mở vị thế)
        active_trades = df[df['Position'] == 1]
        win_rate = (active_trades['Strategy_Return'] > 0).sum() / len(active_trades) if len(active_trades) > 0 else 0

        metrics = {
            "Total Return (AI Strategy)": f"{total_ret_ai * 100:.2f}%",
            "Total Return (Buy & Hold)": f"{total_ret_bh * 100:.2f}%",
            "Sharpe Ratio (AI)": f"{sharpe_ratio:.2f}",
            "Max Drawdown (AI)": f"{max_drawdown_ai * 100:.2f}%",
            "Max Drawdown (Buy & Hold)": f"{max_drawdown_bh * 100:.2f}%",
            "Win Rate": f"{win_rate * 100:.2f}%",
            "Total Trades Executed": int(df['Trade_Occurred'].sum())
        }

        return metrics
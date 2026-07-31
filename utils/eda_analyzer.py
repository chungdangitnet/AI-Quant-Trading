import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

class FinancialEDA:
    """
    Class hỗ trợ Exploratory Data Analysis (EDA) cho dữ liệu tài chính / chuỗi thời gian.
    """
    def __init__(self, df: pd.DataFrame, target_col: str = "Close"):
        self.df = df.copy()
        self.target_col = target_col

    def summary_overview(self) -> pd.DataFrame:
        """
        In báo cáo thống kê mô tả chi tiết các biến số (Bao gồm Skewness & Kurtosis).
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        stats_df = numeric_df.describe().T
        stats_df["skewness"] = numeric_df.skew()
        stats_df["kurtosis"] = numeric_df.kurtosis()
        return stats_df

    def plot_distribution_suite(self, col: str = "Log_Return"):
        """
        Vẽ bộ biểu đồ phân phối: Histogram, KDE và Q-Q Plot.
        """
        if col not in self.df.columns:
            print(f"Cảnh báo: Cột '{col}' không tồn tại trong DataFrame.")
            return

        data = self.df[col].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram & KDE
        sns.histplot(data, kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Phân phối Lợi nhuận ({col})", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(col)

        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[1])
        axes[1].set_title(f"Q-Q Plot ({col})", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.show()

    def plot_correlations(self):
        """
        Vẽ ma trận tương quan Pearson giữa các biến số.
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
        plt.title("Ma trận Tương quan Pearson", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def plot_seasonality_and_anova(self, alpha=0.05):
        """
        Phân tích Tính mùa vụ theo Ngày trong tuần & Kiểm định ANOVA / Kruskal-Wallis.
        """
        if "Day_of_Week" not in self.df.columns or "Log_Return" not in self.df.columns:
            print("Cảnh báo: DataFrame cần chứa cột 'Day_of_Week' và 'Log_Return'.")
            return

        # TỰ ĐỘNG CHUYỂN ĐỔI: Nếu cột đang là số (0-4), ánh xạ sang chữ
        if pd.api.types.is_numeric_dtype(self.df["Day_of_Week"]):
            day_mapping = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
            self.df["Day_of_Week"] = self.df["Day_of_Week"].map(day_mapping)

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        valid_days = [d for d in day_order if d in self.df["Day_of_Week"].dropna().unique()]

        # 1. Chuẩn bị dữ liệu theo đúng thứ tự ngày trong tuần
        days_data = [
            self.df[self.df["Day_of_Week"] == day]["Log_Return"].dropna().values 
            for day in valid_days
        ]

        if not days_data:
            print("Lỗi: Không tìm thấy dữ liệu ngày hợp lệ để phân tích.")
            return

        # 2. Vẽ Boxplot bằng Matplotlib
        plt.figure(figsize=(10, 6))
        # Đã sửa 'tick_labels' thành 'labels' để không bị lỗi trên Matplotlib mới
        plt.boxplot(days_data, labels=valid_days, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.title("Phân phối Lợi nhuận theo Các Ngày trong Tuần", fontsize=14, fontweight="bold")
        plt.ylabel("Log Return")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

        # 3. Kiểm định ANOVA & Kruskal-Wallis
        f_stat, p_val_anova = stats.f_oneway(*days_data)
        h_stat, p_val_kruskal = stats.kruskal(*days_data)

        # 4. In Báo cáo Thống kê Động
        print("=" * 80)
        print("BÁO CÁO PHÂN TÍCH MÙA VỤ & KIỂM ĐỊNH THỐNG KÊ")
        print("=" * 80)
        print(f"Kiểm định ANOVA F-test (F-stat = {f_stat:.4f}): p-value = {p_val_anova:.4f}")
        print(f"Kiểm định Kruskal-Wallis (H-stat = {h_stat:.4f}): p-value = {p_val_kruskal:.4f}")
        print("-" * 80)

        # Đánh giá dựa trên Kruskal-Wallis (ưu tiên do Log Return không chuẩn)
        if p_val_kruskal > alpha:
            print(f"1. Nhận xét: p-value > {alpha} ở kiểm định Kruskal-Wallis.")
            print("2. Giải thích: Không có bằng chứng thống kê cho thấy trung bình lợi nhuận giữa các ngày có sự khác biệt.")
            print("3. Ý nghĩa: Hiệu ứng 'Day-of-the-Week' (ví dụ: Monday Effect) không đủ mạnh để khai thác.")
            print("4. Kết luận: KHÔNG NÊN dựa vào yếu tố Ngày trong tuần làm tín hiệu giao dịch độc lập.")
        else:
            print(f"1. Nhận xét: p-value <= {alpha} (Có ý nghĩa thống kê).")
            print("2. Giải thích: Tồn tại sự khác biệt có ý nghĩa thống kê về lợi nhuận giữa ít nhất 2 ngày trong tuần.")
            print("3. Ý nghĩa: Có dấu hiệu của bất thường mùa vụ (Seasonal Anomaly) trên cổ phiếu này.")
            print("4. Kết luận: CÓ THỂ cân nhắc kết hợp biến Ngày trong tuần làm Feature phụ trợ cho mô hình ML.")
        
        print("=" * 80 + "\n")
    def plot_day_of_year_vs_profit(self, period=60):
        """
        Vẽ biểu đồ Tương quan giữa Ngày trong năm (Day of Year) và Lợi nhuận chu kỳ N ngày.
        """
        df_plot = self.df.copy()
        
        # Trích xuất Ngày trong năm từ DatetimeIndex
        df_plot['Day_of_Year'] = df_plot.index.dayofyear
        
        # Tính Lợi nhuận tích lũy chu kỳ N ngày (%)
        df_plot['Rolling_Profit'] = (df_plot['Close'] - df_plot['Close'].shift(period)) / df_plot['Close'].shift(period) * 100
        df_plot = df_plot.dropna(subset=['Rolling_Profit'])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df_plot['Day_of_Year'], df_plot['Rolling_Profit'], color='purple', s=6, alpha=0.6, edgecolors='none')
        plt.title(f"Tương quan giữa Ngày và Lợi Nhuận chu kỳ {period} ngày", fontsize=14, fontweight="bold")
        plt.xlabel("Day of year", fontsize=11)
        plt.ylabel("Profit (%)", fontsize=11)
        plt.grid(True, linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_macd_vs_profit(self, period=60):
        """
        Vẽ biểu đồ Tương quan giữa MACD và Lợi nhuận chu kỳ N ngày.
        """
        df_plot = self.df.copy()
        
        # Tính MACD (EMA12 - EMA26)
        ema_12 = df_plot['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_plot['Close'].ewm(span=26, adjust=False).mean()
        df_plot['MACD'] = ema_12 - ema_26
        
        # Tính Lợi nhuận tích lũy chu kỳ N ngày (%)
        df_plot['Rolling_Profit'] = (df_plot['Close'] - df_plot['Close'].shift(period)) / df_plot['Close'].shift(period) * 100
        df_plot = df_plot.dropna(subset=['MACD', 'Rolling_Profit'])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df_plot['MACD'], df_plot['Rolling_Profit'], color='purple', s=8, alpha=0.5, edgecolors='none')
        plt.title(f"Tương quan MACD và Lợi Nhuận chu kỳ {period} ngày", fontsize=14, fontweight="bold")
        plt.xlabel("MACD", fontsize=11)
        plt.ylabel("Lợi Nhuận (%)", fontsize=11)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.grid(True, linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()
    def plot_rsi_vs_profit(self, period=60, rsi_period=14, step=10):
        """
        Tính chỉ báo RSI, làm tròn theo bước 'step' (ví dụ: 10, 20, 30...) 
        và vẽ biểu đồ tương quan với Lợi nhuận chu kỳ N ngày.
        """
        df_plot = self.df.copy()
        
        # 1. Tính chỉ báo RSI 14 phiên chuẩn
        delta = df_plot['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        
        rs = gain / loss
        df_plot['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. Làm tròn RSI về mốc bội số của 'step' (mặc định bước 10 -> 10, 20, 30, 40, 50, 60, 70)
        df_plot['RSI_Rounded'] = (df_plot['RSI'] / step).round() * step
        
        # 3. Tính Lợi nhuận tích lũy chu kỳ N ngày (%)
        df_plot['Rolling_Profit'] = (df_plot['Close'] - df_plot['Close'].shift(period)) / df_plot['Close'].shift(period) * 100
        
        # Bỏ dữ liệu NaN
        df_plot = df_plot.dropna(subset=['RSI_Rounded', 'Rolling_Profit'])
        
        # 4. Vẽ biểu đồ Scatter Plot giống hệt hình ảnh mẫu
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_plot['RSI_Rounded'], 
            df_plot['Rolling_Profit'], 
            color='purple', 
            s=8, 
            alpha=0.4, 
            edgecolors='none'
        )
        
        plt.title(f"Tương quan RSI làm tròn và Lợi Nhuận chu kỳ {period} ngày", fontsize=14, fontweight="bold")
        plt.xlabel("RSI", fontsize=11)
        plt.ylabel("Lợi Nhuận (%)", fontsize=11)
        plt.grid(True, linestyle='-', alpha=0.5)
        
        # Đặt mốc trục X cố định theo đúng các khoảng RSI
        ticks = range(10, 80, step)
        plt.xticks(ticks)
        
        plt.tight_layout()
        plt.show()
    def plot_macd_signal_vs_profit(self, period=60, signal_period=9):
        """
        Tính đường Tín hiệu MACD (MACD Signal) và vẽ biểu đồ tương quan 
        với Lợi nhuận chu kỳ N ngày.
        """
        df_plot = self.df.copy()
        
        # 1. Tính đường MACD (EMA12 - EMA26)
        ema_12 = df_plot['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_plot['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        
        # 2. Tính đường MACD Signal (EMA9 của MACD)
        df_plot['MACD_Signal'] = macd.ewm(span=signal_period, adjust=False).mean()
        
        # 3. Tính Lợi nhuận tích lũy chu kỳ N ngày (%)
        df_plot['Rolling_Profit'] = (df_plot['Close'] - df_plot['Close'].shift(period)) / df_plot['Close'].shift(period) * 100
        
        # Bỏ dữ liệu NaN
        df_plot = df_plot.dropna(subset=['MACD_Signal', 'Rolling_Profit'])
        
        # 4. Vẽ biểu đồ Scatter Plot giống hệt ảnh mẫu
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_plot['MACD_Signal'], 
            df_plot['Rolling_Profit'], 
            color='purple', 
            s=8, 
            alpha=0.5, 
            edgecolors='none'
        )
        
        # Cấu hình tiêu đề & nhãn trục
        plt.title(f"Tương quan MACD Signal và Lợi Nhuận chu kỳ {period} ngày", fontsize=14, fontweight="bold")
        plt.xlabel("MACD Signal", fontsize=11)
        plt.ylabel("Lợi Nhuận (%)", fontsize=11)
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.grid(True, linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    def plot_year_vs_profit(self, period=60):
        """
        Trích xuất Năm từ Index và vẽ biểu đồ tương quan 
        với Lợi nhuận chu kỳ N ngày.
        """
        df_plot = self.df.copy()
        
        # 1. Trích xuất Năm từ DatetimeIndex
        df_plot['Year'] = df_plot.index.year
        
        # 2. Tính Lợi nhuận tích lũy chu kỳ N ngày (%)
        df_plot['Rolling_Profit'] = (df_plot['Close'] - df_plot['Close'].shift(period)) / df_plot['Close'].shift(period) * 100
        
        # Bỏ dữ liệu NaN
        df_plot = df_plot.dropna(subset=['Year', 'Rolling_Profit'])
        
        # 3. Vẽ biểu đồ Scatter Plot giống hệt ảnh mẫu
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_plot['Year'], 
            df_plot['Rolling_Profit'], 
            color='purple', 
            s=8, 
            alpha=0.4, 
            edgecolors='none'
        )
        
        # Cấu hình giao diện & nhãn
        plt.title(f"Tương quan giữa Năm và Lợi Nhuận chu kỳ {period} ngày", fontsize=14, fontweight="bold")
        plt.xlabel("Year", fontsize=11)
        plt.ylabel("profit", fontsize=11)
        plt.grid(True, linestyle='-', alpha=0.5)
        
        # Ép trục X hiển thị đúng các số năm nguyên (2017, 2018, 2019...)
        years = sorted(df_plot['Year'].unique())
        plt.xticks(years)
        
        plt.tight_layout()
        plt.show()
    def plot_stoch_k_vs_profit(self, period=60, k_period=14, step=10):
        """
        Tính chỉ báo Stochastic %K (14 phiên), làm tròn theo bước 'step' 
        và vẽ biểu đồ tương quan với Lợi nhuận chu kỳ N ngày.
        """
        df_plot = self.df.copy()
        
        # 1. Tính chỉ báo Stochastic Oscillator %K
        low_min = df_plot['Low'].rolling(window=k_period).min()
        high_max = df_plot['High'].rolling(window=k_period).max()
        
        df_plot['Stoch_K'] = 100 * ((df_plot['Close'] - low_min) / (high_max - low_min))
        
        # 2. Làm tròn Stoch_K về mốc bội số của 'step' (0, 10, 20, ..., 100)
        df_plot['Stoch_K_Rounded'] = (df_plot['Stoch_K'] / step).round() * step
        
        # 3. Tính Lợi nhuận tích lũy chu kỳ N ngày (%)
        df_plot['Rolling_Profit'] = (df_plot['Close'] - df_plot['Close'].shift(period)) / df_plot['Close'].shift(period) * 100
        
        # Bỏ dữ liệu NaN
        df_plot = df_plot.dropna(subset=['Stoch_K_Rounded', 'Rolling_Profit'])
        
        # 4. Vẽ biểu đồ Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df_plot['Stoch_K_Rounded'], 
            df_plot['Rolling_Profit'], 
            color='purple', 
            s=8, 
            alpha=0.4, 
            edgecolors='none'
        )
        
        # Cấu hình giao diện & nhãn
        plt.title(f"Tương quan Stoch_K làm tròn và Lợi Nhuận chu kỳ {period} ngày", fontsize=14, fontweight="bold")
        plt.xlabel("Stoch_K", fontsize=11)
        plt.ylabel("Lợi Nhuận (%)", fontsize=11)
        plt.grid(True, linestyle='-', alpha=0.5)
        
        # Đặt mốc trục X cố định từ 0 đến 100
        ticks = range(0, 110, step)
        plt.xticks(ticks)
        
        plt.tight_layout()
        plt.show()
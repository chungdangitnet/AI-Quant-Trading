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

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        valid_days = [d for d in day_order if d in self.df["Day_of_Week"].unique()]

        # 1. Chuẩn bị dữ liệu theo đúng thứ tự ngày trong tuần
        days_data = [
            self.df[self.df["Day_of_Week"] == day]["Log_Return"].dropna().values 
            for day in valid_days
        ]

        # 2. Vẽ Boxplot bằng Matplotlib
        plt.figure(figsize=(10, 6))
        plt.boxplot(days_data, tick_labels=valid_days, patch_artist=True, boxprops=dict(facecolor="lightblue"))
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
            print(f"1. Nhận xét: p-value > {alpha} ở cả 2 kiểm định ANOVA và Kruskal-Wallis.")
            print("2. Giải thích: Không có bằng chứng thống kê cho thấy trung bình lợi nhuận giữa các ngày có sự khác biệt rõ rệt.")
            print("3. Ý nghĩa: Hiệu ứng 'Day-of-the-Week' (ví dụ: Monday Effect) không đủ mạnh để khai thác.")
            print("4. Kết luận: KHÔNG NÊN dựa vào yếu tố Ngày trong tuần làm tín hiệu giao dịch độc lập.")
        else:
            print(f"1. Nhận xét: p-value <= {alpha} (Có ý nghĩa thống kê).")
            print("2. Giải thích: Tồn tại sự khác biệt có ý nghĩa thống kê về lợi nhuận giữa ít nhất 2 ngày trong tuần.")
            print("3. Ý nghĩa: Có dấu hiệu của bất thường mùa vụ (Seasonal Anomaly) trên cổ phiếu này.")
            print("4. Kết luận: CÓ THỂ cân nhắc kết hợp biến Ngày trong tuần làm Feature phụ trợ cho mô hình ML.")
        
        print("=" * 80 + "\n")
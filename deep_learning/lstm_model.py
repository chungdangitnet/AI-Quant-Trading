# deep_learning/lstm_model.py
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """Mạng Nơ-ron Hồi quy LSTM dự báo chuỗi thời gian chứng khoán."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(StockLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Đầu vào x có dạng: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Lấy kết quả ở bước thời gian cuối cùng (last time-step)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
# config/tickers.py

# Danh mục cổ phiếu Mỹ
US_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"
]

# Danh mục cổ phiếu Việt Nam (VN30 và các mã thanh khoản cao)
VN_WATCHLIST = [
    "HPG.VN", "VCB.VN", "VIC.VN", "VHM.VN", "FPT.VN", 
    "MWG.VN", "TCB.VN", "MBB.VN", "ACB.VN", "STB.VN",
    "SSI.VN", "VND.VN", "VIC.VN", "GAS.VN", "VIC.VN"
]

# Gộp chung hoặc tách riêng tùy ý bạn (Ở đây ta gộp chung để quét một thể)
WATCHLIST = US_WATCHLIST + VN_WATCHLIST
# 從 Hugging Face 下載數據指南

## 數據來源

**Dataset URL:** https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data

**數據信息：**
- 23 個加密貨幣幣種
- 15m 和 1h 兩個時間框架
- 共 46 個文件
- 總數據點：4,819,964
- 總大小：110.57 MB

---

## 快速開始

### 第 1 步：安裝依賴

```bash
pip install -r requirements.txt
```

會安裝:
- `pandas` - 數據處理
- `numpy` - 數值計算
- `huggingface_hub` - 從 HF 下載

### 第 2 步：下載所有數據

```bash
python download_data_from_hf.py
```

**預期輸出：**
```
==========================================
開始下載 46 個文件
==========================================

下載: BTCUSDT 15m
  HF 路徑: klines/BTCUSDT/BTC_15m.parquet
  成功! 數據量: 219,643 行
  文件位置: data/...
  列: ['open_time', 'open', 'high', 'low', 'close', 'volume', ...]
  已保存種 CSV: data/BTCUSDT_15m.csv

...

==========================================
下載完成!
  成功: 46
  失敗: 0
  成功率: 100.0%
==========================================

已下載的文件:

CSV 文件 (46):
  - BTCUSDT_15m.csv (4.23 MB)
  - BTCUSDT_1h.csv (0.87 MB)
  - ETHUSDT_15m.csv (3.12 MB)
  ...
```

### 第 3 步：檢查數據

```python
import pandas as pd

# 讀取已下載的數據
df = pd.read_csv('data/BTCUSDT_15m.csv')
print(df.head())
print(f"數據形狀: {df.shape}")
print(f"列名: {df.columns.tolist()}")
```

---

## 使用方式

### 方式 1：下載所有數據（推薦）

```bash
python download_data_from_hf.py
```

在 `main()` 函數中：
```python
# 已默認選擇
download_all()
```

### 方式 2：只下載特定幣種

編輯 `download_data_from_hf.py`，修改 `main()` 函數：

```python
def main():
    # 只下載 BTC, ETH, BNB
    download_specific_symbols([
        'BTCUSDT',
        'ETHUSDT',
        'BNBUSDT'
    ])
```

然後運行：
```bash
python download_data_from_hf.py
```

### 方式 3：下載單個文件

```python
from download_data_from_hf import HFDataDownloader

downloader = HFDataDownloader()

# 下載單個文件
downloader.download_single_file('BTCUSDT', '15m')

# 查看已下載的文件
downloader.list_available_files()
```

---

## 支持的幣種

```
BTCUSDT   - Bitcoin
ETHUSDT   - Ethereum
BNBUSDT   - Binance Coin
XRPUSDT   - Ripple
ADAUSDT   - Cardano
DOGEUSDT  - Dogecoin
MATICUSDT - Polygon
LTCUSDT   - Litecoin
AVAXUSDT  - Avalanche
SOLUSDT   - Solana
ATOMUSDT  - Cosmos
ARBUSDT   - Arbitrum
OPUSDT    - Optimism
UNIUSDT   - Uniswap
LINKUSDT  - Chainlink
FILUSDT   - Filecoin
ETCUSDT   - Ethereum Classic
ALGOUSDT  - Algorand
AAVEUSDT  - Aave
NEARUSDT  - NEAR Protocol
BCHUSDT   - Bitcoin Cash
DOTUSDT   - Polkadot
```

---

## 數據格式

### 下載後的文件結構

```
data/
├── BTCUSDT_15m.csv
├── BTCUSDT_1h.csv
├── ETHUSDT_15m.csv
├── ETHUSDT_1h.csv
├── ...
└── cache/  (HF 緩存，可刪除)
    └── models--zongowo111--v2-crypto-ohlcv-data/
        └── snapshots/
            └── main/
                └── klines/
                    ├── BTCUSDT/BTC_15m.parquet
                    ├── ETHUSDT/ETH_15m.parquet
                    └── ...
```

### CSV 列結構

```
open_time       - 開盤時間 (Unix timestamp)
open           - 開盤價格
high           - 最高價格
low            - 最低價格
close          - 收盤價格
volume         - 交易量
close_time     - 閉盤時間
quote_asset_volume  - 報價資產交易量
number_of_trades    - 交易筆數
taker_buy_base_asset_volume  - 買方基礎資產交易量
taker_buy_quote_asset_volume - 買方報價資產交易量
ignore         - 忽略欄位
```

---

## 常見問題

### Q: 下載需要多長時間？
A: 取決於網絡速度。
- 所有 46 個文件：5-15 分鐘
- 單個文件：幾秒鐘
- 首次下載會比較慢，之後會緩存

### Q: 下載的文件會保存到哪裡？
A: 
- CSV 文件：`data/` 目錄
- 原始 Parquet：`data/cache/` 目錄（HF 緩存）

### Q: 可以只下載 CSV 不保存 Parquet 嗎？
A: 可以，編輯代碼注釋掉保存 CSV 的部分：

```python
# 注釋掉這行
# csv_file = self.data_dir / f"{symbol}_{timeframe}.csv"
# df.to_csv(csv_file, index=False)
```

### Q: 數據已經下載了，怎樣避免重複下載？
A: 已默認設置 `force_download=False`，會自動使用緩存。

### Q: 網絡中斷了怎麼辦？
A: 直接重新運行下載腳本，會自動跳過已下載的文件。

### Q: 怎樣確認下載的數據完整？
A: 運行腳本後會顯示成功/失敗統計，以及已下載文件列表。

---

## 使用下載的數據

### 與標籤創建集成

下載完成後，可以直接運行標籤創建程序：

```bash
# 第 1 步：下載數據
python download_data_from_hf.py

# 第 2 步：創建標籤
python label_v3_clean.py

# 第 3 步：調優參數
python label_parameter_tuning.py
```

### 讀取和分析數據

```python
import pandas as pd
from pathlib import Path

# 讀取所有已下載的 CSV 文件
data_dir = Path('data')
for csv_file in data_dir.glob('*.csv'):
    df = pd.read_csv(csv_file)
    print(f"{csv_file.name}: {len(df)} 行")
    print(f"時間範圍: {df['open_time'].min()} - {df['open_time'].max()}")
    print()
```

---

## 注意事項

1. **網絡連接**：需要穩定的網絡連接
2. **磁盤空間**：需要至少 200 MB 的磁盤空間（CSV + 緩存）
3. **Python 版本**：需要 Python 3.7+
4. **API 限制**：HF 沒有嚴格的下載限制，但大量並發可能會受限

---

## 故障排除

### 錯誤：`FileNotFoundError: klines/BTCUSDT/BTC_15m.parquet`

**原因**：文件名可能不正確或幣種不支持

**解決**：
1. 檢查 HF 上的實際文件名
2. 確保幣種在支持列表中
3. 檢查拼寫（BTCUSDT 不是 BTC_USDT）

### 錯誤：`No module named 'huggingface_hub'`

**原因**：未安裝依賴

**解決**：
```bash
pip install -r requirements.txt
```

### 錯誤：`Connection timeout`

**原因**：網絡問題或 HF 服務器問題

**解決**：
1. 檢查網絡連接
2. 等待幾分鐘後重試
3. 嘗試下載單個文件測試

---

## 高級選項

### 自定義下載邏輯

```python
from download_data_from_hf import HFDataDownloader

# 創建下載器實例
downloader = HFDataDownloader()

# 自定義下載特定組合
custom_symbols = ['BTCUSDT', 'ETHUSDT']
custom_timeframes = ['15m']

downloader.download_all_data(
    symbols=custom_symbols,
    timeframes=custom_timeframes
)
```

### 檢查下載進度

```python
# 在 download_data_from_hf.py 中查看日誌
# 日誌位置：logs/download_YYYYMMDD_HHMMSS.log

# 或實時查看
downloader.list_available_files()
```

---

## 後續步驟

1. ✅ 下載數據
2. 📊 創建標籤 (`python label_v3_clean.py`)
3. 🔧 調優參數 (`python label_parameter_tuning.py`)
4. 🤖 訓練 ML 模型
5. ✅ 驗證模型性能

---

**數據已準備好！開始創建標籤吧！🚀**

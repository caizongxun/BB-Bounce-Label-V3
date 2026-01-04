# 基於盈利性的標籤創建指南

## 核心理念

與之前的標籤系統不同，新的標籤系統**只標記盈利的 K 棒**，讓模型學會預測真正能賺錢的交易機會。

## 標籤定義

| 標籤 | 含義 | 操作 | 說明 |
|------|------|------|------|
| **1** | 下軌有盈利反彈 | 做多 | 觸碰下軌後持有期間 ≥ 0.1% 盈利 |
| **2** | 上軌有盈利反彈 | 做空 | 觸碰上軌後持有期間 ≥ 0.1% 盈利 |
| **0** | 技文帶假信號 | 不交易 | 觸碰後無盈利（虧損或微利） |
| **-1** | 無觸碰 | 中性 | 完全不在 Bollinger Bands 邊界 |

---

## 快速開始

### 第 1 步：運行標籤創建

```bash
python label_profitability_focused.py
```

### 第 2 步：查看結果

**輸出位置：**
```
outputs/labels/BTCUSDT_15m_profitability_labels.csv
outputs/labels/BTCUSDT_1h_profitability_labels.csv
...
```

**標籤分佈示例：**
```
下軌有盈利反彈 (label=1)：1000 個 K 棒
上軌有盈利反彈 (label=2)：800 個 K 棒
技文帶假信號 (label=0)：500 個 K 棒
無觸碰 (label=-1)：97700 個 K 棒

總信號數：2300
有盈利信號比例：78.3%
```

---

## 參數調整

### 持有時間 (holding_bars)

```python
# 持有 5 個 15m K 棒 = 75 分鐘
holding_bars = 5

# 持有 10 個 1h K 棒 = 10 小時
holding_bars = 10
```

**建議：**
- 15m：3-10 根（45 分鐘～2.5 小時）
- 1h：2-8 根（2～8 小時）

### 盈利閾值 (profit_threshold)

```python
# 最低盈利 0.1%
profit_threshold = 0.1

# 最低盈利 0.5%
profit_threshold = 0.5

# 最低盈利 1%
profit_threshold = 1.0
```

**影響：**
- 更低的閾值 → 更多標籤 1 和 2（但含金量低）
- 更高的閾值 → 更少標籤 1 和 2（但都是優質信號）

**推薦：**
- 交易成本 < 0.05% 時，用 0.1%
- 交易成本 0.05-0.1% 時，用 0.2%
- 追求高質量時，用 0.5%～1%

---

## 完整使用流程

### 步驟 1：準備數據

```bash
# 確保數據已下載並轉換為 CSV
python download_data_from_hf.py
```

### 步驟 2：建立單個幣種的標籤

```python
from label_profitability_focused import ProfitabilityLabelCreator

creator = ProfitabilityLabelCreator()

# 為 BTC 15m 創建標籤
creator.run_full_pipeline(
    symbol='BTCUSDT',
    timeframe='15m',
    holding_bars=5,
    profit_threshold=0.1
)
```

### 步驟 3：批量創建標籤（所有幣種）

```python
from label_profitability_focused import ProfitabilityLabelCreator

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'MATICUSDT', 'LTCUSDT', 'AVAXUSDT', 'SOLUSDT',
    'ATOMUSDT', 'ARBUSDT', 'OPUSDT', 'UNIUSDT', 'LINKUSDT',
    'FILUSDT', 'ETCUSDT', 'ALGOUSDT', 'AAVEUSDT', 'NEARUSDT',
    'BCHUSDT', 'DOTUSDT'
]
TIMEFRAMES = ['15m', '1h']

creator = ProfitabilityLabelCreator()

for symbol in SYMBOLS:
    for timeframe in TIMEFRAMES:
        print(f'\n正在處理 {symbol}_{timeframe}...')
        creator.run_full_pipeline(
            symbol,
            timeframe,
            holding_bars=5,
            profit_threshold=0.1
        )
```

### 步驟 4：分析標籤品質

查看日誌輸出，了解：
- 每個幣種的有盈利信號比例
- 技文信號的準確性
- 各標籤的分佈

---

## 標籤輸出解釋

### CSV 文件結構

```
open_time,open,high,low,close,label
2023-01-01 00:00:00,42000,42100,41900,42050,1
2023-01-01 00:15:00,42050,42200,42000,42150,-1
2023-01-01 00:30:00,42150,42300,42050,42100,0
...
```

### 標籤分析

運行時會輸出：

```
標籤統計：
  下軌有盈利反彈 (label=1)：1000 (10.34%)
  上軌有盈利反彈 (label=2)：800 (8.27%)
  技文帶假信號 (label=0)：500 (5.17%)
  無觸碰 (label=-1)：7700 (76.22%)

總信號數：2300
有盈利信號: 1800
有盈利比例: 78.26%
```

---

## 訓練模型

### 數據平衡

由於 label=-1 的數據很多，模型可能會偏向預測 -1。建議：

```python
# 選項 1：只用有信號的數據
training_df = df[df['label'] != -1].copy()

# 選項 2：進行類別權重調整
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df['label']),
    y=df['label']
)

# 選項 3：使用過採樣或欠採樣
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
```

### 特徵工程

模型已自動計算的特徵：
- `volatility` - 波動度（High - Low）
- `body_size` - K 棒實體大小
- `upper_wick` - 上影線長度
- `lower_wick` - 下影線長度

可以進一步添加：
- RSI - 相對強度指數
- MACD - 指數平滑異同移動平均線
- 成交量變化
- 前 N 根 K 棒的趨勢

---

## 常見問題

### Q: 為什麼我的有盈利信號比例這麼低？

**可能原因：**
1. `profit_threshold` 設太高
2. `holding_bars` 設太短
3. 這段時間的市場行情不好

**解決方案：**
```python
# 嘗試降低盈利閾值
creator.run_full_pipeline(
    symbol='BTCUSDT',
    timeframe='15m',
    holding_bars=5,
    profit_threshold=0.05  # 從 0.1% 降到 0.05%
)
```

### Q: 該如何選擇 holding_bars？

**時間對應：**
- 15m 時間框架
  - holding_bars=3 → 45 分鐘
  - holding_bars=5 → 75 分鐘
  - holding_bars=10 → 150 分鐘（2.5 小時）
  - holding_bars=20 → 300 分鐘（5 小時）

- 1h 時間框架
  - holding_bars=2 → 2 小時
  - holding_bars=4 → 4 小時
  - holding_bars=8 → 8 小時
  - holding_bars=24 → 24 小時

**建議：**
- 短線交易：3-5 根 K 棒
- 中線交易：8-15 根 K 棒
- 長線交易：20+ 根 K 棒

### Q: label=0 的技文信號如何處理？

**理解：**
Label=0 表示價格觸碰了 Bollinger Bands，但隨後沒有達到盈利目標。這些是**假信號**或**需要避免的情況**。

**處理方法：**

**方法 1：用作反面教材**
```python
# 讓模型學會「這種情況下不要交易"
training_data = df[df['label'].isin([0, 1, 2])]  # 只用有信號的
```

**方法 2：分類為 label=0（風險）**
```python
# 讓模型區分「安全的信號（1,2）" vs "風險的信號（0）"
# 這需要模型學會避免假信號
```

**方法 3：進一步分析**
```python
# 分析 label=0 的共同特徵
# 例如：波動度太小、離開軌距離太遠等
# 在預測時添加這些過濾條件
```

### Q: 能否同時處理多個幣種？

**是的，這是推薦做法：**

```python
from pathlib import Path
import pandas as pd

# 合併所有標籤
all_labels = []
for csv_file in Path('outputs/labels').glob('*.csv'):
    df = pd.read_csv(csv_file)
    df['symbol'] = csv_file.stem  # 添加幣種標識
    all_labels.append(df)

combined_df = pd.concat(all_labels, ignore_index=True)
combined_df.to_csv('outputs/combined_labels.csv', index=False)
```

### Q: 如何驗證標籤品質？

```python
# 計算各標籤的實際勝率
df = pd.read_csv('outputs/labels/BTCUSDT_15m_profitability_labels.csv')

# 標籤 1 的勝率
label_1_count = len(df[df['label'] == 1])
label_1_profitable = len(df[(df['label'] == 1) & (df['close'] > df['open'])])  # 簡單示例
print(f'Label 1 勝率: {label_1_profitable / label_1_count * 100:.2f}%')
```

---

## 下一步

1. ✅ 運行 `label_profitability_focused.py` 創建標籤
2. 🔍 檢查輸出的標籤分佈和品質
3. 💾 合併所有標籤數據
4. 🤖 使用標籤訓練機器學習模型
5. 📊 回測模型性能
6. 🚀 在實盤中測試

---

## 注意事項

1. **數據完整性**
   - 確保數據沒有缺失或異常值
   - 驗證 open_time 是否正確排序

2. **參數靈敏性**
   - 不同市場環境需要調整參數
   - 定期重新計算標籤以反映最新市場

3. **過擬合風險**
   - 不要過度調整參數以提高某個幣種的成績
   - 在多個幣種上驗證策略

4. **交易成本**
   - 考慮實際的手續費和滑點
   - 盈利閾值應該高於總交易成本

---

## 技術支持

查看日誌文件：
```bash
cat logs/label_YYYYMMDD_HHMMSS.log
```

遇到問題，檢查：
1. CSV 文件格式是否正確
2. 數據是否包含 NaN 值
3. 時間框架是否符合預期
4. Bollinger Bands 計算是否合理

---

**開始創建盈利導向的標籤吧！🚀**

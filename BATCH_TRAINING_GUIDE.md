# 批量訓練指南 - 全 23 種幣種

## 支持的幣種 (23 種)

```
BTC    ETH    BNB    SOL    ADA
XRP    DOGE   AVAX   LINK   MATIC
LIT    LTC    NEAR   OP     PEPE
SHIB   STX    SUI    TON    UNI
APT    BLAST  FLOKI
```

## 前置準備

### 1️⃣ 確保標籤檔案存在

每個幣種需要對應的標籤 CSV 檔案：

```
outputs/labels/BTCUSDT_15m_profitability_v2.csv      ✓ (已有)
outputs/labels/ETHUSDT_15m_profitability_v2.csv      ✓ (如果有)
outputs/labels/BNBUSDT_15m_profitability_v2.csv      ✓ (如果有)
...
```

**檢查命令**：
```bash
ls -la outputs/labels/*_15m_profitability_v2.csv
```

### 2️⃣ 檔案名稱規則

```
{SYMBOL}USDT_15m_profitability_v2.csv

例如：
- BTCUSDT_15m_profitability_v2.csv
- ETHUSDT_15m_profitability_v2.csv
- SOLUSDT_15m_profitability_v2.csv
```

## 執行訓練

### 方案 A：訓練所有 23 種幣種

```bash
python batch_training_all_symbols.py
```

**預期耗時**：
- 每個幣種：3-5 秒
- 總時間：70-115 秒 (~2 分鐘)

### 方案 B：檢查進度

訓練過程中會實時輸出：
```
[1/23] 訓練 BTCUSDT...
  ✓ BTCUSDT: AUC=0.8190 F1=0.8671 (RandomForest)

[2/23] 訓練 ETHUSDT...
  ✓ ETHUSDT: AUC=0.8234 F1=0.8512 (XGBoost)

[3/23] 訓練 BNBUSDT...
  ✓ BNBUSDT: AUC=0.7845 F1=0.8123 (RandomForest)
...
```

## 輸出結果

### 1️⃣ 訓練後的目錄結構

```
outputs/models/
├── BTCUSDT/
│   ├── model.pkl          ← 訓練好的模型
│   └── scaler.pkl         ← 標準化器
├── ETHUSDT/
│   ├── model.pkl
│   └── scaler.pkl
├── BNBUSDT/
│   ├── model.pkl
│   └── scaler.pkl
...
└── training_results_summary.csv  ← 成績單
```

### 2️⃣ 成績單 (training_results_summary.csv)

```csv
symbol,accuracy,precision,recall,f1,auc_roc
BTCUSVT,0.7997,0.8103,0.9325,0.8671,0.8190
ETHUSVT,0.7834,0.8234,0.8945,0.8512,0.8234
BNBUSVT,0.7656,0.7956,0.8723,0.8123,0.7845
...
```

## 訓練結果分析

### 📊 預期性能指標

```
AUC-ROC 平均: ~80-82%
  ├─ 最佳: 85%+
  └─ 最低: 75%+

F1 Score 平均: ~84-86%
  ├─ 最佳: 88%+
  └─ 最低: 80%+

Precision 平均: ~81-83%
Recall 平均: ~88-92%
```

### 🎯 性能排名

成績單自動按 AUC-ROC 排序，查看頂部表現最好的幣種：

```
性能排名 (AUC-ROC):
  1. 頂級幣 - AUC=0.85+ F1=0.88+
  2. 優秀幣 - AUC=0.82+ F1=0.86+
  3. 良好幣 - AUC=0.80+ F1=0.84+
  4. 可用幣 - AUC=0.75+ F1=0.80+
```

## 使用訓練好的模型

### 1️⃣ 加載特定幣種的模型

```python
import pickle
from pathlib import Path

symbol = 'ETHUSDT'
model_dir = Path(f"outputs/models/{symbol}")

# 加載模型和標準化器
with open(model_dir / "model.pkl", 'rb') as f:
    model = pickle.load(f)

with open(model_dir / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

print(f"✓ 已加載 {symbol} 的模型")
```

### 2️⃣ 進行預測

```python
import pandas as pd

# 準備新數據（格式同訓練數據）
X_new = pd.DataFrame({...})  # 50 個特徵

# 特徵標準化
X_new_scaled = scaler.transform(X_new)

# 預測
y_pred = model.predict(X_new_scaled)  # 0 或 1
y_pred_proba = model.predict_proba(X_new_scaled)[:, 1]  # 概率

# 決策
if y_pred_proba > 0.8:
    print("強信號 - 盈利概率 > 80%")
elif y_pred_proba > 0.6:
    print("中等信號 - 盈利概率 60-80%")
else:
    print("弱信號 - 盈利概率 < 60%")
```

### 3️⃣ 批量預測所有幣種

```python
import pickle
from pathlib import Path
import pandas as pd

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', ...]

for symbol in symbols:
    model_dir = Path(f"outputs/models/{symbol}")
    if not model_dir.exists():
        continue
    
    with open(model_dir / "model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(model_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # 預測該幣種
    X_new_scaled = scaler.transform(X_new[symbol])
    prob = model.predict_proba(X_new_scaled)[:, 1]
    
    print(f"{symbol}: 盈利概率={prob:.2%}")
```

## 疑難排解

### Q: 訓練失敗 - 找不到檔案

**A**: 檢查檔案路徑：
```bash
ls -la outputs/labels/
```

確保檔案存在且名稱正確（區分大小寫）。

### Q: 訓練失敗 - 樣本數過少

**A**: 該幣種樣本不足 1000 行，自動跳過。

檢查有效樣本數：
```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/labels/SYMBOL_15m_profitability_v2.csv'); print(len(df[df['label'].isin([0,1])]))"
```

### Q: 訓練太慢？

**A**: 正常耗時 2 分鐘左右。

加速方案：
```python
# 在 batch_training_all_symbols.py 修改
n_jobs=-1  # 改為 n_jobs=4（根據 CPU 核心數）
```

### Q: 某個幣種訓練失敗，怎麼單獨重訓？

**A**: 修改腳本只訓練該幣種：
```python
symbols = ['ETHUSDT']  # 只訓練 ETH
```

## 最佳實踐

### ✅ 訓練前

1. 檢查所有標籤檔案
2. 確保磁盤空間充足 (>500MB)
3. 關閉其他重 CPU 程序

### ✅ 訓練中

1. 監控進度（實時日誌）
2. 不要中斷訓練（會導致部分模型遺失）
3. 確保電源充足

### ✅ 訓練後

1. 檢查 training_results_summary.csv
2. 篩選高性能幣種（AUC > 0.80）
3. 優先在這些幣種上使用模型

## 性能預期

### 按幣種分類

| 類型 | AUC | F1 | 特點 | 推薦 |
|------|-----|----|----|------|
| **優秀** | 82%+ | 86%+ | 流動性好，趨勢清晰 | 優先使用 |
| **良好** | 80-82% | 84-86% | 正常幣種 | 可正常使用 |
| **可用** | 75-80% | 80-84% | 波動較大 | 謹慎使用 |
| **較差** | <75% | <80% | 噪音多，難預測 | 不推薦 |

## 後續優化

### 📈 可選改進

1. **特徵優化**：移除低重要性特徵
2. **超參數調優**：Grid Search / Bayesian Optimization
3. **集成模型**：XGBoost + RandomForest 加權平均
4. **時間序列交叉驗證**：避免數據洩露
5. **動態重訓**：定期用新數據更新模型

### 🔄 定期維護

```bash
# 每週重訓一次
cron: 0 0 * * 0 python batch_training_all_symbols.py
```

---

**訓練完成後，所有模型已準備好用於實時交易！** 🚀

# V3 快速開始指南 (2 分鐘)

## 1. 安裝依賴

```bash
# 安裝檇包
pip install pandas numpy scikit-learn xgboost talib-ng
```

## 2. 準備標第資椅

確保你介旁副有標第文件:
```
outputs/labels/BTCUSDT_15m_profitability_v2.csv
```

■ 必需一欄位: `open`, `high`, `low`, `close`, `volume`

## 3. 運行訓練

### 方案 A: 簡單遒取
```bash
python feature_engineering_and_training_v3_final.py
```

### 方案 B: 先清理旧模式 + 訓練
```bash
# 清理 V1/V2 文件
python cleanup.py

# 運行 V3
python feature_engineering_and_training_v3_final.py
```

## 4. 等待訓練完成

訓練漈漈沒有需要，提帬是子的統筹 (50K+ 檢查):

```
■■■ 火火火火火火  无丑...
```

5-15 分鐘不等 (根據你的 CPU).

## 5. 查看結果

訓練完成後，控制台會輸出：

```
✅ 訓練完成！
最佳模型: XGBoost (AUC-ROC: 0.73XX)

前 15 個重要特徵:
  1. BB_Position             - 0.0892
  2. ATR                     - 0.0654
  3. Historical_Vol          - 0.0598
  ...

模型已保存:
  ✅ outputs/models/BTCUSDT_15m_model_v3.pkl
  ✅ outputs/models/BTCUSDT_15m_scaler_v3.pkl
```

## 6. 使用訓練好的模型

```python
import pickle
import pandas as pd
import numpy as np

# 加載模型和標準化器
with open('outputs/models/BTCUSDT_15m_model_v3.pkl', 'rb') as f:
    model = pickle.load(f)

with open('outputs/models/BTCUSDT_15m_scaler_v3.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 使用新數據進行預測
# → 下次在你的交易 bot 中整合
```

## 綐立問題?

### Q: 訓練購輈了
**A:** 棄窥旧模式、梨焴絛橄。
```bash
rm outputs/models/BTCUSDT_15m_model_v2.pkl
rm outputs/models/BTCUSDT_15m_scaler_v2.pkl
```

### Q: 檇包不存在
**A:** 確保你有安裝正確的版本。
```bash
pip install --upgrade xgboost scikit-learn
```

### Q: 標第文件找不到
**A:** 確保你介旁副有下面文件、漈減存子、沈埋缶、控試清洛。
```
outputs/labels/BTCUSDT_15m_profitability_v2.csv
```

---

✨ 你已經有了 V3 訓練系統！前進吧。

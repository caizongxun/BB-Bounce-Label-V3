# BB Bounce Label V3 - 完整工作流程指南

## 目前狀態

✅ **標籤生成完成**
- 已處理 22 個交易對
- 總計 3,814,036 行數據
- 有效交易標籤 2,202,265 個
- 成功率範圍: 32.2% - 70.4%

## 工作流程架構

```
┌─────────────────────────────────────────────────────────┐
│ 1️⃣  數據準備層                                          │
├─────────────────────────────────────────────────────────┤
│ • CSV 文件位置: ./data/                                 │
│ • 文件格式: {SYMBOL}_15m.csv                            │
│ • 必要列: open, high, low, close, volume               │
│ • 數據檢查: find_csv_files.py                           │
└─────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────┐
│ 2️⃣  標籤生成層                                          │
├─────────────────────────────────────────────────────────┤
│ • 腳本: label_generation_fix.py                         │
│ • 輸出: ./outputs/labels/{SYMBOL}_15m_profitability_v2.csv │
│ • Bollinger Bands 計算                                  │
│ • 反彈有效性判斷                                        │
│ • 標籤: is_profitable (0/1)                             │
└─────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────┐
│ 3️⃣  質量驗證層                                          │
├─────────────────────────────────────────────────────────┤
│ • 腳本: verify_labels_quality.py                        │
│ • 檢查項:                                               │
│   - 數據完整性 (價格、交易量)                            │
│   - 標籤比例 (建議 > 50%)                                │
│   - 有利/不利比例分布                                   │
│   - Bollinger Bands 有效性                              │
│ • 輸出: ./outputs/analysis/label_quality_summary.csv   │
└─────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────┐
│ 4️⃣  模型訓練層 (下一步)                                 │
├─────────────────────────────────────────────────────────┤
│ • 腳本: trainbbmodel.py                                 │
│ • 輸入: ./outputs/labels/                               │
│ • 輸出: ./outputs/models/                               │
│ • 支持的模型: RandomForest, XGBoost, LightGBM          │
│ • 交叉驗證: k-fold (k=5)                                │
└─────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────┐
│ 5️⃣  模型評估層 (最終)                                   │
├─────────────────────────────────────────────────────────┤
│ • 性能指標: Accuracy, Precision, Recall, F1             │
│ • 混淆矩陣分析                                          │
│ • 特徵重要性排序                                        │
│ • 交易模擬回測                                          │
└─────────────────────────────────────────────────────────┘
```

## 使用指南

### 步驟 1: 驗證 CSV 文件

```bash
# 檢查所有 CSV 文件位置
python find_csv_files.py
```

**預期輸出:**
```
找到 44 個 CSV 檔案
15m 時間框: 22 個
1h 時間框: 22 個
建議的資料目錄位置: data
```

### 步驟 2: 生成標籤 (已完成)

```bash
# 為所有 CSV 文件生成 Bollinger Bands 反彈標籤
python label_generation_fix.py
```

**預期時間:** 約 15-30 分鐘（根據數據量）

**輸出統計:**
| 指標 | 值 |
|------|-----|
| 已處理交易對 | 22 個 |
| 總行數 | 3,814,036 |
| 有效交易標籤 | 2,202,265 |
| 平均成功率 | 56.8% |

### 步驟 3: 驗證標籤質量 (建議)

```bash
# 檢查生成的標籤是否符合訓練要求
python verify_labels_quality.py
```

**驗證項:**
- ✅ 數據完整性 (無缺失值)
- ✅ 價格邏輯 (high >= low)
- ✅ 標籤分布 (建議 50-70% 有利)
- ✅ Bollinger Bands 計算

**輸出文件:**
- `outputs/analysis/label_quality_summary.csv` - 質量摘要表

### 步驟 4: 訓練模型 (下一步)

```bash
# 使用生成的標籤訓練 ML 模型
python trainbbmodel.py
```

**模型選項:**
```python
# 在 trainbbmodel.py 中配置
models = {
    'random_forest': RandomForestClassifier(...),
    'xgboost': XGBClassifier(...),
    'lightgbm': LGBMClassifier(...)
}
```

**輸出:**
- `outputs/models/best_model.pkl` - 最佳模型
- `outputs/models/model_metrics.json` - 性能指標
- `outputs/models/feature_importance.csv` - 特徵重要性

### 步驟 5: 評估和回測 (可選)

```bash
# 評估模型性能並進行交易模擬
python evaluate_model.py
```

**性能指標:**
- 準確率 (Accuracy)
- 精準率 (Precision)
- 召回率 (Recall)
- F1 分數
- ROC-AUC

## 目錄結構

```
BB-Bounce-Label-V3/
├── data/                          # 原始 CSV 數據
│   ├── BTCUSDT_15m.csv
│   ├── ETHUSDT_15m.csv
│   └── ... (22 個交易對)
├── outputs/
│   ├── labels/                    # 生成的標籤
│   │   ├── BTCUSDT_15m_profitability_v2.csv
│   │   ├── ETHUSDT_15m_profitability_v2.csv
│   │   └── ...
│   ├── analysis/                  # 分析報告
│   │   └── label_quality_summary.csv
│   └── models/                    # 訓練的模型 (待生成)
│       ├── best_model.pkl
│       ├── model_metrics.json
│       └── feature_importance.csv
├── find_csv_files.py              # CSV 文件檢查工具
├── label_generation_fix.py        # 標籤生成腳本 ✅
├── verify_labels_quality.py       # 質量驗證工具
├── trainbbmodel.py                # 模型訓練腳本 (下一步)
└── WORKFLOW_GUIDE.md              # 本文檔
```

## 標籤質量統計

### 按交易對分布

| 交易對 | 成功率 | 有效標籤 | 狀態 |
|--------|--------|----------|------|
| ADAUSDT | 69.6% | 152,755 | ✅ 優秀 |
| ETHUSDT | 68.0% | 149,451 | ✅ 優秀 |
| SOLUSDT | 70.4% | 130,219 | ✅ 優秀 |
| BTCUSDT | 60.8% | 133,566 | ✅ 良好 |
| ... | ... | ... | ... |
| ARBUSDT | 32.2% | 30,294 | ⚠️ 待改進 |

**分布特點:**
- 高質量交易對 (>60%): 12 個
- 中等質量 (40-60%): 8 個
- 需改進 (<40%): 2 個

## 常見問題

### Q: 為什麼有些交易對的成功率很低 (如 ARBUSDT 32.2%)?

A: 可能原因:
1. 市場波動性差異 - 某些幣種反彈特性不同
2. 數據質量 - 缺少突變或極端行情
3. BB 參數不適配 - 可調整 period 和 std_dev
4. 樣本量不足 - ARBUSDT 數據量較少

**解決方案:**
```python
# 在 label_generation_fix.py 中調整參數
self.min_bounce_pct = 0.5      # 降低反彈閾值
self.min_candles_recovery = 3  # 減少恢復蠟燭數
```

### Q: 標籤生成失敗怎麼辦?

A: 執行診斷:
```bash
# 1. 檢查 CSV 文件
python find_csv_files.py

# 2. 查看詳細日誌
python label_generation_fix.py 2>&1 | grep ERROR

# 3. 檢查單一文件
python -c "import pandas as pd; df = pd.read_csv('data/BTCUSDT_15m.csv'); print(df.columns)"
```

### Q: 訓練前需要檢查什麼?

A: 運行質量驗證:
```bash
python verify_labels_quality.py
```

檢查項:
- ✅ 所有 CSV 有必要的列
- ✅ 標籤比例 > 50%
- ✅ 無異常數據 (NaN, inf)
- ✅ Bollinger Bands 計算完成

## 性能優化建議

### 標籤生成優化

```python
# 並行處理多個文件
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(generator.generate_labels, csv_files)
```

### 內存優化

```python
# 分批處理大文件
chunksize = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunksize):
    # 處理每一批
    pass
```

## 下一步行動

### 立即可執行

```bash
# 1. 驗證標籤質量
python verify_labels_quality.py

# 2. 準備訓練環境
pip install scikit-learn xgboost lightgbm
```

### 短期計劃

- [ ] 運行 trainbbmodel.py 訓練模型
- [ ] 評估模型性能
- [ ] 優化低成功率交易對的參數

### 中期計劃

- [ ] 實現交易模擬系統
- [ ] 添加風險管理功能
- [ ] 集成實時數據處理

## 技術參考

### Bollinger Bands 參數

- **period**: 20 (默認)
- **std_dev**: 2 (默認)
- **用途**: 識別價格極值

### 反彈判斷邏輯

```
1. 檢測觸碰下軌 (low <= bb_lower * 1.01)
2. 查看後續 5 根蠟燭的反彈
3. 計算反彈百分比 = (max - min) / min * 100
4. 若反彈 >= 0.5%，標記為有利 (is_profitable=1)
```

### 標籤定義

| 標籤 | 含義 | 用途 |
|------|------|------|
| is_profitable = 1 | 有效反彈 | 正樣本 (買入信號) |
| is_profitable = 0 | 無效反彈 | 負樣本 (避免買入) |
| is_profitable = NaN | 未標籤 | 不用於訓練 |

## 相關文件

- `find_csv_files.py` - CSV 文件診斷工具
- `label_generation_fix.py` - 標籤生成核心引擎
- `verify_labels_quality.py` - 質量驗證系統
- `trainbbmodel.py` - 模型訓練框架 (待完善)
- `WORKFLOW_GUIDE.md` - 本文檔

## 支持和反饋

如有問題或建議，請提交:
- Issue: GitHub Issues
- PR: Pull Request
- 討論: GitHub Discussions

---

**最後更新**: 2026-01-04
**狀態**: ✅ 標籤生成完成，可進行質量驗證和訓練

#!/bin/bash

# 訓練脚本: 清理舊檔 + 運行 V3 訓練

echo "開始清理舊檔案..."
python cleanup.py

echo ""
echo "開始 V3 訓練..."
echo "="*70
python feature_engineering_and_training_v3_final.py

echo ""
echo "✅ 訓練完成！"
echo "訓練韓模存儲路徑:"
echo "  - outputs/models/BTCUSDT_15m_model_v3.pkl"
echo "  - outputs/models/BTCUSDT_15m_scaler_v3.pkl"

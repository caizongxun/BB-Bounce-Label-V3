"""
清理腳本 - 刪除所有 V1 和 V2 訓練成果
"""

import os
import shutil
from pathlib import Path

print("開始清理舊版本訓練成果...")

# 待刪除的模型
files_to_remove = [
    "outputs/models/BTCUSDT_15m_profitability_model_v1.pkl",
    "outputs/models/BTCUSDT_15m_scaler_v1.pkl",
    "outputs/models/BTCUSDT_15m_profitability_model_v2.pkl",
    "outputs/models/BTCUSDT_15m_scaler_v2.pkl",
    "feature_engineering_and_training_v1.py",
    "feature_engineering_and_training_v2_optimized.py",
]

for file_path in files_to_remove:
    p = Path(file_path)
    if p.exists():
        try:
            p.unlink()
            print(f"✅ 刪除: {file_path}")
        except Exception as e:
            print(f"❌ 失敗: {file_path} - {e}")
    else:
        print(f"☐ 不存在: {file_path}")

print("\n清理完成！")
print("已保留: feature_engineering_and_training_v3_final.py")

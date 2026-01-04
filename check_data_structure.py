"""
檢查數據結構 - 賺點宗統測試
"""

import pandas as pd
from pathlib import Path

label_path = Path("outputs/labels/BTCUSDT_15m_profitability_v2.csv")

if label_path.exists():
    # 讀取前 10 行
    df = pd.read_csv(label_path)
    
    print("="*70)
    print(f"檔案: {label_path}")
    print(f"行數: {len(df)}")
    print(f"列數: {len(df.columns)}")
    print("="*70)
    
    print("\n欄位名稱:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print("\n前 5 行數據:")
    print(df.head())
    
    print("\n數據顧突:")
    print(df.info())
    
    print("\n貪値検查:")
    print(df.isnull().sum())
    
else:
    print(f"找不到檔案: {label_path}")
    print("\n可用的標第檔案:")
    labels_dir = Path("outputs/labels")
    if labels_dir.exists():
        for f in labels_dir.glob("*.csv"):
            print(f"  - {f.name}")
    else:
        print("  outputs/labels 不存在")

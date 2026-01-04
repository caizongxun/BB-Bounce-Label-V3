"""
檢查可用的檔案 - 確保全部 23 種幣種的標籤檔案存在
"""

import pandas as pd
from pathlib import Path

# 23 種幣種
symbols = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
    'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT',
    'LITUSDT', 'LTCUSDT', 'NEARUSDT', 'OPUSDT', 'PEPEUSDT',
    'SHIBUSDT', 'STXUSDT', 'SUIUSDT', 'TONUSDT', 'UNIUSDT',
    'APTUSDT', 'BLASTUSDT', 'FLOKIUSDT'
]

print("\n" + "="*70)
print("棄查可用的檔案")
print("="*70 + "\n")

available = []
unavailable = []

for symbol in symbols:
    csv_path = Path(f"outputs/labels/{symbol}_15m_profitability_v2.csv")
    
    if csv_path.exists():
        # 棄查檔案大小
        size_mb = csv_path.stat().st_size / (1024*1024)
        
        # 棄查霉數
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        valid_rows = len(df[df['label'].isin([0, 1])])
        
        print(f"✅ {symbol:12s} - {size_mb:6.1f}MB - {total_rows:6d} 行 (有效: {valid_rows:6d})")
        available.append({
            'symbol': symbol,
            'size_mb': size_mb,
            'total_rows': total_rows,
            'valid_rows': valid_rows
        })
    else:
        print(f"❌ {symbol:12s} - 找不到檔案 (outputs/labels/{symbol}_15m_profitability_v2.csv)")
        unavailable.append(symbol)

print("\n" + "="*70)
print(f"总統：{len(available)}/{len(symbols)} 可用")
print("="*70)

if available:
    df_summary = pd.DataFrame(available)
    print(f"\n檔案統計:")
    print(f"  總大小: {df_summary['size_mb'].sum():.1f} MB")
    print(f"  總行數: {df_summary['total_rows'].sum():,d}")
    print(f"  有效行: {df_summary['valid_rows'].sum():,d}")
    print(f"  有效率: {100*df_summary['valid_rows'].sum()/df_summary['total_rows'].sum():.1f}%")
    
    print(f"\n訓練推估：")
    print(f"  每個幣種: 3-5 秒")
    print(f"  總耷時間: {len(available)*4//60} 分 {len(available)*4%60} 秒")
    print(f"  預料: ~2 分鐘")

if unavailable:
    print(f"\n缺少的檔案 ({len(unavailable)} 種):")
    for symbol in unavailable:
        print(f"  - {symbol}_15m_profitability_v2.csv")
    print(f"\n提示: 你需要先推敲這些幣種的標籤檔案。")

print("\n下一步:")
if len(available) >= 20:
    print("✅ 檔案顇完整，可以進行訓練！")
    print("\n運行: python batch_training_all_symbols.py")
else:
    print(f"⚠️ 可用的檔案過少 ({len(available)}/23)，建議先推敲正常頁面")

print("\n" + "="*70 + "\n")

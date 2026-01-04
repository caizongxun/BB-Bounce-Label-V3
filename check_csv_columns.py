"""
棄測 CSV 檔的列名
"""

import pandas as pd
from pathlib import Path

# 棄測第一個 CSV 檔
csv_files = list(Path('data').glob('*.csv'))

if csv_files:
    csv_file = csv_files[0]
    print(f'檔案：{csv_file}')
    print(f'\n列名：')
    
    df = pd.read_csv(csv_file)
    for col in df.columns:
        print(f'  - {col}')
    
    print(f'\n前 5 行：')
    print(df.head())
    
    print(f'\n數據類型：')
    print(df.dtypes)
else:
    print('找不到 CSV 檔!')

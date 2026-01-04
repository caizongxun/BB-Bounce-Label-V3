"""
標籤品質驗證和分析工具
驗證生成的標籤是否符合訓練要求
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelQualityVerifier:
    """驗證生成的標籤品質"""
    
    def __init__(self, labels_dir='outputs/labels'):
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path('outputs/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"標籤目錄不存在: {self.labels_dir}")
    
    def find_label_files(self):
        """找到所有標籤檔案"""
        label_files = list(self.labels_dir.glob('*_15m_profitability_v2.csv'))
        logger.info(f"找到 {len(label_files)} 個標籤檔案")
        return sorted(label_files)
    
    def verify_single_file(self, csv_file):
        """驗證單一標籤檔案"""
        try:
            symbol = csv_file.stem.split('_')[0]
            logger.info(f"\n驗證 {symbol}...")
            
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"{symbol} 檔案為空")
                return None
            
            # 基本檢查
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'is_profitable', 'bounce_type', 'bounce_pct']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"{symbol} 缺少列: {missing_cols}")
                return None
            
            # 統計信息
            total_rows = len(df)
            labeled_rows = df['is_profitable'].notna().sum()
            profitable = (df['is_profitable'] == 1).sum()
            non_profitable = (df['is_profitable'] == 0).sum()
            unlabeled = total_rows - labeled_rows
            
            # 檢查數據範圍
            price_valid = (df['high'] >= df['low']).all()
            volume_valid = (df['volume'] >= 0).all()
            bounce_valid = (df['bounce_pct'] >= 0).all()
            
            # BB 指標檢查
            bb_cols = ['sma', 'bb_upper', 'bb_lower']
            bb_valid = all(df[col].notna().sum() > 0 for col in bb_cols if col in df.columns)
            
            # 反彈類型分布
            bounce_types = df['bounce_type'].value_counts().to_dict()
            
            # 反彈百分比統計
            bounce_pct_stats = {
                'mean': df[df['is_profitable'] == 1]['bounce_pct'].mean(),
                'std': df[df['is_profitable'] == 1]['bounce_pct'].std(),
                'min': df[df['is_profitable'] == 1]['bounce_pct'].min(),
                'max': df[df['is_profitable'] == 1]['bounce_pct'].max(),
            }
            
            result = {
                'symbol': symbol,
                'total_rows': total_rows,
                'labeled_rows': labeled_rows,
                'profitable': profitable,
                'non_profitable': non_profitable,
                'unlabeled': unlabeled,
                'label_ratio': labeled_rows / total_rows if total_rows > 0 else 0,
                'profitable_ratio': profitable / labeled_rows if labeled_rows > 0 else 0,
                'price_valid': price_valid,
                'volume_valid': volume_valid,
                'bounce_valid': bounce_valid,
                'bb_valid': bb_valid,
                'bounce_types': bounce_types,
                'bounce_pct_mean': bounce_pct_stats['mean'],
                'bounce_pct_std': bounce_pct_stats['std'],
                'bounce_pct_range': (bounce_pct_stats['min'], bounce_pct_stats['max']),
            }
            
            # 詳細日誌
            logger.info(f"  總行數: {total_rows:,}")
            logger.info(f"  已標籤: {labeled_rows:,} ({result['label_ratio']*100:.1f}%)")
            logger.info(f"  有利可圖: {profitable:,} ({result['profitable_ratio']*100:.1f}%)")
            logger.info(f"  非有利: {non_profitable:,}")
            logger.info(f"  未標籤: {unlabeled:,}")
            logger.info(f"  數據驗證: 價格={price_valid}, 交易量={volume_valid}, 反彈={bounce_valid}, BB={bb_valid}")
            logger.info(f"  反彈類型: {bounce_types}")
            logger.info(f"  反彈%% - 平均: {bounce_pct_stats['mean']:.2f}%, 範圍: [{bounce_pct_stats['min']:.2f}%, {bounce_pct_stats['max']:.2f}%]")
            
            return result
            
        except Exception as e:
            logger.error(f"驗證 {csv_file.name} 時出錯: {str(e)}")
            return None
    
    def verify_all(self):
        """驗證所有標籤檔案"""
        label_files = self.find_label_files()
        
        if not label_files:
            logger.error("找不到標籤檔案")
            return []
        
        results = []
        for label_file in label_files:
            result = self.verify_single_file(label_file)
            if result:
                results.append(result)
        
        return results
    
    def generate_report(self, results):
        """生成驗證報告"""
        if not results:
            logger.warning("沒有驗證結果")
            return
        
        logger.info("\n" + "="*70)
        logger.info("標籤品質驗證報告")
        logger.info("="*70)
        
        # 全局統計
        total_symbols = len(results)
        total_labeled = sum(r['labeled_rows'] for r in results)
        total_profitable = sum(r['profitable'] for r in results)
        total_rows = sum(r['total_rows'] for r in results)
        
        logger.info(f"\n全局統計:")
        logger.info(f"  交易對數量: {total_symbols}")
        logger.info(f"  總行數: {total_rows:,}")
        logger.info(f"  已標籤行數: {total_labeled:,} ({total_labeled/total_rows*100:.1f}%)")
        logger.info(f"  有利可圖標籤: {total_profitable:,} ({total_profitable/total_labeled*100:.1f}%)")
        
        # 數據驗證狀態
        valid_price = sum(1 for r in results if r['price_valid'])
        valid_volume = sum(1 for r in results if r['volume_valid'])
        valid_bounce = sum(1 for r in results if r['bounce_valid'])
        valid_bb = sum(1 for r in results if r['bb_valid'])
        
        logger.info(f"\n數據驗證狀態:")
        logger.info(f"  價格數據有效: {valid_price}/{total_symbols}")
        logger.info(f"  交易量有效: {valid_volume}/{total_symbols}")
        logger.info(f"  反彈數據有效: {valid_bounce}/{total_symbols}")
        logger.info(f"  BB指標有效: {valid_bb}/{total_symbols}")
        
        # 標籤比例分布
        logger.info(f"\n標籤比例分布:")
        logger.info(f"{'交易對':<12} {'總行數':>10} {'已標籤':>10} {'有利率':>10} {'反彈%%均值':>12}")
        logger.info("-" * 60)
        
        for r in sorted(results, key=lambda x: x['profitable_ratio'], reverse=True):
            logger.info(
                f"{r['symbol']:<12} {r['total_rows']:>10,} {r['labeled_rows']:>10,} "
                f"{r['profitable_ratio']*100:>9.1f}% {r['bounce_pct_mean']:>11.2f}%"
            )
        
        # 警告信息
        logger.info(f"\n質量檢查:")
        warnings = 0
        
        for r in results:
            if r['label_ratio'] < 0.5:
                logger.warning(f"  {r['symbol']}: 標籤比例過低 ({r['label_ratio']*100:.1f}%)")
                warnings += 1
            
            if r['profitable_ratio'] > 0.8 or r['profitable_ratio'] < 0.3:
                logger.warning(f"  {r['symbol']}: 有利率異常 ({r['profitable_ratio']*100:.1f}%)")
                warnings += 1
            
            if not r['price_valid']:
                logger.warning(f"  {r['symbol']}: 價格數據問題 (high < low)")
                warnings += 1
        
        if warnings == 0:
            logger.info("  所有標籤通過質量檢查")
        else:
            logger.info(f"  發現 {warnings} 個警告")
        
        logger.info("\n" + "="*70)
        logger.info("下一步:")
        logger.info("  1. 運行訓練: python trainbbmodel.py")
        logger.info("  2. 評估模型: outputs/models/ 目錄")
        logger.info("="*70)
    
    def export_summary(self, results):
        """匯出摘要到 CSV"""
        if not results:
            return
        
        summary_df = pd.DataFrame([
            {
                'Symbol': r['symbol'],
                'Total_Rows': r['total_rows'],
                'Labeled_Rows': r['labeled_rows'],
                'Label_Ratio': f"{r['label_ratio']*100:.1f}%",
                'Profitable': r['profitable'],
                'Non_Profitable': r['non_profitable'],
                'Profitable_Ratio': f"{r['profitable_ratio']*100:.1f}%",
                'Bounce_Pct_Mean': f"{r['bounce_pct_mean']:.2f}%",
                'Bounce_Pct_Std': f"{r['bounce_pct_std']:.2f}%",
                'Data_Quality': 'PASS' if all([r['price_valid'], r['volume_valid'], r['bounce_valid'], r['bb_valid']]) else 'WARN'
            }
            for r in results
        ])
        
        output_file = self.output_dir / 'label_quality_summary.csv'
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"\n摘要已匯出: {output_file}")


def main():
    try:
        logger.info("開始標籤品質驗證")
        logger.info("="*70)
        
        verifier = LabelQualityVerifier(labels_dir='outputs/labels')
        results = verifier.verify_all()
        
        if results:
            verifier.generate_report(results)
            verifier.export_summary(results)
        else:
            logger.error("驗證失敗")
    
    except FileNotFoundError as e:
        logger.error(f"錯誤: {str(e)}")
        logger.error("請先運行: python label_generation_fix.py")
    except Exception as e:
        logger.error(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

"""
基於盈市算法的盤子標籤創建

都須項目：
- 標您应該导有盆市K棺的窉位
- 標您应該导有技文K棺的窉位
- 標您不應往来走的K棺（技文帶偽）
- 家托分者位置
訓練模型預測：這個次擺援不會是潜在的鼠一壭
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Tuple, List

# 設置日誌
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/label_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProfitabilityLabelCreator:
    """基於盆市算法的盤子標籤創建器"""
    
    def __init__(self, min_reversal_pct=0.1, lookback_period=20):
        """
        Args:
            min_reversal_pct: 最低反轉幅度 (%)
            lookback_period: 憨查回看時期 (根據時間框架)
        """
        self.min_reversal_pct = min_reversal_pct
        self.lookback_period = lookback_period
        self.df = None
        
        Path('outputs/labels').mkdir(parents=True, exist_ok=True)
        Path('outputs/analysis').mkdir(parents=True, exist_ok=True)
    
    def load_data(self, symbol: str, timeframe: str):
        """載入 CSV 數據並自動棄測列名"""
        csv_path = f'data/{symbol}_{timeframe}.csv'
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f'成功加載 {symbol}_{timeframe}: {len(self.df)} 行數據')
            
            # 自動棄測列名（不同的數據源可能有不同的列名）
            logger.info(f'查詢到的列：{self.df.columns.tolist()}')
            
            # 棄測並正正列名
            col_mapping = {}
            
            # 惨找 open_time 列
            time_cols = [col for col in self.df.columns if 'time' in col.lower()]
            if time_cols:
                col_mapping['open_time'] = time_cols[0]
                logger.info(f'找到 open_time 列：{time_cols[0]}')
            
            # 惨找 OHLC 列
            for target_col in ['open', 'high', 'low', 'close']:
                for df_col in self.df.columns:
                    if target_col.lower() == df_col.lower():
                        col_mapping[target_col] = df_col
                        break
            
            # 確保所有必需列都找到了
            required_cols = ['open', 'high', 'low', 'close', 'open_time']
            missing_cols = [col for col in required_cols if col not in col_mapping]
            
            if missing_cols:
                logger.error(f'找不到列：{missing_cols}')
                logger.error(f'可用的列：{self.df.columns.tolist()}')
                return False
            
            # 連置列名
            self.df = self.df.rename(columns=col_mapping)
            logger.info(f'列名棄測完成')
            
            # 轉換 open_time 為日期
            try:
                self.df['open_time'] = pd.to_datetime(self.df['open_time'], unit='ms')
            except:
                try:
                    self.df['open_time'] = pd.to_datetime(self.df['open_time'])
                except:
                    logger.warning('找不到有效的時間列')
            
            self.df = self.df.sort_values('open_time').reset_index(drop=True)
            
            return True
        except FileNotFoundError:
            logger.error(f'找不到數據檔案：{csv_path}')
            return False
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """計算布林様帶"""
        logger.info(f'計算布林様帶 (period={period}, std_dev={std_dev})...')
        
        self.df['sma'] = self.df['close'].rolling(window=period).mean()
        self.df['std'] = self.df['close'].rolling(window=period).std()
        self.df['upper_band'] = self.df['sma'] + (self.df['std'] * std_dev)
        self.df['lower_band'] = self.df['sma'] - (self.df['std'] * std_dev)
        
        logger.info('BB 計算完成')
    
    def detect_touches(self) -> dict:
        """棄測指標触磐點"""
        logger.info('棄測指標触磐點...')
        
        touches = {
            'lower_touches': [],  # [idx, ...]
            'upper_touches': [],  # [idx, ...]
        }
        
        for i in range(1, len(self.df)):
            if pd.isna(self.df.loc[i, 'lower_band']):
                continue
            
            # 棄測下軌触磐
            if (self.df.loc[i-1, 'close'] > self.df.loc[i-1, 'lower_band'] and
                self.df.loc[i, 'low'] <= self.df.loc[i, 'lower_band']):
                touches['lower_touches'].append(i)
            
            # 棄測上軌触磐
            if (self.df.loc[i-1, 'close'] < self.df.loc[i-1, 'upper_band'] and
                self.df.loc[i, 'high'] >= self.df.loc[i, 'upper_band']):
                touches['upper_touches'].append(i)
        
        logger.info(f'棄測到 {len(touches["lower_touches"])} 個下軌触磐')
        logger.info(f'棄測到 {len(touches["upper_touches"])} 個上軌触磐')
        
        return touches
    
    def analyze_profitability(self, touches: dict, holding_bars=5, profit_threshold=0.1):
        """
        分析盆市成總數
        
        Args:
            touches: 由 detect_touches() 輸出
            holding_bars: 持有時間根數 (根據時間框架)
            profit_threshold: 盆市阀值 (%)
        """
        logger.info(f'\n分析盆市性 (holding_bars={holding_bars}, profit_threshold={profit_threshold}%)')
        logger.info('='*60)
        
        results = {
            'lower_profitable': [],  # 下軌有盆市
            'lower_loss': [],         # 下軌搎欷
            'upper_profitable': [],   # 上軌有盆市
            'upper_loss': [],         # 上軌搎欷
        }
        
        # 分析下軌触磐 (做多)
        for touch_idx in touches['lower_touches']:
            if touch_idx + holding_bars >= len(self.df):
                continue
            
            entry_price = self.df.loc[touch_idx, 'close']
            exit_price = self.df.loc[touch_idx + holding_bars, 'close']
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
            
            if profit_pct >= profit_threshold:
                results['lower_profitable'].append(touch_idx)
            else:
                results['lower_loss'].append(touch_idx)
        
        # 分析上軌触磐 (做空)
        for touch_idx in touches['upper_touches']:
            if touch_idx + holding_bars >= len(self.df):
                continue
            
            entry_price = self.df.loc[touch_idx, 'close']
            exit_price = self.df.loc[touch_idx + holding_bars, 'close']
            profit_pct = ((entry_price - exit_price) / entry_price) * 100  # 做空所以翻轉
            
            if profit_pct >= profit_threshold:
                results['upper_profitable'].append(touch_idx)
            else:
                results['upper_loss'].append(touch_idx)
        
        # 輸出統計
        logger.info(f'\n下軌触磐 (做多)')
        logger.info(f'  有盆市（label=1）: {len(results["lower_profitable"])}')
        logger.info(f'  搎欷 (label=0): {len(results["lower_loss"])}')
        if len(results['lower_profitable']) + len(results['lower_loss']) > 0:
            lower_wr = len(results['lower_profitable']) / (len(results['lower_profitable']) + len(results['lower_loss'])) * 100
            logger.info(f'  勝率: {lower_wr:.2f}%')
        
        logger.info(f'\n上軌触磐 (做空)')
        logger.info(f'  有盆市（label=2）: {len(results["upper_profitable"])}')
        logger.info(f'  搎欷 (label=0): {len(results["upper_loss"])}')
        if len(results['upper_profitable']) + len(results['upper_loss']) > 0:
            upper_wr = len(results['upper_profitable']) / (len(results['upper_profitable']) + len(results['upper_loss'])) * 100
            logger.info(f'  勝率: {upper_wr:.2f}%')
        
        logger.info('='*60)
        
        return results
    
    def create_labels(self, touches: dict, profitability_results: dict):
        """
        為所有 K 棺創建標籤
        
        標籤定義：
            1: 下軌触磐 + 有盆市 (應該做多)
            2: 上軌触磐 + 有盆市 (應該做空)
            0: 技文触磐但搎欷 (不應該接)
            -1: 沒有触磐 (中性)
        """
        logger.info('\n為所有 K 棺創建標籤...')
        
        self.df['label'] = -1  # 預設數值：沒有触磐
        
        # 標您有盆市的下軌触磐
        for idx in profitability_results['lower_profitable']:
            self.df.loc[idx, 'label'] = 1
        
        # 標您有盆市的上軌触磐
        for idx in profitability_results['upper_profitable']:
            self.df.loc[idx, 'label'] = 2
        
        # 標您搎欷的触磐 (技文)
        for idx in profitability_results['lower_loss'] + profitability_results['upper_loss']:
            self.df.loc[idx, 'label'] = 0
        
        logger.info('標籤創建完成')
    
    def get_label_statistics(self):
        """獲得標籤統計"""
        logger.info('\n標籤統計：')
        logger.info('='*60)
        
        label_counts = self.df['label'].value_counts().sort_index()
        
        for label, count in label_counts.items():
            pct = count / len(self.df) * 100
            if label == 1:
                desc = '下軌有盆市 (label=1)'
            elif label == 2:
                desc = '上軌有盆市 (label=2)'
            elif label == 0:
                desc = '技文搎欷 (label=0)'
            else:
                desc = '沒有触磐 (label=-1)'
            
            logger.info(f'  {desc}: {count} ({pct:.2f}%)')
        
        total_signals = label_counts.get(1, 0) + label_counts.get(2, 0) + label_counts.get(0, 0)
        total_profitable = label_counts.get(1, 0) + label_counts.get(2, 0)
        
        logger.info(f'\n  總信號數：{total_signals}')
        logger.info(f'  有盆市信號: {total_profitable}')
        
        if total_signals > 0:
            logger.info(f'  有盆市比例: {total_profitable/total_signals*100:.2f}%')
        
        logger.info('='*60)
        
        return label_counts
    
    def save_labels(self, symbol: str, timeframe: str):
        """保存標籤到 CSV"""
        output_path = f'outputs/labels/{symbol}_{timeframe}_profitability_labels.csv'
        
        # 只保存子鎖專用列
        output_df = self.df[['open_time', 'open', 'high', 'low', 'close', 'label']].copy()
        output_df.to_csv(output_path, index=False)
        
        logger.info(f'\n標籤已保存到：{output_path}')
    
    def analyze_features(self, touches: dict):
        """分析有盆市 K 棺的窉位特征"""
        logger.info('\n分析有盆市 K 棺特征...')
        logger.info('='*60)
        
        # 計算設知者推动力（归一化辛加幅度）
        self.df['volatility'] = self.df['high'] - self.df['low']
        self.df['body_size'] = abs(self.df['close'] - self.df['open'])
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        
        profitable_indices = self.df[self.df['label'].isin([1, 2])].index
        loss_indices = self.df[self.df['label'] == 0].index
        
        logger.info(f'\n有盆市 K 棺 (n={len(profitable_indices)})：')
        for col in ['volatility', 'body_size', 'upper_wick', 'lower_wick']:
            if len(profitable_indices) > 0:
                mean_val = self.df.loc[profitable_indices, col].mean()
                logger.info(f'  {col}平均: {mean_val:.6f}')
        
        logger.info(f'\n搎欷 K 棺 (n={len(loss_indices)})：')
        for col in ['volatility', 'body_size', 'upper_wick', 'lower_wick']:
            if len(loss_indices) > 0:
                mean_val = self.df.loc[loss_indices, col].mean()
                logger.info(f'  {col}平均: {mean_val:.6f}')
        
        logger.info('='*60)
    
    def run_full_pipeline(self, symbol: str, timeframe: str, holding_bars=5, profit_threshold=0.1):
        """這行完整標籤創建流程"""
        logger.info(f'\n開始為 {symbol}_{timeframe} 創建標籤')
        logger.info('='*60)
        
        # 1. 載入数据
        if not self.load_data(symbol, timeframe):
            return None
        
        # 2. 計算布林様帶
        self.calculate_bollinger_bands()
        
        # 3. 棄測触磐點
        touches = self.detect_touches()
        
        # 4. 分析盆市性
        profitability_results = self.analyze_profitability(
            touches,
            holding_bars=holding_bars,
            profit_threshold=profit_threshold
        )
        
        # 5. 為所有 K 棺創建標籤
        self.create_labels(touches, profitability_results)
        
        # 6. 獲得統計信息
        self.get_label_statistics()
        
        # 7. 分析特征
        self.analyze_features(touches)
        
        # 8. 保存標籤
        self.save_labels(symbol, timeframe)
        
        logger.info(f'\n{symbol}_{timeframe} 流程完成！')
        logger.info('='*60)
        
        return self.df


def main():
    """主函數"""
    logger.info('='*60)
    logger.info('基於盈市算法的盤子標籤創建')
    logger.info('='*60)
    
    creator = ProfitabilityLabelCreator()
    
    # 收益: 0.1% (可根據需要調整)
    symbol = 'BTCUSDT'
    timeframe = '15m'
    
    creator.run_full_pipeline(
        symbol,
        timeframe,
        holding_bars=5,      # 持有 5 個 15m K 棺 = 75 分鐘
        profit_threshold=0.1  # 最低推能有 0.1% 的成總
    )


if __name__ == '__main__':
    main()

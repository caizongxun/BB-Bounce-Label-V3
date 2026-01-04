"""
基於盈利性的標籤創建 v2

流程：
1. 偵測所有觸碰或接近上下軌的 K 棒 (距離 < 0.05% BB 寶寶)
2. 分類成 有盈利 vs 無盈利
   - 有盈利: 未來10根K棒有盈利 (最高/低算出輊錢)
   - 無盈利: 未來10根K棒沒有盈利
3. 僅會稉有盈利K棒作上團中弟築齐的分類
"""

import pandas as pd
import numpy as np
import logging
import sys
import io
from datetime import datetime
from pathlib import Path

# 設置 UTF-8 編碼
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 設置日誌
Path('logs').mkdir(exist_ok=True)
Path('outputs/labels').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/label_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProfitabilityLabelCreatorV2:
    """第二版本：基於盈利性的標籤創建器"""
    
    def __init__(self, bb_period=20, bb_std=2, touch_threshold_pct=0.05, 
                 holding_bars=5, profit_threshold=0.1):
        """
        Args:
            bb_period: BB 計算週需
            bb_std: BB 標正差倍數
            touch_threshold_pct: 觸碰閾值 (相對於 BB 寶寶)
            holding_bars: 持有時間根數
            profit_threshold: 盈利最低閾值 (%)
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.touch_threshold_pct = touch_threshold_pct
        self.holding_bars = holding_bars
        self.profit_threshold = profit_threshold
        
        self.df = None
        self.touch_indices = []
    
    def load_data(self, symbol: str, timeframe: str):
        """載入數據並自動偵測列名"""
        csv_path = f'data/{symbol}_{timeframe}.csv'
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f'成功加載 {symbol}_{timeframe}: {len(self.df)} 行數據')
            
            # 自動偵測列名
            if 'timestamp' in self.df.columns:
                self.df = self.df.rename(columns={'timestamp': 'open_time'})
            
            # 檢查所有必需列
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                logger.error(f'缺少列：{missing_cols}')
                return False
            
            # 轉換時間格式
            if 'open_time' in self.df.columns:
                try:
                    self.df['open_time'] = pd.to_datetime(self.df['open_time'])
                except:
                    pass
            
            self.df = self.df.sort_values('open_time' if 'open_time' in self.df.columns else self.df.columns[0]).reset_index(drop=True)
            return True
        except FileNotFoundError:
            logger.error(f'找不到數據檔案：{csv_path}')
            return False
    
    def calculate_bollinger_bands(self):
        """計算布林傑帶"""
        logger.info(f'計算布林傑帶 (period={self.bb_period}, std={self.bb_std})...')
        
        self.df['sma'] = self.df['close'].rolling(window=self.bb_period).mean()
        self.df['std'] = self.df['close'].rolling(window=self.bb_period).std()
        self.df['upper_band'] = self.df['sma'] + (self.df['std'] * self.bb_std)
        self.df['lower_band'] = self.df['sma'] - (self.df['std'] * self.bb_std)
        self.df['bb_width'] = self.df['upper_band'] - self.df['lower_band']
        
        logger.info('BB 計算完成')
    
    def detect_touches_and_near(self):
        """
        偵測所有觸碰或接近上下軌的 K 棒
        
        標正閾值：
        - 下軌：(close - lower_band) / bb_width < touch_threshold_pct
        - 上軌：(upper_band - close) / bb_width < touch_threshold_pct
        """
        logger.info(f'\n偵測觸碰及接近 (閾值: {self.touch_threshold_pct}%)...')
        
        touches = []
        touch_types = {}
        
        for i in range(self.bb_period, len(self.df) - self.holding_bars):
            row = self.df.iloc[i]
            
            if pd.isna(row['bb_width']) or row['bb_width'] == 0:
                continue
            
            # 計算距離上下軌的相對距離
            dist_to_lower = (row['close'] - row['lower_band']) / row['bb_width'] * 100
            dist_to_upper = (row['upper_band'] - row['close']) / row['bb_width'] * 100
            
            # 也檢查 high/low
            low_to_lower = (row['low'] - row['lower_band']) / row['bb_width'] * 100
            high_to_upper = (row['upper_band'] - row['high']) / row['bb_width'] * 100
            
            # 觸碰下軌 (做多)
            if low_to_lower <= self.touch_threshold_pct or dist_to_lower <= self.touch_threshold_pct:
                touches.append(i)
                touch_types[i] = 'lower'
            
            # 觸碰上軌 (做空)
            elif high_to_upper <= self.touch_threshold_pct or dist_to_upper <= self.touch_threshold_pct:
                touches.append(i)
                touch_types[i] = 'upper'
        
        self.touch_indices = touches
        self.touch_types = touch_types
        
        logger.info(f'偵測到 {len(touches)} 個觸碰/接近 K 棒')
        logger.info(f'  下軌 (做多): {sum(1 for t in touch_types.values() if t == "lower")}')
        logger.info(f'  上軌 (做空): {sum(1 for t in touch_types.values() if t == "upper")}')
        
        return touches, touch_types
    
    def calculate_profitability(self, touch_idx: int, touch_type: str):
        """
        計算是否有盈利
        
        標正：未來 holding_bars 根K棒的最高/低是否能盈利
        """
        if touch_idx + self.holding_bars >= len(self.df):
            return False, 0, 0
        
        entry_price = self.df.iloc[touch_idx]['close']
        
        # 獲得未來10根K棒
        future_data = self.df.iloc[touch_idx + 1:touch_idx + self.holding_bars + 1]
        max_price = future_data['high'].max()
        min_price = future_data['low'].min()
        
        if touch_type == 'lower':
            # 做多：上潈是否能盈利
            profit_pct = (max_price - entry_price) / entry_price * 100
            is_profitable = profit_pct >= self.profit_threshold
            return is_profitable, profit_pct, max_price
        else:  # upper
            # 做空：下潈是否能盈利
            profit_pct = (entry_price - min_price) / entry_price * 100
            is_profitable = profit_pct >= self.profit_threshold
            return is_profitable, profit_pct, min_price
    
    def create_labels(self):
        """
        為所有 K 棒創建標籤
        
        標籤定義：
            1: 下軌有盈利 (做多有輊錢)
            2: 上軌有盈利 (做空有輊錢)
            0: 觸碰/接近但無盈利 (不應接)
            -1: 沒有觸碰 (中性)
        """
        logger.info('\n為所有 K 棒創建標籤...')
        
        self.df['label'] = -1
        self.df['profit_pct'] = np.nan
        self.df['target_price'] = np.nan
        
        lower_profitable = 0
        lower_unprofitable = 0
        upper_profitable = 0
        upper_unprofitable = 0
        
        for touch_idx in self.touch_indices:
            touch_type = self.touch_types[touch_idx]
            is_profitable, profit_pct, target_price = self.calculate_profitability(touch_idx, touch_type)
            
            if touch_type == 'lower':
                label = 1 if is_profitable else 0
                if is_profitable:
                    lower_profitable += 1
                else:
                    lower_unprofitable += 1
            else:  # upper
                label = 2 if is_profitable else 0
                if is_profitable:
                    upper_profitable += 1
                else:
                    upper_unprofitable += 1
            
            self.df.loc[touch_idx, 'label'] = label
            self.df.loc[touch_idx, 'profit_pct'] = profit_pct
            self.df.loc[touch_idx, 'target_price'] = target_price
        
        logger.info('\n標籤統計:')
        logger.info('='*60)
        logger.info(f'  下軌有盈利 (label=1): {lower_profitable}')
        logger.info(f'  下軌無盈利 (label=0): {lower_unprofitable}')
        if lower_profitable + lower_unprofitable > 0:
            lower_wr = lower_profitable / (lower_profitable + lower_unprofitable) * 100
            logger.info(f'  下軌勝率: {lower_wr:.2f}%')
        
        logger.info()
        logger.info(f'  上軌有盈利 (label=2): {upper_profitable}')
        logger.info(f'  上軌無盈利 (label=0): {upper_unprofitable}')
        if upper_profitable + upper_unprofitable > 0:
            upper_wr = upper_profitable / (upper_profitable + upper_unprofitable) * 100
            logger.info(f'  上軌勝率: {upper_wr:.2f}%')
        
        total_profitable = lower_profitable + upper_profitable
        total_unprofitable = lower_unprofitable + upper_unprofitable
        
        logger.info()
        logger.info(f'  總有盈利 K 棒: {total_profitable}')
        logger.info(f'  總無盈利 K 棒: {total_unprofitable}')
        logger.info(f'  無觸碰 K 棒: {(self.df["label"] == -1).sum()}')
        logger.info('='*60)
    
    def backtest_on_profitable_signals(self):
        """回測：僅在有盈利 K 棒上做交易是否程輻 100%"""
        logger.info('\n回測：僅在有盈利K棒做交易...')
        logger.info('='*60)
        
        profitable_signals = self.df[self.df['label'].isin([1, 2])]
        
        if len(profitable_signals) == 0:
            logger.warning('沒有有盈利的K棒')
            return
        
        # 做多信號 (下軌)
        long_signals = self.df[self.df['label'] == 1]
        if len(long_signals) > 0:
            long_profitable = 0
            for idx in long_signals.index:
                if self.df.loc[idx, 'profit_pct'] >= self.profit_threshold:
                    long_profitable += 1
            long_wr = long_profitable / len(long_signals) * 100
            logger.info(f'\n做多信號 (下軌有盈利)')
            logger.info(f'  信號數: {len(long_signals)}')
            logger.info(f'  輊錢信號: {long_profitable}')
            logger.info(f'  勝率: {long_wr:.2f}%')
        
        # 做空信號 (上軌)
        short_signals = self.df[self.df['label'] == 2]
        if len(short_signals) > 0:
            short_profitable = 0
            for idx in short_signals.index:
                if self.df.loc[idx, 'profit_pct'] >= self.profit_threshold:
                    short_profitable += 1
            short_wr = short_profitable / len(short_signals) * 100
            logger.info(f'\n做空信號 (上軌有盈利)')
            logger.info(f'  信號數: {len(short_signals)}')
            logger.info(f'  輊錢信號: {short_profitable}')
            logger.info(f'  勝率: {short_wr:.2f}%')
        
        total_profitable = len(self.df[self.df['label'].isin([1, 2])])
        total_winning = len(self.df[(self.df['label'].isin([1, 2])) & (self.df['profit_pct'] >= self.profit_threshold)])
        
        if total_profitable > 0:
            overall_wr = total_winning / total_profitable * 100
            logger.info(f'\n整體成總率')
            logger.info(f'  總信號數: {total_profitable}')
            logger.info(f'  總輊錢數: {total_winning}')
            logger.info(f'  成功率: {overall_wr:.2f}%')
            logger.info(f'  目標: 100%')
        
        logger.info('='*60)
    
    def save_labels(self, symbol: str, timeframe: str):
        """保存標籤到 CSV"""
        output_path = f'outputs/labels/{symbol}_{timeframe}_profitability_v2.csv'
        
        # 只保存需要的列
        output_df = self.df[[
            'open', 'high', 'low', 'close', 'label', 'profit_pct', 'target_price'
        ]].copy()
        
        if 'open_time' in self.df.columns:
            output_df = self.df[[
                'open_time', 'open', 'high', 'low', 'close', 'label', 'profit_pct', 'target_price'
            ]].copy()
        
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f'\n標籤已保存到: {output_path}')
    
    def run_full_pipeline(self, symbol: str, timeframe: str):
        """完整流程"""
        logger.info('\n' + '='*60)
        logger.info(f'開始為 {symbol}_{timeframe} 創建標籤 (v2)')
        logger.info('='*60)
        
        # 1. 載入數據
        if not self.load_data(symbol, timeframe):
            return None
        
        # 2. 計算 BB
        self.calculate_bollinger_bands()
        
        # 3. 偵測觸碰及接近
        self.detect_touches_and_near()
        
        # 4. 創建標籤
        self.create_labels()
        
        # 5. 回測上有盈利K棒
        self.backtest_on_profitable_signals()
        
        # 6. 保存標籤
        self.save_labels(symbol, timeframe)
        
        logger.info(f'\n{symbol}_{timeframe} 流程完成！\n')
        
        return self.df


def main():
    logger.info('\n' + '='*60)
    logger.info('基於盈利性的標籤創建 v2')
    logger.info('='*60)
    
    creator = ProfitabilityLabelCreatorV2(
        bb_period=20,
        bb_std=2,
        touch_threshold_pct=0.05,
        holding_bars=5,
        profit_threshold=0.1
    )
    
    creator.run_full_pipeline('BTCUSDT', '15m')


if __name__ == '__main__':
    main()

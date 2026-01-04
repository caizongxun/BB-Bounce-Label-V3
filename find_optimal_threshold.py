"""
自動查找最優閾值

目標：遍歷不同的觸碰閾值（0.01% ~ 1%），找出能捕獲最多有盈利機會的閾值

需求：
- 找出產生最多有盈利信號的閾值
- 同時保持儘可能高的勝率 (>75%)
- 標記出每個閾值的統計數據
"""

import pandas as pd
import numpy as np
import logging
import sys
import io
from datetime import datetime
from pathlib import Path
import json

# 設置 UTF-8 編碼
if sys.platform == 'win32':
    # Windows 上設置 UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 設置日誌
Path('logs').mkdir(exist_ok=True)
Path('outputs/threshold_analysis').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/threshold_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OptimalThresholdFinder:
    """自動查找最優閾值的器"""
    
    def __init__(self, bb_period=20, bb_std=2, holding_bars=5, profit_threshold=0.1):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.holding_bars = holding_bars
        self.profit_threshold = profit_threshold
        
        self.df = None
        self.results = []  # 存儲每個閾值的統計
    
    def load_data(self, symbol: str, timeframe: str):
        """載入數據並自動偵測列名"""
        csv_path = f'data/{symbol}_{timeframe}.csv'
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f'成功加載 {symbol}_{timeframe}: {len(self.df)} 行數據')
            
            # 自動偵測列名
            if 'timestamp' in self.df.columns:
                self.df = self.df.rename(columns={'timestamp': 'open_time'})
            
            # 檢查OHLC列
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                logger.error(f'缺少列：{missing_cols}')
                return False
            
            # 轉換時間
            if 'open_time' in self.df.columns:
                try:
                    self.df['open_time'] = pd.to_datetime(self.df['open_time'])
                except:
                    pass
            
            self.df = self.df.sort_values('open_time' if 'open_time' in self.df.columns else self.df.columns[0]).reset_index(drop=True)
            return True
        except FileNotFoundError:
            logger.error(f'找不到數據文件：{csv_path}')
            return False
    
    def calculate_bollinger_bands(self):
        """計算布林傑帶"""
        self.df['sma'] = self.df['close'].rolling(window=self.bb_period).mean()
        self.df['std'] = self.df['close'].rolling(window=self.bb_period).std()
        self.df['upper_band'] = self.df['sma'] + (self.df['std'] * self.bb_std)
        self.df['lower_band'] = self.df['sma'] - (self.df['std'] * self.bb_std)
        self.df['bb_width'] = self.df['upper_band'] - self.df['lower_band']
    
    def detect_touches_with_threshold(self, threshold_pct: float):
        """使用特定閾值偵測觸碰點"""
        touches = []
        touch_types = {}
        
        for i in range(self.bb_period, len(self.df) - self.holding_bars):
            row = self.df.iloc[i]
            
            if pd.isna(row['bb_width']) or row['bb_width'] == 0:
                continue
            
            # 計算距離上下軌的相對距離
            dist_to_lower = (row['close'] - row['lower_band']) / row['bb_width'] * 100
            dist_to_upper = (row['upper_band'] - row['close']) / row['bb_width'] * 100
            
            low_to_lower = (row['low'] - row['lower_band']) / row['bb_width'] * 100
            high_to_upper = (row['upper_band'] - row['high']) / row['bb_width'] * 100
            
            # 觸碰下軌
            if low_to_lower <= threshold_pct or dist_to_lower <= threshold_pct:
                touches.append(i)
                touch_types[i] = 'lower'
            # 觸碰上軌
            elif high_to_upper <= threshold_pct or dist_to_upper <= threshold_pct:
                touches.append(i)
                touch_types[i] = 'upper'
        
        return touches, touch_types
    
    def calculate_profitability_stats(self, touches: list, touch_types: dict):
        """計算有盈利的統計數據"""
        lower_profitable = 0
        lower_unprofitable = 0
        upper_profitable = 0
        upper_unprofitable = 0
        
        for touch_idx in touches:
            touch_type = touch_types[touch_idx]
            
            if touch_idx + self.holding_bars >= len(self.df):
                continue
            
            entry_price = self.df.iloc[touch_idx]['close']
            future_data = self.df.iloc[touch_idx + 1:touch_idx + self.holding_bars + 1]
            max_price = future_data['high'].max()
            min_price = future_data['low'].min()
            
            if touch_type == 'lower':
                profit_pct = (max_price - entry_price) / entry_price * 100
                is_profitable = profit_pct >= self.profit_threshold
                if is_profitable:
                    lower_profitable += 1
                else:
                    lower_unprofitable += 1
            else:  # upper
                profit_pct = (entry_price - min_price) / entry_price * 100
                is_profitable = profit_pct >= self.profit_threshold
                if is_profitable:
                    upper_profitable += 1
                else:
                    upper_unprofitable += 1
        
        return {
            'lower_profitable': lower_profitable,
            'lower_unprofitable': lower_unprofitable,
            'upper_profitable': upper_profitable,
            'upper_unprofitable': upper_unprofitable,
        }
    
    def search_optimal_threshold(self, threshold_range=(0.01, 1.0), step=0.01):
        """
        搜索最優閾值
        
        Args:
            threshold_range: 閾值範圍 (min, max)
            step: 每次增加的步長
        """
        logger.info('\n' + '='*70)
        logger.info('開始搜索最優閾值...')
        logger.info('='*70)
        
        # 計算 BB
        self.calculate_bollinger_bands()
        
        # 生成閾值段
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        
        logger.info(f'\n每次測試的閾值數量: {len(thresholds)}')
        logger.info(f'閾值範圍: {threshold_range[0]}% ~ {threshold_range[1]}%')
        logger.info(f'步長: {step}%\n')
        
        self.results = []
        best_result = None
        best_total_profitable = 0
        
        for idx, threshold in enumerate(thresholds, 1):
            # 偵測觸碰點
            touches, touch_types = self.detect_touches_with_threshold(threshold)
            
            if len(touches) == 0:
                continue
            
            # 計算統計
            stats = self.calculate_profitability_stats(touches, touch_types)
            
            total_signals = sum([stats['lower_profitable'], stats['lower_unprofitable'],
                               stats['upper_profitable'], stats['upper_unprofitable']])
            total_profitable = stats['lower_profitable'] + stats['upper_profitable']
            
            if total_signals > 0:
                lower_wr = stats['lower_profitable'] / (stats['lower_profitable'] + stats['lower_unprofitable']) * 100 if (stats['lower_profitable'] + stats['lower_unprofitable']) > 0 else 0
                upper_wr = stats['upper_profitable'] / (stats['upper_profitable'] + stats['upper_unprofitable']) * 100 if (stats['upper_profitable'] + stats['upper_unprofitable']) > 0 else 0
                overall_wr = total_profitable / total_signals * 100
            else:
                lower_wr = upper_wr = overall_wr = 0
            
            result = {
                'threshold': round(threshold, 4),
                'total_signals': total_signals,
                'total_profitable': total_profitable,
                'total_unprofitable': total_signals - total_profitable,
                'lower_profitable': stats['lower_profitable'],
                'lower_unprofitable': stats['lower_unprofitable'],
                'lower_wr': lower_wr,
                'upper_profitable': stats['upper_profitable'],
                'upper_unprofitable': stats['upper_unprofitable'],
                'upper_wr': upper_wr,
                'overall_wr': overall_wr,
            }
            
            self.results.append(result)
            
            # 追蹤最優結果
            if total_profitable > best_total_profitable:
                best_total_profitable = total_profitable
                best_result = result
            
            # 進度提示
            if idx % 5 == 0 or idx == len(thresholds):
                logger.info(f'進度: {idx}/{len(thresholds)} - 閾值 {threshold:.4f}% 找到 {total_profitable} 個有盈利信號')
        
        if best_result:
            logger.info('\n最優結果找到！')
            logger.info('='*70)
            self._log_result(best_result, is_best=True)
            logger.info('='*70)
        
        return self.results, best_result
    
    def _log_result(self, result: dict, is_best=False):
        """輸出統計結果"""
        prefix = '[BEST] ' if is_best else '       '
        logger.info(f'{prefix}閾值: {result["threshold"]:.4f}%')
        logger.info(f'       總信號數: {result["total_signals"]}')
        logger.info(f'       有盈利信號: {result["total_profitable"]} ({result["overall_wr"]:.2f}%)')
        logger.info(f'       下軌: {result["lower_profitable"]} 盈利 + {result["lower_unprofitable"]} 虧損 ({result["lower_wr"]:.2f}%)')
        logger.info(f'       上軌: {result["upper_profitable"]} 盈利 + {result["upper_unprofitable"]} 虧損 ({result["upper_wr"]:.2f}%)')
    
    def print_top_results(self, top_n=10):
        """打印前 N 個最好的結果"""
        logger.info('\n' + '='*70)
        logger.info(f'前 {top_n} 個最優的閾值')
        logger.info('='*70)
        
        # 按照有盈利信號數和勝率排序
        sorted_results = sorted(
            self.results,
            key=lambda x: (x['total_profitable'], x['overall_wr']),
            reverse=True
        )
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            logger.info(f'\n{i}. 閾值: {result["threshold"]:.4f}%')
            logger.info(f'   信號: {result["total_signals"]} | 有盈利: {result["total_profitable"]} | 勝率: {result["overall_wr"]:.2f}%')
            logger.info(f'   下軌: {result["lower_profitable"]}/{result["lower_profitable"] + result["lower_unprofitable"]} ({result["lower_wr"]:.2f}%)')
            logger.info(f'   上軌: {result["upper_profitable"]}/{result["upper_profitable"] + result["upper_unprofitable"]} ({result["upper_wr"]:.2f}%)')
        
        logger.info('\n' + '='*70)
    
    def save_analysis(self, symbol: str, timeframe: str):
        """保存分析結果到 JSON 和 CSV"""
        output_dir = Path('outputs/threshold_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存為 CSV
        csv_path = output_dir / f'{symbol}_{timeframe}_threshold_analysis.csv'
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f'\n分析結果已保存到 CSV: {csv_path}')
        
        # 保存為 JSON
        json_path = output_dir / f'{symbol}_{timeframe}_threshold_analysis.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f'分析結果已保存到 JSON: {json_path}')
    
    def run_full_pipeline(self, symbol: str, timeframe: str, 
                         threshold_range=(0.01, 1.0), step=0.01):
        """完整流程"""
        logger.info('\n' + '#'*70)
        logger.info(f'# 搜索最優閾值 - {symbol}_{timeframe}')
        logger.info('#'*70)
        
        # 載入數據
        if not self.load_data(symbol, timeframe):
            return None
        
        # 搜索閾值
        results, best_result = self.search_optimal_threshold(threshold_range, step)
        
        # 打印前 15 個最優結果
        self.print_top_results(top_n=15)
        
        # 保存分析
        self.save_analysis(symbol, timeframe)
        
        logger.info(f'\n{symbol}_{timeframe} 搜索完成！\n')
        
        return results, best_result


def main():
    logger.info('\n' + '#'*70)
    logger.info('# 自動查找最優閾值')
    logger.info('#'*70)
    
    finder = OptimalThresholdFinder(
        bb_period=20,
        bb_std=2,
        holding_bars=5,
        profit_threshold=0.1
    )
    
    # 搜索不同的閾值
    results, best_result = finder.run_full_pipeline(
        symbol='BTCUSDT',
        timeframe='15m',
        threshold_range=(0.01, 1.0),  # 搜查 0.01% ~ 1%
        step=0.05                      # 每次步進 0.05%
    )


if __name__ == '__main__':
    main()

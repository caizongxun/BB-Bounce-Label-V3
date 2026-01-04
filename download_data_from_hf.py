"""
從 Hugging Face 下載加密貨幣 OHLCV 數據

數據來源: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
"""

import os
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging
from datetime import datetime

# 設置日誌
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 設置參數
HF_DATASET = "zongowo111/v2-crypto-ohlcv-data"
HF_REPO_TYPE = "dataset"

# 所有支援的幣種 (需要符合 HF 上的子目錄名)
SYMBOLS = [
    'BTCUSDT',    # Bitcoin
    'ETHUSDT',    # Ethereum
    'BNBUSDT',    # Binance Coin
    'XRPUSDT',    # Ripple
    'ADAUSDT',    # Cardano
    'DOGEUSDT',   # Dogecoin
    'MATICUSDT',  # Polygon
    'LTCUSDT',    # Litecoin
    'AVAXUSDT',   # Avalanche
    'SOLUSDT',    # Solana
    'ATOMUSDT',   # Cosmos
    'ARBUSDT',    # Arbitrum
    'OPUSDT',     # Optimism
    'UNIUSDT',    # Uniswap
    'LINKUSDT',   # Chainlink
    'FILUSDT',    # Filecoin
    'ETCUSDT',    # Ethereum Classic
    'ALGOUSDT',   # Algorand
    'AAVEUSDT',   # Aave
    'NEARUSDT',   # NEAR Protocol
    'BCHUSDT',    # Bitcoin Cash
    'DOTUSDT',    # Polkadot
]

TIMEFRAMES = ['15m', '1h']  # 支援的時間框架


class HFDataDownloader:
    """從 Hugging Face 下載加密貨幣數據"""
    
    def __init__(self):
        # 创建數據目錄
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        logger.info(f'数据目录: {self.data_dir.absolute()}')
    
    def _get_file_name(self, symbol, timeframe):
        """
        根據符號和時間框架生成文件名
        
        有効判斷: HF 上的文件名可能是 BTC_15m.parquet 或 BTCUSDT_15m.parquet
        """
        # 第一種格式：符號不包含 USDT 后罐
        symbol_short = symbol.replace('USDT', '')
        return f"{symbol_short}_{timeframe}.parquet"
    
    def download_single_file(self, symbol, timeframe):
        """
        下載单個文件
        
        Args:
            symbol: 幣種 (e.g., 'BTCUSDT')
            timeframe: 時間框架 (e.g., '15m')
        
        Returns:
            True 如果成功， False 否則
        """
        try:
            file_name = self._get_file_name(symbol, timeframe)
            hf_path = f"klines/{symbol}/{file_name}"
            
            logger.info(f'\n下載: {symbol} {timeframe}')
            logger.info(f'  HF 路徑: {hf_path}')
            
            # 從 HF 下載
            file_path = hf_hub_download(
                repo_id=HF_DATASET,
                filename=hf_path,
                repo_type=HF_REPO_TYPE,
                cache_dir=str(self.data_dir),
                force_download=False,  # 已存在則不重新下載
            )
            
            # 讀取數據檢查
            df = pd.read_parquet(file_path)
            logger.info(f'  成功! 数据量: {len(df)} 行')
            logger.info(f'  文件位置: {file_path}')
            logger.info(f'  列: {list(df.columns)}')
            
            # 轉換成 CSV 格式（可選）
            csv_file = self.data_dir / f"{symbol}_{timeframe}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f'  已保存種 CSV: {csv_file}')
            
            return True
            
        except Exception as e:
            logger.error(f'  下載失敗: {e}')
            return False
    
    def download_all_data(self, symbols=None, timeframes=None):
        """
        下載所有数据
        
        Args:
            symbols: 要下載的符號列表 (None = 下載所有)
            timeframes: 要下載的時間框架 (None = 下載所有)
        """
        if symbols is None:
            symbols = SYMBOLS
        if timeframes is None:
            timeframes = TIMEFRAMES
        
        total = len(symbols) * len(timeframes)
        logger.info(f'\n==========================================')
        logger.info(f'開始下載 {total} 個文件')
        logger.info(f'==========================================')
        
        success_count = 0
        failed_count = 0
        
        for i, symbol in enumerate(symbols, 1):
            for timeframe in timeframes:
                if self.download_single_file(symbol, timeframe):
                    success_count += 1
                else:
                    failed_count += 1
        
        logger.info(f'\n==========================================')
        logger.info(f'下載完成!')
        logger.info(f'  成功: {success_count}')
        logger.info(f'  失敗: {failed_count}')
        logger.info(f'  成功率: {success_count/total*100:.1f}%')
        logger.info(f'==========================================')
        
        return success_count, failed_count
    
    def list_available_files(self):
        """列出已下載的文件"""
        logger.info(f'\n已下載的文件:')
        
        csv_files = list(self.data_dir.glob('*.csv'))
        parquet_files = list(self.data_dir.glob('**/*.parquet'))
        
        if csv_files:
            logger.info(f'\nCSV 文件 ({len(csv_files)}):')
            for f in sorted(csv_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f'  - {f.name} ({size_mb:.2f} MB)')
        
        if parquet_files:
            logger.info(f'\nParquet 文件 ({len(parquet_files)}):')
            for f in sorted(parquet_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f'  - {f.relative_to(self.data_dir)} ({size_mb:.2f} MB)')


def download_specific_symbols(symbols_list):
    """下載特定符號的數據"""
    downloader = HFDataDownloader()
    downloader.download_all_data(symbols=symbols_list)
    downloader.list_available_files()


def download_all():
    """下載所有數據"""
    downloader = HFDataDownloader()
    downloader.download_all_data()
    downloader.list_available_files()


def main():
    """主函数 - 選擇下載模式"""
    import sys
    
    logger.info('='*60)
    logger.info('Hugging Face 加密貨幣數據下載器')
    logger.info('='*60)
    
    # 方案 1: 下載所有数据 (▼詳推)
    download_all()
    
    # 方案 2: 只下載特定符號
    # download_specific_symbols(['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    
    # 方案 3: 自訂下載
    # downloader = HFDataDownloader()
    # downloader.download_single_file('BTCUSDT', '15m')
    # downloader.list_available_files()


if __name__ == '__main__':
    main()

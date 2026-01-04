"""
批量訓練脚本 - 訓練所有 23 種幣種

支持的符號:
BTC, ETH, BNB, SOL, ADA, XRP, DOGE, AVAX, LINK, MATIC, 
LIT, LTC, NEAR, OP, PEPE, SHIB, STX, SUI, TON, UNISWAP,
APT, BLAST, FLOKI

每個符號訓練時間: 3-5 秒
預料總訓練時間: 70-115 秒 (~2 分鐘)
"""

import logging
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 技術指標實現
class TechnicalIndicators:
    @staticmethod
    def RSI(prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices, dtype=float)
        rsi[:period] = 100. - 100. / (1. + rs)
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            upval = delta if delta > 0 else 0.
            downval = -delta if delta < 0 else 0.
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        return rsi
    
    @staticmethod
    def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
        ema_fast = pd.Series(prices).ewm(span=fastperiod).mean().values
        ema_slow = pd.Series(prices).ewm(span=slowperiod).mean().values
        macd = ema_fast - ema_slow
        signal = pd.Series(macd).ewm(span=signalperiod).mean().values
        hist = macd - signal
        return macd, signal, hist
    
    @staticmethod
    def ATR(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.r_[close[0], close[:-1]])
        tr3 = np.abs(low - np.r_[close[0], close[:-1]])
        tr = np.max([tr1, tr2, tr3], axis=0)
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr

# 特徵工程
class AdvancedFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.features = []
        self.indicators = TechnicalIndicators()
    
    def add_all_features(self):
        close = self.df['close'].values
        
        # Momentum
        if HAS_TALIB:
            self.df['RSI'] = talib.RSI(close, timeperiod=14)
        else:
            self.df['RSI'] = self.indicators.RSI(close, period=14)
        self.features.append('RSI')
        
        if HAS_TALIB:
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            macd, macd_signal, macd_hist = self.indicators.MACD(close)
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macd_signal
        self.df['MACD_Hist'] = macd_hist
        self.features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        self.df['ROC_10'] = self.df['close'].pct_change(periods=10) * 100
        self.features.append('ROC_10')
        
        window = 14
        lowest_low = self.df['low'].rolling(window=window).min()
        highest_high = self.df['high'].rolling(window=window).max()
        self.df['Stoch_K'] = ((self.df['close'] - lowest_low) / (highest_high - lowest_low + 1e-8)) * 100
        self.df['Stoch_D'] = self.df['Stoch_K'].rolling(window=3).mean()
        self.features.extend(['Stoch_K', 'Stoch_D'])
        
        # Volatility
        if HAS_TALIB:
            self.df['ATR'] = talib.ATR(self.df['high'].values, self.df['low'].values, close, timeperiod=14)
        else:
            self.df['ATR'] = self.indicators.ATR(self.df['high'].values, self.df['low'].values, close)
        self.features.append('ATR')
        
        self.df['Historical_Vol'] = self.df['close'].pct_change().rolling(window=20).std() * 100
        self.features.append('Historical_Vol')
        
        sma = self.df['close'].rolling(window=20).mean()
        std = self.df['close'].rolling(window=20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        self.df['BB_Position'] = (self.df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
        self.df['BB_Position'] = self.df['BB_Position'].clip(0, 1)
        self.features.append('BB_Position')
        
        hist_vol_ma = self.df['Historical_Vol'].rolling(20).mean()
        self.df['Volatility_Ratio'] = self.df['Historical_Vol'] / (hist_vol_ma + 1e-8)
        self.features.append('Volatility_Ratio')
        
        # Trend
        self.df['EMA_12'] = self.df['close'].ewm(span=12).mean()
        self.df['EMA_26'] = self.df['close'].ewm(span=26).mean()
        self.df['EMA_Diff'] = self.df['EMA_12'] - self.df['EMA_26']
        self.features.extend(['EMA_12', 'EMA_26', 'EMA_Diff'])
        
        self.df['SMA_20'] = self.df['close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['close'].rolling(window=50).mean()
        self.df['SMA_Ratio'] = self.df['SMA_20'] / (self.df['SMA_50'] + 1e-8)
        self.features.extend(['SMA_20', 'SMA_50', 'SMA_Ratio'])
        
        self.df['Price_SMA_Ratio'] = self.df['close'] / (self.df['SMA_20'] + 1e-8)
        self.features.append('Price_SMA_Ratio')
        
        # Volume
        obv = np.where(self.df['close'] > self.df['close'].shift(1), 1, 
                       np.where(self.df['close'] < self.df['close'].shift(1), -1, 0))
        self.df['OBV_Direction'] = obv
        self.features.append('OBV_Direction')
        
        self.df['Body_Size'] = np.abs(self.df['close'] - self.df['open'])
        self.df['True_Range'] = self.df['high'] - self.df['low']
        self.features.extend(['Body_Size', 'True_Range'])
        
        # Candle patterns
        o = self.df['open'].values
        h = self.df['high'].values
        l = self.df['low'].values
        c = self.df['close'].values
        
        self.df['Upper_Wick'] = h - np.maximum(o, c)
        self.features.append('Upper_Wick')
        
        self.df['Lower_Wick'] = np.minimum(o, c) - l
        self.features.append('Lower_Wick')
        
        self.df['True_Range_Ratio'] = (h - l) / (np.abs(c - o) + 1e-8)
        self.features.append('True_Range_Ratio')
        
        self.df['Direction'] = np.where(c >= o, 1, -1)
        self.features.append('Direction')
        
        # Statistical
        returns = self.df['close'].pct_change()
        self.df['Returns_Skew'] = returns.rolling(window=20).skew()
        self.df['Returns_Kurt'] = returns.rolling(window=20).kurt()
        self.features.extend(['Returns_Skew', 'Returns_Kurt'])
        
        self.df['Price_Acceleration'] = returns.diff()
        self.features.append('Price_Acceleration')
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            for col in ['RSI', 'MACD', 'Stoch_K', 'ATR', 'Historical_Vol']:
                if col in self.df.columns:
                    self.df[f'{col}_Lag{lag}'] = self.df[col].shift(lag)
                    self.features.append(f'{col}_Lag{lag}')
        
        # Interactions
        if 'RSI' in self.df.columns and 'ATR' in self.df.columns:
            self.df['RSI_ATR_Interaction'] = self.df['RSI'] * self.df['ATR']
            self.features.append('RSI_ATR_Interaction')
        
        if 'MACD' in self.df.columns and 'ATR' in self.df.columns:
            self.df['MACD_Normalized'] = self.df['MACD'] / (self.df['ATR'] + 1e-8)
            self.features.append('MACD_Normalized')
        
        return self.df, self.features

def train_symbol(symbol, csv_path):
    """訓練单一符號"""
    try:
        # 加載數據
        if not csv_path.exists():
            logger.warning(f"  ☐ {symbol}: 檔案不存在 {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        
        # 過濾有效數據
        df_valid = df[df['label'].isin([0, 1])].copy()
        
        if len(df_valid) < 1000:
            logger.warning(f"  ☐ {symbol}: 有效樣本數過少 ({len(df_valid)})")
            return None
        
        # 特徵工程
        engineer = AdvancedFeatureEngineer(df_valid)
        df_features, features = engineer.add_all_features()
        
        X = df_features[features]
        y = df_features['label']
        
        # 準備數據
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 500:
            logger.warning(f"  ☐ {symbol}: 準備不譜數據過少")
            return None
        
        # 標準化並分割
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        
        # 訓練模型
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        xgb_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # 評估
        results = {}
        for name, model in [('XGBoost', xgb_model), ('RandomForest', rf_model)]:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc_roc': roc_auc_score(y_test, y_pred_proba)
            }
        
        # 選擇最佳模型
        best_name = max(results, key=lambda x: results[x]['auc_roc'])
        best_model = xgb_model if best_name == 'XGBoost' else rf_model
        best_result = results[best_name]
        
        # 保存模型
        output_dir = Path(f"outputs/models/{symbol}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        with open(output_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"  ✅ {symbol}: AUC={best_result['auc_roc']:.4f} F1={best_result['f1']:.4f} ({best_name})")
        
        return best_result
    
    except Exception as e:
        logger.error(f"  ❌ {symbol}: {str(e)}")
        return None

def main():
    # 23 種幣種
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
        'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT',
        'LITUSDT', 'LTCUSDT', 'NEARUSDT', 'OPUSDT', 'PEPEUSDT',
        'SHIBUSDT', 'STXUSDT', 'SUIUSDT', 'TONUSDT', 'UNIUSDT',
        'APTUSDT', 'BLASTUSDT', 'FLOKIUSDT'
    ]
    
    logger.info("\n" + "="*70)
    logger.info("批量訓練所有符號 (23 種)")
    logger.info("="*70 + "\n")
    
    results_summary = {}
    start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/23] 訓練 {symbol}...")
        csv_path = Path(f"outputs/labels/{symbol}_15m_profitability_v2.csv")
        result = train_symbol(symbol, csv_path)
        if result:
            results_summary[symbol] = result
    
    elapsed = time.time() - start_time
    
    # 求和统计
    logger.info("\n" + "="*70)
    logger.info("訓練結果總結")
    logger.info("="*70)
    
    if results_summary:
        df_results = pd.DataFrame(results_summary).T
        df_results = df_results.sort_values('auc_roc', ascending=False)
        
        logger.info(f"\n成功訓練: {len(results_summary)}/{len(symbols)}")
        logger.info(f"总耐時間: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分)")
        logger.info(f"\nAUC-ROC 總統:")
        logger.info(f"  平均: {df_results['auc_roc'].mean():.4f}")
        logger.info(f"  最高: {df_results['auc_roc'].max():.4f}")
        logger.info(f"  最低: {df_results['auc_roc'].min():.4f}")
        logger.info(f"\nF1 Score 總統:")
        logger.info(f"  平均: {df_results['f1'].mean():.4f}")
        logger.info(f"  最高: {df_results['f1'].max():.4f}")
        logger.info(f"  最低: {df_results['f1'].min():.4f}")
        
        logger.info(f"\n性能排名 (AUC-ROC):")
        for idx, (symbol, row) in enumerate(df_results.iterrows(), 1):
            logger.info(f"  {idx:2d}. {symbol:12s} - AUC={row['auc_roc']:.4f} F1={row['f1']:.4f} Pre={row['precision']:.4f} Rec={row['recall']:.4f}")
        
        # 保存經隨
        results_df = pd.DataFrame(results_summary).T.sort_values('auc_roc', ascending=False)
        results_df.to_csv('outputs/models/training_results_summary.csv')
        logger.info(f"\n經隨已保存: outputs/models/training_results_summary.csv")
    else:
        logger.error("\n純不到有效數據，需要先可的標第檔案")
    
    logger.info("\n" + "="*70)
    logger.info("訓練完成！")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()

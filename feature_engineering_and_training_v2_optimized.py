"""
特弶工程 + 模型訓練 (v2 優化版)

優化方面：
1. 加入歷史序列特弶 (前 3 根 K 棒的信息)
2. 添加技術指標 (RSI, ATR, MACD)
3. 切換到 XGBoost (更強大的功率)
4. 調整類別權重 (解決標籤不平衡)
"""

import pandas as pd
import numpy as np
import logging
import sys
import io
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 設置 UTF-8 編碼
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 設置日誌
Path('logs').mkdir(exist_ok=True)
Path('outputs/models').mkdir(parents=True, exist_ok=True)
Path('outputs/analysis').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'logs/model_training_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineerV2:
    """優化版特弶提取器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = []
    
    def calculate_bollinger_bands_first(self, bb_period=20, bb_std=2):
        """計算 Bollinger Bands"""
        logger.info(f'\n計算布林傑帶 (period={bb_period}, std={bb_std})...')
        
        self.df['sma'] = self.df['close'].rolling(window=bb_period).mean()
        self.df['std'] = self.df['close'].rolling(window=bb_period).std()
        self.df['upper_band'] = self.df['sma'] + (self.df['std'] * bb_std)
        self.df['lower_band'] = self.df['sma'] - (self.df['std'] * bb_std)
        self.df['bb_width'] = self.df['upper_band'] - self.df['lower_band']
        
        logger.info('布林傑帶計算完成')
    
    def calculate_candle_features(self):
        """計算 K 棒特弶"""
        logger.info('\n計算 K 棒特弶...')
        
        self.df['body_size'] = self.df['high'] - self.df['low']
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        
        self.df['upper_wick_ratio'] = np.where(
            self.df['body_size'] > 0,
            self.df['upper_wick'] / self.df['body_size'],
            0
        )
        
        self.df['lower_wick_ratio'] = np.where(
            self.df['body_size'] > 0,
            self.df['lower_wick'] / self.df['body_size'],
            0
        )
        
        self.df['direction'] = np.where(self.df['close'] > self.df['open'], 1, -1)
        
        self.df['close_position'] = np.where(
            self.df['body_size'] > 0,
            (self.df['close'] - self.df['low']) / self.df['body_size'],
            0.5
        )
        
        logger.info('K 棒特弶計算完成')
    
    def calculate_volatility_features(self, window=20):
        """計算波幅特弶"""
        logger.info(f'\n計算波幅特弶 (window={window})...')
        
        self.df['volatility'] = self.df['high'].rolling(window).std()
        self.df['normalized_volatility'] = self.df['high'].rolling(window).std() / self.df['close'].rolling(window).mean()
        self.df['price_change'] = self.df['close'].pct_change() * 100
        self.df['hl_ratio'] = self.df['high'] / self.df['low']
        
        logger.info('波幅特弶計算完成')
    
    def calculate_bb_features(self):
        """計算布林傑帶特弶"""
        logger.info('\n計算布林傑帶特弶...')
        
        self.df['bb_position'] = np.where(
            self.df['bb_width'] > 0,
            (self.df['close'] - self.df['lower_band']) / self.df['bb_width'],
            0.5
        )
        
        self.df['bb_width_ratio'] = np.where(
            self.df['close'] > 0,
            self.df['bb_width'] / self.df['close'],
            0
        )
        
        self.df['distance_to_upper'] = np.where(
            self.df['bb_width'] > 0,
            (self.df['upper_band'] - self.df['close']) / self.df['bb_width'] * 100,
            0
        )
        
        self.df['distance_to_lower'] = np.where(
            self.df['bb_width'] > 0,
            (self.df['close'] - self.df['lower_band']) / self.df['bb_width'] * 100,
            0
        )
        
        logger.info('布林傑帶特弶計算完成')
    
    def calculate_historical_features(self, lookback=3):
        """
        ✅ 新加：歷史序列特弶
        這是最重要的优化！
        """
        logger.info(f'\n計算歷史序列特弶 (lookback={lookback})...')
        
        # 前 N 根 K 棒的方向
        for lag in range(1, lookback + 1):
            self.df[f'prev_{lag}_direction'] = self.df['direction'].shift(lag)
            self.df[f'prev_{lag}_body'] = self.df['body'].shift(lag)
            self.df[f'prev_{lag}_volatility'] = self.df['volatility'].shift(lag)
        
        # 棂次方向 (連續 N 根下跌表示暴跌)
        self.df['consecutive_down'] = 0
        for i in range(len(self.df)):
            if i >= lookback:
                down_count = sum([
                    self.df[f'prev_{lag}_direction'].iloc[i] == -1
                    for lag in range(1, lookback + 1)
                ])
                self.df.loc[i, 'consecutive_down'] = down_count
        
        # 當前 K 棒 vs 前一根的相對大小
        self.df['body_size_change'] = self.df['body'] / (self.df['body'].shift(1) + 1e-6)
        self.df['body_size_change'] = self.df['body_size_change'].fillna(1)
        
        # 動量變化（趨勢反轉信號）
        self.df['momentum_reversal'] = (
            (self.df['direction'].shift(1) == -1) &
            (self.df['close'] > self.df['close'].shift(1))
        ).astype(int)
        
        logger.info('歷史序列特弶計算完成')
    
    def calculate_technical_features(self):
        """
        ✅ 新加：技術指標 (RSI, ATR, MACD)
        """
        logger.info('\n計算技術指標...')
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        self.df['rsi'] = self.df['rsi'].fillna(50)
        
        # ATR (真實流動率)
        tr1 = self.df['high'] - self.df['low']
        tr2 = abs(self.df['high'] - self.df['close'].shift())
        tr3 = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(14).mean()
        self.df['atr'] = self.df['atr'].fillna(self.df['atr'].mean())
        
        # MACD
        exp1 = self.df['close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd'] = exp1 - exp2
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd'] = self.df['macd'].fillna(0)
        self.df['macd_signal'] = self.df['macd_signal'].fillna(0)
        
        logger.info('技術指標計算完成')
    
    def get_feature_columns(self):
        """獲得所有特弶列名"""
        features = [
            # K 棒特弶 (8)
            'body_size', 'body', 'upper_wick', 'lower_wick',
            'upper_wick_ratio', 'lower_wick_ratio', 'direction', 'close_position',
            
            # 波幅特弶 (4)
            'volatility', 'normalized_volatility', 'price_change', 'hl_ratio',
            
            # BB 特弶 (4)
            'bb_position', 'bb_width_ratio', 'distance_to_upper', 'distance_to_lower',
            
            # 歷史序列特弶 (9) - ✅ 新加
            'prev_1_direction', 'prev_2_direction', 'prev_3_direction',
            'prev_1_body', 'prev_2_body', 'prev_3_body',
            'prev_1_volatility', 'prev_2_volatility', 'prev_3_volatility',
            'consecutive_down', 'body_size_change', 'momentum_reversal',
            
            # 技術指標 (3) - ✅ 新加
            'rsi', 'atr', 'macd'
        ]
        return features


class ModelTrainerV2:
    """優化版模型訓練器"""
    
    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """準備訓練數據"""
        logger.info('\n準備訓練數據...')
        
        df_clean = self.df.dropna()
        df_clean = df_clean[df_clean['label'] != -1].copy()
        
        logger.info(f'佭冱數量: {len(df_clean)}')
        logger.info(f'  有盈利 (label=1/2): {(df_clean["label"] > 0).sum()}')
        logger.info(f'  無盈利 (label=0): {(df_clean["label"] == 0).sum()}')
        
        X = df_clean[self.feature_cols]
        y = (df_clean['label'] > 0).astype(int)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f'\n訓練集合: {len(self.X_train)}')
        logger.info(f'測試集合: {len(self.X_test)}')
    
    def train_model(self):
        """訓練模型"""
        logger.info('\n訓練模型...')
        
        # 計算類別權重
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = {
            0: class_weights[0],
            1: class_weights[1]
        }
        
        logger.info(f'類別權重: {class_weight_dict}')
        
        # ✅ 使用 XGBoost（更好）或 Random Forest (選擇權)
        if HAS_XGBOOST:
            logger.info('\n使用 XGBoost 模型')
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_weights[0] / class_weights[1],
                random_state=42,
                verbosity=0
            )
        else:
            logger.info('\nXGBoost 未安裝，使用 Random Forest')
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight=class_weight_dict,
                n_jobs=-1
            )
        
        self.model.fit(self.X_train, self.y_train)
        logger.info('模型訓練完成')
    
    def evaluate_model(self):
        """評估模型"""
        logger.info('\n評估模型...')
        logger.info('='*70)
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_test_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # 訓練集合
        logger.info('\n訓練集合性能:')
        train_acc = accuracy_score(self.y_train, y_train_pred)
        logger.info(f'  Accuracy: {train_acc:.4f}')
        
        # 測試集合
        logger.info('\n測試集合性能:')
        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred)
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, y_test_pred_proba)
        
        logger.info(f'  Accuracy:  {test_acc:.4f}')
        logger.info(f'  Precision: {test_precision:.4f}')
        logger.info(f'  Recall:    {test_recall:.4f}')
        logger.info(f'  F1 Score:  {test_f1:.4f}')
        logger.info(f'  AUC-ROC:   {test_auc:.4f}')
        
        # 混淆矩陣
        logger.info('\n混淆矩陣 (Test Set):')
        cm = confusion_matrix(self.y_test, y_test_pred)
        logger.info(f'  TN: {cm[0, 0]}, FP: {cm[0, 1]}')
        logger.info(f'  FN: {cm[1, 0]}, TP: {cm[1, 1]}')
        
        logger.info('='*70)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'auc': test_auc
        }
    
    def get_feature_importance(self, top_n=15):
        """獲得特弶重要度"""
        logger.info(f'\n前 {top_n} 個最重要的特弶:')
        logger.info('='*70)
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i, idx in enumerate(indices[:top_n], 1):
            logger.info(f'{i:2d}. {self.feature_cols[idx]:25s} - {importances[idx]:.4f}')
        
        logger.info('='*70)
        
        return dict(zip([self.feature_cols[i] for i in indices[:top_n]], 
                       importances[indices[:top_n]]))
    
    def save_model(self, symbol: str, timeframe: str):
        """保存模型"""
        model_path = f'outputs/models/{symbol}_{timeframe}_profitability_model_v2.pkl'
        scaler_path = f'outputs/models/{symbol}_{timeframe}_scaler_v2.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f'\n模型已保存到: {model_path}')
        logger.info(f'標正化器已保存到: {scaler_path}')


def main():
    logger.info('\n' + '#'*70)
    logger.info('# 特弶工程 + 模型訓練 (v2 優化版)')
    logger.info('#'*70)
    
    # 載入標籤數據
    logger.info('\n載入標籤數據...')
    label_file = 'outputs/labels/BTCUSDT_15m_profitability_v2.csv'
    
    try:
        df = pd.read_csv(label_file)
        logger.info(f'成功加載: {label_file}')
    except FileNotFoundError:
        logger.error(f'找不到標籤文件: {label_file}')
        logger.error('請先運行 label_profitability_v2.py')
        return
    
    # 提取特弶
    logger.info('\n' + '='*70)
    logger.info('第 1 階段：特弶提取器初始化')
    logger.info('='*70)
    
    fe = FeatureEngineerV2(df)
    fe.calculate_bollinger_bands_first(bb_period=20, bb_std=2)
    fe.calculate_candle_features()
    fe.calculate_volatility_features(window=20)
    fe.calculate_bb_features()
    
    # ✅ 新步驟
    fe.calculate_historical_features(lookback=3)
    fe.calculate_technical_features()
    
    feature_cols = fe.get_feature_columns()
    df = fe.df
    
    logger.info(f'\n特弶數量: {len(feature_cols)}')
    logger.info(f'特弶列: {feature_cols}')
    
    # 訓練模型
    logger.info('\n' + '='*70)
    logger.info('第 2 階段：模型訓練')
    logger.info('='*70)
    
    trainer = ModelTrainerV2(df, feature_cols)
    trainer.prepare_data(test_size=0.2)
    trainer.train_model()
    
    # 評估模型
    logger.info('\n' + '='*70)
    logger.info('第 3 階段：模型評估')
    logger.info('='*70)
    
    metrics = trainer.evaluate_model()
    
    # 特弶重要度
    feature_importance = trainer.get_feature_importance(top_n=15)
    
    # 保存模型
    trainer.save_model('BTCUSDT', '15m')
    
    # 扳述性統計
    logger.info('\n' + '='*70)
    logger.info('模型訓練完成！')
    logger.info('='*70)
    logger.info('\n鏡前后比較：')
    logger.info(f'  AUC-ROC: 68.12% \u2192 {metrics["auc"]:.2%}')
    logger.info(f'  改進: {(metrics["auc"] - 0.6812) * 100:.2f} 百分佋')
    logger.info('')


if __name__ == '__main__':
    main()

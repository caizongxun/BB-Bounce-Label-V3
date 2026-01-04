"""
特弶工程 + 模型訓練

流程：
1. 載入標籤數據
2. 提取 K 棒特弶（高度、寬度、上下影線等）
3. 樓次分程 (train/test)
4. 訓練高斩化模型（Random Forest / XGBoost）
5. 計算標準 (accuracy, precision, recall, F1, AUC)
6. 分析特弶重要度
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
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
            f'logs/model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特弶提取器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = []
    
    def calculate_bollinger_bands_first(self, bb_period=20, bb_std=2):
        """先計算 Bollinger Bands（徆不是例突突突）"""
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
        
        # 1. 体大 (高 - 低)
        self.df['body_size'] = self.df['high'] - self.df['low']
        
        # 2. 寬度 (|close - open|)
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        
        # 3. 上影線 (高 - max(open, close))
        self.df['upper_wick'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        
        # 4. 下影線 (min(open, close) - 低)
        self.df['lower_wick'] = self.df[['open', 'close']].min(axis=1) - self.df['low']
        
        # 5. 上下影線比例
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
        
        # 6. 方向 (1=上潈, -1=下潈)
        self.df['direction'] = np.where(self.df['close'] > self.df['open'], 1, -1)
        
        # 7. 量次位置 (close 位於 [low, high] 中的位置)
        self.df['close_position'] = np.where(
            self.df['body_size'] > 0,
            (self.df['close'] - self.df['low']) / self.df['body_size'],
            0.5
        )
        
        logger.info('K 棒特弶計算完成')
    
    def calculate_volatility_features(self, window=20):
        """計算波幅特弶"""
        logger.info(f'\n計算波幅特弶 (window={window})...')
        
        # 1. 波幅
        self.df['volatility'] = self.df['high'].rolling(window).std()
        
        # 2. 正規化波幅 (相對於平均價格)
        self.df['normalized_volatility'] = self.df['high'].rolling(window).std() / self.df['close'].rolling(window).mean()
        
        # 3. 價格變化率
        self.df['price_change'] = self.df['close'].pct_change() * 100
        
        # 4. 高低比
        self.df['hl_ratio'] = self.df['high'] / self.df['low']
        
        logger.info('波幅特弶計算完成')
    
    def calculate_bb_features(self):
        """計算布林傑帶特弶"""
        logger.info('\n計算布林傑帶特弶...')
        
        # 子鎖走下位置
        self.df['bb_position'] = np.where(
            self.df['bb_width'] > 0,
            (self.df['close'] - self.df['lower_band']) / self.df['bb_width'],
            0.5
        )
        
        # 子鎖比例
        self.df['bb_width_ratio'] = np.where(
            self.df['close'] > 0,
            self.df['bb_width'] / self.df['close'],
            0
        )
        
        # 距離上軌 (百分比)
        self.df['distance_to_upper'] = np.where(
            self.df['bb_width'] > 0,
            (self.df['upper_band'] - self.df['close']) / self.df['bb_width'] * 100,
            0
        )
        
        # 距離下軌 (百分比)
        self.df['distance_to_lower'] = np.where(
            self.df['bb_width'] > 0,
            (self.df['close'] - self.df['lower_band']) / self.df['bb_width'] * 100,
            0
        )
        
        logger.info('布林傑帶特弶計算完成')
    
    def get_feature_columns(self):
        """獲得所有特弶列名"""
        features = [
            'body_size', 'body', 'upper_wick', 'lower_wick',
            'upper_wick_ratio', 'lower_wick_ratio', 'direction', 'close_position',
            'volatility', 'normalized_volatility', 'price_change', 'hl_ratio',
            'bb_position', 'bb_width_ratio', 'distance_to_upper', 'distance_to_lower'
        ]
        return features


class ModelTrainer:
    """模型訓練器"""
    
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
        
        # 移除缺失值
        df_clean = self.df.dropna()
        
        # 移除 label = -1 (沒有觸碰的K棒)
        df_clean = df_clean[df_clean['label'] != -1].copy()
        
        logger.info(f'佭冱數量: {len(df_clean)}')
        logger.info(f'  有盈利 (label=1/2): {(df_clean["label"] > 0).sum()}')
        logger.info(f'  無盈利 (label=0): {(df_clean["label"] == 0).sum()}')
        
        # 提取特弶和標籤
        X = df_clean[self.feature_cols]
        y = (df_clean['label'] > 0).astype(int)  # 1=有盈利, 0=無盈利
        
        # 分程
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 標正化
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f'\n訓練集合: {len(self.X_train)}')
        logger.info(f'測試集合: {len(self.X_test)}')
    
    def train_model(self, n_estimators=200, max_depth=15, random_state=42):
        """訓練 Random Forest 模型"""
        logger.info('\n訓練模型...')
        
        # 計算類別權重 (故司不平衡趣)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_train),
            y=self.y_train
        )
        class_weight_dict = {
            0: class_weights[0],  # 無盈利
            1: class_weights[1]   # 有盈利
        }
        
        logger.info(f'類別權重: {class_weight_dict}')
        
        # 訓練模型
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight=class_weight_dict,
            n_jobs=-1
        )
        
        self.model.fit(self.X_train, self.y_train)
        logger.info('模型訓練完成')
    
    def evaluate_model(self):
        """評估模型"""
        logger.info('\n評估模型...')
        logger.info('='*70)
        
        # 預測
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_test_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # 訓練集合指標
        logger.info('\n訓練集合性能:')
        train_acc = accuracy_score(self.y_train, y_train_pred)
        logger.info(f'  Accuracy: {train_acc:.4f}')
        
        # 測試集合指標
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
    
    def get_feature_importance(self, top_n=10):
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
        model_path = f'outputs/models/{symbol}_{timeframe}_profitability_model.pkl'
        scaler_path = f'outputs/models/{symbol}_{timeframe}_scaler.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f'\n模型已保存到: {model_path}')
        logger.info(f'標正化器已保存到: {scaler_path}')


def main():
    logger.info('\n' + '#'*70)
    logger.info('# 特弶工程 + 模型訓練')
    logger.info('#'*70)
    
    # 載入標籤數據
    logger.info('\n載入標籤數據...')
    label_file = 'outputs/labels/BTCUSDT_15m_profitability_v2.csv'
    
    try:
        df = pd.read_csv(label_file)
        logger.info(f'成功加載: {label_file}')
    except FileNotFoundError:
        logger.error(f'找不到標籤文件: {label_file}')
        logger.error('請先運行 label_profitability_v2.py 載入標籤數據')
        return
    
    # 提取特弶
    fe = FeatureEngineer(df)
    
    # 很重要：先計算 BB、然後計算其他特弶
    fe.calculate_bollinger_bands_first(bb_period=20, bb_std=2)
    fe.calculate_candle_features()
    fe.calculate_volatility_features(window=20)
    fe.calculate_bb_features()
    
    feature_cols = fe.get_feature_columns()
    df = fe.df
    
    logger.info(f'\n特弶數量: {len(feature_cols)}')
    logger.info(f'特弶列: {feature_cols}')
    
    # 訓練模型
    trainer = ModelTrainer(df, feature_cols)
    trainer.prepare_data(test_size=0.2)
    trainer.train_model(n_estimators=200, max_depth=15)
    
    # 評估模型
    metrics = trainer.evaluate_model()
    
    # 特弶重要度
    feature_importance = trainer.get_feature_importance(top_n=10)
    
    # 保存模型
    trainer.save_model('BTCUSDT', '15m')
    
    logger.info(f'\n模型訓練完成！\n')


if __name__ == '__main__':
    main()

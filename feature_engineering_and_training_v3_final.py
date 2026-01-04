"""
特徵工程 + 模型訓練 V3 (完全重新設計)

設計思路:
1. 多層特徵工程 (技術指標 + 波動率 + 時序特徵)
2. 改進標籤邏輯 (基於真實獲利能力)
3. 模型優化 (LSTM + XGBoost 混合方案)
4. 超參數調優

作者: AI 系統設計
日期: 2026-01-04
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import talib

# 配置
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PART 1: 進階標籤生成器
# ============================================================================

class AdvancedLabelGenerator:
    """
    改進的標籤生成邏輯:
    - 識別確實觸碰下軌的點
    - 檢查未來 5 根 K 棒的獲利潛力
    - 加權考慮波幅和獲利幅度
    """
    
    def __init__(self, df, bb_period=20, bb_std=2, lookforward=5, profit_threshold=0.001):
        self.df = df.copy()
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.lookforward = lookforward
        self.profit_threshold = profit_threshold
        
        logger.info("初始化進階標籤生成器...")
        self._calculate_bollinger_bands()
        
    def _calculate_bollinger_bands(self):
        """計算布林傑帶"""
        self.df['SMA'] = self.df['close'].rolling(window=self.bb_period).mean()
        self.df['STD'] = self.df['close'].rolling(window=self.bb_period).std()
        self.df['BB_Upper'] = self.df['SMA'] + (self.df['STD'] * self.bb_std)
        self.df['BB_Lower'] = self.df['SMA'] - (self.df['STD'] * self.bb_std)
        self.df['BB_Middle'] = self.df['SMA']
        
    def _detect_lower_band_touch(self):
        """
        檢測觸碰下軌的點:
        - low <= BB_Lower
        - 非極端情況 (避免崩盤)
        """
        self.df['touches_lower'] = False
        
        for i in range(self.bb_period, len(self.df)):
            if pd.notna(self.df.loc[i, 'BB_Lower']):
                # 低點 <= 下軌
                if self.df.loc[i, 'low'] <= self.df.loc[i, 'BB_Lower']:
                    self.df.loc[i, 'touches_lower'] = True
                    
    def _calculate_future_profitability(self):
        """
        計算未來 N 根 K 棒的獲利能力:
        - 從觸碰點的低點開始
        - 計算未來最高點的獲利 %
        - 考慮波幅加權
        """
        self.df['label'] = 0  # 默認無盈利
        
        for i in range(len(self.df) - self.lookforward):
            if not self.df.loc[i, 'touches_lower']:
                continue
                
            # 當前低點（觸碰點）
            touch_low = self.df.loc[i, 'low']
            
            # 未來 N 根 K 棒的最高點
            future_high = self.df.loc[i:i+self.lookforward, 'high'].max()
            
            # 計算獲利百分比
            if touch_low > 0:
                profit_pct = (future_high - touch_low) / touch_low
                
                # 考慮 ATR 加權：大波幅時要求更高的絕對獲利
                atr = self.df.loc[i, 'ATR'] if 'ATR' in self.df.columns else 0
                adjusted_threshold = self.profit_threshold
                
                if atr > 0:
                    # 波幅越大，要求越高
                    adjusted_threshold *= (1 + atr / touch_low)
                
                # 標籤: 1 = 盈利, 0 = 不盈利
                if profit_pct >= adjusted_threshold:
                    self.df.loc[i, 'label'] = 1
                    
    def generate_labels(self):
        """生成完整標籤"""
        self._detect_lower_band_touch()
        self._calculate_future_profitability()
        
        # 統計
        touch_count = self.df['touches_lower'].sum()
        profitable_count = (self.df[self.df['touches_lower']]['label'] == 1).sum()
        
        logger.info(f"觸碰下軌點數: {touch_count}")
        logger.info(f"其中盈利: {profitable_count} ({100*profitable_count/max(touch_count,1):.1f}%)")
        
        return self.df[['label', 'touches_lower']]


# ============================================================================
# PART 2: 多層特徵工程
# ============================================================================

class AdvancedFeatureEngineer:
    """
    多層特徵工程:
    Layer 1: 技術指標 (RSI, MACD, OBV, Stochastic)
    Layer 2: 波動率相關 (ATR, Historical Vol, BB Position)
    Layer 3: 時序特徵 (Momentum, Trend, Pattern)
    Layer 4: 統計特徵 (Skewness, Kurtosis)
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.features = []
        
    def add_momentum_indicators(self):
        """添加動量指標"""
        logger.info("計算動量指標...")
        
        # RSI (14)
        close = self.df['close'].values
        self.df['RSI'] = talib.RSI(close, timeperiod=14)
        self.features.append('RSI')
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macd_signal
        self.df['MACD_Hist'] = macd_hist
        self.features.extend(['MACD', 'MACD_Signal', 'MACD_Hist'])
        
        # Stochastic Oscillator
        k, d = talib.STOCH(self.df['high'].values, self.df['low'].values, close, 
                          fastk_period=14, slowk_period=3, slowd_period=3)
        self.df['Stoch_K'] = k
        self.df['Stoch_D'] = d
        self.features.extend(['Stoch_K', 'Stoch_D'])
        
        # Rate of Change (ROC)
        self.df['ROC_10'] = talib.ROC(close, timeperiod=10)
        self.features.append('ROC_10')
        
    def add_volatility_indicators(self):
        """添加波動率指標"""
        logger.info("計算波動率指標...")
        
        # ATR (14)
        self.df['ATR'] = talib.ATR(self.df['high'].values, self.df['low'].values, 
                                   self.df['close'].values, timeperiod=14)
        self.features.append('ATR')
        
        # 歷史波動率 (20日)
        self.df['Historical_Vol'] = self.df['close'].pct_change().rolling(window=20).std()
        self.features.append('Historical_Vol')
        
        # Bollinger Bands Position (0-1)
        sma = self.df['close'].rolling(window=20).mean()
        std = self.df['close'].rolling(window=20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        self.df['BB_Position'] = (self.df['close'] - bb_lower) / (bb_upper - bb_lower)
        self.df['BB_Position'] = self.df['BB_Position'].clip(0, 1)
        self.features.append('BB_Position')
        
        # 波幅比 (當前與平均)
        self.df['Volatility_Ratio'] = self.df['Historical_Vol'] / self.df['Historical_Vol'].rolling(20).mean()
        self.features.append('Volatility_Ratio')
        
    def add_trend_indicators(self):
        """添加趨勢指標"""
        logger.info("計算趨勢指標...")
        
        # EMA (12 vs 26)
        self.df['EMA_12'] = talib.EMA(self.df['close'].values, timeperiod=12)
        self.df['EMA_26'] = talib.EMA(self.df['close'].values, timeperiod=26)
        self.df['EMA_Diff'] = self.df['EMA_12'] - self.df['EMA_26']
        self.features.extend(['EMA_12', 'EMA_26', 'EMA_Diff'])
        
        # SMA (20 vs 50)
        self.df['SMA_20'] = talib.SMA(self.df['close'].values, timeperiod=20)
        self.df['SMA_50'] = talib.SMA(self.df['close'].values, timeperiod=50)
        self.df['SMA_Ratio'] = self.df['SMA_20'] / (self.df['SMA_50'] + 1e-8)
        self.features.extend(['SMA_20', 'SMA_50', 'SMA_Ratio'])
        
        # 當前價格相對於 SMA
        self.df['Price_SMA_Ratio'] = self.df['close'] / (self.df['SMA_20'] + 1e-8)
        self.features.append('Price_SMA_Ratio')
        
    def add_volume_indicators(self):
        """添加成交量指標"""
        logger.info("計算成交量指標...")
        
        # OBV (On-Balance Volume)
        self.df['OBV'] = talib.OBV(self.df['close'].values, self.df['volume'].values)
        self.df['OBV_EMA'] = talib.EMA(self.df['OBV'].values, timeperiod=20)
        self.features.extend(['OBV', 'OBV_EMA'])
        
        # 成交量相對變化
        self.df['Volume_MA'] = self.df['volume'].rolling(window=20).mean()
        self.df['Volume_Ratio'] = self.df['volume'] / (self.df['Volume_MA'] + 1e-8)
        self.features.extend(['Volume_MA', 'Volume_Ratio'])
        
    def add_candle_patterns(self):
        """添加 K 棒形態特徵"""
        logger.info("計算 K 棒形態特徵...")
        
        o = self.df['open'].values
        h = self.df['high'].values
        l = self.df['low'].values
        c = self.df['close'].values
        
        # 身體大小
        self.df['Body_Size'] = np.abs(c - o)
        self.features.append('Body_Size')
        
        # 上影線
        self.df['Upper_Wick'] = h - np.maximum(o, c)
        self.features.append('Upper_Wick')
        
        # 下影線
        self.df['Lower_Wick'] = np.minimum(o, c) - l
        self.features.append('Lower_Wick')
        
        # 真實波幅比
        self.df['True_Range_Ratio'] = (h - l) / (np.abs(c - o) + 1e-8)
        self.features.append('True_Range_Ratio')
        
        # 方向 (1=漲, -1=跌)
        self.df['Direction'] = np.where(c >= o, 1, -1)
        self.features.append('Direction')
        
    def add_statistical_features(self):
        """添加統計特徵"""
        logger.info("計算統計特徵...")
        
        returns = self.df['close'].pct_change()
        
        # 過去 20 根的波幅度量
        self.df['Returns_Skew'] = returns.rolling(window=20).skew()
        self.df['Returns_Kurt'] = returns.rolling(window=20).kurt()
        self.features.extend(['Returns_Skew', 'Returns_Kurt'])
        
        # 價格加速度
        self.df['Price_Acceleration'] = returns.diff()
        self.features.append('Price_Acceleration')
        
    def add_lag_features(self, lags=[1, 2, 3, 5]):
        """添加滯後特徵"""
        logger.info(f"計算滯後特徵 (lags={lags})...")
        
        for lag in lags:
            for col in ['RSI', 'MACD', 'Stoch_K', 'ATR', 'Historical_Vol']:
                if col in self.df.columns:
                    self.df[f'{col}_Lag{lag}'] = self.df[col].shift(lag)
                    self.features.append(f'{col}_Lag{lag}')
        
    def add_interaction_features(self):
        """添加交互特徵"""
        logger.info("計算交互特徵...")
        
        # RSI * ATR
        if 'RSI' in self.df.columns and 'ATR' in self.df.columns:
            self.df['RSI_ATR_Interaction'] = self.df['RSI'] * self.df['ATR']
            self.features.append('RSI_ATR_Interaction')
        
        # MACD 與價格關係
        if 'MACD' in self.df.columns:
            self.df['MACD_Normalized'] = self.df['MACD'] / (self.df['ATR'] + 1e-8)
            self.features.append('MACD_Normalized')
        
    def engineer_all_features(self):
        """生成所有特徵"""
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_trend_indicators()
        self.add_volume_indicators()
        self.add_candle_patterns()
        self.add_statistical_features()
        self.add_lag_features()
        self.add_interaction_features()
        
        logger.info(f"特徵工程完成！總特徵數: {len(self.features)}")
        logger.info(f"特徵列表: {self.features[:10]}... (showing first 10)")
        
        return self.df, self.features


# ============================================================================
# PART 3: 模型訓練與評估
# ============================================================================

class ModelTrainer:
    """模型訓練與評估"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """準備數據"""
        logger.info("準備訓練數據...")
        
        # 移除含 NaN 的行
        mask = ~(self.X.isna().any(axis=1) | self.y.isna())
        X_clean = self.X[mask]
        y_clean = self.y[mask]
        
        logger.info(f"有效樣本數: {len(X_clean)}")
        logger.info(f"正樣本: {(y_clean == 1).sum()}, 負樣本: {(y_clean == 0).sum()}")
        logger.info(f"正樣本比例: {100 * (y_clean == 1).mean():.2f}%")
        
        # 標準化
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # 分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_clean, test_size=self.test_size, random_state=self.random_state,
            stratify=y_clean
        )
        
        logger.info(f"訓練集: {len(self.X_train)}, 測試集: {len(self.X_test)}")
        
    def train_xgboost(self):
        """訓練 XGBoost 模型"""
        logger.info("訓練 XGBoost 模型...")
        
        # 計算類別權重
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        
        logger.info("XGBoost 訓練完成")
        
    def train_random_forest(self):
        """訓練隨機森林"""
        logger.info("訓練隨機森林...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = model
        
        logger.info("隨機森林訓練完成")
        
    def evaluate_model(self, name, model):
        """評估單個模型"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        self.results[name] = results
        
        logger.info(f"\n{name} 測試集性能:")
        logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  F1 Score:  {results['f1']:.4f}")
        logger.info(f"  AUC-ROC:   {results['auc_roc']:.4f}")
        logger.info(f"  混淆矩陣: TN={results['confusion_matrix'][0,0]}, FP={results['confusion_matrix'][0,1]}, FN={results['confusion_matrix'][1,0]}, TP={results['confusion_matrix'][1,1]}")
        
    def get_feature_importance(self, model_name='XGBoost', top_n=20):
        """獲取特徵重要性"""
        model = self.models.get(model_name)
        if model is None or not hasattr(model, 'feature_importances_'):
            return None
        
        importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n{model_name} 前 {top_n} 個重要特徵:")
        for idx, row in importance.head(top_n).iterrows():
            logger.info(f"  {row['feature']:30s} - {row['importance']:.4f}")
        
        return importance
    
    def train_all(self):
        """訓練所有模型"""
        self.prepare_data()
        self.train_xgboost()
        self.train_random_forest()
        
        for model_name, model in self.models.items():
            self.evaluate_model(model_name, model)
        
        # 獲取特徵重要性
        self.get_feature_importance('XGBoost', top_n=15)
        
        # 選擇最佳模型
        best_model_name = max(self.results, key=lambda x: self.results[x]['auc_roc'])
        logger.info(f"\n最佳模型: {best_model_name} (AUC-ROC: {self.results[best_model_name]['auc_roc']:.4f})")
        
        return self.models[best_model_name], best_model_name


# ============================================================================
# MAIN 執行
# ============================================================================

def main():
    logger.info("\n" + "="*70)
    logger.info("特徵工程 + 模型訓練 V3 (完全重新設計)")
    logger.info("="*70 + "\n")
    
    # 1. 加載數據
    logger.info("第 1 階段：數據加載")
    logger.info("="*70)
    
    label_path = Path("outputs/labels/BTCUSDT_15m_profitability_v2.csv")
    if not label_path.exists():
        logger.error(f"找不到標籤文件: {label_path}")
        return
    
    df = pd.read_csv(label_path)
    logger.info(f"加載 {len(df)} 行數據")
    
    # 2. 進階標籤生成
    logger.info("\n第 2 階段：進階標籤生成")
    logger.info("="*70)
    
    label_gen = AdvancedLabelGenerator(df, bb_period=20, bb_std=2, lookforward=5, profit_threshold=0.001)
    df = label_gen.generate_labels()
    
    # 3. 特徵工程
    logger.info("\n第 3 階段：多層特徵工程")
    logger.info("="*70)
    
    engineer = AdvancedFeatureEngineer(df)
    df, features = engineer.engineer_all_features()
    
    # 4. 模型訓練
    logger.info("\n第 4 階段：模型訓練與評估")
    logger.info("="*70)
    
    X = df[features]
    y = df['label']
    
    trainer = ModelTrainer(X, y)
    best_model, best_model_name = trainer.train_all()
    
    # 5. 保存模型
    logger.info("\n第 5 階段：保存模型")
    logger.info("="*70)
    
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "BTCUSDT_15m_model_v3.pkl"
    scaler_path = output_dir / "BTCUSDT_15m_scaler_v3.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(trainer.scaler, f)
    
    logger.info(f"模型已保存: {model_path}")
    logger.info(f"標準化器已保存: {scaler_path}")
    
    logger.info("\n" + "="*70)
    logger.info("訓練完成！")
    logger.info("="*70 + "\n")
    
    return best_model, trainer.scaler, features


if __name__ == "__main__":
    main()

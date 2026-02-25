import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
import json
import time
import warnings
import sqlite3
import pickle
import os
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.optimize import minimize_scalar
import hashlib
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import talib
except ImportError:
    logger.error("TA-Lib not installed. Install with: pip install TA-Lib")
    raise

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
except ImportError:
    logger.error("python-telegram-bot not installed. Install with: pip install python-telegram-bot")
    raise

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.feature_selection import mutual_info_classif, SelectKBest
    from sklearn.cluster import KMeans
except ImportError:
    logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
    raise

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Input, Attention, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    tf.get_logger().setLevel('ERROR')
except ImportError:
    logger.warning("TensorFlow not installed. ML features will be limited.")
    tf = None

TWELVE_DATA_API_KEY = "ce0dbe1303af4be6b0cbe593744c01bd"
TELEGRAM_TOKEN = "8463088511:AAFU-8PL31RBVBrRPC3Dr5YiE0CMUGP02Ac"

DB_NAME = "xauusd_ai.db"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TIMEFRAMES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
}

@dataclass
class AdaptiveConfig:
    min_confidence: float = 60.0
    min_adx: float = 25.0
    risk_per_trade: float = 0.02
    atr_multiplier: float = 1.2
    success_threshold: float = 0.65
    retrain_interval: int = 1
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class MetaLearningEngine:
    def __init__(self):
        self.strategy_weights = {
            'smc': 0.25,
            'technical': 0.25,
            'ml_ensemble': 0.30,
            'pattern_memory': 0.20
        }
        self.performance_history = defaultdict(list)
        self.market_regime_stats = {}
        self.feature_importance_history = []
        self.config = AdaptiveConfig()
        self.learning_rate = 0.1
        self.exploration_rate = 0.15
        
    def update_strategy_weights(self, trade_results: List[Dict]):
        if len(trade_results) < 10:
            return
        
        strategy_performance = {}
        for strategy in self.strategy_weights.keys():
            relevant_trades = [
                t for t in trade_results 
                if strategy in t.get('factors', []) or self._strategy_contributed(t, strategy)
            ]
            if relevant_trades:
                wins = sum(1 for t in relevant_trades if t['actual_result'] == 'SUCCESS')
                strategy_performance[strategy] = wins / len(relevant_trades)
            else:
                strategy_performance[strategy] = 0.5
        
        temperature = 0.5
        exp_scores = {
            k: np.exp(v / temperature) 
            for k, v in strategy_performance.items()
        }
        total = sum(exp_scores.values())
        
        new_weights = {k: exp_scores[k] / total for k in exp_scores}
        self.strategy_weights = {
            k: 0.7 * self.strategy_weights[k] + 0.3 * new_weights[k]
            for k in self.strategy_weights
        }
        
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v / total for k, v in self.strategy_weights.items()}
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def _strategy_contributed(self, trade: Dict, strategy: str) -> bool:
        factors = trade.get('factors', [])
        strategy_keywords = {
            'smc': ['OB', 'FVG', 'liquidity', 'order block'],
            'technical': ['RSI', 'MACD', 'EMA', 'ADX'],
            'ml_ensemble': ['DL ensemble', 'LSTM', 'Random Forest'],
            'pattern_memory': ['Historical pattern', 'success rate']
        }
        return any(keyword in str(factors) for keyword in strategy_keywords.get(strategy, []))
    
    def adapt_to_market_regime(self, current_regime: str, trade_history: List[Dict]):
        regime_trades = [
            t for t in trade_history 
            if t.get('market_regime') == current_regime
        ]
        
        if len(regime_trades) < 20:
            return
        
        wins = [t for t in regime_trades if t['actual_result'] == 'SUCCESS']
        win_rate = len(wins) / len(regime_trades)
        
        if current_regime == 'trending':
            if win_rate > 0.6:
                self.config.min_adx = max(20.0, self.config.min_adx * 0.95)
                self.config.min_confidence = max(55.0, self.config.min_confidence * 0.97)
            else:
                self.config.min_adx = min(30.0, self.config.min_adx * 1.05)
                self.config.min_confidence = min(70.0, self.config.min_confidence * 1.03)
        elif current_regime == 'ranging':
            if win_rate > 0.6:
                self.config.min_adx = min(35.0, self.config.min_adx * 1.05)
            else:
                self.config.min_adx = max(15.0, self.config.min_adx * 0.95)
        
        if win_rate > 0.5:
            avg_win = np.mean([t['profit_loss'] for t in wins if t['profit_loss']])
            losses = [t for t in regime_trades if t['actual_result'] != 'SUCCESS']
            avg_loss = np.mean([abs(t['profit_loss']) for t in losses if t['profit_loss']]) or 1
            
            kelly_f = (win_rate - ((1 - win_rate) / (avg_win / avg_loss)))
            self.config.risk_per_trade = np.clip(kelly_f * 0.5, 0.01, 0.05)
        
        self.market_regime_stats[current_regime] = {
            'win_rate': win_rate,
            'optimal_adx': self.config.min_adx,
            'optimal_confidence': self.config.min_confidence,
            'kelly_fraction': self.config.risk_per_trade
        }
        
        logger.info(f"Adapted to {current_regime}: ADX={self.config.min_adx:.1f}, "
                   f"Confidence={self.config.min_confidence:.1f}, Risk={self.config.risk_per_trade:.2%}")
    
    def discover_new_features(self, df: pd.DataFrame, trade_history: List[Dict]) -> List[str]:
        if len(trade_history) < 50:
            return []
        
        features = []
        labels = []
        
        for trade in trade_history:
            if 'market_state' not in trade:
                continue
            
            ms = json.loads(trade['market_state']) if isinstance(trade['market_state'], str) else trade['market_state']
            feature_vec = [
                ms.get('adx', 25) / 100,
                ms.get('atr_percent', 1) / 10,
                ms.get('rsi', 50) / 100,
                ms.get('trend_strength', 0.5),
                ms.get('volatility', 0.5),
                trade.get('hour', 12) / 24,
                trade.get('day_of_week', 0) / 6,
                trade.get('confidence', 60) / 100
            ]
            features.append(feature_vec)
            labels.append(1 if trade['actual_result'] == 'SUCCESS' else 0)
        
        if len(features) < 30:
            return []
        
        X = np.array(features)
        y = np.array(labels)
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        feature_names = ['adx', 'atr', 'rsi', 'trend', 'volatility', 'hour', 'dow', 'confidence']
        
        new_features = []
        for i, score in enumerate(mi_scores):
            if score > 0.1:
                new_features.append({
                    'name': feature_names[i],
                    'importance': score,
                    'correlation': np.corrcoef(X[:, i], y)[0, 1]
                })
        
        self.feature_importance_history.append({
            'timestamp': datetime.now().isoformat(),
            'features': new_features
        })
        
        return [f['name'] for f in sorted(new_features, key=lambda x: x['importance'], reverse=True)]
    
    def generate_trade_signature(self, signal) -> str:
        key_data = {
            'direction': signal.direction,
            'confluence': signal.confluence_count,
            'regime': signal.market_state.get('regime', 'unknown'),
            'session': signal.market_state.get('session', 'unknown'),
            'hour': signal.timestamp.hour if signal.timestamp else 12
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def get_similar_trade_performance(self, signature: str, trade_history: List[Dict]) -> Optional[Dict]:
        similar = [
            t for t in trade_history 
            if self._calculate_similarity(t, signature) > 0.8
        ]
        
        if len(similar) < 5:
            return None
        
        wins = sum(1 for t in similar if t['actual_result'] == 'SUCCESS')
        return {
            'sample_size': len(similar),
            'win_rate': wins / len(similar),
            'avg_profit': np.mean([t['profit_loss'] for t in similar if t['profit_loss']]),
            'confidence_interval': stats.t.interval(
                0.95, len(similar)-1, 
                loc=wins/len(similar), 
                scale=stats.sem([1 if t['actual_result'] == 'SUCCESS' else 0 for t in similar])
            )
        }
    
    def _calculate_similarity(self, trade: Dict, signature: str) -> float:
        trade_sig = self.generate_trade_signature_from_dict(trade)
        return 1.0 if trade_sig == signature else 0.0
    
    def generate_trade_signature_from_dict(self, trade: Dict) -> str:
        key_data = {
            'direction': trade.get('direction'),
            'confluence': trade.get('confluence_count'),
            'regime': trade.get('market_regime'),
            'session': trade.get('session'),
            'hour': trade.get('hour')
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def save_state(self):
        state = {
            'strategy_weights': self.strategy_weights,
            'config': self.config.to_dict(),
            'market_regime_stats': self.market_regime_stats,
            'feature_history': self.feature_importance_history[-100:]
        }
        with open(os.path.join(MODEL_DIR, 'meta_learning_state.pkl'), 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        path = os.path.join(MODEL_DIR, 'meta_learning_state.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)
                self.strategy_weights = state.get('strategy_weights', self.strategy_weights)
                self.config = AdaptiveConfig.from_dict(state.get('config', self.config.to_dict()))
                self.market_regime_stats = state.get('market_regime_stats', {})
                self.feature_importance_history = state.get('feature_history', [])


class SignalResult:
    def __init__(self, direction=None, entry=0.0, stop_loss=0.0, 
                 take_profit_1=0.0, take_profit_2=0.0, confidence=0.0,
                 confluence_count=0, factors=None, timeframe_alignment=None,
                 status="NO_SETUP", timestamp=None, invalid_reason=None,
                 market_state=None, model_predictions=None, strategy_scores=None):
        self.direction = direction
        self.entry = entry
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.confidence = confidence
        self.confluence_count = confluence_count
        self.factors = factors or []
        self.timeframe_alignment = timeframe_alignment or {}
        self.status = status
        self.timestamp = timestamp or datetime.now()
        self.invalid_reason = invalid_reason
        self.market_state = market_state or {}
        self.model_predictions = model_predictions or {}
        self.strategy_scores = strategy_scores or {}
        self.trade_id = None
        self.trade_signature = None
        self.actual_result = None
        self.profit_loss = None
        self.exit_price = None
        self.exit_time = None


class TradeDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()
        self.trade_cache = []
        self.success_patterns = {}
        self.failure_patterns = {}
        self.load_cached_trades()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            direction TEXT,
            entry REAL,
            stop_loss REAL,
            take_profit_1 REAL,
            take_profit_2 REAL,
            confidence REAL,
            confluence_count INTEGER,
            factors TEXT,
            timeframe_alignment TEXT,
            market_state TEXT,
            model_predictions TEXT,
            strategy_scores TEXT,
            trade_signature TEXT,
            status TEXT,
            actual_result TEXT,
            profit_loss REAL,
            exit_price REAL,
            exit_time TEXT,
            market_regime TEXT,
            adx_value REAL,
            atr_percent REAL,
            rsi_value REAL,
            macd_value REAL,
            ema_alignment TEXT,
            session TEXT,
            day_of_week INTEGER,
            hour INTEGER
        )''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS meta_learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            strategy_weights TEXT,
            config TEXT,
            market_regime TEXT,
            adaptation_trigger TEXT
        )''')
        
        c.execute('''CREATE INDEX IF NOT EXISTS idx_trades_signature ON trades(trade_signature)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(actual_result)''')
        
        conn.commit()
        conn.close()
    
    def load_cached_trades(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM trades WHERE actual_result IS NOT NULL ORDER BY timestamp DESC LIMIT 2000")
        rows = c.fetchall()
        conn.close()
        
        for row in rows:
            trade_data = self.row_to_dict(row)
            self.trade_cache.append(trade_data)
            self.analyze_pattern(trade_data)
        
        logger.info(f"Loaded {len(self.trade_cache)} historical trades")
    
    def row_to_dict(self, row):
        columns = ['id', 'timestamp', 'direction', 'entry', 'stop_loss', 'take_profit_1',
                   'take_profit_2', 'confidence', 'confluence_count', 'factors',
                   'timeframe_alignment', 'market_state', 'model_predictions', 'strategy_scores',
                   'trade_signature', 'status', 'actual_result', 'profit_loss', 'exit_price', 
                   'exit_time', 'market_regime', 'adx_value', 'atr_percent', 'rsi_value', 
                   'macd_value', 'ema_alignment', 'session', 'day_of_week', 'hour']
        return {col: val for col, val in zip(columns, row)}
    
    def analyze_pattern(self, trade_data):
        if not trade_data['actual_result']:
            return
        
        pattern_key = json.dumps({
            'session': trade_data['session'],
            'hour': trade_data['hour'],
            'market_regime': trade_data['market_regime'],
            'ema_alignment': trade_data['ema_alignment']
        }, sort_keys=True)
        
        if trade_data['actual_result'] == 'SUCCESS':
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = {'count': 0, 'avg_profit': 0, 'trades': []}
            self.success_patterns[pattern_key]['count'] += 1
            self.success_patterns[pattern_key]['avg_profit'] = (
                (self.success_patterns[pattern_key]['avg_profit'] * 
                 (self.success_patterns[pattern_key]['count'] - 1) +
                 (trade_data['profit_loss'] or 0)) / 
                self.success_patterns[pattern_key]['count']
            )
            self.success_patterns[pattern_key]['trades'].append(trade_data.get('trade_signature'))
        else:
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = {'count': 0, 'avg_loss': 0}
            self.failure_patterns[pattern_key]['count'] += 1
    
    def save_trade(self, signal_result, market_state):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        dt = signal_result.timestamp
        signature = signal_result.trade_signature or "unknown"
        
        c.execute('''INSERT INTO trades (
            timestamp, direction, entry, stop_loss, take_profit_1, take_profit_2,
            confidence, confluence_count, factors, timeframe_alignment, market_state,
            model_predictions, strategy_scores, trade_signature, status, market_regime, 
            adx_value, atr_percent, rsi_value, macd_value, ema_alignment, session, 
            day_of_week, hour
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            dt.isoformat(),
            signal_result.direction,
            signal_result.entry,
            signal_result.stop_loss,
            signal_result.take_profit_1,
            signal_result.take_profit_2,
            signal_result.confidence,
            signal_result.confluence_count,
            json.dumps(signal_result.factors),
            json.dumps(signal_result.timeframe_alignment),
            json.dumps(signal_result.market_state),
            json.dumps(signal_result.model_predictions),
            json.dumps(signal_result.strategy_scores),
            signature,
            signal_result.status,
            market_state.get('regime', 'unknown'),
            market_state.get('adx', 0),
            market_state.get('atr_percent', 0),
            market_state.get('rsi', 50),
            market_state.get('macd', 0),
            market_state.get('ema_alignment', 'neutral'),
            market_state.get('session', 'unknown'),
            dt.weekday(),
            dt.hour
        ))
        
        trade_id = c.lastrowid
        conn.commit()
        conn.close()
        
        signal_result.trade_id = trade_id
        return trade_id
    
    def update_trade_result(self, trade_id, result, profit_loss, exit_price):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''UPDATE trades SET 
            actual_result = ?, profit_loss = ?, exit_price = ?, exit_time = ?
            WHERE id = ?''', (
            result, profit_loss, exit_price, datetime.now().isoformat(), trade_id
        ))
        
        conn.commit()
        conn.close()
        
        self.load_cached_trades()
    
    def log_meta_adaptation(self, meta_engine, trigger: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO meta_learning_log 
                     (timestamp, strategy_weights, config, market_regime, adaptation_trigger)
                     VALUES (?, ?, ?, ?, ?)''', (
            datetime.now().isoformat(),
            json.dumps(meta_engine.strategy_weights),
            json.dumps(meta_engine.config.to_dict()),
            json.dumps(list(meta_engine.market_regime_stats.keys())),
            trigger
        ))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self, limit=5000):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT * FROM trades 
                     WHERE actual_result IS NOT NULL 
                     ORDER BY timestamp DESC LIMIT ?''', (limit,))
        rows = c.fetchall()
        conn.close()
        
        return [self.row_to_dict(row) for row in rows]
    
    def get_trades_by_signature(self, signature: str, limit=100):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT * FROM trades 
                     WHERE trade_signature = ? AND actual_result IS NOT NULL
                     ORDER BY timestamp DESC LIMIT ?''', (signature, limit))
        rows = c.fetchall()
        conn.close()
        return [self.row_to_dict(row) for row in rows]


class DeepLearningModel:
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.meta_model = None
        self.scaler = RobustScaler()
        self.sequence_length = 60
        self.feature_dim = 25
        self.autoencoder = None
        self.load_models()
    
    def load_models(self):
        if tf and os.path.exists(os.path.join(MODEL_DIR, "lstm_model.h5")):
            try:
                self.lstm_model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
                logger.info("Loaded existing LSTM model")
            except Exception as e:
                logger.error(f"Error loading LSTM: {e}")
                self.build_lstm_model()
        elif tf:
            self.build_lstm_model()
        
        if os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")):
            try:
                with open(os.path.join(MODEL_DIR, "rf_model.pkl"), 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info("Loaded existing RF model")
            except Exception as e:
                logger.error(f"Error loading RF: {e}")
                self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        else:
            self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        
        if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
            try:
                with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading scaler: {e}")
        
        if os.path.exists(os.path.join(MODEL_DIR, "meta_model.pkl")):
            try:
                with open(os.path.join(MODEL_DIR, "meta_model.pkl"), 'rb') as f:
                    self.meta_model = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading meta model: {e}")
                self.meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            self.meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    def build_lstm_model(self):
        if not tf:
            return
        
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        attention = Attention()([x, x])
        x = Concatenate()([x, attention])
        
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        self.lstm_model = Model(inputs=inputs, outputs=outputs)
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        logger.info("Built new Attention-LSTM model")
    
    def prepare_sequence_features(self, df):
        features = []
        
        for col in ['close', 'high', 'low', 'open', 'volume',
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'ema_9', 'ema_21', 'ema_50', 'ema_200',
                    'atr', 'atr_percent', 'adx', 'bb_position',
                    'obv', 'obv_slope', 'returns', 'volume_ratio',
                    'ema_slope', 'session_score']:
            if col in df.columns:
                features.append(df[col].fillna(0).values[-self.sequence_length:])
            else:
                features.append(np.zeros(self.sequence_length))
        
        features = np.column_stack(features)
        return self.scaler.fit_transform(features)
    
    def predict(self, df, market_state=None):
        if len(df) < self.sequence_length:
            return {
                'lstm_prob': 0.5,
                'rf_prob': 0.5,
                'meta_prob': 0.5,
                'ensemble': 0.5,
                'confidence': 0.0,
                'uncertainty': 1.0
            }
        
        try:
            X_seq = self.prepare_sequence_features(df)
            X_seq = X_seq.reshape(1, self.sequence_length, self.feature_dim)
            
            lstm_preds = []
            if self.lstm_model and tf:
                for _ in range(10):
                    pred = self.lstm_model(X_seq, training=True)
                    lstm_preds.append(float(pred[0][0]))
                lstm_pred = np.mean(lstm_preds)
                lstm_uncertainty = np.std(lstm_preds)
            else:
                lstm_pred = 0.5
                lstm_uncertainty = 0.5
            
            X_rf = self.extract_rf_features(df)
            rf_pred = 0.5
            rf_uncertainty = 0.5
            if self.rf_model:
                rf_proba = self.rf_model.predict_proba(X_rf)[0]
                rf_pred = rf_proba[1] if len(rf_proba) > 1 else 0.5
                rf_uncertainty = 1 - max(rf_proba)
            
            meta_features = np.array([[lstm_pred, rf_pred, 
                                       market_state.get('adx', 25) / 100,
                                       market_state.get('trend_strength', 0.5),
                                       market_state.get('volatility', 0.5)]])
            
            meta_pred = 0.5
            if self.meta_model:
                meta_proba = self.meta_model.predict_proba(meta_features)[0]
                meta_pred = meta_proba[1] if len(meta_proba) > 1 else 0.5
            
            total_uncertainty = lstm_uncertainty + rf_uncertainty + 0.1
            weights = [
                (1 - lstm_uncertainty) / total_uncertainty,
                (1 - rf_uncertainty) / total_uncertainty,
                0.1 / total_uncertainty
            ]
            weights = np.array(weights) / sum(weights)
            
            ensemble = weights[0] * lstm_pred + weights[1] * rf_pred + weights[2] * meta_pred
            
            agreement = 1 - np.std([lstm_pred, rf_pred, meta_pred])
            confidence = agreement * (1 - np.mean([lstm_uncertainty, rf_uncertainty]))
            
            return {
                'lstm_prob': lstm_pred,
                'rf_prob': rf_pred,
                'meta_prob': meta_pred,
                'ensemble': ensemble,
                'confidence': confidence,
                'uncertainty': np.mean([lstm_uncertainty, rf_uncertainty]),
                'model_weights': weights.tolist()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'lstm_prob': 0.5,
                'rf_prob': 0.5,
                'meta_prob': 0.5,
                'ensemble': 0.5,
                'confidence': 0.0,
                'uncertainty': 1.0
            }
    
    def extract_rf_features(self, df):
        features = []
        row = df.iloc[-1]
        
        feature_list = [
            row.get('rsi', 50) / 100,
            row.get('macd', 0) / 100,
            row.get('adx', 25) / 100,
            row.get('atr_percent', 1) / 10,
            row.get('bb_position', 0.5),
            1 if row.get('ema_9', 0) > row.get('ema_21', 0) else 0,
            1 if row.get('close', 0) > row.get('vwap', 0) else 0,
            row.get('volume_ratio', 1.0),
            row.get('obv_slope', 0),
            row.get('session_score', 0.5),
            row.get('returns', 0) * 100,
            (row.get('high', 0) - row.get('low', 0)) / row.get('close', 1),
            row.get('close', 0) / row.get('ema_200', 1) - 1,
            int(row.get('hour', 12) >= 8 and row.get('hour', 12) <= 17)
        ]
        
        return np.array([feature_list])
    
    def retrain(self, trade_history, meta_engine=None):
        if len(trade_history) < 10:
            logger.info(f"Insufficient trades for retrain: {len(trade_history)}")
            return False
        
        logger.info(f"Retraining models with {len(trade_history)} trades...")
        
        if meta_engine:
            self.sequence_length = min(60, max(30, len(trade_history) // 10))
        
        success = self._retrain_rf(trade_history)
        success = self._retrain_meta(trade_history) and success
        
        if tf:
            success = self._retrain_lstm(trade_history) and success
        
        return success
    
    def _retrain_rf(self, trade_history):
        try:
            X = []
            y = []
            
            for trade in trade_history:
                features = [
                    trade.get('confidence', 50) / 100,
                    trade.get('confluence_count', 0) / 10,
                    trade.get('adx_value', 25) / 100,
                    trade.get('atr_percent', 1) / 10,
                    trade.get('rsi_value', 50) / 100,
                    1 if trade.get('session') in ['london', 'ny_am'] else 0,
                    trade.get('hour', 12) / 24,
                    trade.get('day_of_week', 0) / 6
                ]
                X.append(features)
                y.append(1 if trade['actual_result'] == 'SUCCESS' else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            tscv = TimeSeriesSplit(n_splits=3)
            best_score = 0
            best_model = None
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced'
                )
                
                model.fit(X_train, y_train)
                score = f1_score(y_val, model.predict(X_val))
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            self.rf_model = best_model
            
            with open(os.path.join(MODEL_DIR, "rf_model.pkl"), 'wb') as f:
                pickle.dump(self.rf_model, f)
            
            logger.info(f"RF model retrained. Best F1: {best_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"RF retrain error: {e}")
            return False
    
    def _retrain_meta(self, trade_history):
        try:
            X = []
            y = []
            
            for trade in trade_history:
                features = [
                    trade.get('confidence', 50) / 100,
                    trade.get('confluence_count', 0) / 10,
                    trade.get('adx_value', 25) / 100,
                    trade.get('atr_percent', 1) / 10,
                    trade.get('rsi_value', 50) / 100
                ]
                X.append(features)
                y.append(1 if trade['actual_result'] == 'SUCCESS' else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            self.meta_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            self.meta_model.fit(X, y)
            
            with open(os.path.join(MODEL_DIR, "meta_model.pkl"), 'wb') as f:
                pickle.dump(self.meta_model, f)
            
            logger.info("Meta model retrained")
            return True
            
        except Exception as e:
            logger.error(f"Meta retrain error: {e}")
            return False
    
    def _retrain_lstm(self, trade_history):
        if not tf:
            return True
        
        try:
            sequences = []
            labels = []
            
            for trade in trade_history:
                if 'market_state' not in trade or not trade['market_state']:
                    continue
                
                market_state = json.loads(trade['market_state']) if isinstance(trade['market_state'], str) else trade['market_state']
                
                seq = np.array(market_state.get('sequence', []))
                if len(seq) >= self.sequence_length:
                    sequences.append(seq[-self.sequence_length:])
                    labels.append(1 if trade['actual_result'] == 'SUCCESS' else 0)
            
            if len(sequences) < 10:
                logger.info(f"Insufficient sequences for LSTM: {len(sequences)}")
                return True
            
            X = np.array(sequences)
            y = np.array(labels)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.build_lstm_model()
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            checkpoint = ModelCheckpoint(
                os.path.join(MODEL_DIR, "lstm_best.h5"),
                monitor='val_accuracy',
                save_best_only=True
            )
            
            self.lstm_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stop, lr_reduce, checkpoint],
                verbose=0
            )
            
            loss, accuracy, precision, recall = self.lstm_model.evaluate(X_test, y_test, verbose=0)
            
            save_model(self.lstm_model, os.path.join(MODEL_DIR, "lstm_model.h5"))
            
            logger.info(f"LSTM retrained. Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"LSTM retrain error: {e}")
            return False


class TwelveDataClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = None
        self.rate_limit_remaining = 100
        self.last_request_time = 0

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _fetch_with_retry(self, url, params, max_retries=3):
        min_interval = 60 / 8
        time_since_last = time.time() - self.last_request_time
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, params=params, ssl=False) as response:
                    self.last_request_time = time.time()
                    
                    if response.status == 200:
                        data = await response.json()
                        self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
                        return data
                    elif response.status == 429:
                        wait_time = int(response.headers.get('Retry-After', (attempt + 1) * 5))
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        text = await response.text()
                        logger.error(f"HTTP {response.status}: {text[:200]}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
        return None

    async def get_ohlcv(self, symbol, interval, outputsize=500):
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "order": "asc",
            "timezone": "UTC"
        }

        data = await self._fetch_with_retry(url, params)

        if not data or "values" not in data:
            logger.error(f"No data for {symbol} {interval}")
            return None

        try:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0)

            df = df.dropna(subset=["open", "high", "low", "close"])

            if len(df) < 50:
                logger.warning(f"Insufficient data: {len(df)} rows")
                return None

            return df

        except Exception as e:
            logger.error(f"Data parsing error: {e}")
            return None

    async def get_quote(self, symbol):
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "apikey": self.api_key}
        return await self._fetch_with_retry(url, params)


class SMCAnalyzer:
    def __init__(self):
        self.order_blocks = []
        self.liquidity_zones = []
        self.fvgs = []
        self.fair_value_gaps = []

    def detect_swing_points(self, df, lookback=5):
        if len(df) < lookback * 2 + 1:
            return [], []

        highs = df["high"].values
        lows = df["low"].values

        swing_highs = []
        swing_lows = []

        for i in range(lookback, len(df) - lookback):
            try:
                is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, lookback+1))
                is_swing_high = is_swing_high and all(highs[i] >= highs[i+j] for j in range(1, lookback+1))

                is_swing_low = all(lows[i] <= lows[i-j] for j in range(1, lookback+1))
                is_swing_low = is_swing_low and all(lows[i] <= lows[i+j] for j in range(1, lookback+1))

                if is_swing_high:
                    swing_highs.append(i)
                if is_swing_low:
                    swing_lows.append(i)
            except:
                continue

        return swing_highs, swing_lows

    def identify_order_blocks(self, df):
        self.order_blocks = []

        if len(df) < 3:
            return

        for i in range(2, min(len(df), 100)):
            try:
                current = df.iloc[i]
                prev = df.iloc[i-1]
                prev2 = df.iloc[i-2]

                body_size = abs(current["close"] - current["open"])
                range_size = current["high"] - current["low"]

                if range_size == 0 or body_size < range_size * 0.2:
                    continue

                momentum = abs(prev["close"] - prev2["open"])

                ob = None
                if current["close"] > current["open"] and momentum > body_size * 0.3:
                    strength = min((body_size / range_size) * 0.5 + 0.3, 1.0) if range_size > 0 else 0.3
                    ob = {
                        "high": float(current["high"]),
                        "low": float(current["low"]),
                        "type": "bullish",
                        "strength": strength,
                        "valid": True,
                        "fresh": True,
                        "tested": 0,
                        "index": i
                    }
                elif current["close"] < current["open"] and momentum > body_size * 0.3:
                    strength = min((body_size / range_size) * 0.5 + 0.3, 1.0) if range_size > 0 else 0.3
                    ob = {
                        "high": float(current["high"]),
                        "low": float(current["low"]),
                        "type": "bearish",
                        "strength": strength,
                        "valid": True,
                        "fresh": True,
                        "tested": 0,
                        "index": i
                    }

                if ob:
                    self.order_blocks.append(ob)

            except Exception as e:
                continue

    def update_ob_validity(self, current_price):
        for ob in self.order_blocks:
            if ob["type"] == "bullish":
                if current_price < ob["low"]:
                    ob["valid"] = False
                elif current_price > ob["high"]:
                    ob["tested"] += 1
            else:
                if current_price > ob["high"]:
                    ob["valid"] = False
                elif current_price < ob["low"]:
                    ob["tested"] += 1

            if ob["tested"] >= 3:
                ob["fresh"] = False

    def detect_fvg(self, df):
        self.fvgs = []

        if len(df) < 3:
            return

        for i in range(2, min(len(df), 50)):
            try:
                c1 = df.iloc[i-2]
                c2 = df.iloc[i-1]

                if c2["low"] > c1["high"]:
                    self.fvgs.append({
                        "high": float(c2["low"]),
                        "low": float(c1["high"]),
                        "bullish": True,
                        "filled": False,
                        "index": i
                    })
                elif c2["high"] < c1["low"]:
                    self.fvgs.append({
                        "high": float(c1["low"]),
                        "low": float(c2["high"]),
                        "bullish": False,
                        "filled": False,
                        "index": i
                    })
            except:
                continue

    def detect_liquidity(self, df, swing_highs, swing_lows):
        self.liquidity_zones = []

        if not swing_highs or not swing_lows or len(df) < 10:
            return

        try:
            recent_highs = [df.iloc[i]["high"] for i in swing_highs[-10:] if i < len(df)]
            recent_lows = [df.iloc[i]["low"] for i in swing_lows[-10:] if i < len(df)]

            for high in recent_highs[-5:]:
                cluster = sum(1 for h in recent_highs if abs(h - high) < high * 0.001)
                if cluster >= 2:
                    self.liquidity_zones.append({
                        "price": float(high),
                        "type": "equal_highs",
                        "strength": cluster
                    })

            for low in recent_lows[-5:]:
                cluster = sum(1 for l in recent_lows if abs(l - low) < low * 0.001)
                if cluster >= 2:
                    self.liquidity_zones.append({
                        "price": float(low),
                        "type": "equal_lows",
                        "strength": cluster
                    })
        except:
            pass


class TechnicalAnalyzer:
    def calculate_indicators(self, df):
        try:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            volume = df.get("volume", pd.Series(np.ones(len(df)))).values

            df["ema_9"] = talib.EMA(close, timeperiod=9)
            df["ema_21"] = talib.EMA(close, timeperiod=21)
            df["ema_50"] = talib.EMA(close, timeperiod=50)
            df["ema_200"] = talib.EMA(close, timeperiod=200)

            df["ema_slope"] = df["ema_21"].diff(5) / 5

            df["rsi"] = talib.RSI(close, timeperiod=14)

            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist

            df["atr"] = talib.ATR(high, low, close, timeperiod=14)
            df["atr_percent"] = (df["atr"] / df["close"]) * 100

            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)
            df["bb_upper"] = upper
            df["bb_lower"] = lower
            df["bb_position"] = (df["close"] - lower) / (upper - lower)

            df["adx"] = talib.ADX(high, low, close, timeperiod=14)

            df["obv"] = talib.OBV(close, volume)
            df["obv_slope"] = df["obv"].diff(5)

            df["vwap"] = (df["close"] * df.get("volume", 1)).cumsum() / df.get("volume", 1).cumsum()

            df["returns"] = df["close"].pct_change()

            if "volume" in df.columns:
                df["volume_sma"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)

            df["session_score"] = df.index.map(
                lambda x: 1.0 if 12 <= x.hour < 17 else 0.9 if 8 <= x.hour < 12 else 0.6
            )

            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek

            df["price_momentum"] = df["close"].pct_change(3)
            df["volatility_regime"] = df["atr_percent"].rolling(20).mean()
            
            df = df.ffill().bfill().fillna(0)
            return df

        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return df

    def check_trend_alignment(self, data):
        alignment = {}

        for tf, df in data.items():
            if df.empty or "ema_9" not in df.columns or len(df) < 2:
                alignment[tf] = "NEUTRAL"
                continue

            try:
                close = df["close"].iloc[-1]
                ema9 = df["ema_9"].iloc[-1]
                ema21 = df["ema_21"].iloc[-1] if "ema_21" in df.columns else ema9

                if close > ema9 > ema21:
                    alignment[tf] = "BULLISH"
                elif close < ema9 < ema21:
                    alignment[tf] = "BEARISH"
                else:
                    alignment[tf] = "MIXED"
            except:
                alignment[tf] = "NEUTRAL"

        return alignment
    
    def detect_market_regime(self, df_1h):
        if df_1h.empty or "adx" not in df_1h.columns:
            return "unknown"
        
        adx = df_1h["adx"].iloc[-1]
        atr_percent = df_1h["atr_percent"].iloc[-1] if "atr_percent" in df_1h.columns else 1.0
        
        if adx > 30:
            return "strong_trend"
        elif adx > 25:
            return "trending"
        elif atr_percent < 0.5:
            return "low_volatility"
        else:
            return "ranging"


class ScalpingEngine:
    def __init__(self):
        self.smc = SMCAnalyzer()
        self.tech = TechnicalAnalyzer()
        self.db = TradeDatabase(DB_NAME)
        self.dl_model = DeepLearningModel()
        self.meta_engine = MetaLearningEngine()
        self.meta_engine.load_state()
        self.last_retrain = datetime.now()
        self.trade_monitoring = {}
        self.analysis_count = 0
    
    def analyze(self, data, current_price):
        self.analysis_count += 1
        
        df_1m = data.get("1m", pd.DataFrame())
        df_5m = data.get("5m", pd.DataFrame())
        df_15m = data.get("15m", pd.DataFrame())
        df_30m = data.get("30m", pd.DataFrame())
        df_1h = data.get("1h", pd.DataFrame())

        if df_1m.empty or len(df_1m) < 50:
            return SignalResult(
                direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
                confidence=0, confluence_count=0, factors=[], timeframe_alignment={},
                status="NO_SETUP", invalid_reason="Insufficient 1m data"
            )

        market_regime = self.tech.detect_market_regime(df_1h)
        
        if self.analysis_count % 10 == 0:
            trade_history = self.db.get_training_data(limit=500)
            self.meta_engine.adapt_to_market_regime(market_regime, trade_history)
            self.meta_engine.update_strategy_weights(trade_history)

        config = self.meta_engine.config

        swing_highs, swing_lows = self.smc.detect_swing_points(df_5m)
        self.smc.identify_order_blocks(df_5m)
        self.smc.update_ob_validity(current_price)
        self.smc.detect_fvg(df_5m)
        self.smc.detect_liquidity(df_5m, swing_highs, swing_lows)

        alignment = self.tech.check_trend_alignment(data)

        bullish_count = sum(1 for v in alignment.values() if v == "BULLISH")
        bearish_count = sum(1 for v in alignment.values() if v == "BEARISH")

        if bullish_count >= 3 and bearish_count >= 3:
            trend_bias = "CONFLICTED"
        elif bullish_count >= 4:
            trend_bias = "BULLISH"
        elif bearish_count >= 4:
            trend_bias = "BEARISH"
        elif bullish_count >= 3:
            trend_bias = "BULLISH_WEAK"
        elif bearish_count >= 3:
            trend_bias = "BEARISH_WEAK"
        else:
            trend_bias = "NEUTRAL"

        market_state = {
            'regime': market_regime,
            'adx': df_1h["adx"].iloc[-1] if not df_1h.empty and "adx" in df_1h.columns else 25,
            'atr_percent': df_1m["atr_percent"].iloc[-1] if "atr_percent" in df_1m.columns else 1.0,
            'rsi': df_1m["rsi"].iloc[-1] if "rsi" in df_1m.columns else 50,
            'macd': df_1m["macd"].iloc[-1] if "macd" in df_1m.columns else 0,
            'ema_alignment': 'bullish' if bullish_count >= 4 else 'bearish' if bearish_count >= 4 else 'neutral',
            'session': 'london' if 8 <= datetime.now().hour < 12 else 'ny_am' if 12 <= datetime.now().hour < 17 else 'ny_pm' if 17 <= datetime.now().hour < 21 else 'asia',
            'trend_strength': max(bullish_count, bearish_count) / 5.0,
            'volatility': df_1m["atr_percent"].iloc[-1] / 10 if "atr_percent" in df_1m.columns else 0.5,
            'sequence': df_1m[['close', 'rsi', 'macd', 'adx', 'atr_percent']].values.tolist() if len(df_1m) >= 60 else []
        }

        model_preds = self.dl_model.predict(df_1m, market_state)

        strategy_scores = {}
        
        score = 0.0
        factors = []
        direction = None

        adx_1h = df_1h["adx"].iloc[-1] if not df_1h.empty and "adx" in df_1h.columns else 25
        adx_15m = df_15m["adx"].iloc[-1] if not df_15m.empty and "adx" in df_15m.columns else 25

        if adx_1h < config.min_adx or adx_15m < config.min_adx:
            return SignalResult(
                direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
                confidence=0, confluence_count=0, factors=[], timeframe_alignment=alignment,
                status="NO_SETUP", invalid_reason=f"Low ADX (1h:{adx_1h:.1f}, 15m:{adx_15m:.1f})",
                market_state=market_state, model_predictions=model_preds
            )

        atr = df_1m["atr"].iloc[-1] if "atr" in df_1m.columns else current_price * 0.0005

        rsi_1m = df_1m["rsi"].iloc[-1] if "rsi" in df_1m.columns else 50
        rsi_5m = df_5m["rsi"].iloc[-1] if "rsi" in df_5m.columns else 50

        macd_1m = df_1m["macd"].iloc[-1] if "macd" in df_1m.columns else 0
        macd_signal_1m = df_1m["macd_signal"].iloc[-1] if "macd_signal" in df_1m.columns else 0

        pattern_check = self.db.get_training_data(limit=100)
        similar_performance = None
        
        if "BULLISH" in trend_bias:
            smc_score = 0
            tech_score = 0
            ml_score = 0
            pattern_score = 0
            
            if 20 < rsi_5m < 50:
                smc_score += 15
                factors.append("RSI pullback in uptrend")

            if rsi_1m < 35:
                smc_score += 10
                factors.append("1m RSI oversold")

            if macd_1m > macd_signal_1m:
                tech_score += 10
                factors.append("MACD bullish cross")

            for ob in self.smc.order_blocks:
                if ob["type"] == "bullish" and ob["valid"] and ob["fresh"]:
                    if abs(current_price - ob["low"]) < atr * 2:
                        smc_score += 15 + int(ob["strength"] * 10)
                        factors.append(f"Bullish OB @ {ob['low']:.2f}")
                        break

            for fvg in self.smc.fvgs:
                if fvg["bullish"] and not fvg["filled"]:
                    if fvg["low"] <= current_price <= fvg["high"]:
                        smc_score += 10
                        factors.append("Inside bullish FVG")
                        break

            if model_preds['ensemble'] > 0.6:
                ml_score += 15
                factors.append(f"DL ensemble bullish ({model_preds['ensemble']:.2f})")
            elif model_preds['ensemble'] > 0.5:
                ml_score += 8
                factors.append(f"DL weak bullish ({model_preds['ensemble']:.2f})")

            if bullish_count >= 4:
                tech_score += 15
                factors.append("Strong HTF alignment")

            pattern_check = self.meta_engine.get_similar_trade_performance(
                self.meta_engine.generate_trade_signature_from_dict({
                    'direction': 'LONG',
                    'confluence': len(factors),
                    'market_regime': market_regime,
                    'session': market_state['session'],
                    'hour': datetime.now().hour
                }), pattern_check
            )
            
            if pattern_check and pattern_check['win_rate'] > 0.6:
                pattern_score += 10
                factors.append(f"Similar setups: {pattern_check['win_rate']:.1%} WR")

            weights = self.meta_engine.strategy_weights
            score = (
                smc_score * weights['smc'] +
                tech_score * weights['technical'] +
                ml_score * weights['ml_ensemble'] +
                pattern_score * weights['pattern_memory']
            ) * (1 + len(factors) * 0.05)

            strategy_scores = {
                'smc': smc_score,
                'technical': tech_score,
                'ml_ensemble': ml_score,
                'pattern_memory': pattern_score,
                'raw_total': score
            }

            if score >= config.min_confidence:
                direction = "LONG"

        elif "BEARISH" in trend_bias:
            smc_score = 0
            tech_score = 0
            ml_score = 0
            pattern_score = 0
            
            if 50 < rsi_5m < 80:
                smc_score += 15
                factors.append("RSI bounce in downtrend")

            if rsi_1m > 65:
                smc_score += 10
                factors.append("1m RSI overbought")

            if macd_1m < macd_signal_1m:
                tech_score += 10
                factors.append("MACD bearish cross")

            for ob in self.smc.order_blocks:
                if ob["type"] == "bearish" and ob["valid"] and ob["fresh"]:
                    if abs(current_price - ob["high"]) < atr * 2:
                        smc_score += 15 + int(ob["strength"] * 10)
                        factors.append(f"Bearish OB @ {ob['high']:.2f}")
                        break

            for fvg in self.smc.fvgs:
                if not fvg["bullish"] and not fvg["filled"]:
                    if fvg["low"] <= current_price <= fvg["high"]:
                        smc_score += 10
                        factors.append("Inside bearish FVG")
                        break

            if model_preds['ensemble'] < 0.4:
                ml_score += 15
                factors.append(f"DL ensemble bearish ({model_preds['ensemble']:.2f})")
            elif model_preds['ensemble'] < 0.5:
                ml_score += 8
                factors.append(f"DL weak bearish ({model_preds['ensemble']:.2f})")

            if bearish_count >= 4:
                tech_score += 15
                factors.append("Strong HTF alignment")

            weights = self.meta_engine.strategy_weights
            score = (
                smc_score * weights['smc'] +
                tech_score * weights['technical'] +
                ml_score * weights['ml_ensemble'] +
                pattern_score * weights['pattern_memory']
            ) * (1 + len(factors) * 0.05)

            strategy_scores = {
                'smc': smc_score,
                'technical': tech_score,
                'ml_ensemble': ml_score,
                'pattern_memory': pattern_score,
                'raw_total': score
            }

            if score >= config.min_confidence:
                direction = "SHORT"

        confluence = len(factors)

        if direction is None:
            return SignalResult(
                direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
                confidence=min(score, 100), confluence_count=confluence, factors=factors,
                timeframe_alignment=alignment, status="NO_SETUP",
                invalid_reason=f"No setup (score:{score:.1f}, needed:{config.min_confidence:.1f})",
                market_state=market_state, model_predictions=model_preds,
                strategy_scores=strategy_scores
            )

        entry = current_price
        stop_distance = atr * config.atr_multiplier

        if direction == "LONG":
            stop_loss = entry - stop_distance
            take_profit_1 = entry + (stop_distance * 2.0)
            take_profit_2 = entry + (stop_distance * 3.0)
        else:
            stop_loss = entry + stop_distance
            take_profit_1 = entry - (stop_distance * 2.0)
            take_profit_2 = entry - (stop_distance * 3.0)

        signal = SignalResult(
            direction=direction,
            entry=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(take_profit_1, 2),
            take_profit_2=round(take_profit_2, 2),
            confidence=min(score, 100),
            confluence_count=confluence,
            factors=factors,
            timeframe_alignment=alignment,
            status="VALID",
            market_state=market_state,
            model_predictions=model_preds,
            strategy_scores=strategy_scores
        )

        signal.trade_signature = self.meta_engine.generate_trade_signature(signal)
        similar = self.db.get_trades_by_signature(signal.trade_signature)
        if len(similar) > 5:
            wins = sum(1 for t in similar if t['actual_result'] == 'SUCCESS')
            signal.factors.append(f"History: {wins}/{len(similar)} similar wins")

        trade_id = self.db.save_trade(signal, market_state)
        signal.trade_id = trade_id

        self.trade_monitoring[trade_id] = {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'direction': direction,
            'entry_time': datetime.now()
        }

        return signal
    
    async def check_retrain_needed(self):
        hours_since_retrain = (datetime.now() - self.last_retrain).total_seconds() / 3600
        
        if hours_since_retrain >= self.meta_engine.config.retrain_interval:
            logger.info(f"Retrain interval reached: {hours_since_retrain:.1f} hours")
            
            trade_history = self.db.get_training_data(limit=5000)
            
            if len(trade_history) >= 10:
                success = self.dl_model.retrain(trade_history, self.meta_engine)
                
                self.meta_engine.update_strategy_weights(trade_history)
                
                new_features = self.meta_engine.discover_new_features(None, trade_history)
                if new_features:
                    logger.info(f"Discovered important features: {new_features}")
                
                self.db.log_meta_adaptation(self.meta_engine, "scheduled_retrain")
                self.meta_engine.save_state()
                
                if success:
                    self.last_retrain = datetime.now()
                    logger.info("Retraining completed successfully")
                    return True
            else:
                logger.info(f"Insufficient trades for retrain: {len(trade_history)}")
        
        return False
    
    def monitor_open_trades(self, current_price):
        for trade_id, trade_info in list(self.trade_monitoring.items()):
            direction = trade_info['direction']
            entry = trade_info['entry']
            sl = trade_info['stop_loss']
            tp1 = trade_info['take_profit_1']
            tp2 = trade_info['take_profit_2']
            
            result = None
            pnl = 0
            exit_price = current_price
            
            if direction == "LONG":
                if current_price <= sl:
                    result = "STOP_LOSS"
                    pnl = (sl - entry) / entry * 100
                elif current_price >= tp2:
                    result = "SUCCESS"
                    pnl = (tp2 - entry) / entry * 100
                    exit_price = tp2
                elif current_price >= tp1:
                    continue
            else:
                if current_price >= sl:
                    result = "STOP_LOSS"
                    pnl = (entry - sl) / entry * 100
                elif current_price <= tp2:
                    result = "SUCCESS"
                    pnl = (entry - tp2) / entry * 100
                    exit_price = tp2
                elif current_price <= tp1:
                    continue
            
            if result:
                self.db.update_trade_result(trade_id, result, pnl, exit_price)
                del self.trade_monitoring[trade_id]
                logger.info(f"Trade {trade_id} closed: {result}, PnL: {pnl:.2f}%")


class XAUUSDBot:
    def __init__(self):
        self.data_client = TwelveDataClient(TWELVE_DATA_API_KEY)
        self.engine = ScalpingEngine()
        self.last_signal = None
        self.monitoring = False
        self.chat_id = None
        self.retrain_task = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.chat_id = update.effective_chat.id

        keyboard = [
            [InlineKeyboardButton("🔥 Start Signal Stream", callback_data="stream")],
            [InlineKeyboardButton("📊 Single Analysis", callback_data="analyze")],
            [InlineKeyboardButton("⏹️ Stop Stream", callback_data="stop")],
            [InlineKeyboardButton("📈 View Stats", callback_data="stats")],
            [InlineKeyboardButton("🧠 Meta-Learning Status", callback_data="meta")]
        ]

        await update.message.reply_text(
            "🏆 *XAUUSD Self-Learning AI Scalper v2.0*\n\n"
            "🧠 Features:\n"
            "• Adaptive Strategy Weights\n"
            "• Market Regime Detection\n"
            "• Auto Feature Discovery\n"
            "• Kelly Criterion Sizing\n"
            "• Uncertainty Quantification\n"
            "• Pattern Memory\n\n"
            "The AI evolves with every trade!",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()

        if query.data == "stream":
            await self.start_stream(query, context)
        elif query.data == "analyze":
            await self.single_analysis(query)
        elif query.data == "stop":
            await self.stop_stream(query, context)
        elif query.data == "stats":
            await self.show_stats(query)
        elif query.data == "meta":
            await self.show_meta_status(query)

    async def fetch_all_timeframes(self):
        data = {}
        current_price = None

        async with self.data_client as client:
            for tf_name, interval in TIMEFRAMES.items():
                try:
                    df = await client.get_ohlcv("XAU/USD", interval, 500)
                    if df is not None and not df.empty:
                        data[tf_name] = self.engine.tech.calculate_indicators(df)
                        if tf_name == "1m" and current_price is None:
                            current_price = df["close"].iloc[-1]
                    else:
                        logger.error(f"Failed to fetch {tf_name}")
                except Exception as e:
                    logger.error(f"Error fetching {tf_name}: {e}")

                await asyncio.sleep(0.5)

            if current_price is None:
                try:
                    quote = await client.get_quote("XAU/USD")
                    if quote and "price" in quote:
                        current_price = float(quote["price"])
                except Exception as e:
                    logger.error(f"Error fetching quote: {e}")

        return data, current_price

    async def single_analysis(self, query):
        await query.edit_message_text("🔬 Analyzing with Adaptive AI...")

        try:
            data, current_price = await self.fetch_all_timeframes()

            if not data or current_price is None:
                await query.edit_message_text(
                    "❌ Data unavailable. Retrying...",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Retry", callback_data="analyze")]])
                )
                return

            signal = self.engine.analyze(data, current_price)
            self.last_signal = signal

            await self.send_signal_message(query, signal, is_stream=False)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await query.edit_message_text(
                f"❌ Error: {str(e)[:100]}",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Retry", callback_data="analyze")]])
            )

    async def start_stream(self, query, context: ContextTypes.DEFAULT_TYPE):
        self.monitoring = True

        await query.edit_message_text(
            "🔔 *ADAPTIVE STREAM STARTED*\n\n"
            "AI is learning and evolving...\n"
            "Strategy weights auto-adjust every 10 trades\n"
            "Models retrain hourly with new data\n"
            "Risk sizing adapts to market regime",
            parse_mode="Markdown"
        )

        context.application.create_task(self.stream_loop(context))
        self.retrain_task = context.application.create_task(self.retrain_loop())

    async def stop_stream(self, query, context: ContextTypes.DEFAULT_TYPE):
        self.monitoring = False
        
        if self.retrain_task:
            self.retrain_task.cancel()

        keyboard = [
            [InlineKeyboardButton("🔥 Start Stream", callback_data="stream")],
            [InlineKeyboardButton("📊 Single Analysis", callback_data="analyze")]
        ]

        await query.edit_message_text(
            "⏹️ Stream stopped. AI state saved.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def retrain_loop(self):
        while self.monitoring:
            try:
                await asyncio.sleep(3600)
                if self.monitoring:
                    await self.engine.check_retrain_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retrain loop error: {e}")

    async def stream_loop(self, context: ContextTypes.DEFAULT_TYPE):
        last_message_time = 0

        while self.monitoring:
            try:
                data, current_price = await self.fetch_all_timeframes()

                if not data or current_price is None:
                    await asyncio.sleep(30)
                    continue

                self.engine.monitor_open_trades(current_price)

                signal = self.engine.analyze(data, current_price)
                self.last_signal = signal

                now = time.time()

                if signal.status == "VALID" and signal.confidence >= 80:
                    if self.chat_id:
                        await context.bot.send_message(
                            chat_id=self.chat_id,
                            text=self.format_signal_text(signal, is_alarm=True),
                            parse_mode="Markdown"
                        )
                    last_message_time = now

                elif now - last_message_time > 300:
                    if self.chat_id:
                        await context.bot.send_message(
                            chat_id=self.chat_id,
                            text=self.format_signal_text(signal, is_alarm=False),
                            parse_mode="Markdown"
                        )
                    last_message_time = now

                await asyncio.sleep(45)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(60)
    
    async def show_stats(self, query):
        history = self.engine.db.get_training_data(limit=100)
        
        if not history:
            await query.edit_message_text("No trade history yet.")
            return
        
        total = len(history)
        wins = sum(1 for t in history if t['actual_result'] == 'SUCCESS')
        win_rate = wins / total if total > 0 else 0
        
        avg_profit = np.mean([t['profit_loss'] for t in history if t['profit_loss']]) if history else 0
        
        returns = [t['profit_loss'] for t in history if t['profit_loss'] is not None]
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        text = (
            f"📊 *AI PERFORMANCE STATS*\n\n"
            f"Total Trades: {total}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Avg PnL: {avg_profit:.2f}%\n"
            f"Risk-Adj Return: {sharpe:.2f}\n"
            f"Last Retrain: {(datetime.now() - self.engine.last_retrain).total_seconds()/3600:.1f}h ago\n\n"
            f"Success Patterns: {len(self.engine.db.success_patterns)}\n"
            f"Failure Patterns: {len(self.engine.db.failure_patterns)}"
        )
        
        await query.edit_message_text(
            text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="stop")]])
        )
    
    async def show_meta_status(self, query):
        meta = self.engine.meta_engine
        
        text = (
            f"🧠 *META-LEARNING STATUS*\n\n"
            f"*Strategy Weights:*\n"
            f"  SMC: {meta.strategy_weights['smc']:.2%}\n"
            f"  Technical: {meta.strategy_weights['technical']:.2%}\n"
            f"  ML Ensemble: {meta.strategy_weights['ml_ensemble']:.2%}\n"
            f"  Pattern Memory: {meta.strategy_weights['pattern_memory']:.2%}\n\n"
            f"*Adaptive Config:*\n"
            f"  Min Confidence: {meta.config.min_confidence:.1f}%\n"
            f"  Min ADX: {meta.config.min_adx:.1f}\n"
            f"  Risk/Trade: {meta.config.risk_per_trade:.2%}\n\n"
            f"*Market Regimes Learned:*\n"
        )
        
        for regime, stats in meta.market_regime_stats.items():
            text += f"  {regime}: {stats.get('win_rate', 0):.1%} WR\n"
        
        text += f"\nAnalyses: {self.engine.analysis_count}"
        
        await query.edit_message_text(
            text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="stop")]])
        )

    def format_signal_text(self, signal, is_alarm=False):
        if signal.status != "VALID":
            emoji = "⚪"
            header = f"{emoji} *XAUUSD ANALYSIS* {emoji}"

            text = (
                f"{header}\n\n"
                f"Status: *NO TRADE*\n"
                f"Confidence: *{signal.confidence:.1f}%*\n"
                f"Confluence: {signal.confluence_count}/10\n"
                f"DL Confidence: {signal.model_predictions.get('confidence', 0):.2f}\n\n"
                f"Reason: {signal.invalid_reason or 'No setup'}\n\n"
                f"*Timeframe Alignment:*\n"
            )

            for tf, align in signal.timeframe_alignment.items():
                em = "🟢" if align == "BULLISH" else "🔴" if align == "BEARISH" else "⚪"
                text += f"{tf}: {em} {align}\n"

            return text

        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        alarm_emoji = "🚨 " if is_alarm else ""

        strat_scores = signal.strategy_scores or {}
        
        text = (
            f"{alarm_emoji}{emoji} *{signal.direction} SIGNAL* {emoji}{alarm_emoji}\n\n"
            f"Confidence: *{signal.confidence:.1f}%* 🎯\n"
            f"Confluence: {signal.confluence_count} factors\n"
            f"DL Ensemble: {signal.model_predictions.get('ensemble', 0.5):.2f}\n"
            f"Uncertainty: {signal.model_predictions.get('uncertainty', 0.5):.2f}\n\n"
            f"📍 Entry: `{signal.entry}`\n"
            f"🛑 SL: `{signal.stop_loss}`\n"
            f"🎯 TP1: `{signal.take_profit_1}`\n"
            f"🎯 TP2: `{signal.take_profit_2}`\n\n"
            f"*Strategy Breakdown:*\n"
            f"SMC: {strat_scores.get('smc', 0):.0f}\n"
            f"Tech: {strat_scores.get('technical', 0):.0f}\n"
            f"ML: {strat_scores.get('ml_ensemble', 0):.0f}\n"
            f"Pattern: {strat_scores.get('pattern_memory', 0):.0f}\n\n"
            f"*Factors:*\n"
        )

        for i, factor in enumerate(signal.factors[:6], 1):
            text += f"{i}. {factor}\n"

        text += f"\n*Alignment:*\n"
        for tf in ["1m", "5m", "15m", "30m", "1h"]:
            align = signal.timeframe_alignment.get(tf, "NEUTRAL")
            em = "🟢" if align == "BULLISH" else "🔴" if align == "BEARISH" else "⚪"
            text += f"{em}"

        return text

    async def send_signal_message(self, query, signal, is_stream=False):
        text = self.format_signal_text(signal, is_alarm=signal.confidence >= 80)

        keyboard = [
            [InlineKeyboardButton("🔄 Refresh", callback_data="analyze")],
            [InlineKeyboardButton("🔔 Start Stream", callback_data="stream")],
            [InlineKeyboardButton("⬅️ Back", callback_data="stop")]
        ]

        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )


def main():
    bot = XAUUSDBot()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CallbackQueryHandler(bot.button_handler))

    logger.info("XAUUSD Adaptive AI Scalper starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

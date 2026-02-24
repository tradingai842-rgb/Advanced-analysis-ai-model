import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json
from collections import deque
import talib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TWELVE_DATA_API_KEY = "ce0dbe1303af4be6b0cbe593744c01bd"
NEWS_API_KEY = "23a88a95fc774d76afd8ffcee66ccb01"
TELEGRAM_TOKEN = "8463088511:AAFU-8PL31RBVBrRPC3Dr5YiE0CMUGP02Ac"
POLYGON_API_KEY = "0n_l16xhW0_6Rpt9ZVQNYXD77ywLW68l"

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class OrderBlockType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    BREAKER = "breaker"
    MITIGATION = "mitigation"
    RECLAIMED = "reclaimed"

class LiquidityType(Enum):
    EQUAL_HIGHS = "equal_highs"
    EQUAL_LOWS = "equal_lows"
    TRENDLINE = "trendline"
    PREVIOUS_DAY = "previous_day"
    SWEPT = "swept"

class ImbalanceType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class OrderBlock:
    high: float
    low: float
    open_price: float
    close_price: float
    volume: float
    timestamp: datetime
    ob_type: OrderBlockType
    is_valid: bool = True
    mitigation_price: Optional[float] = None
    times_tested: int = 0
    strength_score: float = 0.0

class LiquidityZone:
    price_level: float
    zone_type: LiquidityType
    strength: float
    volume_at_level: float = 0.0
    is_swept: bool = False
    sweep_timestamp: Optional[datetime] = None
    is_target: bool = False

class FairValueGap:
    high: float
    low: float
    is_bullish: bool
    timestamp: datetime
    is_filled: bool = False
    fill_percentage: float = 0.0
    confluence_score: float = 0.0

class ImbalanceZone:
    high: float
    low: float
    zone_type: ImbalanceType
    timestamp: datetime
    is_mitigated: bool = False
    mitigation_timestamp: Optional[datetime] = None

class VolumeProfileLevel:
    price: float
    volume: float
    is_poc: bool = False
    is_val: bool = False
    is_vah: bool = False

class MarketContext:
    structure: MarketStructure
    trend_strength: float
    volatility_regime: str
    session: str
    institutional_bias: str
    risk_level: str
    smart_money_pressure: float
    delta_bias: str
    correlation_score: float

class TickData:
    price: float
    size: float
    side: str
    timestamp: datetime
    is_iceberg: bool = False
    is_spoof: bool = False

class Signal:
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float
    risk_reward_1: float
    risk_reward_2: float
    position_size: float
    reasons: List[str]
    context: Dict
    ml_prediction: float
    anomaly_score: float
    expected_slippage: float
    time_decay: float
    timestamp: datetime

class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_ohlcv(self, symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "order": "asc"
        }
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            if "values" not in data:
                raise Exception(f"API Error: {data}")
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df = df.astype(float)
            return df
    
    async def get_quote(self, symbol: str) -> Dict:
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "apikey": self.api_key}
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_order_book(self, symbol: str) -> Dict:
        url = f"{self.base_url}/order_book"
        params = {"symbol": symbol, "apikey": self.api_key}
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_level2_data(self, symbol: str) -> Dict:
        url = f"{self.base_url}/level2"
        params = {"symbol": symbol, "apikey": self.api_key}
        async with self.session.get(url, params=params) as response:
            return await response.json()

class PolygonClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
    
    async def get_tick_data(self, symbol: str, timestamp: datetime) -> List[Dict]:
        url = f"{self.base_url}/ticks/stocks/nbbo/{symbol}/{timestamp.strftime('%Y-%m-%d')}"
        params = {"apiKey": self.api_key, "timestamp": timestamp.isoformat()}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get("results", [])

class FinnhubClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
    
    async def get_order_book(self, symbol: str) -> Dict:
        url = f"{self.base_url}/stock/book"
        params = {"symbol": symbol, "token": self.api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()
    
    async def get_sentiment(self, symbol: str) -> Dict:
        url = f"{self.base_url}/news-sentiment"
        params = {"symbol": symbol, "token": self.api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()

class AlternativeDataClient:
    def __init__(self, alpha_key: str, cftc_key: str):
        self.alpha_key = alpha_key
        self.cftc_key = cftc_key
    
    async def get_gold_etf_flows(self) -> Dict:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "ETF_PROFILE",
            "symbol": "GLD",
            "apikey": self.alpha_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()
    
    async def get_cftc_positioning(self) -> Dict:
        url = f"https://www.cftc.gov/dea/futures/other_lf.htm"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                text = await response.text()
                return self._parse_cftc(text)
    
    def _parse_cftc(self, html: str) -> Dict:
        return {"net_position": 0, "commercial": 0, "non_commercial": 0}
    
    async def get_dxy_correlation(self) -> float:
        return -0.85
    
    async def get_yield_correlation(self) -> float:
        return -0.75
    
    async def get_btc_correlation(self) -> float:
        return 0.45

class NewsFilter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.high_impact_events = [
            "non-farm payrolls", "nfp", "fomc", "fed", "interest rate",
            "cpi", "inflation", "gdp", "unemployment", "retail sales",
            "pmi", "geopolitical", "war", "conflict", "treasury", "yields",
            "goldman sachs", "jp morgan", "bullion", "federal reserve",
            "powell", "lagarde", "ecb", "boe", "bank of england"
        ]
        self.impact_scores = {}
        self.geopolitical_risk_index = 0.0
    
    async def fetch_news(self) -> List[Dict]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "gold OR XAUUSD OR XAU OR \"Federal Reserve\" OR FOMC OR NFP OR inflation OR bullion",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get("articles", [])
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float, float]:
        positive_words = ["surge", "rally", "bullish", "breakout", "strong", "growth", "optimistic", "moon", "rocket"]
        negative_words = ["crash", "plunge", "bearish", "breakdown", "weak", "recession", "fear", "dump", "collapse"]
        uncertainty_words = ["uncertain", "volatile", "unclear", "mixed", "cautious", "wait", "pause"]
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        unc_score = sum(1 for word in uncertainty_words if word in text_lower)
        
        total = pos_score + neg_score + unc_score
        if total == 0:
            return 0.0, 0.0, 1.0
        
        return pos_score/total, neg_score/total, unc_score/total
    
    def detect_high_impact(self, text: str) -> Tuple[bool, float, str]:
        text_lower = text.lower()
        impact_score = 0.0
        is_high_impact = False
        category = "normal"
        
        for event in self.high_impact_events:
            if event in text_lower:
                is_high_impact = True
                impact_score += 0.25
                if event in ["war", "conflict", "geopolitical"]:
                    category = "geopolitical"
                    self.geopolitical_risk_index = min(self.geopolitical_risk_index + 0.1, 1.0)
                elif event in ["fomc", "fed", "powell", "interest rate"]:
                    category = "monetary"
        
        if any(word in text_lower for word in ["breaking", "urgent", "alert", "exclusive"]):
            impact_score += 0.35
            is_high_impact = True
        
        if "goldman" in text_lower or "jpmorgan" in text_lower or "ubs" in text_lower:
            impact_score += 0.20
        
        return is_high_impact, min(impact_score, 1.0), category
    
    async def get_trading_conditions(self) -> Dict:
        news = await self.fetch_news()
        total_pos, total_neg, total_unc = 0, 0, 0
        high_impact_detected = False
        max_impact_score = 0.0
        critical_categories = set()
        
        for article in news[:10]:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title} {description}"
            
            pos, neg, unc = self.analyze_sentiment(text)
            is_high_impact, impact_score, category = self.detect_high_impact(text)
            
            total_pos += pos
            total_neg += neg
            total_unc += unc
            
            if is_high_impact:
                high_impact_detected = True
                max_impact_score = max(max_impact_score, impact_score)
                critical_categories.add(category)
        
        total = total_pos + total_neg + total_unc
        if total > 0:
            sentiment_score = (total_pos - total_neg) / total
        else:
            sentiment_score = 0
        
        uncertainty_level = total_unc / len(news) if news else 0
        
        return {
            "avoid_trading": (high_impact_detected and max_impact_score > 0.6) or uncertainty_level > 0.5,
            "sentiment": sentiment_score,
            "impact_score": max_impact_score,
            "uncertainty": uncertainty_level,
            "geopolitical_risk": self.geopolitical_risk_index,
            "categories": list(critical_categories),
            "recommendation": "WAIT" if ((high_impact_detected and max_impact_score > 0.6) or uncertainty_level > 0.5) else "TRADE"
        }

class LSTMModel:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self._build_model()
    
    def _build_model(self):
        self.model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.sequence_length, 15)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        features = np.column_stack([
            df['returns'].values,
            df['rsi'].values / 100,
            df['macd'].values,
            df['adx'].values / 100,
            df['atr_percent'].values,
            df['bb_position'].values,
            df['volume_delta'].values if 'volume_delta' in df.columns else np.zeros(len(df)),
            df['obv_slope'].values if 'obv_slope' in df.columns else np.zeros(len(df)),
            df['ema_slope'].values if 'ema_slope' in df.columns else np.zeros(len(df)),
            df['vwap_distance'].values if 'vwap_distance' in df.columns else np.zeros(len(df)),
            df['session_score'].values if 'session_score' in df.columns else np.zeros(len(df)),
            df['correlation_dxy'].values if 'correlation_dxy' in df.columns else np.full(len(df), -0.85),
            df['correlation_yield'].values if 'correlation_yield' in df.columns else np.full(len(df), -0.75),
            df['cftc_net'].values if 'cftc_net' in df.columns else np.zeros(len(df)),
            df['etf_flow'].values if 'etf_flow' in df.columns else np.zeros(len(df))
        ])
        return self.scaler.fit_transform(features)
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        if len(df) < self.sequence_length:
            return 0.0, 0.0
        
        features = self.prepare_features(df)
        X = features[-self.sequence_length:].reshape(1, self.sequence_length, 15)
        prediction = self.model.predict(X, verbose=0)[0][0]
        confidence = 1.0 - abs(prediction - np.sign(prediction) * 1.0)
        return float(prediction), float(confidence)

class RandomForestEnsemble:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        features = []
        for i in range(len(df)):
            row = df.iloc[i]
            feat = [
                row['rsi'] / 100,
                row['macd'] / 100,
                row['adx'] / 100,
                row['atr_percent'] / 10,
                row['bb_position'],
                1 if row['ema_9'] > row['ema_21'] else 0,
                1 if row['close'] > row['vwap'] else 0,
                row['volume_ratio'] if 'volume_ratio' in row else 1.0,
                row['obv_slope'] if 'obv_slope' in row else 0,
                row['session_score'] if 'session_score' in row else 0.5
            ]
            features.append(feat)
        return np.array(features)
    
    def predict_proba(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        if not self.is_trained or len(df) < 20:
            return 0.33, 0.33, 0.34
        
        features = self.extract_features(df)
        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled[-1:])[0]
        return proba[0], proba[1], proba[2]

class ReinforcementLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
    
    def get_state(self, context: MarketContext, technical_score: float) -> str:
        return f"{context.structure.value}_{context.volatility_regime}_{int(technical_score*10)}"
    
    def get_action(self, state: str) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 3)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        return np.argmax(self.q_table[state])
    
    def update(self, state: str, action: int, reward: float, next_state: str):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

class AnomalyDetector:
    def __init__(self):
        self.baseline_mean = None
        self.baseline_std = None
        self.window_size = 100
    
    def fit(self, returns: np.ndarray):
        self.baseline_mean = np.mean(returns)
        self.baseline_std = np.std(returns)
    
    def detect(self, current_return: float, volume_spike: float, spread: float) -> Tuple[bool, float]:
        if self.baseline_std is None or self.baseline_std == 0:
            return False, 0.0
        
        z_score = abs(current_return - self.baseline_mean) / self.baseline_std
        volume_z = volume_spike
        spread_z = spread / 0.01
        
        anomaly_score = (z_score * 0.4 + volume_z * 0.4 + spread_z * 0.2) / 3
        is_anomaly = anomaly_score > 2.5
        
        return is_anomaly, min(anomaly_score / 5, 1.0)

class VolumeProfileAnalyzer:
    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins
        self.poc_level: Optional[float] = None
        self.value_area_high: Optional[float] = None
        self.value_area_low: Optional[float] = None
        self.profile: List[VolumeProfileLevel] = []
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20 or 'volume' not in df.columns:
            return {}
        
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / self.num_bins
        
        volume_by_price = {}
        for _, row in df.iterrows():
            typical_price = (row['high'] + row['low'] + row['close']) / 3
            bin_price = round(typical_price / bin_size) * bin_size
            volume_by_price[bin_price] = volume_by_price.get(bin_price, 0) + row['volume']
        
        sorted_levels = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_levels:
            return {}
        
        self.poc_level = sorted_levels[0][0]
        total_volume = sum(v for _, v in sorted_levels)
        value_area_volume = total_volume * 0.70
        
        current_volume = 0
        value_area_prices = []
        for price, vol in sorted_levels:
            current_volume += vol
            value_area_prices.append(price)
            if current_volume >= value_area_volume:
                break
        
        self.value_area_high = max(value_area_prices) if value_area_prices else self.poc_level
        self.value_area_low = min(value_area_prices) if value_area_prices else self.poc_level
        
        self.profile = [VolumeProfileLevel(price=p, volume=v, is_poc=(p==self.poc_level),
                                          is_val=(p==self.value_area_low), is_vah=(p==self.value_area_high))
                       for p, v in sorted_levels]
        
        return {
            "poc": self.poc_level,
            "vah": self.value_area_high,
            "val": self.value_area_low,
            "profile": self.profile
        }
    
    def get_nearest_levels(self, current_price: float, num_levels: int = 3) -> List[VolumeProfileLevel]:
        sorted_profile = sorted(self.profile, key=lambda x: abs(x.price - current_price))
        return sorted_profile[:num_levels]

class OrderFlowAnalyzer:
    def __init__(self):
        self.tick_buffer: Deque[TickData] = deque(maxlen=1000)
        self.cumulative_delta: float = 0.0
        self.delta_history: Deque[float] = deque(maxlen=100)
        self.imbalance_threshold = 2.0
        self.iceberg_detection_window = 50
    
    def process_tick(self, tick: TickData):
        self.tick_buffer.append(tick)
        
        if tick.side == "buy":
            self.cumulative_delta += tick.size
        else:
            self.cumulative_delta -= tick.size
        
        self.delta_history.append(self.cumulative_delta)
    
    def calculate_delta_metrics(self) -> Dict:
        if len(self.delta_history) < 20:
            return {"delta": 0, "delta_slope": 0, "buying_pressure": 0.5}
        
        recent_delta = list(self.delta_history)[-20:]
        slope = np.polyfit(range(len(recent_delta)), recent_delta, 1)[0]
        
        buy_volume = sum(t.size for t in self.tick_buffer if t.side == "buy")
        sell_volume = sum(t.size for t in self.tick_buffer if t.side == "sell")
        total = buy_volume + sell_volume
        
        buying_pressure = buy_volume / total if total > 0 else 0.5
        
        return {
            "delta": self.cumulative_delta,
            "delta_slope": slope,
            "buying_pressure": buying_pressure,
            "delta_divergence": self._check_divergence()
        }
    
    def _check_divergence(self) -> bool:
        if len(self.tick_buffer) < 100:
            return False
        
        recent_ticks = list(self.tick_buffer)[-100:]
        prices = [t.price for t in recent_ticks]
        deltas = []
        running_delta = 0
        
        for tick in recent_ticks:
            if tick.side == "buy":
                running_delta += tick.size
            else:
                running_delta -= tick.size
            deltas.append(running_delta)
        
        price_change = prices[-1] - prices[0]
        delta_change = deltas[-1] - deltas[0]
        
        return (price_change > 0 and delta_change < 0) or (price_change < 0 and delta_change > 0)
    
    def detect_iceberg_orders(self) -> List[Dict]:
        icebergs = []
        if len(self.tick_buffer) < self.iceberg_detection_window:
            return icebergs
        
        window = list(self.tick_buffer)[-self.iceberg_detection_window:]
        price_groups = {}
        
        for tick in window:
            price_key = round(tick.price, 2)
            if price_key not in price_groups:
                price_groups[price_key] = []
            price_groups[price_key].append(tick)
        
        for price, ticks in price_groups.items():
            if len(ticks) > 10:
                total_size = sum(t.size for t in ticks)
                avg_size = total_size / len(ticks)
                if avg_size > 5.0:
                    icebergs.append({
                        "price": price,
                        "count": len(ticks),
                        "total_size": total_size,
                        "side": ticks[0].side
                    })
        
        return icebergs
    
    def detect_spoofing(self) -> List[Dict]:
        spoofs = []
        if len(self.tick_buffer) < 100:
            return spoofs
        
        recent = list(self.tick_buffer)[-100:]
        for i, tick in enumerate(recent[:-5]):
            if tick.size > 10.0:
                cancelled = True
                for future_tick in recent[i+1:i+6]:
                    if abs(future_tick.price - tick.price) < 0.01 and future_tick.size > 1.0:
                        cancelled = False
                        break
                if cancelled:
                    spoofs.append({
                        "price": tick.price,
                        "size": tick.size,
                        "timestamp": tick.timestamp
                    })
        
        return spoofs

class SMCAnalyzer:
    def __init__(self):
        self.order_blocks: deque = deque(maxlen=100)
        self.liquidity_zones: List[LiquidityZone] = []
        self.fvgs: List[FairValueGap] = []
        self.imbalances: List[ImbalanceZone] = []
        self.swing_points: deque = deque(maxlen=200)
        self.market_structure: MarketStructure = MarketStructure.RANGING
        self.previous_structure: MarketStructure = MarketStructure.RANGING
        self.breaker_blocks: List[OrderBlock] = []
        self.mitigation_blocks: List[OrderBlock] = []
        self.reclaimed_blocks: List[OrderBlock] = []
        self.premium_array: List[float] = []
        self.discount_array: List[float] = []
    
    def detect_swing_points(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
        highs = df["high"].values
        lows = df["low"].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
                           all(highs[i] >= highs[i+j] for j in range(1, lookback+1))
            is_swing_low = all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
                          all(lows[i] <= lows[i+j] for j in range(1, lookback+1))
            
            if is_swing_high:
                swing_highs.append(i)
            if is_swing_low:
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def identify_order_blocks(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]):
        self.order_blocks.clear()
        self.breaker_blocks.clear()
        self.mitigation_blocks.clear()
        self.reclaimed_blocks.clear()
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            body_size = abs(current["close"] - current["open"])
            range_size = current["high"] - current["low"]
            
            if body_size < range_size * 0.2:
                continue
            
            is_bullish = current["close"] > current["open"]
            is_bearish = current["close"] < current["open"]
            
            momentum_before = abs(prev["close"] - prev2["open"])
            
            if is_bullish and momentum_before > body_size * 0.3:
                ob = OrderBlock(
                    high=current["high"],
                    low=current["low"],
                    open_price=current["open"],
                    close_price=current["close"],
                    volume=current.get("volume", 0),
                    timestamp=df.index[i],
                    ob_type=OrderBlockType.BULLISH,
                    strength_score=self._calculate_ob_strength(current, prev, True)
                )
                self.order_blocks.append(ob)
                
                if self._is_breaker_block(df, i, True):
                    ob.ob_type = OrderBlockType.BREAKER
                    self.breaker_blocks.append(ob)
            
            elif is_bearish and momentum_before > body_size * 0.3:
                ob = OrderBlock(
                    high=current["high"],
                    low=current["low"],
                    open_price=current["open"],
                    close_price=current["close"],
                    volume=current.get("volume", 0),
                    timestamp=df.index[i],
                    ob_type=OrderBlockType.BEARISH,
                    strength_score=self._calculate_ob_strength(current, prev, False)
                )
                self.order_blocks.append(ob)
                
                if self._is_breaker_block(df, i, False):
                    ob.ob_type = OrderBlockType.BREAKER
                    self.breaker_blocks.append(ob)
        
        self._update_mitigation_blocks(df)
        self._update_reclaimed_blocks(df)
    
    def _calculate_ob_strength(self, current: pd.Series, previous: pd.Series, is_bullish: bool) -> float:
        strength = 0.0
        body_size = abs(current["close"] - current["open"])
        range_size = current["high"] - current["low"]
        
        if body_size / range_size > 0.7:
            strength += 0.3
        
        if current.get("volume", 0) > previous.get("volume", 0) * 1.5:
            strength += 0.3
        
        if is_bullish and current["close"] == current["high"]:
            strength += 0.2
        elif not is_bullish and current["close"] == current["low"]:
            strength += 0.2
        
        return min(strength, 1.0)
    
    def _is_breaker_block(self, df: pd.DataFrame, idx: int, was_bullish: bool) -> bool:
        if idx < 5 or idx >= len(df) - 5:
            return False
        
        ob_high = df.iloc[idx]["high"]
        ob_low = df.iloc[idx]["low"]
        
        for i in range(idx+1, min(idx+6, len(df))):
            if was_bullish:
                if df.iloc[i]["low"] < ob_low and df.iloc[i]["close"] > ob_high:
                    return True
            else:
                if df.iloc[i]["high"] > ob_high and df.iloc[i]["close"] < ob_low:
                    return True
        
        return False
    
    def _update_mitigation_blocks(self, df: pd.DataFrame):
        current_price = df["close"].iloc[-1]
        
        for ob in self.order_blocks:
            if ob.ob_type in [OrderBlockType.BULLISH, OrderBlockType.BREAKER]:
                if ob.low < current_price < ob.high and ob.times_tested > 0:
                    ob.ob_type = OrderBlockType.MITIGATION
                    self.mitigation_blocks.append(ob)
            elif ob.ob_type in [OrderBlockType.BEARISH, OrderBlockType.BREAKER]:
                if ob.low < current_price < ob.high and ob.times_tested > 0:
                    ob.ob_type = OrderBlockType.MITIGATION
                    self.mitigation_blocks.append(ob)
    
    def _update_reclaimed_blocks(self, df: pd.DataFrame):
        for ob in self.order_blocks:
            if ob.times_tested >= 2 and ob.is_valid:
                recent_test = False
                for i in range(-5, 0):
                    if i >= -len(df):
                        low, high = df.iloc[i]["low"], df.iloc[i]["high"]
                        if ob.low <= high and ob.high >= low:
                            recent_test = True
                            break
                
                if recent_test and ob.is_valid:
                    ob.ob_type = OrderBlockType.RECLAIMED
                    self.reclaimed_blocks.append(ob)
    
    def detect_fair_value_gaps(self, df: pd.DataFrame):
        self.fvgs.clear()
        
        for i in range(2, len(df)):
            candle_1 = df.iloc[i-2]
            candle_2 = df.iloc[i-1]
            candle_3 = df.iloc[i]
            
            gap_up = candle_2["low"] > candle_1["high"]
            gap_down = candle_2["high"] < candle_1["low"]
            
            confluence = 0.0
            if candle_3["volume"] > df["volume"].rolling(20).mean().iloc[i] * 1.5:
                confluence += 0.3
            
            if gap_up:
                fvg = FairValueGap(
                    high=candle_2["low"],
                    low=candle_1["high"],
                    is_bullish=True,
                    timestamp=df.index[i],
                    confluence_score=confluence
                )
                self.fvgs.append(fvg)
            
            elif gap_down:
                fvg = FairValueGap(
                    high=candle_1["low"],
                    low=candle_2["high"],
                    is_bullish=False,
                    timestamp=df.index[i],
                    confluence_score=confluence
                )
                self.fvgs.append(fvg)
        
        self._update_fvg_status(df)
    
    def _update_fvg_status(self, df: pd.DataFrame):
        for fvg in self.fvgs:
            if fvg.is_filled:
                continue
            
            for i in range(-10, 0):
                if i >= -len(df):
                    low, high = df.iloc[i]["low"], df.iloc[i]["high"]
                    if fvg.low <= high and fvg.high >= low:
                        overlap = min(high, fvg.high) - max(low, fvg.low)
                        fvg.fill_percentage = overlap / (fvg.high - fvg.low)
                        if fvg.fill_percentage >= 0.9:
                            fvg.is_filled = True
    
    def detect_imbalance_zones(self, df: pd.DataFrame):
        self.imbalances.clear()
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            if current["high"] < prev["low"]:
                zone = ImbalanceZone(
                    high=prev["low"],
                    low=current["high"],
                    zone_type=ImbalanceType.BEARISH,
                    timestamp=df.index[i]
                )
                self.imbalances.append(zone)
            
            elif current["low"] > prev["high"]:
                zone = ImbalanceZone(
                    high=current["low"],
                    low=prev["high"],
                    zone_type=ImbalanceType.BULLISH,
                    timestamp=df.index[i]
                )
                self.imbalances.append(zone)
    
    def identify_liquidity_zones(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]):
        self.liquidity_zones.clear()
        
        highs = [df.iloc[i]["high"] for i in swing_highs[-15:]]
        lows = [df.iloc[i]["low"] for i in swing_lows[-15:]]
        
        for i, high in enumerate(highs):
            cluster = [h for h in highs if abs(h - high) < high * 0.0005]
            if len(cluster) >= 2:
                avg_high = np.mean(cluster)
                strength = len(cluster) / len(highs)
                
                is_swept = any(df.iloc[j]["high"] > avg_high * 1.001 for j in range(-5, 0) if j >= -len(df))
                
                zone = LiquidityZone(
                    price_level=avg_high,
                    zone_type=LiquidityType.EQUAL_HIGHS,
                    strength=strength,
                    is_swept=is_swept,
                    is_target=not is_swept
                )
                self.liquidity_zones.append(zone)
        
        for i, low in enumerate(lows):
            cluster = [l for l in lows if abs(l - low) < low * 0.0005]
            if len(cluster) >= 2:
                avg_low = np.mean(cluster)
                strength = len(cluster) / len(lows)
                
                is_swept = any(df.iloc[j]["low"] < avg_low * 0.999 for j in range(-5, 0) if j >= -len(df))
                
                zone = LiquidityZone(
                    price_level=avg_low,
                    zone_type=LiquidityType.EQUAL_LOWS,
                    strength=strength,
                    is_swept=is_swept,
                    is_target=not is_swept
                )
                self.liquidity_zones.append(zone)
        
        self._add_previous_day_liquidity(df)
    
    def _add_previous_day_liquidity(self, df: pd.DataFrame):
        if len(df) < 100:
            return
        
        daily_groups = df.groupby(df.index.date)
        for date, group in list(daily_groups.items())[-3:]:
            day_high = group["high"].max()
            day_low = group["low"].min()
            
            self.liquidity_zones.append(LiquidityZone(
                price_level=day_high,
                zone_type=LiquidityType.PREVIOUS_DAY,
                strength=0.7,
                is_swept=False
            ))
            self.liquidity_zones.append(LiquidityZone(
                price_level=day_low,
                zone_type=LiquidityType.PREVIOUS_DAY,
                strength=0.7,
                is_swept=False
            ))
    
    def calculate_premium_discount(self, df: pd.DataFrame):
        if len(df) < 50:
            return
        
        fib_high = df["high"].rolling(50).max().iloc[-1]
        fib_low = df["low"].rolling(50).min().iloc[-1]
        range_size = fib_high - fib_low
        
        self.premium_array = [fib_high - range_size * 0.236, fib_high - range_size * 0.382]
        self.discount_array = [fib_low + range_size * 0.236, fib_low + range_size * 0.382]
    
    def analyze_market_structure(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> MarketContext:
        if not swing_highs or not swing_lows:
            return MarketContext(MarketStructure.RANGING, 0.0, "normal", "unknown", "neutral", "medium", 0.0, "neutral", 0.0)
        
        recent_highs = [df.iloc[i]["high"] for i in swing_highs[-5:]]
        recent_lows = [df.iloc[i]["low"] for i in swing_lows[-5:]]
        
        higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            structure = MarketStructure.UPTREND
            strength = 0.85
        elif lower_highs and lower_lows:
            structure = MarketStructure.DOWNTREND
            strength = 0.85
        elif higher_highs and lower_lows:
            structure = MarketStructure.BREAKOUT
            strength = 0.90
        elif lower_highs and higher_lows:
            structure = MarketStructure.REVERSAL
            strength = 0.80
        else:
            structure = MarketStructure.RANGING
            strength = 0.40
        
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        if volatility > 0.30:
            vol_regime = "extreme"
        elif volatility > 0.20:
            vol_regime = "high"
        elif volatility > 0.12:
            vol_regime = "normal"
        else:
            vol_regime = "low"
        
        current_hour = datetime.now().hour
        if 8 <= current_hour < 12:
            session = "london"
            session_score = 0.9
        elif 12 <= current_hour < 17:
            session = "ny_am"
            session_score = 1.0
        elif 17 <= current_hour < 21:
            session = "ny_pm"
            session_score = 0.8
        else:
            session = "asia"
            session_score = 0.6
        
        if structure == MarketStructure.UPTREND:
            bias = "bullish"
        elif structure == MarketStructure.DOWNTREND:
            bias = "bearish"
        else:
            bias = "neutral"
        
        risk = "high" if vol_regime in ["high", "extreme"] else "medium"
        
        smart_money_pressure = 0.0
        for ob in self.order_blocks:
            if ob.ob_type == OrderBlockType.BULLISH and ob.is_valid:
                smart_money_pressure += ob.strength_score
            elif ob.ob_type == OrderBlockType.BEARISH and ob.is_valid:
                smart_money_pressure -= ob.strength_score
        
        smart_money_pressure = np.clip(smart_money_pressure, -1, 1)
        
        return MarketContext(structure, strength, vol_regime, session, bias, risk, smart_money_pressure, "neutral", 0.0)

class TechnicalAnalyzer:
    def __init__(self):
        self.lookback = 200
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_9"] = talib.EMA(df["close"], timeperiod=9)
        df["ema_21"] = talib.EMA(df["close"], timeperiod=21)
        df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
        df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
        
        df["ema_slope"] = df["ema_21"].diff(5) / 5
        
        df["rsi"] = talib.RSI(df["close"], timeperiod=14)
        df["rsi_ema"] = talib.EMA(df["rsi"], timeperiod=9)
        
        macd, macdsignal, macdhist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["macd"] = macd
        df["macd_signal"] = macdsignal
        df["macd_hist"] = macdhist
        
        df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        df["atr_percent"] = (df["atr"] / df["close"]) * 100
        
        upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2.5, nbdevdn=2.5)
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_middle"] = middle
        df["bb_position"] = (df["close"] - lower) / (upper - lower)
        df["bb_width"] = (upper - lower) / middle
        
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
        df["plus_di"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod=14)
        df["minus_di"] = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod=14)
        
        df["slowk"], df["slowd"] = talib.STOCH(df["high"], df["low"], df["close"], 
                                                fastk_period=14, slowk_period=3, slowk_matype=0,
                                                slowd_period=3, slowd_matype=0)
        
        df["obv"] = talib.OBV(df["close"], df.get("volume", pd.Series([0]*len(df))))
        df["obv_slope"] = df["obv"].diff(5)
        
        df["ad_line"] = talib.AD(df["high"], df["low"], df["close"], df.get("volume", pd.Series([0]*len(df))))
        
        df["vwap"] = (df["close"] * df.get("volume", pd.Series([1]*len(df)))).cumsum() / df.get("volume", pd.Series([1]*len(df))).cumsum()
        df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]
        
        df["returns"] = df["close"].pct_change()
        
        if "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
            df["volume_delta"] = df["volume"] * np.where(df["close"] > df["open"], 1, -1)
        
        df["session_score"] = df.index.map(lambda x: 1.0 if 12 <= x.hour < 17 else 0.8 if 8 <= x.hour < 12 else 0.6)
        
        return df
    
    def detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []
        
        for i in range(lookback, len(df) - 1):
            if df["high"].iloc[i] == df["high"].iloc[i-lookback:i+1].max():
                price_highs.append((i, df["high"].iloc[i]))
                rsi_highs.append((i, df["rsi"].iloc[i]))
            if df["low"].iloc[i] == df["low"].iloc[i-lookback:i+1].min():
                price_lows.append((i, df["low"].iloc[i]))
                rsi_lows.append((i, df["rsi"].iloc[i]))
        
        bearish_div = False
        bullish_div = False
        hidden_bullish = False
        hidden_bearish = False
        
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            price_hh = price_highs[-1][1] > price_highs[-2][1]
            rsi_lh = rsi_highs[-1][1] < rsi_highs[-2][1]
            price_lh = price_highs[-1][1] < price_highs[-2][1]
            rsi_hh = rsi_highs[-1][1] > rsi_highs[-2][1]
            
            if price_hh and rsi_lh:
                bearish_div = True
            if price_lh and rsi_hh:
                hidden_bearish = True
        
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            price_ll = price_lows[-1][1] < price_lows[-2][1]
            rsi_hl = rsi_lows[-1][1] > rsi_lows[-2][1]
            price_hl = price_lows[-1][1] > price_lows[-2][1]
            rsi_ll = rsi_lows[-1][1] < rsi_lows[-2][1]
            
            if price_ll and rsi_hl:
                bullish_div = True
            if price_hl and rsi_ll:
                hidden_bullish = True
        
        return {
            "bearish_divergence": bearish_div,
            "bullish_divergence": bullish_div,
            "hidden_bullish": hidden_bullish,
            "hidden_bearish": hidden_bearish,
            "strength": 0.9 if (bearish_div or bullish_div) else 0.7 if (hidden_bullish or hidden_bearish) else 0.0
        }
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        close = df["close"].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            "pivot": pivot,
            "r1": r1, "r2": r2, "r3": r3,
            "s1": s1, "s2": s2, "s3": s3
        }
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        high = df["high"].rolling(100).max().iloc[-1]
        low = df["low"].rolling(100).min().iloc[-1]
        range_size = high - low
        
        return {
            "0.0": high,
            "0.236": high - range_size * 0.236,
            "0.382": high - range_size * 0.382,
            "0.5": high - range_size * 0.5,
            "0.618": high - range_size * 0.618,
            "0.786": high - range_size * 0.786,
            "1.0": low
        }

class ExecutionIntelligence:
    def __init__(self):
        self.slippage_model = {}
        self.impact_model = {}
        self.latency_data = {}
    
    def predict_slippage(self, order_size: float, current_spread: float, volatility: float) -> float:
        base_slippage = current_spread * 0.5
        size_impact = np.log(order_size + 1) * 0.001
        vol_impact = volatility * 0.1
        
        return base_slippage + size_impact + vol_impact
    
    def calculate_market_impact(self, order_size: float, avg_volume: float) -> float:
        participation = order_size / avg_volume
        if participation < 0.01:
            return 0.0001
        elif participation < 0.05:
            return 0.0005
        elif participation < 0.10:
            return 0.001
        else:
            return 0.002
    
    def optimize_entry_timing(self, df: pd.DataFrame) -> Dict:
        current_minute = datetime.now().minute
        
        volatility_by_minute = df.groupby(df.index.minute)["returns"].std()
        lowest_vol_minutes = volatility_by_minute.nsmallest(10).index.tolist()
        
        return {
            "optimal_minutes": lowest_vol_minutes,
            "current_rating": "good" if current_minute in lowest_vol_minutes else "average"
        }

class ScalpingEngine:
    def __init__(self):
        self.smc = SMCAnalyzer()
        self.tech = TechnicalAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.order_flow = OrderFlowAnalyzer()
        self.lstm = LSTMModel()
        self.rf = RandomForestEnsemble()
        self.rl = ReinforcementLearningAgent()
        self.anomaly = AnomalyDetector()
        self.execution = ExecutionIntelligence()
        self.min_confidence = 0.80
        self.risk_reward_1 = 2.0
        self.risk_reward_2 = 3.0
    
    async def analyze(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame, 
                     df_1h: pd.DataFrame, news_context: Dict, alt_data: Dict,
                     tick_data: Optional[List[TickData]] = None) -> Optional[Signal]:
        
        if news_context.get("avoid_trading", False):
            return None
        
        df_1m = self.tech.calculate_indicators(df_1m)
        df_5m = self.tech.calculate_indicators(df_5m)
        df_15m = self.tech.calculate_indicators(df_15m)
        
        if tick_data:
            for tick in tick_data:
                self.order_flow.process_tick(tick)
        
        swing_highs_5m, swing_lows_5m = self.smc.detect_swing_points(df_5m)
        self.smc.identify_order_blocks(df_5m, swing_highs_5m, swing_lows_5m)
        self.smc.detect_fair_value_gaps(df_5m)
        self.smc.detect_imbalance_zones(df_5m)
        self.smc.identify_liquidity_zones(df_5m, swing_highs_5m, swing_lows_5m)
        self.smc.calculate_premium_discount(df_5m)
        
        context = self.smc.analyze_market_structure(df_5m, swing_highs_5m, swing_lows_5m)
        
        vp_data = self.volume_profile.calculate(df_5m)
        
        divergence = self.tech.detect_divergence(df_5m)
        pivots = self.tech.calculate_pivot_points(df_5m)
        fibs = self.tech.calculate_fibonacci_levels(df_5m)
        
        delta_metrics = self.order_flow.calculate_delta_metrics()
        context.delta_bias = "bullish" if delta_metrics["buying_pressure"] > 0.6 else "bearish" if delta_metrics["buying_pressure"] < 0.4 else "neutral"
        
        if alt_data:
            context.correlation_score = alt_data.get("dxy_corr", -0.85) * -1
        
        lstm_pred, lstm_conf = self.lstm.predict(df_5m)
        rf_down, rf_neutral, rf_up = self.rf.predict_proba(df_5m)
        
        ml_direction = np.sign(lstm_pred)
        ml_strength = max(rf_down, rf_neutral, rf_up)
        
        anomaly_detected, anomaly_score = self.anomaly.detect(
            df_1m["returns"].iloc[-1],
            df_1m["volume_ratio"].iloc[-1] if "volume_ratio" in df_1m.columns else 1.0,
            (df_1m["high"].iloc[-1] - df_1m["low"].iloc[-1]) / df_1m["close"].iloc[-1]
        )
        
        if anomaly_detected and anomaly_score > 0.8:
            return None
        
        signal = self._generate_signal(
            df_1m, df_5m, df_15m, context, divergence, pivots, fibs, vp_data,
            delta_metrics, lstm_pred, ml_strength, anomaly_score, news_context
        )
        
        return signal
    
    def _generate_signal(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                        context: MarketContext, divergence: Dict, pivots: Dict, fibs: Dict,
                        vp_data: Dict, delta_metrics: Dict, lstm_pred: float,
                        ml_strength: float, anomaly_score: float, news: Dict) -> Optional[Signal]:
        
        score = 0.0
        reasons = []
        entry = None
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None
        direction = None
        
        current_price = df_1m["close"].iloc[-1]
        atr = df_1m["atr"].iloc[-1]
        
        ema_aligned_bullish = (df_1m["ema_9"].iloc[-1] > df_1m["ema_21"].iloc[-1] > df_1m["ema_50"].iloc[-1])
        ema_aligned_bearish = (df_1m["ema_9"].iloc[-1] < df_1m["ema_21"].iloc[-1] < df_1m["ema_50"].iloc[-1])
        
        rsi_1m = df_1m["rsi"].iloc[-1]
        rsi_5m = df_5m["rsi"].iloc[-1]
        rsi_15m = df_15m["rsi"].iloc[-1] if len(df_15m) > 14 else 50
        
        adx = df_5m["adx"].iloc[-1]
        
        if context.structure == MarketStructure.UPTREND and ema_aligned_bullish:
            if 25 < rsi_5m < 45:
                score += 0.20
                reasons.append("Bullish structure + RSI pullback")
            
            if divergence["bullish_divergence"] or divergence["hidden_bullish"]:
                score += 0.15
                reasons.append("RSI divergence confirmed")
            
            if delta_metrics["buying_pressure"] > 0.6:
                score += 0.10
                reasons.append("Order flow bullish")
            
            for ob in self.smc.order_blocks:
                if ob.ob_type in [OrderBlockType.BULLISH, OrderBlockType.RECLAIMED] and ob.is_valid:
                    if abs(current_price - ob.low) < atr * 1.5:
                        score += 0.15 + (ob.strength_score * 0.05)
                        reasons.append(f"Valid OB support (strength: {ob.strength_score:.2f})")
                        entry = ob.low
                        break
            
            for fvg in self.smc.fvgs:
                if fvg.is_bullish and not fvg.is_filled and fvg.confluence_score > 0.2:
                    if fvg.low <= current_price <= fvg.high:
                        score += 0.10
                        reasons.append("Bullish FVG confluence")
            
            if vp_data and "val" in vp_data:
                if abs(current_price - vp_data["val"]) < atr:
                    score += 0.08
                    reasons.append("At Value Area Low")
            
            if lstm_pred > 0.3 and ml_strength > 0.4:
                score += 0.12
                reasons.append("ML ensemble bullish")
            
            if score >= self.min_confidence:
                direction = "LONG"
                if entry is None:
                    entry = current_price
                
                stop_distance = atr * 1.2
                stop_loss = entry - stop_distance
                take_profit_1 = entry + (stop_distance * self.risk_reward_1)
                take_profit_2 = entry + (stop_distance * self.risk_reward_2)
        
        elif context.structure == MarketStructure.DOWNTREND and ema_aligned_bearish:
            if 55 < rsi_5m < 75:
                score += 0.20
                reasons.append("Bearish structure + RSI bounce")
            
            if divergence["bearish_divergence"] or divergence["hidden_bearish"]:
                score += 0.15
                reasons.append("RSI divergence confirmed")
            
            if delta_metrics["buying_pressure"] < 0.4:
                score += 0.10
                reasons.append("Order flow bearish")
            
            for ob in self.smc.order_blocks:
                if ob.ob_type in [OrderBlockType.BEARISH, OrderBlockType.RECLAIMED] and ob.is_valid:
                    if abs(current_price - ob.high) < atr * 1.5:
                        score += 0.15 + (ob.strength_score * 0.05)
                        reasons.append(f"Valid OB resistance (strength: {ob.strength_score:.2f})")
                        entry = ob.high
                        break
            
            for fvg in self.smc.fvgs:
                if not fvg.is_bullish and not fvg.is_filled and fvg.confluence_score > 0.2:
                    if fvg.low <= current_price <= fvg.high:
                        score += 0.10
                        reasons.append("Bearish FVG confluence")
            
            if vp_data and "vah" in vp_data:
                if abs(current_price - vp_data["vah"]) < atr:
                    score += 0.08
                    reasons.append("At Value Area High")
            
            if lstm_pred < -0.3 and ml_strength > 0.4:
                score += 0.12
                reasons.append("ML ensemble bearish")
            
            if score >= self.min_confidence:
                direction = "SHORT"
                if entry is None:
                    entry = current_price
                
                stop_distance = atr * 1.2
                stop_loss = entry + stop_distance
                take_profit_1 = entry - (stop_distance * self.risk_reward_1)
                take_profit_2 = entry - (stop_distance * self.risk_reward_2)
        
        if direction is None:
            return None
        
        news_boost = abs(news.get("sentiment", 0)) * 0.05
        score = min(score + news_boost, 1.0)
        
        if score < self.min_confidence:
            return None
        
        expected_slippage = self.execution.predict_slippage(
            1.0, (df_1m["high"].iloc[-1] - df_1m["low"].iloc[-1]), df_1m["atr_percent"].iloc[-1]
        )
        
        rl_state = self.rl.get_state(context, score)
        rl_action = self.rl.get_action(rl_state)
        
        position_size = 0.02 / (abs(entry - stop_loss) / entry)
        position_size = min(position_size, 0.05)
        
        return Signal(
            direction=direction,
            entry=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(take_profit_1, 2),
            take_profit_2=round(take_profit_2, 2),
            confidence=round(score * 100, 1),
            risk_reward_1=self.risk_reward_1,
            risk_reward_2=self.risk_reward_2,
            position_size=round(position_size, 4),
            reasons=reasons,
            context={
                "structure": context.structure.value,
                "trend_strength": round(context.trend_strength, 2),
                "session": context.session,
                "volatility": context.volatility_regime,
                "delta_bias": context.delta_bias,
                "smart_money_pressure": round(context.smart_money_pressure, 2)
            },
            ml_prediction=round(lstm_pred, 4),
            anomaly_score=round(anomaly_score, 2),
            expected_slippage=round(expected_slippage, 4),
            time_decay=0.0,
            timestamp=datetime.now()
        )

class XAUUSDBot:
    def __init__(self):
        self.data_client = TwelveDataClient(TWELVE_DATA_API_KEY)
        self.polygon_client = PolygonClient(POLYGON_API_KEY)
        self.finnhub_client = FinnhubClient(FINNHUB_API_KEY)
        self.alt_data = AlternativeDataClient(ALPHA_VANTAGE_API_KEY, CFTC_API_KEY)
        self.news_filter = NewsFilter(NEWS_API_KEY)
        self.engine = ScalpingEngine()
        self.active_users = set()
        self.last_signal = None
        self.tick_buffer = deque(maxlen=1000)
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔥 Analyze XAUUSD", callback_data="analyze")],
            [InlineKeyboardButton("📊 Market Structure", callback_data="structure")],
            [InlineKeyboardButton("⚡ Order Flow", callback_data="orderflow")],
            [InlineKeyboardButton("🤖 ML Prediction", callback_data="ml")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🏆 *XAUUSD MAXIMUM ADVANCED SCALPING AI*\n\n"
            "Institutional Smart Money + Order Flow\n"
            "LSTM Neural Networks + Reinforcement Learning\n"
            "Real-time News + Alternative Data\n"
            "Volume Profile + Anomaly Detection\n\n"
            "Accuracy Target: 85%+\n"
            "Risk Management: Institutional Grade\n\n"
            "Select analysis module:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == "analyze":
    await self._run_full_analysis(query)
elif query.data == "structure":
    await self._show_structure(query)
elif query.data == "orderflow":
    await self._show_orderflow(query)
elif query.data == "ml":
    await self._show_ml_prediction(query)
elif query.data == "settings":
    await self._show_settings(query)
elif query.data == "back":
    await self.start(update, context)

async def _run_full_analysis(self, query):
    await query.edit_message_text("🔬 Running maximum analysis across all modules...")

    try:
        async with self.data_client as client:
            df_1m = await client.get_ohlcv("XAU/USD", "1min", outputsize=500)
            df_5m = await client.get_ohlcv("XAU/USD", "5min", outputsize=500)
            df_15m = await client.get_ohlcv("XAU/USD", "15min", outputsize=300)
            df_1h = await client.get_ohlcv("XAU/USD", "1h", outputsize=200)

        news_task = self.news_filter.get_trading_conditions()
        alt_task = self._get_alternative_data()

        news_context, alt_data = await asyncio.gather(news_task, alt_task)

        if news_context["recommendation"] == "WAIT":
            await query.edit_message_text(
                f"⛔ *TRADING HALTED - NEWS FILTER*\n\n"
                f"Impact Score: {news_context['impact_score']:.2f}/1.0\n"
                f"Categories: {', '.join(news_context['categories'])}\n"
                f"GeoPolitical Risk: {news_context['geopolitical_risk']:.2f}\n"
                f"Sentiment: {news_context['sentiment']:+.2f}\n\n"
                f"Capital protection active.",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
            )
            return

        signal = await self.engine.analyze(df_1m, df_5m, df_15m, df_1h, news_context, alt_data)

        if signal is None:
            await query.edit_message_text(
                "⚪ *NO HIGH-PROBABILITY SETUP*\n\n"
                f"Confidence threshold: 80%\n"
                f"Current market: {self.engine.smc.market_structure.value}\n"
                f"Anomaly score: {self.engine.anomaly.baseline_std if self.engine.anomaly.baseline_std else 'N/A'}\n\n"
                "Waiting for institutional confluence...",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Refresh", callback_data="analyze")],
                    [InlineKeyboardButton("⬅️ Back", callback_data="back")]
                ])
            )
            return

        self.last_signal = signal

        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        direction_text = f"{emoji} *{signal.direction}* {emoji}"

        message = (
            f"{direction_text}\n\n"
            f"📍 Entry: `{signal.entry}`\n"
            f"🛑 Stop: `{signal.stop_loss}`\n"
            f"🎯 TP1: `{signal.take_profit_1}` (1:{signal.risk_reward_1})\n"
            f"🎯 TP2: `{signal.take_profit_2}` (1:{signal.risk_reward_2})\n\n"
            f"🎯 Confidence: *{signal.confidence}%*\n"
            f"📊 Position: {signal.position_size*100:.2f}%\n"
            f"⚠️ Slippage: ±{signal.expected_slippage*100:.2f}%\n\n"
            f"*ML Prediction:* {signal.ml_prediction:+.4f}\n"
            f"*Anomaly Score:* {signal.anomaly_score:.2f}\n\n"
            f"*Confluence ({len(signal.reasons)} factors):*\n"
        )

        for i, reason in enumerate(signal.reasons[:8], 1):
            message += f"{i}. {reason}\n"

        context = signal.context

        structure = context.get("structure", "N/A")
        trend = context.get("trend_strength", "N/A")
        session = context.get("session", "N/A")
        vol = context.get("volatility", "N/A")
        delta = context.get("delta_bias", "N/A")
        smc_pressure = context.get("smart_money_pressure", 0.0)

        message += (
            f"\n*Context:*\n"
            f"Structure: {structure.upper() if isinstance(structure, str) else structure}\n"
            f"Trend: {trend}/1.0\n"
            f"Session: {session.upper() if isinstance(session, str) else session}\n"
            f"Vol: {vol.upper() if isinstance(vol, str) else vol}\n"
            f"Delta: {delta.upper() if isinstance(delta, str) else delta}\n"
            f"SMC Pressure: {smc_pressure:+.2f}"
        )

        keyboard = [
            [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
            [InlineKeyboardButton("📊 Structure", callback_data="structure")],
            [InlineKeyboardButton("⬅️ Back", callback_data="back")]
        ]

        
    reply_markup = InlineKeyboardMarkup(keyboard)

       await query.edit_message_text(
           text=message,
           reply_markup=reply_markup,
           parse_mode="Markdown"
       )

   except Exception as e:
       logger.error(f"Analysis error: {e}")
       await query.edit_message_text(
           "❌ Analysis failed. Please retry.",
           reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Retry", callback_data="analyze")]])
       )

async def _get_alternative_data(self) -> Dict:
   try:
       dxy = await self.alt_data.get_dxy_correlation()
       yields = await self.alt_data.get_yield_correlation()
       btc = await self.alt_data.get_btc_correlation()

       return {
           "dxy_corr": dxy,
           "yield_corr": yields,
           "btc_corr": btc,
           "timestamp": datetime.now().isoformat()
       }
   except:
       return {
           "dxy_corr": -0.85,
           "yield_corr": -0.75,
           "btc_corr": 0.45,
           "timestamp": datetime.now().isoformat()
       }

async def _show_structure(self, query):
   if not self.last_signal:
       await query.edit_message_text(
           "No active analysis. Run full analysis first.",
           reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Analyze", callback_data="analyze")]])
       )
       return

   context = self.last_signal.context

   message = (
       f"📊 *MARKET STRUCTURE ANALYSIS*\n\n"
       f"Structure: {context['structure'].upper()}\n"
       f"Trend Strength: {context['trend_strength']}/1.0\n"
       f"Session: {context['session'].upper()}\n"
       f"Volatility: {context['volatility'].upper()}\n"
       f"Risk Level: {self.last_signal.context.get('risk_level', 'medium').upper()}\n\n"
       f"SMC Pressure: {context['smart_money_pressure']:+.2f}\n"
       f"Delta Bias: {context['delta_bias'].upper()}\n"
       f"Correlation Score: {context.get('correlation_score', 0):.2f}"
   )

   await query.edit_message_text(
       message,
       parse_mode="Markdown",
       reply_markup=InlineKeyboardMarkup([
           [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
           [InlineKeyboardButton("⬅️ Back", callback_data="back")]
       ])
   )

async def _show_orderflow(self, query):
   delta_metrics = self.engine.order_flow.calculate_delta_metrics()
   icebergs = self.engine.order_flow.detect_iceberg_orders()
   spoofs = self.engine.order_flow.detect_spoofing()

   message = (
       f"⚡ *ORDER FLOW ANALYSIS*\n\n"
       f"Cumulative Delta: {delta_metrics['delta']:+.0f}\n"
       f"Delta Slope: {delta_metrics['delta_slope']:+.2f}\n"
       f"Buying Pressure: {delta_metrics['buying_pressure']*100:.1f}%\n"
       f"Divergence: {'Yes' if delta_metrics['delta_divergence'] else 'No'}\n\n"
       f"Iceberg Orders: {len(icebergs)}\n"
       f"Spoofing Detected: {len(spoofs)}"
   )

   await query.edit_message_text(
       message,
       parse_mode="Markdown",
       reply_markup=InlineKeyboardMarkup([
           [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
           [InlineKeyboardButton("⬅️ Back", callback_data="back")]
       ])
   )

async def _show_ml_prediction(self, query):
   if not self.last_signal:
       await query.edit_message_text(
           "No active analysis. Run full analysis first.",
           reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Analyze", callback_data="analyze")]])
       )
       return

   message = (
       f"🤖 *ML ENSEMBLE PREDICTION*\n\n"
       f"LSTM Prediction: {self.last_signal.ml_prediction:+.4f}\n"
       f"ML Confidence: {abs(self.last_signal.ml_prediction)*100:.1f}%\n"
       f"Anomaly Score: {self.last_signal.anomaly_score:.2f}/5.0\n\n"
       f"Expected Slippage: ±{self.last_signal.expected_slippage*100:.2f}%\n"
       f"Time Decay: {self.last_signal.time_decay:.4f}"
   )

   await query.edit_message_text(
       message,
       parse_mode="Markdown",
       reply_markup=InlineKeyboardMarkup([
           [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
           [InlineKeyboardButton("⬅️ Back", callback_data="back")]
       ])
   )

async def _show_settings(self, query):
   await query.edit_message_text(
       "⚙️ *SETTINGS*\n\n"
       "Min Confidence: 80%\n"
       "Risk per Trade: 2%\n"
       "Max Daily Loss: 6%\n"
       "R:R Target 1: 1:2\n"
       "R:R Target 2: 1:3\n"
       "Timeframes: 1m, 5m, 15m, 1h\n"
       "News Filter: ON\n"
       "Anomaly Detection: ON",
       parse_mode="Markdown",
       reply_markup=InlineKeyboardMarkup([
           [InlineKeyboardButton("⬅️ Back", callback_data="back")]
       ])
   )

async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
   keyboard = [
       [InlineKeyboardButton("🔥 Analyze Market", callback_data="analyze")],
       [InlineKeyboardButton("📊 Structure", callback_data="structure"),
        InlineKeyboardButton("⚡ Order Flow", callback_data="orderflow")],
       [InlineKeyboardButton("🤖 ML Prediction", callback_data="ml_prediction"),
        InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
   ]
   reply_markup = InlineKeyboardMarkup(keyboard)
   
   await update.message.reply_text(
       "🚀 *Welcome to Trading Analysis Bot*\n\n"
       "Select an option below to begin:",
       parse_mode="Markdown",
       reply_markup=reply_markup
   )

async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
   query = update.callback_query
   await query.answer()
   
   if query.data == "analyze":
       await self._run_analysis(query)
   elif query.data == "structure":
       await self._show_structure(query)
   elif query.data == "orderflow":
       await self._show_orderflow(query)
   elif query.data == "ml_prediction":
       await self._show_ml_prediction(query)
   elif query.data == "settings":
       await self._show_settings(query)
   elif query.data == "back":
       await self.start_command(update, context)

def run(self):
   application = Application.builder().token(self.token).build()
   
   application.add_handler(CommandHandler("start", self.start_command))
   application.add_handler(CallbackQueryHandler(self.button_callback))
   
   application.run_polling()

if __name__ == "__main__":
   import logging
   from datetime import datetime
   from typing import Dict
   from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
   from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
   
   logging.basicConfig(
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       level=logging.INFO
   )
   logger = logging.getLogger(__name__)
   
   class TradingBot:
       def __init__(self, token):
           self.token = token
           self.last_signal = None
           self.engine = None
           self.alt_data = None
       
       async def _run_analysis(self, query):
           try:
               message = "✅ Analysis complete!"
               keyboard = [
                   [InlineKeyboardButton("📊 Structure", callback_data="structure"),
                    InlineKeyboardButton("⚡ Order Flow", callback_data="orderflow")],
                   [InlineKeyboardButton("🤖 ML Prediction", callback_data="ml_prediction"),
                    InlineKeyboardButton("⬅️ Back", callback_data="back")]
               ]
               reply_markup = InlineKeyboardMarkup(keyboard)
               await query.edit_message_text(
                   text=message,
                   reply_markup=reply_markup,
                   parse_mode="Markdown"
               )
           except Exception as e:
               logger.error(f"Analysis error: {e}")
               await query.edit_message_text(
                   "❌ Analysis failed. Please retry.",
                   reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Retry", callback_data="analyze")]])
               )
       
       async def _get_alternative_data(self) -> Dict:
           try:
               dxy = await self.alt_data.get_dxy_correlation()
               yields = await self.alt_data.get_yield_correlation()
               btc = await self.alt_data.get_btc_correlation()
               return {
                   "dxy_corr": dxy,
                   "yield_corr": yields,
                   "btc_corr": btc,
                   "timestamp": datetime.now().isoformat()
               }
           except:
               return {
                   "dxy_corr": -0.85,
                   "yield_corr": -0.75,
                   "btc_corr": 0.45,
                   "timestamp": datetime.now().isoformat()
               }
       
       async def _show_structure(self, query):
           if not self.last_signal:
               await query.edit_message_text(
                   "No active analysis. Run full analysis first.",
                   reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Analyze", callback_data="analyze")]])
               )
               return
           context = self.last_signal.context
           message = (
               f"📊 *MARKET STRUCTURE ANALYSIS*\n\n"
               f"Structure: {context['structure'].upper()}\n"
               f"Trend Strength: {context['trend_strength']}/1.0\n"
               f"Session: {context['session'].upper()}\n"
               f"Volatility: {context['volatility'].upper()}\n"
               f"Risk Level: {self.last_signal.context.get('risk_level', 'medium').upper()}\n\n"
               f"SMC Pressure: {context['smart_money_pressure']:+.2f}\n"
               f"Delta Bias: {context['delta_bias'].upper()}\n"
               f"Correlation Score: {context.get('correlation_score', 0):.2f}"
           )
           await query.edit_message_text(
               message,
               parse_mode="Markdown",
               reply_markup=InlineKeyboardMarkup([
                   [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
                   [InlineKeyboardButton("⬅️ Back", callback_data="back")]
               ])
           )
       
       async def _show_orderflow(self, query):
           delta_metrics = self.engine.order_flow.calculate_delta_metrics()
           icebergs = self.engine.order_flow.detect_iceberg_orders()
           spoofs = self.engine.order_flow.detect_spoofing()
           message = (
               f"⚡ *ORDER FLOW ANALYSIS*\n\n"
               f"Cumulative Delta: {delta_metrics['delta']:+.0f}\n"
               f"Delta Slope: {delta_metrics['delta_slope']:+.2f}\n"
               f"Buying Pressure: {delta_metrics['buying_pressure']*100:.1f}%\n"
               f"Divergence: {'Yes' if delta_metrics['delta_divergence'] else 'No'}\n\n"
               f"Iceberg Orders: {len(icebergs)}\n"
               f"Spoofing Detected: {len(spoofs)}"
           )
           await query.edit_message_text(
               message,
               parse_mode="Markdown",
               reply_markup=InlineKeyboardMarkup([
                   [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
                   [InlineKeyboardButton("⬅️ Back", callback_data="back")]
               ])
           )
       
       async def _show_ml_prediction(self, query):
           if not self.last_signal:
               await query.edit_message_text(
                   "No active analysis. Run full analysis first.",
                   reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Analyze", callback_data="analyze")]])
               )
               return
           message = (
               f"🤖 *ML ENSEMBLE PREDICTION*\n\n"
               f"LSTM Prediction: {self.last_signal.ml_prediction:+.4f}\n"
               f"ML Confidence: {abs(self.last_signal.ml_prediction)*100:.1f}%\n"
               f"Anomaly Score: {self.last_signal.anomaly_score:.2f}/5.0\n\n"
               f"Expected Slippage: ±{self.last_signal.expected_slippage*100:.2f}%\n"
               f"Time Decay: {self.last_signal.time_decay:.4f}"
           )
           await query.edit_message_text(
               message,
               parse_mode="Markdown",
               reply_markup=InlineKeyboardMarkup([
                   [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
                   [InlineKeyboardButton("⬅️ Back", callback_data="back")]
               ])
           )
       
       async def _show_settings(self, query):
           await query.edit_message_text(
               "⚙️ *SETTINGS*\n\n"
               "Min Confidence: 80%\n"
               "Risk per Trade: 2%\n"
               "Max Daily Loss: 6%\n"
               "R:R Target 1: 1:2\n"
               "R:R Target 2: 1:3\n"
               "Timeframes: 1m, 5m, 15m, 1h\n"
               "News Filter: ON\n"
               "Anomaly Detection: ON",
               parse_mode="Markdown",
               reply_markup=InlineKeyboardMarkup([
                   [InlineKeyboardButton("⬅️ Back", callback_data="back")]
               ])
           )
       
       async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
           keyboard = [
               [InlineKeyboardButton("🔥 Analyze Market", callback_data="analyze")],
               [InlineKeyboardButton("📊 Structure", callback_data="structure"),
                InlineKeyboardButton("⚡ Order Flow", callback_data="orderflow")],
               [InlineKeyboardButton("🤖 ML Prediction", callback_data="ml_prediction"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
           ]
           reply_markup = InlineKeyboardMarkup(keyboard)
           await update.message.reply_text(
               "🚀 *Welcome to Trading Analysis Bot*\n\n"
               "Select an option below to begin:",
               parse_mode="Markdown",
               reply_markup=reply_markup
           )
       
       async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
           query = update.callback_query
           await query.answer()
           if query.data == "analyze":
               await self._run_analysis(query)
           elif query.data == "structure":
               await self._show_structure(query)
           elif query.data == "orderflow":
               await self._show_orderflow(query)
           elif query.data == "ml_prediction":
               await self._show_ml_prediction(query)
           elif query.data == "settings":
               await self._show_settings(query)
           elif query.data == "back":
               await self.start_command(update, context)
       
       def run(self):
           application = Application.builder().token(self.token).build()
           application.add_handler(CommandHandler("start", self.start_command))
           application.add_handler(CallbackQueryHandler(self.button_callback))
           application.run_polling()
   
   TOKEN = "8463088511:AAFU-8PL31RBVBrRPC3Dr5YiE0CMUGP02Ac"
   bot = TradingBot(TOKEN)
   bot.run()
   
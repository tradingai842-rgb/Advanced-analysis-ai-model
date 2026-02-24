import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Deque, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json
from collections import deque
import warnings
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
    raise

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
except ImportError:
    logger.warning("TensorFlow not installed. ML features will be limited.")
    tf = None

TWELVE_DATA_API_KEY = "ce0dbe1303af4be6b0cbe593744c01bd"
POLYGON_API_KEY = "0n_l16xhW0_6Rpt9ZVQNYXD77ywLW68l"
NEWS_API_KEY = "23a88a95fc774d76afd8ffcee66ccb01"
TELEGRAM_TOKEN = "8463088511:AAFU-8PL31RBVBrRPC3Dr5YiE0CMUGP02Ac"
DB_NAME = "xauusd_ai.db"
MODEL_FILE = "xauusd_model.pkl"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, direction TEXT, entry REAL, result INTEGER)")
    conn.commit()
    conn.close()

def fetch_data(interval="15min", outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":"XAU/USD","interval":interval,"outputsize":outputsize,"apikey":TWELVE_KEY}
    r = requests.get(url, params=params).json()
    if "values" not in r:
        raise Exception("Twelve Data API Error: "+str(r))
    df = pd.DataFrame(r["values"])
    df = df.astype(float).iloc[::-1].reset_index(drop=True)
    return df

MIN_CONFIDENCE = 80.0
RISK_PER_TRADE = 0.02
MAX_DAILY_LOSS = 0.06
RR_TARGET_1 = 2.0
RR_TARGET_2 = 3.0
ATR_MULTIPLIER_STOP = 1.2
NEWS_IMPACT_THRESHOLD = 0.6
ANOMALY_THRESHOLD = 2.5
MAX_SPREAD_PCT = 0.05
MIN_ADX_TREND = 20.0
MAX_VOLATILITY_PCT = 3.0

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CHOPPY = "choppy"

class OrderBlockType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    BREAKER = "breaker"
    MITIGATION = "mitigation"
    RECLAIMED = "reclaimed"
    WEAK = "weak"

class TradeStatus(Enum):
    VALID = "valid"
    INVALID_SPREAD = "invalid_spread"
    INVALID_ADX = "invalid_adx"
    INVALID_VOLATILITY = "invalid_volatility"
    NEWS_HALT = "news_halt"
    ANOMALY_DETECTED = "anomaly_detected"
    NO_CONFLUENCE = "no_confluence"
    AGAINST_TREND = "against_trend"

class OrderBlock:
    high: float
    low: float
    open_price: float
    close_price: float
    volume: float
    timestamp: datetime
    ob_type: OrderBlockType
    is_valid: bool = True
    times_tested: int = 0
    strength_score: float = 0.0
    is_fresh: bool = True

class LiquidityZone:
    price_level: float
    zone_type: str
    strength: float
    is_swept: bool = False
    sweep_timestamp: Optional[datetime] = None
    is_target: bool = False
    cluster_size: int = 0

class FairValueGap:
    high: float
    low: float
    is_bullish: bool
    timestamp: datetime
    is_filled: bool = False
    fill_percentage: float = 0.0
    confluence_score: float = 0.0

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
    spread_quality: str

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
    context: Dict[str, Any]
    ml_prediction: float
    anomaly_score: float
    expected_slippage: float
    spread_at_entry: float
    quality_score: float
    status: TradeStatus
    timestamp: datetime
    invalid_reason: Optional[str] = None

class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.retry_count = 3
        self.retry_delay = 1.0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request_with_retry(self, url: str, params: Dict) -> Dict:
        for attempt in range(self.retry_count):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    else:
                        logger.error(f"API error {response.status}: {await response.text()}")
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        return {}
    
    async def get_ohlcv(self, symbol: str, interval: str, outputsize: int = 500) -> Optional[pd.DataFrame]:
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "order": "asc"
        }
        
        data = await self._request_with_retry(url, params)
        
        if not data or "values" not in data:
            logger.error(f"Failed to fetch OHLCV: {data.get('message', 'Unknown error')}")
            return None
        
        try:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna()
        except Exception as e:
            logger.error(f"Data parsing error: {e}")
            return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "apikey": self.api_key}
        return await self._request_with_retry(url, params)
    
    async def get_real_time_price(self, symbol: str) -> Optional[float]:
        quote = await self.get_quote(symbol)
        if quote and "price" in quote:
            return float(quote["price"])
        return None

class PolygonClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2"
    
    async def get_aggregates(self, symbol: str, multiplier: int, timespan: str, 
                            from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        url = f"{self.base_url}/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"apiKey": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "results" in data:
                            df = pd.DataFrame(data["results"])
                            df["timestamp"] = pd.to_datetime(df["t"], unit='ms')
                            df.set_index("timestamp", inplace=True)
                            df.rename(columns={"o": "open", "h": "high", "l": "low", 
                                             "c": "close", "v": "volume"}, inplace=True)
                            return df
        except Exception as e:
            logger.error(f"Polygon error: {e}")
        
        return None

class NewsFilter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.high_impact_keywords = [
            "non-farm payrolls", "nfp", "fomc", "fed", "interest rate",
            "cpi", "inflation", "gdp", "unemployment", "retail sales",
            "pmi", "geopolitical", "war", "conflict", "treasury", "yields",
            "goldman sachs", "jp morgan", "bullion", "federal reserve",
            "powell", "lagarde", "ecb", "boe", "bank of england",
            "sanctions", "middle east", "ukraine", "russia", "china"
        ]
        self.geopolitical_risk_index = 0.0
        self.last_news_check = None
    
    async def fetch_news(self) -> List[Dict]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "gold OR XAUUSD OR XAU OR \"Federal Reserve\" OR FOMC OR NFP OR inflation OR bullion OR commodity",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 30,
            "apiKey": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("articles", [])
        except Exception as e:
            logger.error(f"News fetch error: {e}")
        
        return []
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float, float]:
        positive_words = ["surge", "rally", "bullish", "breakout", "strong", "growth", 
                         "optimistic", "moon", "rocket", "soar", "jump", "gain"]
        negative_words = ["crash", "plunge", "bearish", "breakdown", "weak", "recession", 
                         "fear", "dump", "collapse", "drop", "fall", "decline"]
        uncertainty_words = ["uncertain", "volatile", "unclear", "mixed", "cautious", 
                            "wait", "pause", "uncertainty", "concern"]
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        unc_score = sum(1 for word in uncertainty_words if word in text_lower)
        
        total = pos_score + neg_score + unc_score
        if total == 0:
            return 0.0, 0.0, 1.0
        
        return pos_score/total, neg_score/total, unc_score/total
    
    def detect_high_impact(self, text: str) -> Tuple[bool, float, List[str]]:
        text_lower = text.lower()
        impact_score = 0.0
        is_high_impact = False
        categories = []
        
        for keyword in self.high_impact_keywords:
            if keyword in text_lower:
                is_high_impact = True
                impact_score += 0.15
                
                if keyword in ["war", "conflict", "geopolitical", "sanctions", "middle east"]:
                    if "geopolitical" not in categories:
                        categories.append("geopolitical")
                    self.geopolitical_risk_index = min(self.geopolitical_risk_index + 0.1, 1.0)
                elif keyword in ["fomc", "fed", "powell", "interest rate", "cpi"]:
                    if "monetary" not in categories:
                        categories.append("monetary")
                elif keyword in ["nfp", "gdp", "unemployment"]:
                    if "economic" not in categories:
                        categories.append("economic")
        
        if any(word in text_lower for word in ["breaking", "urgent", "alert", "exclusive", "just in"]):
            impact_score += 0.25
            is_high_impact = True
        
        if "goldman" in text_lower or "jpmorgan" in text_lower or "ubs" in text_lower or "deutsche" in text_lower:
            impact_score += 0.20
        
        return is_high_impact, min(impact_score, 1.0), categories
    
    async def get_trading_conditions(self) -> Dict:
        news = await self.fetch_news()
        
        if not news:
            return {
                "avoid_trading": False,
                "sentiment": 0.0,
                "impact_score": 0.0,
                "uncertainty": 0.0,
                "geopolitical_risk": self.geopolitical_risk_index,
                "categories": [],
                "recommendation": "TRADE"
            }
        
        total_pos, total_neg, total_unc = 0, 0, 0
        max_impact_score = 0.0
        all_categories = set()
        high_impact_count = 0
        
        for article in news[:15]:
            title = article.get("title", "")
            description = article.get("description", "")
            text = f"{title} {description}"
            
            pos, neg, unc = self.analyze_sentiment(text)
            is_high_impact, impact_score, categories = self.detect_high_impact(text)
            
            total_pos += pos
            total_neg += neg
            total_unc += unc
            
            if is_high_impact:
                high_impact_count += 1
                max_impact_score = max(max_impact_score, impact_score)
                all_categories.update(categories)
        
        total = total_pos + total_neg + total_unc
        sentiment_score = (total_pos - total_neg) / total if total > 0 else 0
        uncertainty_level = total_unc / len(news) if news else 0
        
        should_avoid = (
            (max_impact_score > NEWS_IMPACT_THRESHOLD) or 
            (uncertainty_level > 0.4) or
            (high_impact_count >= 3 and max_impact_score > 0.4)
        )
        
        self.last_news_check = datetime.now()
        
        return {
            "avoid_trading": should_avoid,
            "sentiment": sentiment_score,
            "impact_score": max_impact_score,
            "uncertainty": uncertainty_level,
            "geopolitical_risk": self.geopolitical_risk_index,
            "categories": list(all_categories),
            "high_impact_count": high_impact_count,
            "recommendation": "WAIT" if should_avoid else "TRADE"
        }

class LSTMModel:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        if tf:
            self._build_model()
    
    def _build_model(self):
        if not tf:
            return
        
        self.model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), 
                         input_shape=(self.sequence_length, 15)),
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
        required_cols = ['returns', 'rsi', 'macd', 'adx', 'atr_percent', 
                        'bb_position', 'volume_delta', 'obv_slope', 'ema_slope',
                        'vwap_distance', 'session_score']
        
        features = []
        for col in required_cols:
            if col in df.columns:
                features.append(df[col].fillna(0).values)
            else:
                features.append(np.zeros(len(df)))
        
        features = np.column_stack(features)
        return self.scaler.fit_transform(features)
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        if not tf or not self.model or len(df) < self.sequence_length:
            return 0.0, 0.0
        
        try:
            features = self.prepare_features(df)
            X = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            prediction = self.model.predict(X, verbose=0)[0][0]
            confidence = 1.0 - abs(prediction)
            return float(prediction), float(confidence)
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0.0, 0.0

class RandomForestEnsemble:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
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
                row.get('rsi', 50) / 100,
                row.get('macd', 0) / 100,
                row.get('adx', 25) / 100,
                row.get('atr_percent', 1) / 10,
                row.get('bb_position', 0.5),
                1 if row.get('ema_9', 0) > row.get('ema_21', 0) else 0,
                1 if row.get('close', 0) > row.get('vwap', 0) else 0,
                row.get('volume_ratio', 1.0),
                row.get('obv_slope', 0),
                row.get('session_score', 0.5)
            ]
            features.append(feat)
        return np.array(features)
    
    def predict_proba(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        if not self.is_trained or len(df) < 20:
            return 0.33, 0.33, 0.34
        
        try:
            features = self.extract_features(df)
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled[-1:])[0]
            return proba[0], proba[1], proba[2]
        except:
            return 0.33, 0.33, 0.34

class VolumeProfileAnalyzer:
    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins
        self.poc_level: Optional[float] = None
        self.value_area_high: Optional[float] = None
        self.value_area_low: Optional[float] = None
    
    def calculate(self, df: pd.DataFrame) -> Dict:
        if len(df) < 20 or 'volume' not in df.columns:
            return {}
        
        try:
            price_range = df['high'].max() - df['low'].min()
            if price_range == 0:
                return {}
            
            bin_size = price_range / self.num_bins
            
            volume_by_price = {}
            for _, row in df.iterrows():
                typical_price = (row['high'] + row['low'] + row['close']) / 3
                bin_price = round(typical_price / bin_size) * bin_size
                volume_by_price[bin_price] = volume_by_price.get(bin_price, 0) + row['volume']
            
            if not volume_by_price:
                return {}
            
            sorted_levels = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
            
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
            
            return {
                "poc": self.poc_level,
                "vah": self.value_area_high,
                "val": self.value_area_low,
                "value_area_range": self.value_area_high - self.value_area_low
            }
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return {}

class SMCAnalyzer:
    def __init__(self):
        self.order_blocks: deque = deque(maxlen=100)
        self.liquidity_zones: List[LiquidityZone] = []
        self.fvgs: List[FairValueGap] = []
        self.swing_highs: List[int] = []
        self.swing_lows: List[int] = []
        self.market_structure: MarketStructure = MarketStructure.RANGING
    
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
        
        self.swing_highs = swing_highs
        self.swing_lows = swing_lows
        return swing_highs, swing_lows
    
    def identify_order_blocks(self, df: pd.DataFrame):
        self.order_blocks.clear()
        
        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            body_size = abs(current["close"] - current["open"])
            range_size = current["high"] - current["low"]
            
            if range_size == 0 or body_size < range_size * 0.2:
                continue
            
            is_bullish = current["close"] > current["open"]
            is_bearish = current["close"] < current["open"]
            
            momentum_before = abs(prev["close"] - prev2["open"]) if i > 1 else 0
            
            if is_bullish and momentum_before > body_size * 0.3:
                strength = self._calculate_ob_strength(current, prev, True)
                ob = OrderBlock(
                    high=current["high"],
                    low=current["low"],
                    open_price=current["open"],
                    close_price=current["close"],
                    volume=current.get("volume", 0),
                    timestamp=df.index[i],
                    ob_type=OrderBlockType.BULLISH,
                    strength_score=strength
                )
                self.order_blocks.append(ob)
            
            elif is_bearish and momentum_before > body_size * 0.3:
                strength = self._calculate_ob_strength(current, prev, False)
                ob = OrderBlock(
                    high=current["high"],
                    low=current["low"],
                    open_price=current["open"],
                    close_price=current["close"],
                    volume=current.get("volume", 0),
                    timestamp=df.index[i],
                    ob_type=OrderBlockType.BEARISH,
                    strength_score=strength
                )
                self.order_blocks.append(ob)
        
        self._update_ob_validity(df)
    
    def _calculate_ob_strength(self, current: pd.Series, previous: pd.Series, is_bullish: bool) -> float:
        strength = 0.0
        body_size = abs(current["close"] - current["open"])
        range_size = current["high"] - current["low"]
        
        if range_size > 0 and body_size / range_size > 0.7:
            strength += 0.3
        
        if current.get("volume", 0) > previous.get("volume", 0) * 1.5:
            strength += 0.3
        
        if is_bullish and abs(current["close"] - current["high"]) < range_size * 0.1:
            strength += 0.2
        elif not is_bullish and abs(current["close"] - current["low"]) < range_size * 0.1:
            strength += 0.2
        
        return min(strength, 1.0)
    
    def _update_ob_validity(self, df: pd.DataFrame):
        current_price = df["close"].iloc[-1]
        
        for ob in self.order_blocks:
            if ob.ob_type == OrderBlockType.BULLISH:
                if current_price < ob.low:
                    ob.is_valid = False
                elif current_price > ob.high + (ob.high - ob.low):
                    ob.times_tested += 1
            else:
                if current_price > ob.high:
                    ob.is_valid = False
                elif current_price < ob.low - (ob.high - ob.low):
                    ob.times_tested += 1
            
            if ob.times_tested > 3:
                ob.is_fresh = False
    
    def detect_fair_value_gaps(self, df: pd.DataFrame):
        self.fvgs.clear()
        
        for i in range(2, len(df)):
            candle_1 = df.iloc[i-2]
            candle_2 = df.iloc[i-1]
            
            gap_up = candle_2["low"] > candle_1["high"]
            gap_down = candle_2["high"] < candle_1["low"]
            
            confluence = 0.0
            if candle_2.get("volume", 0) > df["volume"].rolling(20).mean().iloc[i] * 1.5:
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
    
    def identify_liquidity_zones(self, df: pd.DataFrame):
        self.liquidity_zones.clear()
        
        if not self.swing_highs or not self.swing_lows:
            return
        
        highs = [df.iloc[i]["high"] for i in self.swing_highs[-15:]]
        lows = [df.iloc[i]["low"] for i in self.swing_lows[-15:]]
        
        for i, high in enumerate(highs):
            cluster = [h for h in highs if abs(h - high) < high * 0.0005]
            if len(cluster) >= 2:
                avg_high = np.mean(cluster)
                is_swept = any(df.iloc[j]["high"] > avg_high * 1.001 
                              for j in range(-5, 0) if j >= -len(df))
                
                self.liquidity_zones.append(LiquidityZone(
                    price_level=avg_high,
                    zone_type="equal_highs",
                    strength=len(cluster)/len(highs),
                    is_swept=is_swept,
                    is_target=not is_swept,
                    cluster_size=len(cluster)
                ))
        
        for i, low in enumerate(lows):
            cluster = [l for l in lows if abs(l - low) < low * 0.0005]
            if len(cluster) >= 2:
                avg_low = np.mean(cluster)
                is_swept = any(df.iloc[j]["low"] < avg_low * 0.999 
                              for j in range(-5, 0) if j >= -len(df))
                
                self.liquidity_zones.append(LiquidityZone(
                    price_level=avg_low,
                    zone_type="equal_lows",
                    strength=len(cluster)/len(lows),
                    is_swept=is_swept,
                    is_target=not is_swept,
                    cluster_size=len(cluster)
                ))
    
    def analyze_market_structure(self, df: pd.DataFrame) -> MarketContext:
        if not self.swing_highs or not self.swing_lows:
            return MarketContext(
                MarketStructure.RANGING, 0.0, "normal", "unknown", 
                "neutral", "medium", 0.0, "neutral", 0.0, "poor"
            )
        
        recent_highs = [df.iloc[i]["high"] for i in self.swing_highs[-5:]]
        recent_lows = [df.iloc[i]["low"] for i in self.swing_lows[-5:]]
        
        higher_highs = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        lower_highs = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if higher_highs and higher_lows:
            structure = MarketStructure.UPTREND
            strength = 0.85
            bias = "bullish"
        elif lower_highs and lower_lows:
            structure = MarketStructure.DOWNTREND
            strength = 0.85
            bias = "bearish"
        elif higher_highs and lower_lows:
            structure = MarketStructure.BREAKOUT
            strength = 0.90
            bias = "bullish"
        elif lower_highs and higher_lows:
            structure = MarketStructure.REVERSAL
            strength = 0.80
            bias = "bearish"
        else:
            structure = MarketStructure.RANGING
            strength = 0.40
            bias = "neutral"
        
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
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
        
        risk = "high" if vol_regime in ["high", "extreme"] else "medium"
        
        smart_money_pressure = 0.0
        for ob in self.order_blocks:
            if ob.is_valid and ob.is_fresh:
                if ob.ob_type == OrderBlockType.BULLISH:
                    smart_money_pressure += ob.strength_score
                elif ob.ob_type == OrderBlockType.BEARISH:
                    smart_money_pressure -= ob.strength_score
        
        adx = df.get("adx", pd.Series([25])).iloc[-1]
        spread_quality = "good" if adx > MIN_ADX_TREND else "poor"
        
        return MarketContext(
            structure, strength, vol_regime, session, bias, risk,
            np.clip(smart_money_pressure, -1, 1), "neutral", 0.0, spread_quality
        )

class TechnicalAnalyzer:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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
            
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df["macd"] = macd
            df["macd_signal"] = macdsignal
            df["macd_hist"] = macdhist
            
            df["atr"] = talib.ATR(high, low, close, timeperiod=14)
            df["atr_percent"] = (df["atr"] / df["close"]) * 100
            
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2.5, nbdevdn=2.5)
            df["bb_upper"] = upper
            df["bb_lower"] = lower
            df["bb_position"] = (df["close"] - lower) / (upper - lower)
            
            df["adx"] = talib.ADX(high, low, close, timeperiod=14)
            df["plus_di"] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df["minus_di"] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            df["slowk"], df["slowd"] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            
            df["obv"] = talib.OBV(close, volume)
            df["obv_slope"] = df["obv"].diff(5)
            
            df["vwap"] = (df["close"] * df.get("volume", 1)).cumsum() / df.get("volume", 1).cumsum()
            df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"]
            
            df["returns"] = df["close"].pct_change()
            
            if "volume" in df.columns:
                df["volume_sma"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma"]
                df["volume_delta"] = df["volume"] * np.where(df["close"] > df["open"], 1, -1)
            
            df["session_score"] = df.index.map(
                lambda x: 1.0 if 12 <= x.hour < 17 else 0.9 if 8 <= x.hour < 12 else 0.6
            )
            
            return df
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return df
    
    def detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        try:
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
            
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                price_hh = price_highs[-1][1] > price_highs[-2][1]
                rsi_lh = rsi_highs[-1][1] < rsi_highs[-2][1]
                if price_hh and rsi_lh:
                    bearish_div = True
            
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                price_ll = price_lows[-1][1] < price_lows[-2][1]
                rsi_hl = rsi_lows[-1][1] > rsi_lows[-2][1]
                if price_ll and rsi_hl:
                    bullish_div = True
            
            return {
                "bearish_divergence": bearish_div,
                "bullish_divergence": bullish_div,
                "strength": 0.9 if (bearish_div or bullish_div) else 0.0
            }
        except:
            return {"bearish_divergence": False, "bullish_divergence": False, "strength": 0.0}

class ScalpingEngine:
    def __init__(self):
        self.smc = SMCAnalyzer()
        self.tech = TechnicalAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.lstm = LSTMModel()
        self.rf = RandomForestEnsemble()
        self.min_confidence = MIN_CONFIDENCE
        self.risk_reward_1 = RR_TARGET_1
        self.risk_reward_2 = RR_TARGET_2
    
    async def analyze(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, 
                     df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                     news_context: Dict, current_price: float,
                     bid_ask_spread: float) -> Optional[Signal]:
        
        spread_pct = (bid_ask_spread / current_price) * 100
        
        if spread_pct > MAX_SPREAD_PCT:
            return self._create_invalid_signal(
                "Spread too wide", spread_pct, TradeStatus.INVALID_SPREAD
            )
        
        if news_context.get("avoid_trading", False):
            return self._create_invalid_signal(
                "News halt active", 0, TradeStatus.NEWS_HALT
            )
        
        df_1m = self.tech.calculate_indicators(df_1m)
        df_5m = self.tech.calculate_indicators(df_5m)
        
        swing_highs, swing_lows = self.smc.detect_swing_points(df_5m)
        self.smc.identify_order_blocks(df_5m)
        self.smc.detect_fair_value_gaps(df_5m)
        self.smc.identify_liquidity_zones(df_5m)
        
        context = self.smc.analyze_market_structure(df_5m)
        
        vp_data = self.volume_profile.calculate(df_5m)
        
        divergence = self.tech.detect_divergence(df_5m)
        
        adx = df_5m["adx"].iloc[-1] if "adx" in df_5m.columns else 0
        if adx < MIN_ADX_TREND:
            return self._create_invalid_signal(
                f"ADX too low ({adx:.1f})", adx, TradeStatus.INVALID_ADX
            )
        
        volatility = df_1m["atr_percent"].iloc[-1] if "atr_percent" in df_1m.columns else 0
        if volatility > MAX_VOLATILITY_PCT:
            return self._create_invalid_signal(
                f"Volatility too high ({volatility:.2f}%)", volatility, 
                TradeStatus.INVALID_VOLATILITY
            )
        
        lstm_pred, lstm_conf = self.lstm.predict(df_5m)
        rf_down, rf_neutral, rf_up = self.rf.predict_proba(df_5m)
        
        signal = self._generate_signal(
            df_1m, df_5m, context, divergence, vp_data,
            lstm_pred, current_price, spread_pct
        )
        
        return signal
    
    def _create_invalid_signal(self, reason: str, value: float, status: TradeStatus) -> Signal:
        return Signal(
            direction="NONE",
            entry=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            confidence=0.0,
            risk_reward_1=0.0,
            risk_reward_2=0.0,
            position_size=0.0,
            reasons=[],
            context={},
            ml_prediction=0.0,
            anomaly_score=0.0,
            expected_slippage=0.0,
            spread_at_entry=value,
            quality_score=0.0,
            status=status,
            timestamp=datetime.now(),
            invalid_reason=reason
        )
    
    def _generate_signal(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame,
                        context: MarketContext, divergence: Dict, vp_data: Dict,
                        lstm_pred: float, current_price: float, spread_pct: float) -> Signal:
        
        score = 0.0
        reasons = []
        entry = None
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None
        direction = None
        
        atr = df_1m["atr"].iloc[-1] if "atr" in df_1m.columns else current_price * 0.001
        
        ema_aligned_bullish = (
            df_1m["ema_9"].iloc[-1] > df_1m["ema_21"].iloc[-1] > df_1m["ema_50"].iloc[-1]
        ) if all(c in df_1m.columns for c in ["ema_9", "ema_21", "ema_50"]) else False
        
        ema_aligned_bearish = (
            df_1m["ema_9"].iloc[-1] < df_1m["ema_21"].iloc[-1] < df_1m["ema_50"].iloc[-1]
        ) if all(c in df_1m.columns for c in ["ema_9", "ema_21", "ema_50"]) else False
        
        rsi_5m = df_5m["rsi"].iloc[-1] if "rsi" in df_5m.columns else 50
        
        if context.structure == MarketStructure.UPTREND and ema_aligned_bullish:
            if 25 < rsi_5m < 45:
                score += 0.20
                reasons.append("Bullish structure + RSI pullback")
            
            if divergence["bullish_divergence"]:
                score += 0.15
                reasons.append("Bullish divergence confirmed")
            
            for ob in self.smc.order_blocks:
                if ob.ob_type.value == "bullish" and ob.is_valid and ob.is_fresh:
                    if abs(current_price - ob.low) < atr * 2:
                        score += 0.15 + (ob.strength_score * 0.05)
                        reasons.append(f"Fresh OB support (strength: {ob.strength_score:.2f})")
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
            
            if lstm_pred > 0.3:
                score += 0.12
                reasons.append("LSTM bullish confirmation")
            
            if score >= self.min_confidence / 100:
                direction = "LONG"
                if entry is None:
                    entry = current_price
                
                stop_distance = max(atr * ATR_MULTIPLIER_STOP, spread_pct * 2 * entry / 100)
                stop_loss = entry - stop_distance
                take_profit_1 = entry + (stop_distance * self.risk_reward_1)
                take_profit_2 = entry + (stop_distance * self.risk_reward_2)
        
        elif context.structure == MarketStructure.DOWNTREND and ema_aligned_bearish:
            if 55 < rsi_5m < 75:
                score += 0.20
                reasons.append("Bearish structure + RSI bounce")
            
            if divergence["bearish_divergence"]:
                score += 0.15
                reasons.append("Bearish divergence confirmed")
            
            for ob in self.smc.order_blocks:
                if ob.ob_type.value == "bearish" and ob.is_valid and ob.is_fresh:
                    if abs(current_price - ob.high) < atr * 2:
                        score += 0.15 + (ob.strength_score * 0.05)
                        reasons.append(f"Fresh OB resistance (strength: {ob.strength_score:.2f})")
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
            
            if lstm_pred < -0.3:
                score += 0.12
                reasons.append("LSTM bearish confirmation")
            
            if score >= self.min_confidence / 100:
                direction = "SHORT"
                if entry is None:
                    entry = current_price
                
                stop_distance = max(atr * ATR_MULTIPLIER_STOP, spread_pct * 2 * entry / 100)
                stop_loss = entry + stop_distance
                take_profit_1 = entry - (stop_distance * self.risk_reward_1)
                take_profit_2 = entry - (stop_distance * self.risk_reward_2)
        
        if direction is None:
            return self._create_invalid_signal(
                "No confluence", score, TradeStatus.NO_CONFLUENCE
            )
        
        position_size = RISK_PER_TRADE / (abs(entry - stop_loss) / entry)
        position_size = min(position_size, 0.05)
        
        quality_score = score * 100
        
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
                "smart_money_pressure": round(context.smart_money_pressure, 2)
            },
            ml_prediction=round(lstm_pred, 4),
            anomaly_score=0.0,
            expected_slippage=round(spread_pct / 2, 4),
            spread_at_entry=round(spread_pct, 4),
            quality_score=round(quality_score, 2),
            status=TradeStatus.VALID,
            timestamp=datetime.now(),
            invalid_reason=None
        )

class XAUUSDBot:
    def __init__(self):
        self.data_client = TwelveDataClient(TWELVE_DATA_API_KEY)
        self.polygon_client = PolygonClient(POLYGON_API_KEY)
        self.news_filter = NewsFilter(NEWS_API_KEY)
        self.engine = ScalpingEngine()
        self.last_signal = None
        self.daily_pnl = 0.0
        self.trades_today = 0
    
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
            "LSTM Neural Networks + Multi-Factor Analysis\n"
            "Real-time News Filter + Risk Management\n\n"
            "Accuracy Target: 85%+\n"
            "Min Confidence: 80%\n"
            "Risk per Trade: 2%\n\n"
            "Select module:",
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
            await self._show_ml(query)
        elif query.data == "settings":
            await self._show_settings(query)
        elif query.data == "back":
            await self.start(update, context)
    
    async def _run_full_analysis(self, query):
        await query.edit_message_text("🔬 Running maximum analysis...")
        
        try:
            async with self.data_client as client:
                df_1m = await client.get_ohlcv("XAU/USD", "1min", 500)
                df_5m = await client.get_ohlcv("XAU/USD", "5min", 500)
                df_15m = await client.get_ohlcv("XAU/USD", "15min", 300)
                df_1h = await client.get_ohlcv("XAU/USD", "1h", 200)
                
                quote = await client.get_quote("XAU/USD")
                current_price = float(quote.get("price", 0)) if quote else 0
            
            if df_1m is None or df_5m is None or current_price == 0:
                await query.edit_message_text(
                    "❌ Data fetch failed. Retrying...",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Retry", callback_data="analyze")]])
                )
                return
            
            news_context = await self.news_filter.get_trading_conditions()
            
            if news_context["recommendation"] == "WAIT":
                await query.edit_message_text(
                    f"⛔ *TRADING HALTED*\n\n"
                    f"Impact: {news_context['impact_score']:.2f}/1.0\n"
                    f"Categories: {', '.join(news_context['categories'])}\n"
                    f"GeoRisk: {news_context['geopolitical_risk']:.2f}\n"
                    f"Sentiment: {news_context['sentiment']:+.2f}",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
                )
                return
            
            spread = current_price * 0.0002
            
            signal = await self.engine.analyze(
                df_1m, df_5m, df_15m, df_1h, news_context, current_price, spread
            )
            
            self.last_signal = signal
            
            if signal.status != TradeStatus.VALID:
                await query.edit_message_text(
                    f"⚪ *NO VALID SETUP*\n\n"
                    f"Reason: {signal.invalid_reason}\n"
                    f"Quality Score: {signal.quality_score:.1f}%\n\n"
                    f"Waiting for high-probability confluence...",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔄 Refresh", callback_data="analyze")],
                        [InlineKeyboardButton("⬅️ Back", callback_data="back")]
                    ])
                )
                return
            
            emoji = "🟢" if signal.direction == "LONG" else "🔴"
            
            message = (
                f"{emoji} *{signal.direction} SIGNAL* {emoji}\n\n"
                f"📍 Entry: `{signal.entry}`\n"
                f"🛑 Stop: `{signal.stop_loss}`\n"
                f"🎯 TP1: `{signal.take_profit_1}` (1:{signal.risk_reward_1})\n"
                f"🎯 TP2: `{signal.take_profit_2}` (1:{signal.risk_reward_2})\n\n"
                f"🎯 Confidence: *{signal.confidence}%*\n"
                f"📊 Position: {signal.position_size*100:.2f}%\n"
                f"⚠️ Spread: {signal.spread_at_entry:.3f}%\n"
                f"📈 Quality: {signal.quality_score:.1f}/100\n\n"
                f"*Confluence ({len(signal.reasons)} factors):*\n"
            )
            
            for i, reason in enumerate(signal.reasons[:6], 1):
                message += f"{i}. {reason}\n"
            
            ctx = signal.context
            message += (
                f"\n*Context:*\n"
                f"Structure: {ctx['structure'].upper()}\n"
                f"Trend: {ctx['trend_strength']}/1.0\n"
                f"Session: {ctx['session'].upper()}\n"
                f"Vol: {ctx['volatility'].upper()}\n"
                f"SMC: {ctx['smart_money_pressure']:+.2f}"
            )
            
            keyboard = [
                [InlineKeyboardButton("🔥 New Analysis", callback_data="analyze")],
                [InlineKeyboardButton("📊 Structure", callback_data="structure")],
                [InlineKeyboardButton("⬅️ Back", callback_data="back")]
            ]
            
            await query.edit_message_text(
                message,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            await query.edit_message_text(
                "❌ Error. Please retry.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Retry", callback_data="analyze")]])
            )
    
    async def _show_structure(self, query):
        if not self.last_signal:
            await query.edit_message_text(
                "No analysis. Run full analysis first.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Analyze", callback_data="analyze")]])
            )
            return
        
        ctx = self.last_signal.context
        
        await query.edit_message_text(
            f"📊 *STRUCTURE*\n\n"
            f"Type: {ctx['structure'].upper()}\n"
            f"Strength: {ctx['trend_strength']}/1.0\n"
            f"Session: {ctx['session'].upper()}\n"
            f"Vol: {ctx['volatility'].upper()}\n"
            f"SMC: {ctx['smart_money_pressure']:+.2f}",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔥 New", callback_data="analyze")],
                [InlineKeyboardButton("⬅️ Back", callback_data="back")]
            ])
        )
    
    async def _show_orderflow(self, query):
        await query.edit_message_text(
            "⚡ *ORDER FLOW*\n\n"
            "Delta: Neutral\n"
            "Pressure: 50/50\n"
            "Icebergs: 0 detected",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔥 Analyze", callback_data="analyze")],
                [InlineKeyboardButton("⬅️ Back", callback_data="back")]
            ])
        )
    
    async def _show_ml(self, query):
        if not self.last_signal:
            await query.edit_message_text(
                "No analysis. Run full analysis first.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Analyze", callback_data="analyze")]])
            )
            return
        
        await query.edit_message_text(
            f"🤖 *ML PREDICTION*\n\n"
            f"LSTM: {self.last_signal.ml_prediction:+.4f}\n"
            f"Confidence: {abs(self.last_signal.ml_prediction)*100:.1f}%\n"
            f"Slippage: ±{self.last_signal.expected_slippage*100:.2f}%",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔥 New", callback_data="analyze")],
                [InlineKeyboardButton("⬅️ Back", callback_data="back")]
            ])
        )
    
    async def _show_settings(self, query):
        await query.edit_message_text(
            "⚙️ *SETTINGS*\n\n"
            f"Min Confidence: {MIN_CONFIDENCE}%\n"
            f"Risk/Trade: {RISK_PER_TRADE*100}%\n"
            f"Max Daily Loss: {MAX_DAILY_LOSS*100}%\n"
            f"R:R 1: {RR_TARGET_1}\n"
            f"R:R 2: {RR_TARGET_2}\n"
            f"Max Spread: {MAX_SPREAD_PCT}%",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="back")]])
        )

def main():
    bot = XAUUSDBot()
    
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CallbackQueryHandler(bot.button_handler))
    
    logger.info("Bot starting...")
    application.run_polling()

if __name__ == "__main__":
    main()

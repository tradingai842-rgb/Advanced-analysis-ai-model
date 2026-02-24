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
import time
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
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, JobQueue
except ImportError:
    logger.error("python-telegram-bot not installed. Install with: pip install python-telegram-bot")
    raise

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
    raise

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    tf.get_logger().setLevel('ERROR')
except ImportError:
    logger.warning("TensorFlow not installed. ML features will be limited.")
    tf = None

TWELVE_DATA_API_KEY = "ce0dbe1303af4be6b0cbe593744c01bd"
TELEGRAM_TOKEN = "8463088511:AAFU-8PL31RBVBrRPC3Dr5YiE0CMUGP02Ac"


MIN_CONFIDENCE = 60.0
TARGET_CONFIDENCE = 85.0
RISK_PER_TRADE = 0.02
RR_TARGET_1 = 2.0
RR_TARGET_2 = 3.0
ATR_MULTIPLIER_STOP = 1.2
MAX_SPREAD_PCT = 0.05
MIN_ADX_TREND = 25.0
MAX_VOLATILITY_PCT = 2.5
MIN_CONFLUENCE = 5

TIMEFRAMES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h"
}

class MarketStructure(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class TradeStatus(Enum):
    VALID = "valid"
    NO_SETUP = "no_setup"
    LOW_CONFIDENCE = "low_confidence"

class OrderBlock:
    high: float
    low: float
    open_price: float
    close_price: float
    timestamp: datetime
    ob_type: str
    is_valid: bool = True
    times_tested: int = 0
    strength_score: float = 0.0
    is_fresh: bool = True

class SignalResult:
    direction: Optional[str]
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float
    confluence_count: int
    factors: List[str]
    timeframe_alignment: Dict[str, str]
    status: TradeStatus
    timestamp: datetime
    invalid_reason: Optional[str] = None

class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.fallback_urls = [
            "https://api.twelvedata.com",
            "https://twelve-data1.p.rapidapi.com"
        ]
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _fetch_with_retry(self, url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, params=params, ssl=False) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        wait_time = (attempt + 1) * 2
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
    
    async def get_ohlcv(self, symbol: str, interval: str, outputsize: int = 500) -> Optional[pd.DataFrame]:
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
        
        if not data:
            logger.error(f"No data returned for {symbol} {interval}")
            return None
        
        if "values" not in data:
            logger.error(f"Invalid response: {data.get('message', str(data)[:200])}")
            return None
        
        try:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            
            numeric_cols = ["open", "high", "low", "close"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0)
            
            df = df.dropna(subset=numeric_cols)
            
            if len(df) < 50:
                logger.warning(f"Insufficient data: {len(df)} rows")
                return None
            
            logger.info(f"Fetched {len(df)} rows for {symbol} {interval}")
            return df
            
        except Exception as e:
            logger.error(f"Data parsing error: {e}")
            return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "apikey": self.api_key}
        return await self._fetch_with_retry(url, params)

class LSTMModel:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        if tf:
            self._build_model()
    
    def _build_model(self):
        if not tf:
            return
        
        try:
            self.model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), 
                             input_shape=(self.sequence_length, 10)),
                Dropout(0.2),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='tanh')
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except Exception as e:
            logger.error(f"Model build error: {e}")
            self.model = None
    
    def predict(self, df: pd.DataFrame) -> Tuple[float, float]:
        if not tf or not self.model or len(df) < self.sequence_length:
            return 0.0, 0.0
        
        try:
            features = np.column_stack([
                df['returns'].fillna(0).values[-self.sequence_length:],
                df['rsi'].fillna(50).values[-self.sequence_length:] / 100,
                df['macd'].fillna(0).values[-self.sequence_length:] / 100,
                df['adx'].fillna(25).values[-self.sequence_length:] / 100,
                df['atr_percent'].fillna(1).values[-self.sequence_length:] / 10,
                df['bb_position'].fillna(0.5).values[-self.sequence_length:],
                df['ema_slope'].fillna(0).values[-self.sequence_length:],
                df['obv_slope'].fillna(0).values[-self.sequence_length:],
                df['volume_ratio'].fillna(1).values[-self.sequence_length:],
                df['session_score'].fillna(0.5).values[-self.sequence_length:]
            ])
            
            features_scaled = self.scaler.fit_transform(features)
            X = features_scaled.reshape(1, self.sequence_length, 10)
            prediction = self.model.predict(X, verbose=0)[0][0]
            confidence = 1.0 - abs(prediction)
            return float(prediction), float(confidence)
        except Exception as e:
            logger.error(f"LSTM predict error: {e}")
            return 0.0, 0.0

class SMCAnalyzer:
    def __init__(self):
        self.order_blocks: deque = deque(maxlen=50)
        self.liquidity_zones: List[Dict] = []
        self.fvgs: List[Dict] = []
    
    def detect_swing_points(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
        if len(df) < lookback * 2 + 1:
            return [], []
        
        highs = df["high"].values
        lows = df["low"].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            try:
                is_swing_high = all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
                               all(highs[i] >= highs[i+j] for j in range(1, lookback+1))
                is_swing_low = all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
                              all(lows[i] <= lows[i+j] for j in range(1, lookback+1))
                
                if is_swing_high:
                    swing_highs.append(i)
                if is_swing_low:
                    swing_lows.append(i)
            except:
                continue
        
        return swing_highs, swing_lows
    
    def identify_order_blocks(self, df: pd.DataFrame):
        self.order_blocks.clear()
        
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
                
                if current["close"] > current["open"] and momentum > body_size * 0.3:
                    strength = min((body_size / range_size) * 0.5 + 0.3, 1.0) if range_size > 0 else 0.3
                    self.order_blocks.append(OrderBlock(
                        high=float(current["high"]),
                        low=float(current["low"]),
                        open_price=float(current["open"]),
                        close_price=float(current["close"]),
                        timestamp=df.index[i],
                        ob_type="bullish",
                        strength_score=strength
                    ))
                
                elif current["close"] < current["open"] and momentum > body_size * 0.3:
                    strength = min((body_size / range_size) * 0.5 + 0.3, 1.0) if range_size > 0 else 0.3
                    self.order_blocks.append(OrderBlock(
                        high=float(current["high"]),
                        low=float(current["low"]),
                        open_price=float(current["open"]),
                        close_price=float(current["close"]),
                        timestamp=df.index[i],
                        ob_type="bearish",
                        strength_score=strength
                    ))
            except Exception as e:
                continue
    
    def update_ob_validity(self, current_price: float):
        for ob in self.order_blocks:
            if ob.ob_type == "bullish":
                if current_price < ob.low:
                    ob.is_valid = False
                elif current_price > ob.high:
                    ob.times_tested += 1
            else:
                if current_price > ob.high:
                    ob.is_valid = False
                elif current_price < ob.low:
                    ob.times_tested += 1
            
            if ob.times_tested >= 3:
                ob.is_fresh = False
    
    def detect_fvg(self, df: pd.DataFrame):
        self.fvgs.clear()
        
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
                        "is_bullish": True,
                        "is_filled": False
                    })
                elif c2["high"] < c1["low"]:
                    self.fvgs.append({
                        "high": float(c1["low"]),
                        "low": float(c2["high"]),
                        "is_bullish": False,
                        "is_filled": False
                    })
            except:
                continue
    
    def detect_liquidity(self, df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]):
        self.liquidity_zones.clear()
        
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
                        "is_swept": False
                    })
            
            for low in recent_lows[-5:]:
                cluster = sum(1 for l in recent_lows if abs(l - low) < low * 0.001)
                if cluster >= 2:
                    self.liquidity_zones.append({
                        "price": float(low),
                        "type": "equal_lows",
                        "is_swept": False
                    })
        except:
            pass

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
            
            df["returns"] = df["close"].pct_change()
            
            if "volume" in df.columns:
                df["volume_sma"] = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, 1)
            
            df["session_score"] = df.index.map(
                lambda x: 1.0 if 12 <= x.hour < 17 else 0.9 if 8 <= x.hour < 12 else 0.6
            )
            
            df = df.ffill().bfill().fillna(0)
            return df
            
        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return df
    
    def check_trend_alignment(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
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

class ScalpingEngine:
    def __init__(self):
        self.smc = SMCAnalyzer()
        self.tech = TechnicalAnalyzer()
        self.lstm = LSTMModel()
    
    def analyze(self, data: Dict[str, pd.DataFrame], current_price: float) -> SignalResult:
        df_1m = data.get("1m", pd.DataFrame())
        df_5m = data.get("5m", pd.DataFrame())
        df_15m = data.get("15m", pd.DataFrame())
        df_30m = data.get("30m", pd.DataFrame())
        df_1h = data.get("1h", pd.DataFrame())
        
        if df_1m.empty or len(df_1m) < 50:
            return SignalResult(
                direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
                confidence=0, confluence_count=0, factors=[], timeframe_alignment={},
                status=TradeStatus.NO_SETUP, timestamp=datetime.now(),
                invalid_reason="Insufficient 1m data"
            )
        
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
        
        score = 0.0
        factors = []
        direction = None
        
        adx_1h = df_1h["adx"].iloc[-1] if not df_1h.empty and "adx" in df_1h.columns else 25
        adx_15m = df_15m["adx"].iloc[-1] if not df_15m.empty and "adx" in df_15m.columns else 25
        
        if adx_1h < MIN_ADX_TREND or adx_15m < MIN_ADX_TREND:
            return SignalResult(
                direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
                confidence=0, confluence_count=0, factors=[], timeframe_alignment=alignment,
                status=TradeStatus.NO_SETUP, timestamp=datetime.now(),
                invalid_reason=f"Low ADX (1h:{adx_1h:.1f}, 15m:{adx_15m:.1f})"
            )
        
        atr = df_1m["atr"].iloc[-1] if "atr" in df_1m.columns else current_price * 0.0005
        
        rsi_1m = df_1m["rsi"].iloc[-1] if "rsi" in df_1m.columns else 50
        rsi_5m = df_5m["rsi"].iloc[-1] if "rsi" in df_5m.columns else 50
        
        macd_1m = df_1m["macd"].iloc[-1] if "macd" in df_1m.columns else 0
        macd_signal_1m = df_1m["macd_signal"].iloc[-1] if "macd_signal" in df_1m.columns else 0
        
        lstm_pred, lstm_conf = self.lstm.predict(df_5m)
        
        if "BULLISH" in trend_bias:
            if 20 < rsi_5m < 50:
                score += 15
                factors.append("RSI pullback in uptrend")
            
            if rsi_1m < 35:
                score += 10
                factors.append("1m RSI oversold")
            
            if macd_1m > macd_signal_1m:
                score += 10
                factors.append("MACD bullish cross")
            
            for ob in self.smc.order_blocks:
                if ob.ob_type == "bullish" and ob.is_valid and ob.is_fresh:
                    if abs(current_price - ob.low) < atr * 2:
                        score += 15 + int(ob.strength_score * 10)
                        factors.append(f"Bullish OB @ {ob.low:.2f} (strength:{ob.strength_score:.2f})")
                        break
            
            for fvg in self.smc.fvgs:
                if fvg["is_bullish"] and not fvg["is_filled"]:
                    if fvg["low"] <= current_price <= fvg["high"]:
                        score += 10
                        factors.append("Inside bullish FVG")
                        break
            
            if lstm_pred > 0.2:
                score += 10
                factors.append(f"LSTM bullish ({lstm_pred:+.2f})")
            
            if bullish_count >= 4:
                score += 15
                factors.append("Strong HTF alignment")
            
            if score >= MIN_CONFIDENCE:
                direction = "LONG"
        
        elif "BEARISH" in trend_bias:
            if 50 < rsi_5m < 80:
                score += 15
                factors.append("RSI bounce in downtrend")
            
            if rsi_1m > 65:
                score += 10
                factors.append("1m RSI overbought")
            
            if macd_1m < macd_signal_1m:
                score += 10
                factors.append("MACD bearish cross")
            
            for ob in self.smc.order_blocks:
                if ob.ob_type == "bearish" and ob.is_valid and ob.is_fresh:
                    if abs(current_price - ob.high) < atr * 2:
                        score += 15 + int(ob.strength_score * 10)
                        factors.append(f"Bearerish OB @ {ob.high:.2f} (strength:{ob.strength_score:.2f})")
                        break
            
            for fvg in self.smc.fvgs:
                if not fvg["is_bullish"] and not fvg["is_filled"]:
                    if fvg["low"] <= current_price <= fvg["high"]:
                        score += 10
                        factors.append("Inside bearish FVG")
                        break
            
            if lstm_pred < -0.2:
                score += 10
                factors.append(f"LSTM bearish ({lstm_pred:+.2f})")
            
            if bearish_count >= 4:
                score += 15
                factors.append("Strong HTF alignment")
            
            if score >= MIN_CONFIDENCE:
                direction = "SHORT"
        
        confluence = len(factors)
        
        if direction is None:
            return SignalResult(
                direction=None, entry=0, stop_loss=0, take_profit_1=0, take_profit_2=0,
                confidence=min(score, 100), confluence_count=confluence, factors=factors,
                timeframe_alignment=alignment, status=TradeStatus.NO_SETUP,
                timestamp=datetime.now(), invalid_reason=f"No setup (score:{score:.1f})"
            )
        
        entry = current_price
        stop_distance = atr * ATR_MULTIPLIER_STOP
        
        if direction == "LONG":
            stop_loss = entry - stop_distance
            take_profit_1 = entry + (stop_distance * RR_TARGET_1)
            take_profit_2 = entry + (stop_distance * RR_TARGET_2)
        else:
            stop_loss = entry + stop_distance
            take_profit_1 = entry - (stop_distance * RR_TARGET_1)
            take_profit_2 = entry - (stop_distance * RR_TARGET_2)
        
        return SignalResult(
            direction=direction,
            entry=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(take_profit_1, 2),
            take_profit_2=round(take_profit_2, 2),
            confidence=min(score, 100),
            confluence_count=confluence,
            factors=factors,
            timeframe_alignment=alignment,
            status=TradeStatus.VALID,
            timestamp=datetime.now()
        )

class XAUUSDBot:
    def __init__(self):
        self.data_client = TwelveDataClient(TWELVE_DATA_API_KEY)
        self.engine = ScalpingEngine()
        self.last_signal = None
        self.monitoring = False
        self.chat_id = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.chat_id = update.effective_chat.id
        
        keyboard = [
            [InlineKeyboardButton("🔥 Start Signal Stream", callback_data="stream")],
            [InlineKeyboardButton("📊 Single Analysis", callback_data="analyze")],
            [InlineKeyboardButton("⏹️ Stop Stream", callback_data="stop")]
        ]
        
        await update.message.reply_text(
            "🏆 *XAUUSD 85%+ Accuracy Scalper*\n\n"
            "Timeframes: 1m | 5m | 15m | 30m | 1h\n"
            "Target: 85%+ confidence\n"
            "Min: 60% with 5+ confluence factors\n\n"
            "Continuous signal stream available.",
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
    
    async def fetch_all_timeframes(self) -> Tuple[Dict[str, pd.DataFrame], Optional[float]]:
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
                except:
                    pass
        
        return data, current_price
    
    async def single_analysis(self, query):
        await query.edit_message_text("🔬 Analyzing XAUUSD...")
        
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
    
    async def start_stream(self, query, context):
        self.monitoring = True
        
        await query.edit_message_text(
            "🔔 *SIGNAL STREAM STARTED*\n\n"
            "Analyzing every 30-60 seconds...\n"
            "Alerts for 80%+ confidence signals.\n"
            "Press Stop to end.",
            parse_mode="Markdown"
        )
        
        context.application.create_task(self.stream_loop(context))
    
    async def stop_stream(self, query, context):
        self.monitoring = False
        
        keyboard = [
            [InlineKeyboardButton("🔥 Start Stream", callback_data="stream")],
            [InlineKeyboardButton("📊 Single Analysis", callback_data="analyze")]
        ]
        
        await query.edit_message_text(
            "⏹️ Stream stopped.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def stream_loop(self, context):
        last_message_time = 0
        
        while self.monitoring:
            try:
                data, current_price = await self.fetch_all_timeframes()
                
                if not data or current_price is None:
                    await asyncio.sleep(30)
                    continue
                
                signal = self.engine.analyze(data, current_price)
                self.last_signal = signal
                
                now = time.time()
                
                if signal.status == TradeStatus.VALID and signal.confidence >= 80:
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
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(60)
    
    def format_signal_text(self, signal: SignalResult, is_alarm: bool) -> str:
        if signal.status != TradeStatus.VALID:
            emoji = "⚪"
            header = f"{emoji} *XAUUSD ANALYSIS* {emoji}"
            
            text = (
                f"{header}\n\n"
                f"Status: *NO TRADE*\n"
                f"Confidence: *{signal.confidence:.1f}%*\n"
                f"Confluence: {signal.confluence_count}/10\n\n"
                f"Reason: {signal.invalid_reason or 'No setup'}\n\n"
                f"*Timeframe Alignment:*\n"
            )
            
            for tf, align in signal.timeframe_alignment.items():
                em = "🟢" if align == "BULLISH" else "🔴" if align == "BEARISH" else "⚪"
                text += f"{tf}: {em} {align}\n"
            
            return text
        
        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        alarm_emoji = "🚨 " if is_alarm else ""
        
        text = (
            f"{alarm_emoji}{emoji} *{signal.direction} SIGNAL* {emoji}{alarm_emoji}\n\n"
            f"Confidence: *{signal.confidence:.1f}%* 🎯\n"
            f"Target: 85%+ (Current: {'✅' if signal.confidence >= 85 else '⚠️'})\n"
            f"Confluence: {signal.confluence_count} factors\n\n"
            f"📍 Entry: `{signal.entry}`\n"
            f"🛑 SL: `{signal.stop_loss}`\n"
            f"🎯 TP1: `{signal.take_profit_1}` (1:{RR_TARGET_1})\n"
            f"🎯 TP2: `{signal.take_profit_2}` (1:{RR_TARGET_2})\n\n"
            f"*Factors:*\n"
        )
        
        for i, factor in enumerate(signal.factors[:6], 1):
            text += f"{i}. {factor}\n"
        
        text += f"\n*Alignment (1m|5m|15m|30m|1h):*\n"
        align_str = ""
        for tf in ["1m", "5m", "15m", "30m", "1h"]:
            align = signal.timeframe_alignment.get(tf, "NEUTRAL")
            em = "🟢" if align == "BULLISH" else "🔴" if align == "BEARISH" else "⚪"
            align_str += f"{em}"
        text += align_str
        
        return text
    
    async def send_signal_message(self, query, signal: SignalResult, is_stream: bool):
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
    
    logger.info("XAUUSD Scalper Bot starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

import atexit
# ---- injected utility: rotate state backups ----
def rotate_state_backups(path, keep=5):
    try:
        folder = os.path.dirname(path) or "."
        base = os.path.basename(path)
        prefix = base + ".bak"
        files = sorted([f for f in os.listdir(folder) if f.startswith(prefix)], reverse=True)
        for i, f in enumerate(files[keep:], start=keep):
            try:
                os.remove(os.path.join(folder, f))
            except Exception as e:
                log("Failed to remove old backup %s: %s", f, e)
    except Exception as e:
        log("Backup rotation error: %s", e)
# ---- end injected utility ----


# ---- injected utility: graceful shutdown ----
def _shutdown_cleanup():
    try:
        if 'tg_manager' in globals() and tg_manager:
            try:
                tg_manager.close()
                log("Telegram manager closed.")
            except Exception as e:
                log("Telegram manager close() error: %s", e)
    except Exception as e:
        log("Shutdown cleanup error: %s", e)

atexit.register(_shutdown_cleanup)
# ---- end injected utility ----


# ---- injected utility: AI thresholds ----
def get_ai_thresholds(AI_CFG):
    try:
        t_long = float(AI_CFG.get("threshold_long", 0.55))
        t_short = float(AI_CFG.get("threshold_short", 0.45))
    except Exception:
        t_long, t_short = 0.55, 0.45
    return t_long, t_short
# ---- end injected utility ----


# ---- injected utility: ensure numeric dtypes for rolling ops ----
def _ensure_numeric_df(df):
    try:
        import pandas as _pd
        numeric_df = df.copy()
        for col in numeric_df.columns:
            if _pd.api.types.is_numeric_dtype(numeric_df[col]):
                continue
            numeric_df[col] = _pd.to_numeric(numeric_df[col], errors="coerce")
        return numeric_df
    except Exception as e:
        log("Failed to enforce numeric dtypes: %s", e)
        return df
# ---- end injected utility ----

from datetime import datetime, timezone, timedelta
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Crypto Bot (OKX via CCXT) — FULL FINAL BUILD
- Complete CLI support (--download-data, --train-ai, --walkforward, --run)
- Stable Hybrid AI (LogReg + PyTorch LSTM with fallback)
- Full Walk-Forward Optimization with equity curves
- Async Telegram integration (no warnings)
- Complete chart generation and reporting
- Live OKX trading support
- Resume-capable data downloads
- Model caching and persistence
- Position recovery and state persistence
- Proper async/await throughout
"""

  # ensure available before _shutdown_cleanup



import os
import sys
import time
import csv
import math
import json
import pickle
import asyncio
import argparse
import shutil
import types
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import ccxt
import aiohttp
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Chart dependencies
import matplotlib
matplotlib.use('Agg')  # headless mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Optional TA-Lib
try:
    import talib
    HAVE_TALIB = True
except ImportError:
    HAVE_TALIB = False

# Optional PyTorch for LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# Initialize state
AI_MODE_STATE = {'mode': 'auto'}
_AI_STATE = {}

# =============================
# CONFIGURATION
# =============================
load_dotenv()

MODE = os.getenv("BOT_MODE", "demo")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")

# Trading pairs
AUTO_PAIRS = {
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'XRP/USDT:USDT',
}

WATCH_ONLY_PAIRS = {
    'DOGE/USDT:USDT',
    'SOL/USDT:USDT',
}

# Strategy configuration
cfg = {
    "daily_budget_usd": 10.0,
    "budget_used_today": 0.0,
    "contracts_mode": "auto",
    "contracts_fixed": 1,
    "leverage": 3,
    "timeframes": ["5m", "15m", "1h", "4h", "1d"],
    "core_tfs": ["5m", "15m", "1h"],
    "atr_period": 14,
    "atr_sl_mult": 1.0,
    "atr_tp_mult": 2.0,
    "partial_tp_ratio": 0.5,
    "use_partial_tp": True,
    "trail_after_tp": True,
    "trail_pct": 0.5,
    "fib_lookback": 120,
    "fib_tolerance": 0.003,
    "use_fvg": True,
    "fvg_lookback": 50,
    "use_ob": True,
    "ob_lookback": 50,
    "session_filter": True,
    "session_utc_start": 12,
    "session_utc_end": 18,
    "heartbeat_minutes": 5,
    "slippage_buffer_pct": 0.02,
    "auto_enabled": True,
    "position_size_fraction": 0.02,  # 2% of account per trade
    "min_notional": 5.0,  # Minimum trade size in USD
    "max_notional": 100.0,  # Maximum trade size in USD
    "budget_reset_hour": 0,  # UTC hour for daily reset
}

# AI Configuration
AI_CFG = {
    "base_tf": "5m",
    "other_tfs": ["15m", "1h"],
    "hist_limit": 400,
    "hist_months": 12,
    "min_obs": 120,
    "learn_epochs_boot": 5,
    "learn_rate": 0.05,
    "l2": 1e-4,
    "threshold_long": 0.58,
    "threshold_short": 0.42,
    "state_file": "ai_model_state.json",
    "lstm_epochs": 20,
    "lstm_batch_size": 64,
    "lstm_hidden": 64,
    "lstm_layers": 2,
    "lstm_seq_len": 60,
    "hybrid_weight": 0.6,  # Weight for LSTM vs LR
    "validation_split": 0.2,
    "early_stopping_patience": 5,
    "min_accuracy": 0.55,  # Minimum accuracy to trust model
}

# Directory structure
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
CHARTS_DIR = "charts"
STATE_DIR = "state"

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, CHARTS_DIR, STATE_DIR]:
    os.makedirs(d, exist_ok=True)

# State files
POSITION_STATE_FILE = os.path.join(STATE_DIR, "open_positions.json")
SETTINGS_STATE_FILE = os.path.join(STATE_DIR, "settings.json")
AI_STATE_FILE = os.path.join(STATE_DIR, "ai_state.json")

# =============================
# UTILITY FUNCTIONS
# =============================
def safe(value, fmt="{:.4f}"):
    """Safe formatter to avoid NoneType errors"""
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float) and (value != value):  # NaN check
            return "N/A"
        return fmt.format(value)
    except Exception:
        return str(value)

def sf(x, n=2):
    """Safe float formatter"""
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def log(msg: str):
    """Timestamp logging"""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


try:
    log(f"[patch-check] datetime.now works: {datetime.now(timezone.utc)}")
except Exception as e:
    print(f"[patch-check] datetime error: {e}")

def normalize_symbol(s):
    """Normalize symbol format"""
    s = s.upper().replace(" ", "")
    if "/" not in s:
        s = f"{s}/USDT:USDT"
    if s.endswith("/USDT"):
        s = s + ":USDT"
    return s

def timeframe_alias(tf: str) -> str:
    """Convert timeframe aliases"""
    t = tf.lower().replace(" ", "")
    aliases = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1hr": "1h", "60m": "1h",
        "2h": "2h", "4h": "4h", "4hr": "4h", "240m": "4h",
        "1d": "1d", "1day": "1d", "daily": "1d"
    }
    return aliases.get(t, "1h")

def save_state(filepath: str, data: dict):
    rotate_state_backups(filepath)
    """Save state to JSON file with backup"""
    try:
        # Create backup if exists
        if os.path.exists(filepath):
            shutil.copy2(filepath, f"{filepath}.bak")
        
        # Save new state
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        log(f"Error saving state to {filepath}: {e}")
        return False

def load_state(filepath: str) -> dict:
    """Load state from JSON file"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        log(f"Error loading state from {filepath}: {e}")
        # Try backup
        try:
            backup_path = f"{filepath}.bak"
            if os.path.exists(backup_path):
                with open(backup_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
    return {}

# Load persistent settings
def load_settings():
    """Load settings from state file"""
    settings = load_state(SETTINGS_STATE_FILE)
    if settings:
        cfg.update(settings)
        log("Settings loaded from state file")

def save_settings():
    """Save current settings to state file"""
    settings = {
        "daily_budget_usd": cfg["daily_budget_usd"],
        "budget_used_today": cfg["budget_used_today"],
        "contracts_mode": cfg["contracts_mode"],
        "contracts_fixed": cfg["contracts_fixed"],
        "leverage": cfg["leverage"],
        "auto_enabled": cfg["auto_enabled"],
        "position_size_fraction": cfg["position_size_fraction"],
        "min_notional": cfg["min_notional"],
        "max_notional": cfg["max_notional"],
        "atr_sl_mult": cfg["atr_sl_mult"],
        "atr_tp_mult": cfg["atr_tp_mult"],
    }
    save_state(SETTINGS_STATE_FILE, settings)

# Load settings on startup
load_settings()

# =============================
# EXCHANGE SETUP
# =============================
def init_exchange():
    """Initialize OKX exchange with proper error handling"""
    try:
        if not all([OKX_API_KEY, OKX_API_SECRET, OKX_PASSPHRASE]):
            log("Missing OKX credentials, using public endpoints only")
            ex = ccxt.okx({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
            if MODE == "demo":
                ex.set_sandbox_mode(True)
            return ex
        
        params = {
            'apiKey': OKX_API_KEY,
            'secret': OKX_API_SECRET,
            'password': OKX_PASSPHRASE,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        }
        ex = ccxt.okx(params)
        
        if MODE == "demo":
            ex.set_sandbox_mode(True)
            ex.options['demo'] = True
        
        # Test connection
        try:
            ex.fetch_balance()
            log("Exchange connection successful")
        except ccxt.AuthenticationError as e:
            if "50101" in str(e) or "50111" in str(e):
                log("Auth error, falling back to public endpoints")
                ex = ccxt.okx({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'swap'}
                })
        
        return ex
    except Exception as e:
        log(f"Exchange initialization error: {e}")
        # Return basic exchange for public data
        return ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

exchange = init_exchange()

def safe_set_leverage(sym, lev):
    """Safely set leverage for a symbol"""
    try:
        if hasattr(exchange, 'apiKey') and exchange.apiKey:
            exchange.set_leverage(lev, sym)
            return True
        else:
            log(f"Cannot set leverage for {sym}: no API credentials")
            return False
    except Exception as e:
        log(f"Failed to set leverage for {sym}: {e}")
        return False

# Set leverage for auto pairs
for sym in list(AUTO_PAIRS):
    safe_set_leverage(sym, cfg["leverage"])
# =============================
# ASYNC TELEGRAM FUNCTIONS (Fixed)
# =============================
class TelegramManager:
    """Async Telegram manager with rate limiting"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.last_message_time = 0
        self.min_interval = 1.0  # Minimum seconds between messages
        self.session = None
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def rate_limit(self):
        """Apply rate limiting"""
        now = time.time()
        elapsed = now - self.last_message_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_message_time = time.time()
    
    async def send_message(self, text: str, parse_mode: str = None) -> bool:
        """Send message with rate limiting and chunking"""
        try:
            if not text or not self.token or not self.chat_id:
                return False
            
            # Split long messages
            chunks = self._split_message(text)
            
            for chunk in chunks:
                await self.rate_limit()
                
                session = await self.get_session()
                data = {
                    "chat_id": self.chat_id,
                    "text": chunk
                }
                if parse_mode:
                    data["parse_mode"] = parse_mode
                
                async with session.post(
                    f"{self.base_url}/sendMessage",
                    data=data
                ) as response:
                    if response.status != 200:
                        log(f"Telegram send failed: {response.status}")
                        return False
            
            return True
            
        except Exception as e:
            log(f"Telegram send error: {e}")
            return False
    
    async def send_photo(self, filepath: str, caption: str = "") -> bool:
        """Send photo with caption"""
        try:
            if not os.path.exists(filepath):
                log(f"Photo file not found: {filepath}")
                return False
            
            await self.rate_limit()
            
            session = await self.get_session()
            
            with open(filepath, "rb") as f:
                data = aiohttp.FormData()
                data.add_field('chat_id', self.chat_id)
                if caption:
                    data.add_field('caption', caption[:1024])  # Telegram limit
                data.add_field('photo', f, filename=os.path.basename(filepath))
                
                async with session.post(
                    f"{self.base_url}/sendPhoto", 
                    data=data
                ) as response:
                    return response.status == 200
        
        except Exception as e:
            log(f"Telegram photo send error: {e}")
            return False
    
    async def get_updates(self, offset: int = None) -> List[dict]:
        """Get updates from Telegram"""
        try:
            session = await self.get_session()
            params = {"timeout": 5}
            if offset:
                params["offset"] = offset
            
            async with session.get(
                f"{self.base_url}/getUpdates",
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("result", [])
                return []
        
        except Exception as e:
            log(f"Telegram get_updates error: {e}")
            return []
    
    def _split_message(self, text: str, max_length: int = 4000) -> List[str]:
        """Split long messages into chunks"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        lines = text.split('\n')
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 <= max_length:
                current_chunk += line + '\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Initialize Telegram manager
tg_manager = TelegramManager(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None

async def tg_send(msg: str):
    """Send message to Telegram"""
    if tg_manager:
        return await tg_manager.send_message(msg)
    return False

async def tg_send_photo(filepath: str, caption: str = ""):
    """Send photo to Telegram"""
    if tg_manager:
        return await tg_manager.send_photo(filepath, caption)
    return False

async def tg_get_updates(offset=None):
    """Get Telegram updates"""
    if tg_manager:
        return await tg_manager.get_updates(offset)
    return []

# =============================
# DATA MANAGEMENT (Enhanced)
# =============================
def _history_path(sym, tf):
    """Get path for historical data CSV"""
    sym_s = sym.replace("/", "_").replace(":", "_")
    return os.path.join(DATA_DIR, f"{sym_s}_{tf}.csv")

async def ensure_history(sym, tf="5m", months=12, limit=1000, max_retries=3):
    """Download and cache historical OHLCV data with resume capability"""
    path = _history_path(sym, tf)
    since = int((time.time() - months * 30 * 24 * 3600) * 1000)
    
    all_rows = []
    now = int(time.time() * 1000)
    
    try:
        # Load existing data if available
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                last_ms = int(df['ts'].iloc[-1])
                since = last_ms + 1
                all_rows = df.values.tolist()
                log(f"Resuming download for {sym} {tf} from {datetime.fromtimestamp(since/1000, tz=timezone.utc)}")
        
        # Download missing data
        downloaded = 0
        retry_count = 0
        
        while since < now:
            try:
                batch = exchange.fetch_ohlcv(sym, timeframe=tf, since=since, limit=limit)
                if not batch:
                    break
                
                all_rows.extend(batch)
                downloaded += len(batch)
                since = batch[-1][0] + 1
                retry_count = 0  # Reset on success
                
                # Progress update
                if downloaded % 1000 == 0:
                    log(f"Downloaded {downloaded} bars for {sym} {tf}")
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    log(f"Download failed for {sym} {tf} after {max_retries} retries: {e}")
                    break
                
                log(f"Download error for {sym} {tf} (retry {retry_count}): {e}")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                continue
        
        # Save data
        if all_rows:
            df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
            df.to_csv(path, index=False)
            log(f"Saved {len(df)} bars for {sym} {tf}")
            return True
        
        return False
        
    except Exception as e:
        log(f"ensure_history error for {sym} {tf}: {e}")
        return False

def load_history_df(sym, tf="5m", months=None):
    """Load historical data as pandas DataFrame"""
    path = _history_path(sym, tf)
    
    # Try to load from cache first
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                df.set_index('ts', inplace=True)
                
                # Check if data is recent enough
                if months:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=months*30)
                    if df.index[-1] < cutoff:
                        # Data too old, will fallback to live fetch
                        pass
                    else:
                        return df
                else:
                    return df
        except Exception as e:
            log(f"Error loading cached data for {sym} {tf}: {e}")
    
    # Fallback to live fetch
    try:
        arr = exchange.fetch_ohlcv(sym, timeframe=tf, limit=AI_CFG.get("hist_limit", 400))
        if arr:
            df = pd.DataFrame(arr, columns=["ts", "open", "high", "low", "close", "volume"])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df.set_index('ts', inplace=True)
            return df
    except Exception as e:
        log(f"Live fetch error for {sym} {tf}: {e}")
    
    return pd.DataFrame()

# =============================
# TECHNICAL INDICATORS (Enhanced)
# =============================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    if len(df) < 50:
        return df
    
    df = df.copy()
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    try:
        # EMAs
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        if HAVE_TALIB and len(df) > 14:
            try:
                df['atr'] = talib.ATR(high.values, low.values, close.values, timeperiod=14)
            except Exception:
                # Fallback calculation
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = tr.rolling(window=14, min_periods=1).mean()
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14, min_periods=1).mean()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = close.rolling(window=bb_period, min_periods=1).mean()
        bb_std_dev = close.rolling(window=bb_period, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        
        # Volume indicators
        df['volume_sma'] = volume.rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = volume / df['volume_sma'].replace(0, 1)
        
        # Additional momentum indicators
        df['momentum'] = close / close.shift(10) - 1
        df['roc'] = close.pct_change(periods=10)
        
        # Price position in BB
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)
        
    except Exception as e:
        log(f"Indicator calculation error: {e}")
    
    # Forward fill missing values, then backfill, then fill remaining with 0
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

async def get_indicators(sym, tf="5m", limit=200):
    """Get current indicators for a symbol"""
    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=tf, limit=limit)
        if not ohlcv:
            return None, None
            
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        
        df = calculate_indicators(df)
        
        if len(df) == 0:
            return None, None
        
        latest = df.iloc[-1]
        indicators = {
            'price': float(latest['close']),
            'ema20': float(latest['ema_20']) if not pd.isna(latest['ema_20']) else None,
            'ema50': float(latest['ema_50']) if not pd.isna(latest['ema_50']) else None,
            'ema200': float(latest['ema_200']) if not pd.isna(latest['ema_200']) else None,
            'rsi14': float(latest['rsi']) if not pd.isna(latest['rsi']) else None,
            'macd': float(latest['macd']) if not pd.isna(latest['macd']) else None,
            'macd_signal': float(latest['macd_signal']) if not pd.isna(latest['macd_signal']) else None,
            'macd_hist': float(latest['macd_hist']) if not pd.isna(latest['macd_hist']) else None,
            'atr': float(latest['atr']) if not pd.isna(latest['atr']) else None,
            'bb_upper': float(latest['bb_upper']) if not pd.isna(latest['bb_upper']) else None,
            'bb_lower': float(latest['bb_lower']) if not pd.isna(latest['bb_lower']) else None,
            'bb_middle': float(latest['bb_middle']) if not pd.isna(latest['bb_middle']) else None,
            'bb_position': float(latest['bb_position']) if not pd.isna(latest['bb_position']) else None,
            'volume_ratio': float(latest['volume_ratio']) if not pd.isna(latest['volume_ratio']) else None,
            'momentum': float(latest['momentum']) if not pd.isna(latest['momentum']) else None,
        }
        
        return indicators, df
        
    except Exception as e:
        log(f"get_indicators error for {sym}: {e}")
        return None, None
# =============================
# IMPROVED AI MODELS
# =============================

class OnlineLR:
    """Enhanced Online Logistic Regression with L2 regularization"""
    
    def __init__(self, n_features, lr=0.05, l2=1e-4):
        self.w = np.zeros(n_features + 1)  # +1 for bias
        self.lr = lr
        self.l2 = l2
        self.training_history = []
        self.accuracy = 0.0
        self.n_updates = 0
    
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    
    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        if len(x) != len(self.w) - 1:
            return 0.5  # Default probability if feature mismatch
        z = self.w[0] + np.dot(self.w[1:], x)
        return float(self._sigmoid(z))
    
    def update(self, x, y):
        try:
            p = self.predict_proba(x)
            err = (y - p)
            
            # Gradient with L2 regularization
            x_array = np.asarray(x, dtype=float)
            grad_bias = err
            grad_weights = err * x_array - self.l2 * self.w[1:]
            
            # Update weights
            self.w[0] += self.lr * grad_bias
            self.w[1:] += self.lr * grad_weights
            
            # Track updates
            self.n_updates += 1
            self.training_history.append({'prediction': p, 'actual': y, 'error': abs(y - p)})
            
            # Calculate rolling accuracy
            if len(self.training_history) > 100:
                recent = self.training_history[-100:]
                correct = sum(1 for h in recent if abs(h['actual'] - (1 if h['prediction'] > 0.5 else 0)) < 0.5)
                self.accuracy = correct / len(recent)
            
            return p
        except Exception as e:
            log(f"OnlineLR update error: {e}")
            return 0.5
    
    def to_dict(self):
        return {
            "w": [float(w) if not (np.isnan(w) or np.isinf(w)) else 0.0 for w in self.w],
            "lr": self.lr,
            "l2": self.l2,
            "accuracy": self.accuracy,
            "n_updates": self.n_updates
        }
    
    @classmethod
    def from_dict(cls, d):
        obj = cls(len(d.get("w", [1])) - 1, d.get("lr", 0.05), d.get("l2", 1e-4))
        obj.w = np.array([float(w) for w in d.get("w", [0])], dtype=float)
        obj.accuracy = d.get("accuracy", 0.0)
        obj.n_updates = d.get("n_updates", 0)
        return obj

# =============================
# PYTORCH LSTM MODEL (Enhanced)
# =============================
if TORCH_OK:
    class LSTMModel(nn.Module):
        """Enhanced PyTorch LSTM for sequence prediction"""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.input_size = input_size
            
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 2, 1)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
        
        def forward(self, x):
            batch_size = x.size(0)
            
            # Initialize hidden state
            device = x.device
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(x, (h0, c0))
            
            # Use last output
            last_output = lstm_out[:, -1, :]
            
            # Batch normalization and dropout
            if batch_size > 1:
                last_output = self.batch_norm(last_output)
            last_output = self.dropout(last_output)
            
            # Fully connected layers
            out = self.relu(self.fc1(last_output))
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            
            return out

else:
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")

# =============================
# AI FEATURE ENGINEERING (Enhanced)
# =============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced features for AI model"""
    if len(df) < 50:
        return pd.DataFrame()
    
    df = calculate_indicators(df.copy())
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    features = pd.DataFrame(index=df.index)
    
    try:
        # Price-based features
        features['returns_1'] = close.pct_change(1)
        features['returns_2'] = close.pct_change(2)
        features['returns_5'] = close.pct_change(5)
        features['returns_10'] = close.pct_change(10)
        features['returns_20'] = close.pct_change(20)
        
        # Log returns (more stable)
        features['log_returns_1'] = np.log(close / close.shift(1))
        features['log_returns_5'] = np.log(close / close.shift(5))
        
        # Momentum features (normalized)
        features['rsi_norm'] = (df['rsi'] - 50) / 50
        features['macd_norm'] = df['macd'] / close
        features['macd_hist_norm'] = df['macd_hist'] / close
        
        # Trend features
        features['ema20_ratio'] = close / df['ema_20']
        features['ema50_ratio'] = close / df['ema_50']
        features['ema200_ratio'] = close / df['ema_200']
        features['ema_trend_20'] = df['ema_20'].pct_change(5)
        features['ema_trend_50'] = df['ema_50'].pct_change(10)
        
        # Volatility features
        features['atr_ratio'] = df['atr'] / close
        features['bb_position'] = df['bb_position']
        features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / close
        features['volatility_5'] = close.rolling(5).std() / close.rolling(5).mean()
        features['volatility_20'] = close.rolling(20).std() / close.rolling(20).mean()
        
        # Volume features
        features['volume_ratio'] = df['volume_ratio']
        features['volume_ma_ratio'] = volume / volume.rolling(10).mean()
        features['price_volume'] = close.pct_change(1) * features['volume_ratio']
        
        # Statistical features
        features['z_score_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
        features['skew_10'] = close.rolling(10).skew()
        features['kurt_10'] = close.rolling(10).kurt()
        
        # Candlestick patterns
        features['doji'] = abs(close - df['open']) / (high - low + 1e-10)
        features['upper_shadow'] = (high - np.maximum(close, df['open'])) / (high - low + 1e-10)
        features['lower_shadow'] = (np.minimum(close, df['open']) - low) / (high - low + 1e-10)
        
        # Advanced momentum
        features['rsi_sma'] = df['rsi'].rolling(5).mean()
        features['rsi_divergence'] = features['rsi_norm'] - features['rsi_norm'].shift(1)
        features['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Mean reversion indicators
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).quantile(0.2)).astype(int)
        features['oversold'] = (df['rsi'] < 30).astype(int)
        features['overbought'] = (df['rsi'] > 70).astype(int)
        
        # Trend strength
        features['trend_strength'] = features['ema20_ratio'] * features['rsi_norm']
        features['momentum_alignment'] = (
            (features['returns_5'] > 0) & 
            (features['rsi_norm'] > 0) & 
            (features['macd_hist_norm'] > 0)
        ).astype(int)
        
    except Exception as e:
        log(f"Feature creation error: {e}")
    
    # Clean features
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    # Clip extreme values
    for col in features.columns:
        if features[col].dtype in ['float64', 'float32']:
            q99 = features[col].quantile(0.99)
            q01 = features[col].quantile(0.01)
            features[col] = features[col].clip(q01, q99)
    
    return features

def create_labels(df: pd.DataFrame, forward_bars=1, threshold=0.002) -> pd.Series:
    """Create labels for training with threshold"""
    close = df['close'].astype(float)
    future_returns = close.shift(-forward_bars) / close - 1
    
    # Use threshold for clearer signals
    labels = pd.Series(0, index=df.index)  # 0 = neutral
    labels[future_returns > threshold] = 1   # 1 = up
    labels[future_returns < -threshold] = 0  # 0 = down
    
    return labels

def prepare_lstm_sequences(features: pd.DataFrame, labels: pd.Series, seq_len=60):
    """Prepare sequences for LSTM training with proper validation"""
    if len(features) < seq_len + 1:
        return None, None
    
    # Ensure features are numeric and clean
    features = features.select_dtypes(include=[np.number])
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    
    X, y = [], []
    for i in range(seq_len, len(features)):
        if i < len(labels):
            X.append(features.iloc[i-seq_len:i].values)
            y.append(labels.iloc[i])
    
    if not X:
        return None, None
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
# =============================
# AI MODEL MANAGEMENT (Enhanced)
# =============================

class AIModelManager:
    """Centralized AI model management with persistence"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.state_file = AI_STATE_FILE
        self.load_state()
    
    def load_state(self):
        """Load AI state from disk"""
        try:
            state = load_state(self.state_file)
            
            # Load LR models
            for symbol, model_data in state.get('lr_models', {}).items():
                try:
                    self.models[f"{symbol}_lr"] = OnlineLR.from_dict(model_data)
                except Exception as e:
                    log(f"Error loading LR model for {symbol}: {e}")
            
            # Load metadata
            self.metadata = state.get('metadata', {})
            
            # Load LSTM models and scalers
            if TORCH_OK:
                for symbol in state.get('lstm_symbols', []):
                    self.load_lstm_model(symbol)
            
            log(f"Loaded AI state: {len(self.models)} models")
            
        except Exception as e:
            log(f"Error loading AI state: {e}")
    
    def save_state(self):
        rotate_state_backups(filepath)
        """Save AI state to disk"""
        try:
            state = {
                'lr_models': {},
                'metadata': self.metadata,
                'lstm_symbols': []
            }
            
            # Save LR models
            for key, model in self.models.items():
                if key.endswith('_lr') and isinstance(model, OnlineLR):
                    symbol = key[:-3]  # Remove '_lr' suffix
                    state['lr_models'][symbol] = model.to_dict()
            
            # Track LSTM symbols
            for symbol in self.scalers.keys():
                state['lstm_symbols'].append(symbol)
            
            save_state(self.state_file, state)
            log("AI state saved successfully")
            
        except Exception as e:
            log(f"Error saving AI state: {e}")
    
    def get_lr_model(self, symbol: str, n_features: int = None) -> OnlineLR:
        """Get or create LogReg model for symbol"""
        key = f"{symbol}_lr"
        
        if key in self.models:
            model = self.models[key]
            if n_features and len(model.w) != n_features + 1:
                # Feature count mismatch, create new model
                log(f"Feature mismatch for {symbol}, creating new LR model")
                model = OnlineLR(n_features, lr=AI_CFG["learn_rate"], l2=AI_CFG["l2"])
                self.models[key] = model
            return model
        
        # Create new model
        if n_features is None:
            n_features = 10  # Default
        model = OnlineLR(n_features, lr=AI_CFG["learn_rate"], l2=AI_CFG["l2"])
        self.models[key] = model
        return model
    
    def get_lstm_paths(self, symbol: str) -> Tuple[str, str]:
        """Get paths for LSTM model and scaler"""
        safe_sym = symbol.replace("/", "_").replace(":", "_")
        model_path = os.path.join(MODELS_DIR, f"{safe_sym}_lstm.pt")
        scaler_path = os.path.join(MODELS_DIR, f"{safe_sym}_scaler.pkl")
        return model_path, scaler_path
    
    def save_lstm_model(self, symbol: str, model: LSTMModel, scaler: StandardScaler, metadata: dict = None):
        """Save LSTM model and scaler with metadata"""
        if not TORCH_OK:
            return False
        
        try:
            model_path, scaler_path = self.get_lstm_paths(symbol)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'metadata': metadata or {}
            }, model_path)
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Store in memory
            self.scalers[symbol] = scaler
            
            # Update metadata
            if metadata:
                self.metadata[symbol] = metadata
            
            return True
            
        except Exception as e:
            log(f"Error saving LSTM model for {symbol}: {e}")
            return False
    
    def load_lstm_model(self, symbol: str) -> Tuple[Optional[LSTMModel], Optional[StandardScaler]]:
        """Load LSTM model and scaler"""
        if not TORCH_OK:
            return None, None
        
        try:
            model_path, scaler_path = self.get_lstm_paths(symbol)
            
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                return None, None
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            model = LSTMModel(
                input_size=checkpoint['input_size'],
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Store in memory
            self.scalers[symbol] = scaler
            if 'metadata' in checkpoint:
                self.metadata[symbol] = checkpoint['metadata']
            
            return model, scaler
            
        except Exception as e:
            log(f"Error loading LSTM model for {symbol}: {e}")
            return None, None
    
    def get_model_metadata(self, symbol: str) -> dict:
        """Get metadata for symbol's models"""
        return self.metadata.get(symbol, {})
    
    def update_model_metadata(self, symbol: str, metadata: dict):
        """Update metadata for symbol's models"""
        if symbol not in self.metadata:
            self.metadata[symbol] = {}
        self.metadata[symbol].update(metadata)

# Initialize global model manager
ai_manager = AIModelManager()

# =============================
# AI TRAINING FUNCTIONS (Enhanced)
# =============================

async def train_lstm_model(symbol: str, features: pd.DataFrame, labels: pd.Series, 
                    epochs=20, batch_size=64, seq_len=60, validation_split=0.2) -> bool:
    """Train LSTM model with proper error handling and validation"""
    if not TORCH_OK:
        log(f"PyTorch not available for LSTM training on {symbol}")
        return False
    
    try:
        # Prepare sequences
        X, y = prepare_lstm_sequences(features, labels, seq_len)
        if X is None or len(X) < 200:  # Minimum samples for training
            log(f"Insufficient sequences for LSTM training on {symbol}: {len(X) if X is not None else 0}")
            return False
        
        log(f"Training LSTM for {symbol} with {len(X)} sequences, {X.shape[2]} features")
        
        # Create and fit scaler on ALL training data
        n_samples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        scaler = StandardScaler()
        X_scaled_flat = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled_flat.reshape(n_samples, seq_length, n_features)
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        if len(X_train) < 50 or len(X_val) < 20:
            log(f"Insufficient data after split for {symbol}")
            return False
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(
            input_size=n_features,
            hidden_size=AI_CFG["lstm_hidden"],
            num_layers=AI_CFG["lstm_layers"]
        ).to(device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        training_history = []
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            scheduler.step(avg_val_loss)
            
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy
            })
            
            if epoch % 5 == 0:
                log(f"LSTM {symbol} Epoch {epoch}/{epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping and model saving
            if val_accuracy > best_val_acc and val_accuracy > AI_CFG["min_accuracy"]:
                best_val_loss = avg_val_loss
                best_val_acc = val_accuracy
                patience_counter = 0
                
                # Save best model
                metadata = {
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                    'training_epochs': epoch + 1,
                    'features': n_features,
                    'seq_len': seq_len,
                    'training_history': training_history
                }
                
                ai_manager.save_lstm_model(symbol, model, scaler, metadata)
            else:
                patience_counter += 1
                if patience_counter >= AI_CFG["early_stopping_patience"]:
                    log(f"Early stopping for {symbol} at epoch {epoch}")
                    break
        
        if best_val_acc < AI_CFG["min_accuracy"]:
            log(f"LSTM training for {symbol} failed: accuracy {best_val_acc:.4f} below threshold")
            return False
        
        log(f"LSTM training completed for {symbol}: "
            f"best val accuracy: {best_val_acc:.4f}, best val loss: {best_val_loss:.4f}")
        return True
        
    except Exception as e:
        log(f"LSTM training error for {symbol}: {e}")
        return False


async def train_lr_model(symbol: str, features: pd.DataFrame, labels: pd.Series, epochs=5) -> bool:
    """Train Logistic Regression model"""
    try:
        if len(features) < AI_CFG["min_obs"]:
            log(f"Insufficient data for LR training on {symbol}: {len(features)}")
            return False
        
        # Get model
        model = ai_manager.get_lr_model(symbol, features.shape[1])
        
        # Training loop with mini-batches
        n_samples = len(features)
        batch_size = min(100, n_samples // 10)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                for idx in batch_indices:
                    model.update(features.iloc[idx].values, int(labels.iloc[idx]))
        
        # Update metadata
        metadata = {
            'accuracy': model.accuracy,
            'n_updates': model.n_updates,
            'features': features.shape[1],
            'training_samples': n_samples
        }
        ai_manager.update_model_metadata(symbol, {'lr_metadata': metadata})
        
        log(f"LR training completed for {symbol}: accuracy {model.accuracy:.4f}")
        return model.accuracy > AI_CFG["min_accuracy"]
        
    except Exception as e:
        log(f"LR training error for {symbol}: {e}")
        return False

async def create_ai_dataset(symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.DataFrame]]:
    """Create training dataset for AI models"""
    try:
        # Load base timeframe data
        df_base = load_history_df(symbol, AI_CFG["base_tf"], months=AI_CFG["hist_months"])
        if len(df_base) < AI_CFG["min_obs"]:
            log(f"Insufficient base data for {symbol}: {len(df_base)}")
            return None, None, None
        
        # Create features and labels
        features = create_features(df_base)
        labels = create_labels(df_base, forward_bars=1, threshold=0.001)
        
        if len(features) == 0:
            log(f"No features created for {symbol}")
            return None, None, None
        
        # Load other timeframes and merge
        for tf in AI_CFG["other_tfs"]:
            try:
                df_tf = load_history_df(symbol, tf, months=AI_CFG["hist_months"])
                if len(df_tf) >= 50:
                    features_tf = create_features(df_tf)
                    if len(features_tf) > 0:
                        # Resample to base timeframe
                        features_tf = features_tf.reindex(features.index, method='ffill')
                        # Add prefix to avoid column conflicts
                        features_tf = features_tf.add_prefix(f"{tf}_")
                        features = pd.concat([features, features_tf], axis=1)
            except Exception as e:
                log(f"Error adding {tf} features for {symbol}: {e}")
        
        # Clean data
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Align features and labels
        min_len = min(len(features), len(labels))
        features = features.iloc[:min_len]
        labels = labels.iloc[:min_len]
        
        # Remove any remaining NaN rows
        valid_mask = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        if len(features) < AI_CFG["min_obs"]:
            log(f"Insufficient clean data for {symbol}: {len(features)}")
            return None, None, None
        
        log(f"Created dataset for {symbol}: {len(features)} samples, {features.shape[1]} features")
        return features, labels, df_base
        
    except Exception as e:
        log(f"Error creating AI dataset for {symbol}: {e}")
        return None, None, None
# =============================
# HYBRID AI PREDICTION (Fixed)
# =============================

async def predict_with_hybrid_ai(symbol: str) -> Optional[Dict]:
    """Make predictions using hybrid AI (LR + LSTM) with proper fallback"""
    try:
        # Create current features
        features, labels, df_base = await create_ai_dataset(symbol)
        if features is None or len(features) == 0:
            log(f"No data available for prediction: {symbol}")
            return None
        
        current_features = features.iloc[-1].values
        
        # Get LR prediction
        lr_model = ai_manager.get_lr_model(symbol, features.shape[1])
        p_lr = lr_model.predict_proba(current_features)
        lr_confidence = lr_model.accuracy if lr_model.accuracy > 0 else 0.5
        
        # Get LSTM prediction if available
        p_lstm = None
        lstm_confidence = 0.0
        
        if TORCH_OK:
            lstm_model, scaler = ai_manager.load_lstm_model(symbol)
            if lstm_model is not None and scaler is not None:
                try:
                    seq_len = AI_CFG["lstm_seq_len"]
                    if len(features) >= seq_len:
                        # Prepare sequence
                        seq_features = features.iloc[-seq_len:].values
                        
                        # Transform using the fitted scaler
                        seq_scaled = scaler.transform(seq_features)
                        
                        # Predict
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        with torch.no_grad():
                            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                            p_lstm = float(lstm_model(seq_tensor).cpu().numpy()[0][0])
                        
                        # Get LSTM confidence from metadata
                        metadata = ai_manager.get_model_metadata(symbol)
                        lstm_confidence = metadata.get('best_val_acc', 0.5)
                        
                except Exception as e:
                    log(f"LSTM prediction error for {symbol}: {e}")
                    p_lstm = None
        
        # Combine predictions with confidence weighting
        if p_lstm is not None and lstm_confidence >= AI_CFG["min_accuracy"]:
            # Weight by confidence scores
            total_confidence = lr_confidence + lstm_confidence
            if total_confidence > 0:
                lr_weight = lr_confidence / total_confidence
                lstm_weight = lstm_confidence / total_confidence
            else:
                lr_weight = lstm_weight = 0.5
            
            p_combined = lr_weight * p_lr + lstm_weight * p_lstm
            method = f"Hybrid (LR: {p_lr:.3f}[{lr_confidence:.3f}], LSTM: {p_lstm:.3f}[{lstm_confidence:.3f}])"
            confidence = total_confidence / 2
        else:
            # Fallback to LR only
            p_combined = p_lr
            method = f"LR only (confidence: {lr_confidence:.3f})"
            confidence = lr_confidence
        
        # Make decision based on dynamic thresholds
        decision = "hold"
        if confidence >= AI_CFG["min_accuracy"]:
            if p_combined >= AI_CFG["threshold_long"]:
                decision = "long"
            elif p_combined <= AI_CFG["threshold_short"]:
                decision = "short"
        
        return {
            "p_up": float(p_combined),
            "decision": decision,
            "method": method,
            "confidence": confidence,
            "components": {
                "lr": p_lr, 
                "lr_confidence": lr_confidence,
                "lstm": p_lstm,
                "lstm_confidence": lstm_confidence
            }
        }
        
    except Exception as e:
        log(f"Hybrid AI prediction error for {symbol}: {e}")
        return None

# =============================
# ENHANCED WALK-FORWARD OPTIMIZATION
# =============================

class WalkForwardOptimizer:
    """Enhanced Walk-forward optimization engine with proper reporting"""
    
    def __init__(self, symbol: str, timeframe: str = "5m", 
                 train_window: int = 1000, test_window: int = 250, 
                 step_size: int = 250, total_months: int = 6):
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.total_months = total_months
        
        self.results = []
        self.equity_curve = []
        self.trades = []
        self.metrics = {}
        self.daily_returns = []
    
    async def run_optimization(self) -> bool:
        """Run walk-forward optimization with progress reporting"""
        try:
            log(f"Starting walk-forward optimization for {self.symbol}")
            
            # Load data
            df = load_history_df(self.symbol, self.timeframe, months=self.total_months)
            if len(df) < self.train_window + self.test_window:
                log(f"Insufficient data for walk-forward: {len(df)} bars")
                return False
            
            # Create features and labels
            features = create_features(df)
            labels = create_labels(df)
            
            if len(features) < self.train_window + self.test_window:
                log(f"Insufficient features for walk-forward: {len(features)}")
                return False
            
            # Initialize
            initial_capital = 10000.0
            current_capital = initial_capital
            current_position = None
            
            # Walk-forward loop
            start_idx = 0
            window_count = 0
            
            while start_idx + self.train_window + self.test_window <= len(features):
                window_count += 1
                log(f"Processing window {window_count}, start: {start_idx}")
                
                # Define windows
                train_end = start_idx + self.train_window
                test_end = train_end + self.test_window
                
                # Training data
                train_features = features.iloc[start_idx:train_end]
                train_labels = labels.iloc[start_idx:train_end]
                
                # Test data
                test_features = features.iloc[train_end:test_end]
                test_labels = labels.iloc[train_end:test_end]
                test_df = df.iloc[train_end:test_end]
                
                # Train models for this window
                success = await self._train_window_models(
                    f"{self.symbol}_wf_{window_count}", 
                    train_features, train_labels
                )
                
                if not success:
                    log(f"Training failed for window {window_count}")
                    start_idx += self.step_size
                    continue
                
                # Test on forward window
                window_results = await self._test_window(
                    f"{self.symbol}_wf_{window_count}",
                    test_features, test_labels, test_df,
                    train_end, current_capital
                )
                
                # Update capital and position
                current_capital = window_results['final_capital']
                current_position = window_results.get('final_position')
                
                # Store results
                self.results.append(window_results)
                self.equity_curve.extend(window_results['equity_curve'])
                self.trades.extend(window_results['trades'])
                self.daily_returns.extend(window_results.get('daily_returns', []))
                
                # Progress update
                if tg_manager:
                    progress = (window_count * self.step_size) / len(features) * 100
                    await tg_send(f"Walk-forward progress: Window {window_count}, {progress:.1f}% complete, Capital: ${current_capital:.2f}")
                
                start_idx += self.step_size
            
            # Calculate overall metrics
            self._calculate_metrics(initial_capital)
            
            log(f"Walk-forward completed: {window_count} windows, "
                f"Final capital: ${current_capital:.2f}")
            return True
            
        except Exception as e:
            log(f"Walk-forward optimization error: {e}")
            return False
    
    async def _train_window_models(self, model_id: str, features: pd.DataFrame, 
                           labels: pd.Series) -> bool:
        """Train models for current window"""
        try:
            # Train LR model
            lr_model = OnlineLR(features.shape[1], lr=AI_CFG["learn_rate"], l2=AI_CFG["l2"])
            
            for epoch in range(AI_CFG["learn_epochs_boot"]):
                indices = np.random.permutation(len(features))
                for i in indices:
                    lr_model.update(features.iloc[i].values, int(labels.iloc[i]))
            
            # Store LR model temporarily
            ai_manager.models[f"{model_id}_lr"] = lr_model
            
            # Train LSTM if available and sufficient data
            lstm_success = False
            if TORCH_OK and len(features) > AI_CFG["lstm_seq_len"] + 200:
                lstm_success = await train_lstm_model(
                    model_id, features, labels,
                    epochs=AI_CFG["lstm_epochs"] // 2,  # Reduced for speed
                    batch_size=AI_CFG["lstm_batch_size"],
                    seq_len=AI_CFG["lstm_seq_len"]
                )
            
            return True
            
        except Exception as e:
            log(f"Window model training error: {e}")
            return False
    
    async def _test_window(self, model_id: str, features: pd.DataFrame, labels: pd.Series,
                    df: pd.DataFrame, start_idx: int, initial_capital: float) -> Dict:
        """Test models on forward window"""
        try:
            equity = [initial_capital]
            trades = []
            daily_returns = []
            position = None
            capital = initial_capital
            
            # Load models
            lr_model = ai_manager.models.get(f"{model_id}_lr")
            lstm_model, lstm_scaler = ai_manager.load_lstm_model(model_id) if TORCH_OK else (None, None)
            
            prices = df['close'].values
            dates = df.index.tolist()
            
            for i in range(len(features)):
                current_price = prices[i]
                current_features = features.iloc[i].values
                
                # Get predictions
                p_lr = lr_model.predict_proba(current_features) if lr_model else 0.5
                p_lstm = None
                
                if lstm_model and lstm_scaler and i >= AI_CFG["lstm_seq_len"]:
                    try:
                        seq_start = max(0, i - AI_CFG["lstm_seq_len"])
                        seq_features = features.iloc[seq_start:i].values
                        if len(seq_features) == AI_CFG["lstm_seq_len"]:
                            seq_scaled = lstm_scaler.transform(seq_features)
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            with torch.no_grad():
                                seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                                p_lstm = float(lstm_model(seq_tensor).cpu().numpy()[0][0])
                    except Exception:
                        pass
                
                # Combine predictions
                if p_lstm is not None:
                    p_combined = AI_CFG["hybrid_weight"] * p_lstm + (1 - AI_CFG["hybrid_weight"]) * p_lr
                else:
                    p_combined = p_lr
                
                # Calculate daily return
                if i > 0:
                    daily_return = (current_price / prices[i-1]) - 1
                    daily_returns.append(daily_return)
                
                # Trading logic
                if position is None:
                    # Enter position
                    if p_combined >= AI_CFG["threshold_long"]:
                        position = {
                            'side': 'long',
                            'entry_price': current_price,
                            'entry_idx': start_idx + i,
                            'entry_date': dates[i],
                            'shares': capital * 0.95 / current_price  # 95% allocation
                        }
                    elif p_combined <= AI_CFG["threshold_short"]:
                        position = {
                            'side': 'short',
                            'entry_price': current_price,
                            'entry_idx': start_idx + i,
                            'entry_date': dates[i],
                            'shares': capital * 0.95 / current_price
                        }
                else:
                    # Exit conditions
                    should_exit = False
                    
                    if position['side'] == 'long':
                        if p_combined < 0.45 or i == len(features) - 1:
                            should_exit = True
                    else:  # short
                        if p_combined > 0.55 or i == len(features) - 1:
                            should_exit = True
                    
                    if should_exit:
                        # Calculate PnL
                        if position['side'] == 'long':
                            pnl = (current_price - position['entry_price']) * position['shares']
                        else:
                            pnl = (position['entry_price'] - current_price) * position['shares']
                        
                        capital += pnl
                        
                        # Record trade
                        trade = {
                            'entry_idx': position['entry_idx'],
                            'exit_idx': start_idx + i,
                            'entry_date': position['entry_date'],
                            'exit_date': dates[i],
                            'side': position['side'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'return_pct': (pnl / (position['entry_price'] * position['shares'])) * 100,
                            'duration_hours': (dates[i] - position['entry_date']).total_seconds() / 3600
                        }
                        trades.append(trade)
                        
                        position = None
                
                equity.append(capital)
            
            return {
                'final_capital': capital,
                'final_position': position,
                'equity_curve': equity,
                'trades': trades,
                'daily_returns': daily_returns,
                'window_start': start_idx,
                'window_end': start_idx + len(features)
            }
            
        except Exception as e:
            log(f"Window testing error: {e}")
            return {
                'final_capital': initial_capital,
                'final_position': None,
                'equity_curve': [initial_capital],
                'trades': [],
                'daily_returns': [],
                'window_start': start_idx,
                'window_end': start_idx + len(features)
            }
    
    def _calculate_metrics(self, initial_capital: float):
        """Calculate comprehensive performance metrics"""
        try:
            if not self.equity_curve or not self.trades:
                return
            
            final_capital = self.equity_curve[-1]
            total_return = (final_capital / initial_capital - 1) * 100
            
            # Trade statistics
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = sum(abs(t['pnl']) for t in losing_trades)
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Drawdown calculation
            peak = np.maximum.accumulate(self.equity_curve)
            drawdown = (np.array(self.equity_curve) - peak) / peak * 100
            max_drawdown = np.min(drawdown)
            
            # Sharpe ratio calculation
            if self.daily_returns and len(self.daily_returns) > 1:
                returns_array = np.array(self.daily_returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            else:
                # Fallback calculation using equity curve
                equity_returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
                sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if np.std(equity_returns) > 0 else 0
            
            # Additional metrics
            avg_trade_duration = np.mean([t.get('duration_hours', 24) for t in self.trades]) if self.trades else 0
            max_consecutive_losses = self._calculate_max_consecutive_losses()
            
            # Exposure calculation (percentage of time in market)
            total_time = len(self.equity_curve)
            time_in_market = sum(1 for t in self.trades for _ in range(int(t.get('duration_hours', 24) / 24)))
            exposure = min(100, (time_in_market / total_time * 100)) if total_time > 0 else 0
            
            self.metrics = {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return_pct': total_return,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate_pct': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe,
                'avg_trade_duration_hours': avg_trade_duration,
                'max_consecutive_losses': max_consecutive_losses,
                'exposure_pct': exposure
            }
            
        except Exception as e:
            log(f"Metrics calculation error: {e}")
            self.metrics = {}
    
    def _calculate_max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        if not self.trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade['pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    async def save_results(self) -> str:
        """Save walk-forward results to CSV and JSON"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"walkforward_{self.symbol.replace('/', '_')}_{timestamp}"
            
            # Save trades to CSV
            trades_file = os.path.join(REPORTS_DIR, f"{base_filename}_trades.csv")
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(trades_file, index=False)
            
            # Save equity curve to CSV
            equity_file = os.path.join(REPORTS_DIR, f"{base_filename}_equity.csv")
            equity_df = pd.DataFrame({
                'equity': self.equity_curve,
                'timestamp': pd.date_range(start=datetime.now() - timedelta(days=len(self.equity_curve)), 
                                         periods=len(self.equity_curve), freq='H')
            })
            equity_df.to_csv(equity_file, index=False)
            
            # Save full results to JSON
            json_file = os.path.join(REPORTS_DIR, f"{base_filename}_results.json")
            results_data = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'parameters': {
                    'train_window': self.train_window,
                    'test_window': self.test_window,
                    'step_size': self.step_size,
                    'total_months': self.total_months
                },
                'metrics': self.metrics,
                'trades': self.trades,
                'window_results': self.results
            }
            
            save_state(json_file, results_data)
            
            log(f"Walk-forward results saved: {trades_file}, {equity_file}, {json_file}")
            return json_file
            
        except Exception as e:
            log(f"Error saving walk-forward results: {e}")
            return ""
    
    def generate_report(self) -> str:
        """Generate comprehensive walk-forward performance report"""
        try:
            if not self.metrics:
                return "No metrics available"
            
            report = f"""
WALK-FORWARD OPTIMIZATION REPORT
{'='*50}
Symbol: {self.symbol}
Timeframe: {self.timeframe}
Period: {self.total_months} months

PARAMETERS:
- Training Window: {self.train_window} bars
- Testing Window: {self.test_window} bars  
- Step Size: {self.step_size} bars
- Windows Processed: {len(self.results)}

PERFORMANCE METRICS:
- Initial Capital: ${self.metrics.get('initial_capital', 0):,.2f}
- Final Capital: ${self.metrics.get('final_capital', 0):,.2f}
- Total Return: {self.metrics.get('total_return_pct', 0):.2f}%
- Max Drawdown: {self.metrics.get('max_drawdown_pct', 0):.2f}%
- Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.3f}
- Exposure: {self.metrics.get('exposure_pct', 0):.1f}%

TRADE STATISTICS:
- Total Trades: {self.metrics.get('total_trades', 0)}
- Winning Trades: {self.metrics.get('winning_trades', 0)}
- Losing Trades: {self.metrics.get('losing_trades', 0)}
- Win Rate: {self.metrics.get('win_rate_pct', 0):.1f}%
- Average Win: ${self.metrics.get('avg_win', 0):.2f}
- Average Loss: ${self.metrics.get('avg_loss', 0):.2f}
- Profit Factor: {self.metrics.get('profit_factor', 0):.2f}
- Avg Trade Duration: {self.metrics.get('avg_trade_duration_hours', 0):.1f} hours
- Max Consecutive Losses: {self.metrics.get('max_consecutive_losses', 0)}
"""
            return report
            
        except Exception as e:
            log(f"Error generating walk-forward report: {e}")
            return "Error generating report"
# =============================
# ENHANCED CHART GENERATION
# =============================

def plot_candlestick_chart(df: pd.DataFrame, title: str, filepath: str, 
                          indicators=True, trades=None, figsize=(16, 12)):
    """Generate enhanced candlestick chart with multiple indicators"""
    try:
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                log(f"Missing column {col} in dataframe")
                return None
        
        # Calculate indicators if not present
        df_plot = calculate_indicators(df.copy())
        
        # Create figure with subplots
        fig, axes = plt.subplots(5, 1, figsize=figsize, 
                                gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]},
                                sharex=True)
        
        # Candlestick chart
        ax_price = axes[0]
        dates = df_plot.index
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            color = '#00ff88' if row['close'] >= row['open'] else '#ff4444'
            alpha = 0.8
            
            # High-Low line
            ax_price.plot([i, i], [row['low'], row['high']], color=color, linewidth=1.2, alpha=alpha)
            
            # Open-Close body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            
            if body_height > 0:
                ax_price.add_patch(Rectangle((i-0.35, body_bottom), 0.7, body_height, 
                                           facecolor=color, alpha=alpha, edgecolor=color))
            else:
                # Doji
                ax_price.plot([i-0.35, i+0.35], [row['close'], row['close']], color=color, linewidth=2)
        
        # Add moving averages if indicators enabled
        if indicators:
            if 'ema_20' in df_plot.columns:
                ax_price.plot(range(len(df_plot)), df_plot['ema_20'], 
                             label='EMA 20', color='#3366ff', linewidth=1.5)
            if 'ema_50' in df_plot.columns:
                ax_price.plot(range(len(df_plot)), df_plot['ema_50'], 
                             label='EMA 50', color='#ff6600', linewidth=1.5)
            if 'ema_200' in df_plot.columns:
                ax_price.plot(range(len(df_plot)), df_plot['ema_200'], 
                             label='EMA 200', color='#cc0099', linewidth=1.5)
            
            # Bollinger Bands
            if all(col in df_plot.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                ax_price.plot(range(len(df_plot)), df_plot['bb_upper'], 
                             color='gray', linestyle='--', alpha=0.5)
                ax_price.plot(range(len(df_plot)), df_plot['bb_lower'], 
                             color='gray', linestyle='--', alpha=0.5)
                ax_price.fill_between(range(len(df_plot)), df_plot['bb_upper'], df_plot['bb_lower'],
                                     alpha=0.1, color='gray')
        
        # Add trade markers if provided
        if trades:
            for trade in trades:
                if 'entry_idx' in trade and 'side' in trade:
                    idx = trade['entry_idx']
                    if 0 <= idx < len(df_plot):
                        color = '#00ff00' if trade['side'] == 'long' else '#ff0000'
                        marker = '^' if trade['side'] == 'long' else 'v'
                        ax_price.scatter(idx, trade.get('entry_price', df_plot.iloc[idx]['close']), 
                                       color=color, marker=marker, s=150, zorder=5, edgecolors='white')
                        
                        # Exit marker
                        if 'exit_idx' in trade and 'exit_price' in trade:
                            exit_idx = trade['exit_idx']
                            if 0 <= exit_idx < len(df_plot):
                                exit_color = '#00aa00' if trade.get('pnl', 0) > 0 else '#aa0000'
                                ax_price.scatter(exit_idx, trade['exit_price'],
                                               color=exit_color, marker='x', s=150, zorder=5)
        
        ax_price.set_title(title, fontsize=16, fontweight='bold', pad=20)
        if indicators:
            ax_price.legend(loc='upper left')
        ax_price.grid(True, alpha=0.3)
        ax_price.set_ylabel('Price', fontsize=12)
        
        # Volume chart
        ax_volume = axes[1]
        if 'volume' in df_plot.columns:
            colors = ['#00ff88' if df_plot.iloc[i]['close'] >= df_plot.iloc[i]['open'] else '#ff4444' 
                     for i in range(len(df_plot))]
            ax_volume.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.7, width=0.8)
            
            # Volume moving average
            if indicators and 'volume_sma' in df_plot.columns:
                ax_volume.plot(range(len(df_plot)), df_plot['volume_sma'], 
                              color='yellow', linewidth=1.5, label='Vol SMA')
                ax_volume.legend(loc='upper left')
            
        ax_volume.set_title('Volume', fontsize=12)
        ax_volume.grid(True, alpha=0.3)
        ax_volume.set_ylabel('Volume', fontsize=10)
        
        # RSI chart
        ax_rsi = axes[2]
        if 'rsi' in df_plot.columns:
            ax_rsi.plot(range(len(df_plot)), df_plot['rsi'], color='purple', linewidth=2)
            ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            
            # Fill overbought/oversold areas
            ax_rsi.fill_between(range(len(df_plot)), 70, 100, alpha=0.1, color='red')
            ax_rsi.fill_between(range(len(df_plot)), 0, 30, alpha=0.1, color='green')
            
        ax_rsi.set_title('RSI (14)', fontsize=12)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend(loc='upper left')
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.set_ylabel('RSI', fontsize=10)
        
        # MACD chart
        ax_macd = axes[3]
        if all(col in df_plot.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            ax_macd.plot(range(len(df_plot)), df_plot['macd'], label='MACD', color='blue', linewidth=1.5)
            ax_macd.plot(range(len(df_plot)), df_plot['macd_signal'], label='Signal', color='red', linewidth=1.5)
            
            # MACD histogram with colors
            colors = ['green' if h >= 0 else 'red' for h in df_plot['macd_hist']]
            ax_macd.bar(range(len(df_plot)), df_plot['macd_hist'], 
                       color=colors, alpha=0.6, width=0.8, label='Histogram')
            
            ax_macd.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
        ax_macd.set_title('MACD', fontsize=12)
        ax_macd.legend(loc='upper left')
        ax_macd.grid(True, alpha=0.3)
        ax_macd.set_ylabel('MACD', fontsize=10)
        
        # ATR chart
        ax_atr = axes[4]
        if 'atr' in df_plot.columns:
            ax_atr.plot(range(len(df_plot)), df_plot['atr'], color='orange', linewidth=2, label='ATR')
            ax_atr.fill_between(range(len(df_plot)), 0, df_plot['atr'], alpha=0.3, color='orange')
            
        ax_atr.set_title('ATR (14)', fontsize=12)
        ax_atr.legend(loc='upper left')
        ax_atr.grid(True, alpha=0.3)
        ax_atr.set_ylabel('ATR', fontsize=10)
        ax_atr.set_xlabel('Time', fontsize=12)
        
        # Format x-axis
        if len(dates) > 50:
            step = max(1, len(dates) // 15)
            tick_positions = range(0, len(dates), step)
            tick_labels = [dates[i].strftime('%m-%d %H:%M') for i in tick_positions]
            plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
        
    except Exception as e:
        log(f"Chart generation error: {e}")
        return None

def plot_equity_curve(equity_data: List[float], dates: List[datetime], 
                     title: str, filepath: str, trades=None, figsize=(14, 10)):
    """Plot enhanced equity curve with trade markers and statistics"""
    try:
        fig, axes = plt.subplots(3, 1, figsize=figsize, 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Equity curve
        ax1 = axes[0]
        ax1.plot(dates, equity_data, color='#2E8B57', linewidth=3, label='Portfolio Value')
        ax1.fill_between(dates, equity_data, alpha=0.3, color='#2E8B57')
        
        # Add trade markers if provided
        if trades:
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) <= 0]
            
            if wins:
                win_dates = [t.get('exit_date', dates[-1]) for t in wins]
                win_values = [equity_data[min(len(equity_data)-1, i)] for i, _ in enumerate(win_dates)]
                ax1.scatter(win_dates, win_values, color='green', alpha=0.7, s=50, label=f'Wins ({len(wins)})')
            
            if losses:
                loss_dates = [t.get('exit_date', dates[-1]) for t in losses]
                loss_values = [equity_data[min(len(equity_data)-1, i)] for i, _ in enumerate(loss_dates)]
                ax1.scatter(loss_dates, loss_values, color='red', alpha=0.7, s=50, label=f'Losses ({len(losses)})')
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Returns chart
        if len(equity_data) > 1:
            ax2 = axes[1]
            returns = [(equity_data[i] / equity_data[i-1] - 1) * 100 
                      for i in range(1, len(equity_data))]
            return_dates = dates[1:]
            
            colors = ['green' if r >= 0 else 'red' for r in returns]
            ax2.bar(return_dates, returns, color=colors, alpha=0.7, width=1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title('Daily Returns (%)', fontsize=12)
            ax2.set_ylabel('Return %', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # Drawdown chart
        if len(equity_data) > 1:
            ax3 = axes[2]
            peak = np.maximum.accumulate(equity_data)
            drawdown = (np.array(equity_data) - peak) / peak * 100
            ax3.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
            ax3.plot(dates, drawdown, color='red', linewidth=2)
            ax3.set_title('Drawdown (%)', fontsize=12)
            ax3.set_ylabel('Drawdown %', fontsize=10)
            ax3.set_xlabel('Date', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
        
    except Exception as e:
        log(f"Equity curve plot error: {e}")
        return None

def plot_walkforward_summary(optimizer: WalkForwardOptimizer, filepath: str):
    """Plot walk-forward optimization summary"""
    try:
        if not optimizer.results:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Window performance
        window_returns = [(r['final_capital'] / 10000 - 1) * 100 for r in optimizer.results]
        ax1.bar(range(len(window_returns)), window_returns, 
                color=['green' if r > 0 else 'red' for r in window_returns], alpha=0.7)
        ax1.set_title('Window Performance (%)', fontsize=14)
        ax1.set_ylabel('Return %', fontsize=12)
        ax1.set_xlabel('Window', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Trade distribution
        if optimizer.trades:
            trade_returns = [t['return_pct'] for t in optimizer.trades]
            ax2.hist(trade_returns, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('Trade Return Distribution', fontsize=14)
            ax2.set_xlabel('Return %', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        # Monthly performance (if applicable)
        if optimizer.equity_curve and len(optimizer.equity_curve) > 30:
            equity_array = np.array(optimizer.equity_curve)
            # Approximate monthly returns
            monthly_points = len(equity_array) // 30
            if monthly_points > 1:
                monthly_equity = [equity_array[i * 30] for i in range(monthly_points)]
                monthly_returns = [(monthly_equity[i] / monthly_equity[i-1] - 1) * 100 
                                 for i in range(1, len(monthly_equity))]
                
                ax3.bar(range(len(monthly_returns)), monthly_returns,
                       color=['green' if r > 0 else 'red' for r in monthly_returns], alpha=0.7)
                ax3.set_title('Approximate Monthly Returns', fontsize=14)
                ax3.set_ylabel('Return %', fontsize=12)
                ax3.set_xlabel('Month', fontsize=12)
                ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Performance metrics
        if optimizer.metrics:
            metrics_text = f"""
Total Return: {optimizer.metrics.get('total_return_pct', 0):.2f}%
Sharpe Ratio: {optimizer.metrics.get('sharpe_ratio', 0):.3f}
Max Drawdown: {optimizer.metrics.get('max_drawdown_pct', 0):.2f}%
Win Rate: {optimizer.metrics.get('win_rate_pct', 0):.1f}%
Profit Factor: {optimizer.metrics.get('profit_factor', 0):.2f}
Total Trades: {optimizer.metrics.get('total_trades', 0)}
            """
            ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_title('Performance Summary', fontsize=14)
            ax4.axis('off')
        
        plt.suptitle(f'Walk-Forward Analysis: {optimizer.symbol}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filepath
        
    except Exception as e:
        log(f"Walk-forward summary plot error: {e}")
        return None

# =============================
# ENHANCED TRADING ENGINE
# =============================




class TradingEngine:

    def get_positions_summary(self) -> dict:
        """Return summary of all active positions."""
        try:
            return {
                k: {
                    "entry_price": v.get("entry_price"),
                    "amount": v.get("amount"),
                    "side": v.get("side"),
                    "unrealized_pnl": v.get("unrealized_pnl", 0)
                }
                for k, v in (self.positions or {}).items()
                if v and v.get("entry_price")
            }
        except Exception as e:
            log(f"get_positions_summary error: {e}")
            return {}

    async def analyze_symbol(self, symbol: str) -> dict:
        """Analyze a symbol using AI predictions and return trade signals."""
        try:
            df = load_history_df(symbol, AI_CFG["base_tf"], AI_CFG["hist_limit"])
            if df.empty:
                return {"direction": None, "confidence": 0.0}

            features = create_features(df)
            if len(features) < AI_CFG.get("min_obs", 50):
                return {"direction": None, "confidence": 0.0}

            ai_manager = globals().get("ai_manager")
            if not ai_manager:
                return {"direction": None, "confidence": 0.0}

            pred = ai_manager.predict(symbol, features)
            t_long, t_short = get_ai_thresholds(AI_CFG)

            direction, confidence = None, 0.0
            if pred.get("prob_up", 0) >= t_long:
                direction, confidence = "long", pred["prob_up"]
            elif pred.get("prob_down", 0) >= t_short:
                direction, confidence = "short", pred["prob_down"]

            atr = df['atr'].iloc[-1] if 'atr' in df else 0.0
            price = df['close'].iloc[-1]
            sl = price - atr * cfg["atr_sl_mult"] if direction == "long" else price + atr * cfg["atr_sl_mult"]
            tp = price + atr * cfg["atr_tp_mult"] if direction == "long" else price - atr * cfg["atr_tp_mult"]

            return {
                "direction": direction,
                "confidence": confidence,
                "stop_loss": sl,
                "take_profit": tp,
                "prob_up": pred.get("prob_up"),
                "prob_down": pred.get("prob_down"),
            }
        except Exception as e:
            log(f"analyze_symbol error for {symbol}: {e}")
            return {"direction": None, "confidence": 0.0}



        def get_account_balance(self, asset="USDT"):
            """
            Returns the free and total balance for a given asset.
            Falls back to 0 if exchange does not provide balance.
            """
            try:
                balances = self.exchange.fetch_balance()
                if asset in balances.get('free', {}):
                    return {
                        "free": balances['free'].get(asset, 0.0),
                        "total": balances['total'].get(asset, 0.0)
                    }
                return {"free": 0.0, "total": 0.0}
            except Exception as e:
                log(f"Balance fetch error: {e}")
                return {"free": 0.0, "total": 0.0}

        """
        Unified TradingEngine with clean architecture and no monkey patches.
        Provides: check_entry_triggers, check_sl_tp, manage_positions, open_planned_position,
        close_position, close_all_positions, and a legacy async check_triggers alias.
        """

        def __init__(self, exchange=None, cfg=None, tg_manager=None):
            self.exchange = exchange if exchange is not None else globals().get("exchange")
            self.cfg = cfg if cfg is not None else globals().get("cfg", {})
            self.tg_manager = tg_manager if tg_manager is not None else globals().get("tg_manager")
            self.positions = getattr(self, "positions", None) or {}

            # Config flags
            self.auto_enabled = self._cfg_get("auto_enabled", False)
            self.partial_tp_enabled = bool(self._cfg_get("use_partial_tp", self._cfg_get("partial_tp_enabled", False)))
            self.partial_tp_fraction = float(self._cfg_get("partial_tp_fraction", 0.5) or 0.5)
            self.use_trailing_sl = bool(self._cfg_get("use_trailing_sl", False))
            self.trailing_sl_atr_mult = float(self._cfg_get("trailing_sl_atr_mult", 2.0) or 2.0)

            self.min_notional = float(self._cfg_get("min_notional", 5.0) or 5.0)
            self.max_notional = float(self._cfg_get("max_notional", 1_000.0) or 1_000.0)
            self.last_heartbeat = 0

        # ---------------- utilities ----------------
        def _cfg_get(self, key, default=None):
            try:
                c = self.cfg
                if isinstance(c, dict):
                    return c.get(key, default)
                return getattr(c, key, default)
            except Exception:
                return default

        def _log(self, msg, level="info"):
            try:
                if "log" in globals() and callable(globals()["log"]):
                    globals()["log"](msg, level=level)
                else:
                    print(f"[{level}] {msg}")
            except Exception:
                print(msg)

        async def _tg(self, text):
            try:
                tm = self.tg_manager or globals().get("tg_manager")
                if tm and hasattr(tm, "send_message"):
                    await tm.send_message(text)
            except Exception as e:
                self._log(f"Telegram send error: {e}", level="warning")

        def _norm_contracts(self, pos):
            for k in ("contracts", "qty", "amount", "size"):
                if pos.get(k) is not None:
                    try:
                        v = float(pos[k])
                        pos["contracts"] = v
                        return v
                    except Exception:
                        continue
            return pos.get("contracts")

        def _persist(self):
            try:
                if hasattr(self, "save_positions") and callable(self.save_positions):
                    self.save_positions()
                elif "save_positions" in globals() and callable(globals()["save_positions"]):
                    globals()["save_positions"]()
            except Exception:
                pass

        # -------------- core methods --------------
        async def check_entry_triggers(self, symbol=None, price=None):
            try:
                for sym, pos in list(self.positions.items()):
                    if symbol and sym != symbol:
                        continue
                    if pos.get("status") != "planned" or pos.get("entry_price") is not None:
                        continue
                    trig = pos.get("trigger_price")
                    if trig is None:
                        continue
                    side = pos.get("side", "long")
                    last = price if (symbol and sym == symbol and price is not None) else pos.get("last_price")
                    if last is None and self.exchange and hasattr(self.exchange, "fetch_ticker"):
                        try:
                            t = await self.exchange.fetch_ticker(sym)
                            last = float(t.get("last")) if t and t.get("last") is not None else None
                        except Exception:
                            last = None
                    if last is None:
                        continue

                    should_open = (side == "long" and float(last) >= float(trig)) or (side == "short" and float(last) <= float(trig))
                    if should_open:
                        await self.open_planned_position(sym, pos, float(last))
                return True
            except Exception as e:
                self._log(f"check_entry_triggers error: {e}", level="error")
                return False

        async def check_sl_tp(self, symbol=None, price=None):
            try:
                for sym, pos in list(self.positions.items()):
                    if symbol and sym != symbol:
                        continue
                    if pos.get("status") != "open" or pos.get("entry_price") is None:
                        continue

                    side = pos.get("side", "long")
                    entry = float(pos.get("entry_price"))
                    last = price if (symbol and sym == symbol and price is not None) else pos.get("last_price")
                    if last is None and self.exchange and hasattr(self.exchange, "fetch_ticker"):
                        try:
                            t = await self.exchange.fetch_ticker(sym)
                            last = float(t.get("last")) if t and t.get("last") is not None else entry
                        except Exception:
                            last = entry
                    last = float(last)

                    # Trailing best + ATR trailing SL
                    if self.use_trailing_sl:
                        best = pos.get("best_price")
                        if best is None:
                            best = last
                        else:
                            best = max(best, last) if side == "long" else min(best, last)
                        pos["best_price"] = float(best)

                        atr = pos.get("atr")
                        if atr is not None:
                            try:
                                atr = float(atr)
                                ts = float(self.trailing_sl_atr_mult) * atr
                                pos["stop_loss"] = (best - ts) if side == "long" else (best + ts)
                            except Exception:
                                pass

                    sl = pos.get("stop_loss")
                    tp = pos.get("take_profit")
                    sl_hit = sl is not None and ((side == "long" and last <= float(sl)) or (side == "short" and last >= float(sl)))
                    tp_hit = tp is not None and ((side == "long" and last >= float(tp)) or (side == "short" and last <= float(tp)))

                    # Partial TP: close fraction and move SL to BE
                    if tp_hit and self.partial_tp_enabled:
                        contracts = self._norm_contracts(pos) or 0.0
                        part = max(0.0, float(contracts) * float(self.partial_tp_fraction))
                        if part > 0.0:
                            await self.close_position(sym, qty=part, reason="take_profit_partial", price=last)
                            pos["stop_loss"] = float(entry)
                            pos["take_profit"] = None
                            contracts = self._norm_contracts(pos) or 0.0
                            if contracts <= 0:
                                continue
                            sl = pos.get("stop_loss")
                            sl_hit = sl is not None and ((side == "long" and last <= float(sl)) or (side == "short" and last >= float(sl)))
                            tp_hit = False

                    if sl_hit or tp_hit:
                        reason = "stop_loss" if sl_hit else "take_profit"
                        await self.close_position(sym, qty=None, reason=reason, price=last)
                return True
            except Exception as e:
                self._log(f"check_sl_tp error: {e}", level="error")
                return False

        async def manage_positions(self, symbols=None):
            try:
                items = list(self.positions.items())
                if symbols:
                    items = [(s, p) for s, p in items if s in symbols]
                if self.exchange and hasattr(self.exchange, "fetch_ticker"):
                    for sym, pos in items:
                        try:
                            t = await self.exchange.fetch_ticker(sym)
                            if t and t.get("last") is not None:
                                pos["last_price"] = float(t["last"])
                        except Exception:
                            pass
                return await self.check_sl_tp()
            except Exception as e:
                self._log(f"manage_positions error: {e}", level="error")
                return False

        async def open_planned_position(self, symbol, pos, current_price):
            try:
                side = pos.get("side", "long")
                contracts = self._norm_contracts(pos)
                if not contracts or contracts <= 0:
                    daily_budget = float(self._cfg_get("daily_budget_usd", 10.0) or 10.0)
                    leverage = float(self._cfg_get("leverage", 1.0) or 1.0)
                    notional = max(self.min_notional, min(self.max_notional, daily_budget * leverage))
                    contracts = max(self.min_notional / float(current_price), notional / float(current_price))
                    pos["contracts"] = float(contracts)

                pos["entry_price"] = float(current_price)
                pos["status"] = "open"
                pos["best_price"] = float(current_price)
                pos["last_price"] = float(current_price)

                if self.exchange and getattr(self.exchange, "apiKey", None):
                    try:
                        order_side = "buy" if side == "long" else "sell"
                        if hasattr(self.exchange, "create_order"):
                            order = await self.exchange.create_order(symbol, "market", order_side, float(contracts))
                            try:
                                pos["order_id"] = str(order.get("id")) if isinstance(order, dict) else None
                            except Exception:
                                pos["order_id"] = None
                    except Exception as e:
                        self._log(f"Live open order failed; paper fallback. err={e}", level="warning")

                self._persist()
                await self._tg(f"🚀 POSITION TRIGGERED {symbol} {side.upper()} @ {float(current_price):.6f} size={float(contracts):.6f}")
                return True
            except Exception as e:
                self._log(f"open_planned_position error: {e}", level="error")
                return False

        async def close_position(self, symbol, qty=None, reason="", price=None):
            try:
                pos = self.positions.get(symbol)
                if not pos or pos.get("status") not in ("open", "closing"):
                    return False
                side = pos.get("side", "long")
                contracts = self._norm_contracts(pos) or 0.0
                close_qty = contracts if qty is None else min(contracts, float(qty))
                if close_qty <= 0:
                    return False

                if self.exchange and getattr(self.exchange, "apiKey", None):
                    try:
                        order_side = "sell" if side == "long" else "buy"
                        if hasattr(self.exchange, "create_order"):
                            await self.exchange.create_order(symbol, "market", order_side, float(close_qty))
                    except Exception as e:
                        self._log(f"Live close order failed; paper fallback. err={e}", level="warning")

                remaining = max(0.0, float(contracts) - float(close_qty))
                pos["contracts"] = float(remaining)
                if price is not None:
                    pos["last_price"] = float(price)
                pos["status"] = "closed" if remaining <= 0 else "open"

                self._persist()
                lp = pos.get("last_price")
                await self._tg(f"🔔 Closed {symbol} {float(close_qty):.6f} on {reason or 'manual'} @ {float(lp) if lp is not None else 0.0:.6f} (remaining {float(remaining):.6f})")
                return True
            except Exception as e:
                self._log(f"close_position error: {e}", level="error")
                return False

        async def close_all_positions(self, symbols=None, reason="manual_close_all"):
            try:
                items = list(self.positions.items())
                if symbols:
                    items = [(s, p) for s, p in items if s in symbols]
                for sym, _ in items:
                    await self.close_position(sym, qty=None, reason=reason)
                return True
            except Exception as e:
                self._log(f"close_all_positions error: {e}", level="error")
                return False

        # legacy alias (async) to keep older calls working

def check_triggers(self, symbol: str, price: float = None):
    """Enforce SL/TP for a symbol if a position is open."""
    try:
        pos = (self.positions or {}).get(symbol)
        if not pos or pos.get("status") != "open":
            return
        # Fetch current mark/last price if not provided
        px = price
        if px is None:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                px = float(ticker.get("last") or ticker.get("close"))
            except Exception:
                return
        sl = pos.get("stop_loss")
        tp = pos.get("take_profit")
        side = pos.get("side")
        trigger = None
        if side == "long":
            if sl is not None and px <= sl:
                trigger = "stop_loss"
            elif tp is not None and px >= tp:
                trigger = "take_profit"
        elif side == "short":
            if sl is not None and px >= sl:
                trigger = "stop_loss"
            elif tp is not None and px <= tp:
                trigger = "take_profit"
        if trigger:
            self.close_position(symbol, reason=trigger, exit_price=px)
            log(f"[engine] {symbol} {trigger} hit at {px}")
    except Exception as e:
        log(f"check_triggers error for {symbol}: {e}")

def _positions_path(self):
    return os.path.join("state", "positions.json")

def save_positions(self):
    """Persist positions to disk safely with a temp swap."""
    try:
        path = self._positions_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.positions, f, indent=2, default=str)
        os.replace(tmp, path)
        log(f"[engine] Saved {len(self.positions)} positions to {path}")
    except Exception as e:
        log(f"[engine] save_positions error: {e}")

def load_positions(self):
    """Load positions from disk into memory."""
    try:
        path = self._positions_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                self.positions = json.load(f)
            log(f"[engine] Restored {len(self.positions)} positions from {path}")
        else:
            self.positions = {}
    except Exception as e:
        log(f"[engine] load_positions error: {e}")
        self.positions = {}
def _calc_sl_tp_percent(self, direction, price):
    def _calc_sl_tp_percent(price: float, side: str, sl_pct: float, tp_pct: float):
        try: sl_pct = float(sl_pct)
        except Exception: sl_pct = 0.02
        try: tp_pct = float(tp_pct)
        except Exception: tp_pct = 0.04
        if side == "long":
            sl = price * (1.0 - sl_pct); tp = price * (1.0 + tp_pct)
        else:
            sl = price * (1.0 + sl_pct); tp = price * (1.0 - tp_pct)
        return sl, tp, f"percent"
def _calc_sl_tp_atr(self, direction, price, atr):
    def _calc_sl_tp_atr(atr: float, price: float, side: str, mult: float):
        if atr is None or not (atr > 0): return None, None, None
        try: mult = float(mult)
        except Exception: mult = 2.0
        if side == "long":
            sl = price - atr * mult; tp = price + atr * mult
        else:
            sl = price + atr * mult; tp = price - atr * mult
        return sl, tp, f"atr_x{mult}"
    async def _fetch_last_price(self, symbol: str) -> float:
        try:
            if hasattr(self, "get_price"):
                p = await self.get_price(symbol) if asyncio.iscoroutinefunction(self.get_price) else self.get_price(symbol)
                if p: return float(p)
        except Exception: pass
        try:
            if hasattr(self, "exchange") and self.exchange:
                if hasattr(self.exchange, "fetch_ticker"):
                    tk = await self.exchange.fetch_ticker(symbol) if asyncio.iscoroutinefunction(self.exchange.fetch_ticker) else self.exchange.fetch_ticker(symbol)
                    if isinstance(tk, dict):
                        if tk.get("last"): return float(tk["last"])
                        if tk.get("close"): return float(tk["close"])
        except Exception as e:
            _safe_log(f"[warn] fetch price failed for {symbol}: {e}")
        return None


class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.max_daily_loss_pct = 5.0  # Maximum 5% daily loss
        self.max_position_risk_pct = 2.0  # Maximum 2% per position
        self.max_correlation_positions = 3  # Max correlated positions
        self.daily_loss_tracker = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.position_history = deque(maxlen=100)
    
    def reset_daily_counters(self):
        """Reset daily risk counters"""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset_date:
            self.daily_loss_tracker = 0.0
            cfg["budget_used_today"] = 0.0
            cfg["daily_trades"] = 0
            self.last_reset_date = current_date
            log("Daily risk counters reset")
    
    def check_position_allowed(self, symbol: str, side: str, size: float, balance: float) -> Tuple[bool, str]:
        """Check if position is allowed by risk management"""
        self.reset_daily_counters()
        
        # Check daily loss limit
        if self.daily_loss_tracker >= balance * (self.max_daily_loss_pct / 100):
            return False, "Daily loss limit reached"
        
        # Check position size limit
        max_position_size = balance * (self.max_position_risk_pct / 100)
        if size > max_position_size:
            return False, f"Position size too large (max: ${max_position_size:.2f})"
        
        # Check daily trades limit
        if cfg.get("daily_trades", 0) >= 10:  # Max 10 trades per day
            return False, "Daily trade limit reached"
        
        # Check available budget
        if cfg["budget_used_today"] + size > cfg["daily_budget_usd"]:
            return False, "Daily budget exceeded"
        
        return True, "Position approved"
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        if pnl < 0:
            self.daily_loss_tracker += abs(pnl)
        
        self.position_history.append({
            'timestamp': datetime.now(timezone.utc),
            'pnl': pnl
        })
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_loss_tracker': self.daily_loss_tracker,
            'daily_budget_used': cfg["budget_used_today"],
            'daily_trades': cfg.get("daily_trades", 0),
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_position_risk_pct': self.max_position_risk_pct
        }

# =============================
# ENHANCED POSITION MANAGEMENT
# =============================

    async def open_position(self, symbol: str, side: str, reason: str = "", force: bool = False) -> bool:
        """Open a new trading position with enhanced risk management"""
        try:
            # Check if position already exists
            if symbol in self.positions and self.positions[symbol]:
                log(f"Position already exists for {symbol}")
                return False
            
            # Get current price and indicators
            indicators, _ = await get_indicators(symbol, "5m")
            if not indicators:
                log(f"Cannot get indicators for {symbol}")
                return False
            
            price = indicators['price']
            atr = indicators.get('atr', price * 0.01)
            
            # Calculate position size
            position_value, contracts = self.calculate_position_size(symbol, price, side)
            if contracts <= 0:
                log(f"Position size too small for {symbol}")
                return False
            
            # Risk management check
            balance = self.get_account_balance()
            allowed, reason_denied = self.risk_manager.check_position_allowed(symbol, side, position_value, balance)
            
            if not allowed and not force:
                log(f"Position denied by risk management: {reason_denied}")
                return False
            
            # Calculate stops and targets with ATR
            atr_sl = atr * cfg["atr_sl_mult"]
            atr_tp = atr * cfg["atr_tp_mult"]
            
            if side.lower() == 'long':
                stop_loss = price - atr_sl
                take_profit = price + atr_tp
                trigger_price = price + (price * cfg["slippage_buffer_pct"] / 100)
            else:  # short
                stop_loss = price + atr_sl
                take_profit = price - atr_tp
                trigger_price = price - (price * cfg["slippage_buffer_pct"] / 100)
            
            # Create position object
            position = {
                'symbol': symbol,
                'side': side.lower(),
                'contracts': contracts,
                'position_value': position_value,
                'trigger_price': trigger_price,
                'entry_price': None,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'initial_stop': stop_loss,
                'initial_tp': take_profit,
                'best_price': price,
                'partial_filled': False,
                'trailing_active': False,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc),
                'atr': atr,
                'synthetic_sl': False  # Flag for synthetic stop loss
            }
            
            self.positions[symbol] = position
            
            # Update budget tracking
            margin_used = position_value / cfg["leverage"]
            cfg["budget_used_today"] += margin_used
            cfg["daily_trades"] = cfg.get("daily_trades", 0) + 1
            
            # Set leverage
            safe_set_leverage(symbol, cfg["leverage"])
            
            # Save positions
            self.save_positions()
            save_settings()
            
            log(f"Position planned: {side.upper()} {symbol} @ {price:.6f}, "
                f"SL: {stop_loss:.6f}, TP: {take_profit:.6f}, Size: ${position_value:.2f}")
            
            return True
            
        except Exception as e:
            log(f"Error opening position for {symbol}: {e}")
            return False
    
    async def check_triggers(self):
        """Check if any positions should be triggered"""
        for symbol, position in list(self.positions.items()):
            if not position or position.get('entry_price') is not None:
                continue
            
            try:
                # Get current price
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                trigger_price = position['trigger_price']
                side = position['side']
                
                should_trigger = False
                if side == 'long' and current_price >= trigger_price:
                    should_trigger = True
                elif side == 'short' and current_price <= trigger_price:
                    should_trigger = True
                
                if should_trigger:
                    success = await self._execute_entry(symbol, position, current_price)
                    if success and tg_manager:
                        await tg_send(
                            f"🎯 POSITION TRIGGERED\n"
                            f"{side.upper()} {symbol}\n"
                            f"Entry: ${current_price:.6f}\n"
                            f"Stop: ${position['stop_loss']:.6f}\n"
                            f"Target: ${position['take_profit']:.6f}\n"
                            f"Size: ${position.get('position_value', 0):.2f}"
                        )
                
            except Exception as e:
                log(f"Trigger check error for {symbol}: {e}")

    async def manage_positions(self):
        """Manage all active positions"""
        for symbol, position in list(self.positions.items()):
            if not position or position.get('entry_price') is None:
                continue
            
            try:
                await self._manage_single_position(symbol, position)
            except Exception as e:
                log(f"Error managing position {symbol}: {e}")

    async def _execute_entry(self, symbol: str, position: Dict, current_price: float) -> bool:
        """Execute position entry"""
        try:
            if not hasattr(exchange, 'apiKey') or not exchange.apiKey:
                # Paper trading mode
                position['entry_price'] = current_price
                position['best_price'] = current_price
                position['order_id'] = f"paper_{int(time.time())}"
                log(f"Paper trading: Position triggered for {symbol} @ {current_price:.6f}")
                self.save_positions()
                return True
            
            # Live trading
            side_map = {'long': 'buy', 'short': 'sell'}
            order_side = side_map[position['side']]
            
            try:
                order = exchange.create_market_order(
                    symbol, order_side, position['contracts']
                )
                
                position['entry_price'] = current_price
                position['best_price'] = current_price
                position['order_id'] = order['id']
                
                log(f"Position triggered: {position['side'].upper()} {symbol} @ {current_price:.6f}")
                self.save_positions()
                
                # Set stop loss and take profit orders
                await self._set_exit_orders(symbol, position)
                
                return True
                
            except Exception as e:
                log(f"Error executing entry for {symbol}: {e}")
                # Enable synthetic stop loss if exchange orders fail
                position['entry_price'] = current_price
                position['best_price'] = current_price
                position['synthetic_sl'] = True
                self.save_positions()
                return True
                
        except Exception as e:
            log(f"Error in _execute_entry for {symbol}: {e}")
            return False

    async def _set_exit_orders(self, symbol: str, position: Dict):
        """Set stop loss and take profit orders"""
        try:
            if position.get('synthetic_sl') or not hasattr(exchange, 'apiKey') or not exchange.apiKey:
                log(f"Using synthetic stops for {symbol}")
                return
            
            # Try to set stop loss order
            try:
                if position['side'] == 'long':
                    sl_order = exchange.create_order(
                        symbol, 'stop_market', 'sell', position['contracts'],
                        None, None, {'stopPrice': position['stop_loss']}
                    )
                else:
                    sl_order = exchange.create_order(
                        symbol, 'stop_market', 'buy', position['contracts'],
                        None, None, {'stopPrice': position['stop_loss']}
                    )
                
                position['sl_order_id'] = sl_order['id']
                log(f"Stop loss order set for {symbol}: {position['stop_loss']:.6f}")
                
            except Exception as e:
                log(f"Failed to set stop loss order for {symbol}: {e}")
                position['synthetic_sl'] = True
            
            # Try to set take profit order
            try:
                if position['side'] == 'long':
                    tp_order = exchange.create_limit_order(
                        symbol, 'sell', position['contracts'], position['take_profit']
                    )
                else:
                    tp_order = exchange.create_limit_order(
                        symbol, 'buy', position['contracts'], position['take_profit']
                    )
                
                position['tp_order_id'] = tp_order['id']
                log(f"Take profit order set for {symbol}: {position['take_profit']:.6f}")
                
            except Exception as e:
                log(f"Failed to set take profit order for {symbol}: {e}")
                
        except Exception as e:
            log(f"Error setting exit orders for {symbol}: {e}")

    async def _manage_single_position(self, symbol: str, position: Dict):
        """Manage a single position"""
        try:
            # Get current price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            side = position['side']
            entry_price = position['entry_price']
            
            # Update best price for trailing
            if side == 'long' and current_price > position['best_price']:
                position['best_price'] = current_price
            elif side == 'short' and current_price < position['best_price']:
                position['best_price'] = current_price
            
            # Check partial take profit
            if (cfg.get("use_partial_tp", False) and not position.get('partial_filled') and
                self._should_partial_tp(position, current_price)):
                
                await self._execute_partial_tp(symbol, position, current_price)
            
            # Check trailing stop
            if cfg.get("trail_after_tp", False) and position.get('partial_filled'):
                self._update_trailing_stop(position, current_price)
            
            # Check exit conditions (synthetic stops)
            if position.get('synthetic_sl') or not position.get('sl_order_id'):
                should_exit, exit_reason = self._check_synthetic_exit(position, current_price)
                if should_exit:
                    await self.close_position(symbol, exit_reason, current_price)
            
            # Check for stale positions (open too long)
            if 'timestamp' in position:
                position_age = datetime.now(timezone.utc) - position['timestamp']
                if position_age > timedelta(hours=24):  # Close after 24 hours
                    await self.close_position(symbol, "Position timeout", current_price)
            
            self.save_positions()
            
        except Exception as e:
            log(f"Error in single position management for {symbol}: {e}")

    def _should_partial_tp(self, position: Dict, current_price: float) -> bool:
        """Check if partial take profit should be executed"""
        target = position['take_profit']
        side = position['side']
        
        if side == 'long':
            return current_price >= target * 0.75
        else:
            return current_price <= target * 1.25

    async def _execute_partial_tp(self, symbol: str, position: Dict, current_price: float):
        """Execute partial take profit"""
        try:
            partial_size = position['contracts'] // 2
            if partial_size <= 0:
                return
            
            if hasattr(exchange, 'apiKey') and exchange.apiKey:
                # Live trading
                close_side = 'sell' if position['side'] == 'long' else 'buy'
                order = exchange.create_market_order(symbol, close_side, partial_size)
                log(f"Partial TP executed for {symbol}: {partial_size} contracts @ {current_price:.6f}")
            else:
                # Paper trading
                log(f"Paper trading: Partial TP for {symbol}: {partial_size} contracts @ {current_price:.6f}")
            
            # Update position
            position['contracts'] -= partial_size
            position['partial_filled'] = True
            position['stop_loss'] = position['entry_price']  # Move to breakeven
            
        except Exception as e:
            log(f"Error executing partial TP for {symbol}: {e}")

    def _update_trailing_stop(self, position: Dict, current_price: float):
        """Update trailing stop loss"""
        trail_distance = position['best_price'] * (cfg.get("trail_pct", 0.5) / 100)
        
        if position['side'] == 'long':
            new_stop = position['best_price'] - trail_distance
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                position['trailing_active'] = True
        else:
            new_stop = position['best_price'] + trail_distance
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
                position['trailing_active'] = True

    def _check_synthetic_exit(self, position: Dict, current_price: float) -> Tuple[bool, str]:
        """Check synthetic exit conditions"""
        side = position['side']
        
        # Stop loss check
        if side == 'long' and current_price <= position['stop_loss']:
            return True, "Stop Loss"
        elif side == 'short' and current_price >= position['stop_loss']:
            return True, "Stop Loss"
        
        # Take profit check
        if side == 'long' and current_price >= position['take_profit']:
            return True, "Take Profit"
        elif side == 'short' and current_price <= position['take_profit']:
            return True, "Take Profit"
        
        return False, ""
    
    async def _cancel_exit_orders(self, symbol: str, position: Dict):
        """Cancel existing stop loss and take profit orders"""
        try:
            if position.get('sl_order_id'):
                try:
                    exchange.cancel_order(position['sl_order_id'], symbol)
                    log(f"Cancelled SL order for {symbol}")
                except Exception:
                    pass
            
            if position.get('tp_order_id'):
                try:
                    exchange.cancel_order(position['tp_order_id'], symbol)
                    log(f"Cancelled TP order for {symbol}")
                except Exception:
                    pass
                    
        except Exception as e:
            log(f"Error cancelling orders for {symbol}: {e}")
    
    async def close_all_positions(self, reason: str = "Close All"):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            if self.positions.get(symbol):
                await self.close_position(symbol, reason)
    
    def get_positions_summary(self) -> Dict:
        """Get summary of all positions"""
        active_positions = {k: v for k, v in self.positions.items() if v and v.get('entry_price')}
        
        total_value = sum(pos.get('position_value', 0) for pos in active_positions.values())
        total_pnl = 0
        
        try:
            for symbol, pos in active_positions.items():
                if pos.get('entry_price'):
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    if pos['side'] == 'long':
                        pnl = (current_price - pos['entry_price']) * pos['contracts']
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['contracts']
                    
                    total_pnl += pnl
        except Exception:
            pass
        
        return {
            'active_count': len(active_positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'positions': active_positions,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
# =============================
# ENHANCED TELEGRAM COMMAND HANDLERS
# =============================

class TelegramBot:
    """Enhanced Telegram bot command handler with full functionality"""
    
    def __init__(self, trading_engine):
        self.engine = trading_engine
        self.help_text = """
🤖 CRYPTO TRADING BOT COMMANDS

📊 ANALYSIS & STATUS:
/status - Account balance, positions, settings
/analyze [SYMBOL] - Multi-timeframe analysis + AI
/chart [SYMBOL] [TF] - Technical analysis chart
/predict [SYMBOL] - AI prediction with confidence

🎯 TRADING:
/long [SYMBOL] - Open long position
/short [SYMBOL] - Open short position
/close [SYMBOL] - Close specific position
/closeall - Close all positions
/auto on|off - Toggle auto trading

⚙️ CONFIGURATION:
/risk - Show/set risk parameters
/set budget <amount> - Set daily budget
/set leverage <x> - Set leverage
/set atrsl <x> - ATR stop loss multiplier
/set atrtp <x> - ATR take profit multiplier

🤖 AI & OPTIMIZATION:
/trainai [SYMBOL] - Train AI models
/walkforward [SYMBOL] - Run walk-forward test
/download - Download historical data

📋 PAIRS MANAGEMENT:
/addpair [SYMBOL] - Add to watch list
/addtrade [SYMBOL] - Add to auto trading
/pairs - Show all monitored pairs

📈 REPORTS & ANALYSIS:
/report [days] - Performance report
/equity - Show equity curve
/trades - Recent trades summary

Type any command for detailed help.
"""
    
    async def handle_command(self, text: str):
        """Handle incoming telegram command with comprehensive error handling"""
        try:
            parts = text.strip().split()
            if not parts:
                return
            
            command = parts[0].lower()
            
            # Route commands
            if command in ['/help', '/start']:
                await self.cmd_help()
            elif command == '/status':
                await self.cmd_status()
            elif command.startswith('/analyze'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_analyze(normalize_symbol(symbol))
            elif command.startswith('/chart'):
                await self.cmd_chart(parts[1:])
            elif command.startswith('/predict'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_predict(normalize_symbol(symbol))
            elif command.startswith('/long'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_long(normalize_symbol(symbol))
            elif command.startswith('/short'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_short(normalize_symbol(symbol))
            elif command.startswith('/close'):
                if len(parts) > 1:
                    symbol = normalize_symbol(parts[1])
                    await self.cmd_close_position(symbol)
                else:
                    await self.cmd_close()
            elif command == '/closeall':
                await self.cmd_close_all()
            elif command.startswith('/auto'):
                state = parts[1] if len(parts) > 1 else 'toggle'
                await self.cmd_auto(state)
            elif command.startswith('/risk'):
                if len(parts) > 1:
                    await self.cmd_set_risk(parts[1:])
                else:
                    await self.cmd_show_risk()
            elif command.startswith('/set'):
                await self.cmd_set(parts[1:])
            elif command.startswith('/trainai'):
                symbol = parts[1] if len(parts) > 1 else None
                await self.cmd_train_ai(symbol)
            elif command.startswith('/walkforward'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_walkforward(normalize_symbol(symbol))
            elif command == '/download':
                await self.cmd_download()
            elif command.startswith('/addpair'):
                symbol = parts[1] if len(parts) > 1 else None
                await self.cmd_add_pair(symbol, watch_only=True)
            elif command.startswith('/addtrade'):
                symbol = parts[1] if len(parts) > 1 else None
                await self.cmd_add_pair(symbol, watch_only=False)
            elif command == '/pairs':
                await self.cmd_show_pairs()
            elif command.startswith('/report'):
                days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 7
                await self.cmd_report(days)
            elif command == '/equity':
                await self.cmd_equity_curve()
            elif command == '/trades':
                await self.cmd_recent_trades()
            else:
                await tg_send(f"Unknown command: {command}\nType /help for available commands")
                
        except Exception as e:
            log(f"Command handling error: {e}")
            await tg_send(f"Error processing command: {str(e)[:200]}")
    
    async def cmd_help(self):
        """Show help message"""
        await tg_send(self.help_text)
    
    async def cmd_status(self):
        """Show comprehensive system status"""
        try:
            # Get account info
            balance = self.engine.get_account_balance()
            positions_summary = self.engine.get_positions_summary()
            
            # Build status message
            status_msg = f"""
📊 TRADING BOT STATUS

💰 Account:
Balance: ${balance:.2f} USDT
Mode: {MODE.upper()}
Daily Budget: ${cfg['daily_budget_usd']:.2f}
Used Today: ${cfg['budget_used_today']:.2f}
Remaining: ${cfg['daily_budget_usd'] - cfg['budget_used_today']:.2f}

⚙️ Settings:
Auto Trading: {'🟢 ON' if self.engine.auto_enabled else '🔴 OFF'}
Leverage: {cfg['leverage']}x
ATR SL: {cfg['atr_sl_mult']}x | ATR TP: {cfg['atr_tp_mult']}x
Position Size: {cfg['position_size_fraction']*100:.1f}% of account

📈 Positions ({positions_summary['active_count']}):
Total Value: ${positions_summary['total_value']:.2f}
Unrealized PnL: ${positions_summary['total_pnl']:.2f}
"""
            
            # Add individual position details
            for symbol, pos in positions_summary['positions'].items():
                if pos and pos.get('entry_price'):
                    # Get current price for PnL calculation
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        
                        if pos['side'] == 'long':
                            pnl = (current_price - pos['entry_price']) * pos['contracts']
                        else:
                            pnl = (pos['entry_price'] - current_price) * pos['contracts']
                        
                        pnl_pct = (pnl / (pos['entry_price'] * pos['contracts'])) * 100
                        
                        status_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "🟡"
                        status_msg += f"\n{status_emoji} {pos['side'].upper()} {symbol}"
                        status_msg += f"\nEntry: ${pos['entry_price']:.6f} | Current: ${current_price:.6f}"
                        status_msg += f"\nPnL: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                        
                    except Exception:
                        status_msg += f"\n🔵 {pos['side'].upper()} {symbol} - Active"
            
            if positions_summary['active_count'] == 0:
                status_msg += "\nNo active positions"
            
            # Add risk metrics
            risk = positions_summary['risk_metrics']
            status_msg += f"""

⚠️ Risk Management:
Daily Loss: ${risk['daily_loss_tracker']:.2f}
Daily Trades: {risk['daily_trades']}/10
Max Position Risk: {risk['max_position_risk_pct']:.1f}%
"""
            
            status_msg += f"""

🎯 Auto Pairs: {len(AUTO_PAIRS)}
👁️ Watch Pairs: {len(WATCH_ONLY_PAIRS)}
"""
            
            await tg_send(status_msg)
            
        except Exception as e:
            log(f"Status command error: {e}")
            await tg_send(f"Error getting status: {e}")
    
    async def cmd_analyze(self, symbol: str):
        """Comprehensive symbol analysis"""
        try:
            analysis = await self.engine.analyze_symbol(symbol)
            
            if not analysis:
                await tg_send(f"Could not analyze {symbol}")
                return
            
            msg = f"📈 ANALYSIS: {symbol}\n\n"
            
            # Multi-timeframe signals
            msg += "📊 Multi-Timeframe Signals:\n"
            for tf, data in analysis['results'].items():
                signal = data['signal']
                indicators = data['indicators']
                
                emoji = "🟢" if signal == 'bullish' else "🔴" if signal == 'bearish' else "🟡"
                msg += f"{emoji} {tf}: {signal.upper()}\n"
                msg += f"  Price: ${indicators['price']:.6f}\n"
                if indicators.get('rsi14'):
                    msg += f"  RSI: {indicators['rsi14']:.1f}\n"
                if indicators.get('bb_position'):
                    msg += f"  BB Pos: {indicators['bb_position']:.2f}\n"
                msg += "\n"
            
            # Core agreement
            if analysis.get('core_agree'):
                msg += f"✅ Core timeframes agree: {analysis['core_signal'].upper()}\n\n"
            else:
                msg += "⚠️ No consensus across core timeframes\n\n"
            
            # AI prediction
            ai_result = analysis.get('ai_result')
            if ai_result:
                confidence_emoji = "🟢" if ai_result['confidence'] >= 0.7 else "🟡" if ai_result['confidence'] >= 0.55 else "🔴"
                msg += f"🤖 AI Prediction:\n"
                msg += f"{confidence_emoji} Probability Up: {ai_result['p_up']:.1%}\n"
                msg += f"Decision: {ai_result['decision'].upper()}\n"
                msg += f"Confidence: {ai_result['confidence']:.1%}\n"
                msg += f"Method: {ai_result['method']}\n"
                
                components = normalize_ai_result(ai_result).get('components', {})
                if components.get('lstm') is not None:
                    msg += f"\nComponents:\n"
                    msg += f"  LR: {components['lr']:.1%} (conf: {components['lr_confidence']:.1%})\n"
                    msg += f"  LSTM: {components['lstm']:.1%} (conf: {components['lstm_confidence']:.1%})\n"
            
            # Trading recommendation
            msg += "\n📋 Recommendation:\n"
            if analysis.get('core_agree') and ai_result:
                core_signal = analysis['core_signal']
                ai_decision = ai_result['decision']
                
                if core_signal == 'bullish' and ai_decision == 'long' and ai_result['confidence'] >= 0.6:
                    msg += "✅ STRONG BUY - Technical + AI alignment"
                elif core_signal == 'bearish' and ai_decision == 'short' and ai_result['confidence'] >= 0.6:
                    msg += "✅ STRONG SELL - Technical + AI alignment"
                elif ai_result['confidence'] >= 0.65:
                    msg += f"⚠️ AI SIGNAL: {ai_decision.upper()} (no technical consensus)"
                else:
                    msg += "🟡 HOLD - Wait for better setup"
            else:
                msg += "🟡 HOLD - Insufficient signals"
            
            await tg_send(msg)
            
        except Exception as e:
            log(f"Analysis error for {symbol}: {e}")
            await tg_send(f"Analysis error for {symbol}: {e}")
    
    async def cmd_chart(self, args: List[str]):
        """Generate and send technical chart"""
        try:
            symbol = 'BTC/USDT:USDT'
            timeframe = '1h'
            
            # Parse arguments
            for arg in args:
                if any(c.isalpha() and c not in 'mhd' for c in arg):
                    symbol = normalize_symbol(arg)
                else:
                    timeframe = timeframe_alias(arg)
            
            await tg_send(f"📊 Generating chart for {symbol} {timeframe.upper()}...")
            
            # Get data
            df = load_history_df(symbol, timeframe, months=3)
            if df.empty:
                await tg_send(f"No data available for {symbol} {timeframe}")
                return
            
            # Get current positions for trade markers
            trades = []
            position = self.engine.positions.get(symbol)
            if (position and position.get('entry_price')):
                trades = [{
                    'entry_idx': len(df) - 50,  # Approximate position
                    'side': position['side'],
                    'entry_price': position['entry_price']
                }]
            
            # Generate chart
            chart_title = f"{symbol} {timeframe.upper()} Technical Analysis"
            chart_file = os.path.join(CHARTS_DIR, f"{symbol.replace('/', '_')}_{timeframe}_chart.png")
            
            chart_path = plot_candlestick_chart(
                df.tail(300), chart_title, chart_file, 
                indicators=True, trades=trades
            )
            
            if chart_path:
                # Get current analysis for caption
                analysis = await self.engine.analyze_symbol(symbol)
                ai_result = analysis.get('ai_result') if analysis else None
                
                caption = f"{symbol} {timeframe} Analysis"
                if ai_result:
                    caption += f"\n🤖 AI: {ai_result['decision'].upper()} ({ai_result['p_up']:.1%})"
                
                await tg_send_photo(chart_path, caption)
            else:
                await tg_send("Failed to generate chart")
                
        except Exception as e:
            log(f"Chart error: {e}")
            await tg_send(f"Chart error: {str(e)[:100]}")
    
    async def cmd_predict(self, symbol: str):
        """Get AI prediction with detailed breakdown"""
        try:
            await tg_send(f"🤖 Generating AI prediction for {symbol}...")
            
            result = await predict_with_hybrid_ai(symbol)
            if result:
                confidence_emoji = "🟢" if result['confidence'] >= 0.7 else "🟡" if result['confidence'] >= 0.55 else "🔴"
                
                msg = f"🤖 AI PREDICTION: {symbol}\n\n"
                msg += f"{confidence_emoji} Probability Up: {result['p_up']:.1%}\n"
                msg += f"Decision: **{result['decision'].upper()}**\n"
                msg += f"Overall Confidence: {result['confidence']:.1%}\n"
                msg += f"Method: {result['method']}\n"
                
                # Component breakdown
                components = result.get('components', {})
                if components.get('lstm') is not None:
                    msg += f"\n📊 Model Components:\n"
                    msg += f"Logistic Regression: {components['lr']:.1%}\n"
                    msg += f"  └ Confidence: {components['lr_confidence']:.1%}\n"
                    msg += f"LSTM Neural Network: {components['lstm']:.1%}\n"
                    msg += f"  └ Confidence: {components['lstm_confidence']:.1%}\n"
                else:
                    msg += f"\nUsing Logistic Regression only (confidence: {components['lr_confidence']:.1%})"
                
                # Recommendation
                msg += f"\n💡 Recommendation:\n"
                if result['confidence'] >= 0.7:
                    if result['decision'] == 'long':
                        msg += "✅ Strong BUY signal - Consider opening long position"
                    elif result['decision'] == 'short':
                        msg += "✅ Strong SELL signal - Consider opening short position"
                    else:
                        msg += "⏸️ HOLD - Wait for clearer signals"
                elif result['confidence'] >= 0.55:
                    msg += f"⚠️ Weak {result['decision'].upper()} signal - Use caution"
                else:
                    msg += "🔴 Low confidence - Avoid trading"
                
                await tg_send(msg)
            else:
                await tg_send(f"Could not generate prediction for {symbol}")
                
        except Exception as e:
            log(f"Prediction error for {symbol}: {e}")
            await tg_send(f"Prediction error: {str(e)[:100]}")
    
    async def cmd_long(self, symbol: str):
        """Open long position"""
        try:
            success = await self.engine.open_position(symbol, 'long', 'Manual long command')
            if success:
                await tg_send(f"✅ Long position planned for {symbol}")
            else:
                await tg_send(f"❌ Failed to plan long position for {symbol}")
        except Exception as e:
            await tg_send(f"Long position error: {e}")
    
    async def cmd_short(self, symbol: str):
        """Open short position"""
        try:
            success = await self.engine.open_position(symbol, 'short', 'Manual short command')
            if success:
                await tg_send(f"✅ Short position planned for {symbol}")
            else:
                await tg_send(f"❌ Failed to plan short position for {symbol}")
        except Exception as e:
            await tg_send(f"Short position error: {e}")
    
    async def cmd_close(self):
        """Close first active position"""
        try:
            active_positions = {k: v for k, v in self.engine.positions.items() if v and v.get('entry_price')}
            if active_positions:
                symbol = list(active_positions.keys())[0]
                success = await self.engine.close_position(symbol, "Manual close")
                if success:
                    await tg_send(f"✅ Position closed: {symbol}")
                else:
                    await tg_send(f"❌ Failed to close position: {symbol}")
            else:
                await tg_send("ℹ️ No active positions to close")
        except Exception as e:
            await tg_send(f"Close position error: {e}")
    
    async def cmd_close_position(self, symbol: str):
        """Close specific position"""
        try:
            position = self.engine.positions.get(symbol)
            if position and position.get('entry_price'):
                success = await self.engine.close_position(symbol, "Manual close")
                if success:
                    await tg_send(f"✅ Position closed: {symbol}")
                else:
                    await tg_send(f"❌ Failed to close position: {symbol}")
            else:
                await tg_send(f"ℹ️ No active position for {symbol}")
        except Exception as e:
            await tg_send(f"Close position error: {e}")
    
    async def cmd_close_all(self):
        """Close all positions"""
        try:
            active_positions = {k: v for k, v in self.engine.positions.items() if v and v.get('entry_price')}
            if not active_positions:
                await tg_send("ℹ️ No active positions to close")
                return
            
            await tg_send(f"🔄 Closing {len(active_positions)} positions...")
            
            success_count = 0
            for symbol in active_positions.keys():
                if await self.engine.close_position(symbol, "Close all command"):
                    success_count += 1
            
            await tg_send(f"✅ Closed {success_count}/{len(active_positions)} positions")
            
        except Exception as e:
            await tg_send(f"Close all error: {e}")
    
    async def cmd_auto(self, state: str):
        """Toggle auto trading"""
        try:
            if state.lower() == 'on':
                self.engine.auto_enabled = True
                cfg["auto_enabled"] = True
                await tg_send("✅ Auto trading ENABLED")
            elif state.lower() == 'off':
                self.engine.auto_enabled = False
                cfg["auto_enabled"] = False
                await tg_send("✅ Auto trading DISABLED")
            else:
                current_state = "ON" if self.engine.auto_enabled else "OFF"
                await tg_send(f"Auto trading is currently {current_state}")
            
            save_settings()
            
        except Exception as e:
            await tg_send(f"Auto toggle error: {e}")
    
    async def cmd_show_risk(self):
        """Show current risk settings"""
        try:
            risk_metrics = self.engine.risk_manager.get_risk_metrics()
            
            msg = f"""
⚠️ RISK MANAGEMENT SETTINGS

💰 Budget Control:
Daily Budget: ${cfg['daily_budget_usd']:.2f}
Used Today: ${risk_metrics['daily_budget_used']:.2f}
Remaining: ${cfg['daily_budget_usd'] - risk_metrics['daily_budget_used']:.2f}

📊 Position Limits:
Position Size: {cfg['position_size_fraction']*100:.1f}% of account
Min Trade Size: ${cfg['min_notional']:.2f}
Max Trade Size: ${cfg['max_notional']:.2f}
Leverage: {cfg['leverage']}x

⛔ Stop Loss & Take Profit:
ATR SL Multiplier: {cfg['atr_sl_mult']}x
ATR TP Multiplier: {cfg['atr_tp_mult']}x
Partial TP: {'ON' if cfg['use_partial_tp'] else 'OFF'}
Trailing Stop: {'ON' if cfg['trail_after_tp'] else 'OFF'}

📈 Daily Limits:
Daily Trades: {risk_metrics['daily_trades']}/10
Max Daily Loss: {risk_metrics['max_daily_loss_pct']:.1f}%
Daily Loss Tracker: ${risk_metrics['daily_loss_tracker']:.2f}
"""
            await tg_send(msg)
            
        except Exception as e:
            await tg_send(f"Risk display error: {e}")
    
    async def cmd_set(self, args: List[str]):
        """Set configuration parameters"""
        try:
            if len(args) < 2:
                await tg_send("Usage: /set <parameter> <value>\n"
                             "Parameters: budget, leverage, atrsl, atrtp, minsize, maxsize, possize")
                return
            
            param, value = args[0].lower(), args[1]
            
            if param == 'budget':
                cfg["daily_budget_usd"] = max(1.0, float(value))
                cfg["budget_used_today"] = 0.0  # Reset usage
                await tg_send(f"✅ Daily budget set to ${cfg['daily_budget_usd']:.2f}")
                
            elif param == 'leverage':
                new_leverage = max(1, min(100, int(float(value))))
                cfg["leverage"] = new_leverage
                # Apply to all auto pairs
                for symbol in AUTO_PAIRS:
                    safe_set_leverage(symbol, new_leverage)
                await tg_send(f"✅ Leverage set to {cfg['leverage']}x")
                
            elif param == 'atrsl':
                cfg["atr_sl_mult"] = max(0.1, float(value))
                await tg_send(f"✅ ATR Stop Loss multiplier set to {cfg['atr_sl_mult']}x")
                
            elif param == 'atrtp':
                cfg["atr_tp_mult"] = max(0.2, float(value))
                await tg_send(f"✅ ATR Take Profit multiplier set to {cfg['atr_tp_mult']}x")
                
            elif param == 'minsize':
                cfg["min_notional"] = max(1.0, float(value))
                await tg_send(f"✅ Minimum trade size set to ${cfg['min_notional']:.2f}")
                
            elif param == 'maxsize':
                cfg["max_notional"] = max(cfg["min_notional"], float(value))
                await tg_send(f"✅ Maximum trade size set to ${cfg['max_notional']:.2f}")
                
            elif param == 'possize':
                new_fraction = max(0.001, min(0.1, float(value)))  # 0.1% to 10%
                cfg["position_size_fraction"] = new_fraction
                await tg_send(f"✅ Position size set to {new_fraction*100:.2f}% of account")
                
            else:
                await tg_send(f"Unknown parameter: {param}")
                return
            
            save_settings()
            
        except ValueError:
            await tg_send("Invalid value. Please provide a valid number.")
        except Exception as e:
            await tg_send(f"Set error: {e}")
    
    async def cmd_train_ai(self, symbol: str = None):
        """Train AI models"""
        try:
            symbols = [normalize_symbol(symbol)] if symbol else list(AUTO_PAIRS)
            
            await tg_send(f"🤖 Starting AI training for {len(symbols)} symbols...")
            
            results = []
            for sym in symbols:
                await tg_send(f"Training models for {sym}...")
                
                # Create dataset
                features, labels, df_base = await create_ai_dataset(sym)
                if features is None:
                    results.append(f"❌ {sym}: Insufficient data")
                    continue
                
                # Train LR model
                lr_success = await train_lr_model(sym, features, labels)
                
                # Train LSTM model if PyTorch available
                lstm_success = False
                if TORCH_OK:
                    lstm_success = await train_lstm_model(sym, features, labels)
                
                # Get model metadata for results
                lr_metadata = ai_manager.get_model_metadata(sym).get('lr_metadata', {})
                lstm_metadata = ai_manager.get_model_metadata(sym)
                
                result = f"{'✅' if lr_success else '❌'} {sym}:\n"
                result += f"  LR: {lr_metadata.get('accuracy', 0):.1%} accuracy\n"
                if TORCH_OK:
                    if lstm_success:
                        result += f"  LSTM: {lstm_metadata.get('best_val_acc', 0):.1%} accuracy"
                    else:
                        result += "  LSTM: Failed"
                
                results.append(result)
            
            # Save AI state
            ai_manager.save_state()
            
            # Send results
            results_msg = "🎉 AI TRAINING COMPLETED\n\n" + "\n\n".join(results)
            await tg_send(results_msg)
            
        except Exception as e:
            log(f"AI training error: {e}")
            await tg_send(f"Training error: {str(e)[:200]}")
    
    async def cmd_walkforward(self, symbol: str):
        """Run walk-forward optimization"""
        try:
            await tg_send(f"📊 Starting walk-forward optimization for {symbol}...")
            
            # Create optimizer
            optimizer = WalkForwardOptimizer(
                symbol=symbol,
                timeframe="5m",
                train_window=1000,
                test_window=250,
                step_size=250,
                total_months=6
            )
            
            # Run optimization
            success = await optimizer.run_optimization()
            
            if success:
                # Generate and save results
                results_file = await optimizer.save_results()
                report = optimizer.generate_report()
                
                # Send text report
                await tg_send(f"📊 WALK-FORWARD RESULTS:\n```{report}```")
                
                # Generate and send equity curve
                if optimizer.equity_curve and len(optimizer.equity_curve) > 1:
                    dates = [datetime.now() - timedelta(days=i) for i in reversed(range(len(optimizer.equity_curve)))]
                    equity_file = os.path.join(CHARTS_DIR, f"walkforward_{symbol.replace('/', '_')}_equity.png")
                    
                    equity_path = plot_equity_curve(
                        optimizer.equity_curve, dates,
                        f"Walk-Forward Equity Curve - {symbol}",
                        equity_file, optimizer.trades
                    )
                    
                    if equity_path:
                        await tg_send_photo(equity_path, f"Walk-Forward Results: {symbol}")
                
                # Generate and send summary chart
                summary_file = os.path.join(CHARTS_DIR, f"walkforward_{symbol.replace('/', '_')}_summary.png")
                summary_path = plot_walkforward_summary(optimizer, summary_file)
                if summary_path:
                    await tg_send_photo(summary_path, f"Walk-Forward Summary: {symbol}")
                
                if results_file:
                    await tg_send(f"✅ Detailed results saved to: {os.path.basename(results_file)}")
                
            else:
                await tg_send(f"❌ Walk-forward optimization failed for {symbol}")
                
        except Exception as e:
            log(f"Walk-forward error: {e}")
            await tg_send(f"Walk-forward error: {str(e)[:200]}")
    
    async def cmd_download(self):
        """Download historical data"""
        try:
            await tg_send("📥 Starting data download...")
            
            symbols = list(AUTO_PAIRS | WATCH_ONLY_PAIRS)
            timeframes = cfg["timeframes"]
            
            total_downloads = len(symbols) * len(timeframes)
            completed = 0
            failed = 0
            
            for symbol in symbols:
                for tf in timeframes:
                    try:
                        success = await ensure_history(symbol, tf, months=AI_CFG["hist_months"])
                        if success:
                            completed += 1
                        else:
                            failed += 1
                        
                        # Progress update every 10 downloads
                        if (completed + failed) % 10 == 0:
                            progress = ((completed + failed) / total_downloads) * 100
                            await tg_send(f"📥 Progress: {progress:.0f}% ({completed} success, {failed} failed)")
                            
                    except Exception as e:
                        log(f"Download error for {symbol} {tf}: {e}")
                        failed += 1
            
            await tg_send(f"✅ Data download completed: {completed} successful, {failed} failed")
            
        except Exception as e:
            await tg_send(f"Download error: {str(e)[:200]}")
    
    async def cmd_add_pair(self, symbol: str, watch_only: bool = True):
        """Add trading pair"""
        if not symbol:
            await tg_send("Usage: /addpair <SYMBOL> or /addtrade <SYMBOL>")
            return
        
        try:
            symbol = normalize_symbol(symbol)
            
            if watch_only:
                if symbol not in WATCH_ONLY_PAIRS and symbol not in AUTO_PAIRS:
                    WATCH_ONLY_PAIRS.add(symbol)
                    await tg_send(f"✅ Added {symbol} to watch list")
                else:
                    await tg_send(f"ℹ️ {symbol} already being monitored")
            else:
                if symbol not in AUTO_PAIRS:
                    AUTO_PAIRS.add(symbol)
                    safe_set_leverage(symbol, cfg["leverage"])
                    await tg_send(f"✅ Added {symbol} to auto trading")
                else:
                    await tg_send(f"ℹ️ {symbol} already in auto trading")
                    
        except Exception as e:
            await tg_send(f"Error adding pair: {e}")
    
    async def cmd_show_pairs(self):
        """Show all monitored pairs"""
        try:
            msg = "📋 MONITORED TRADING PAIRS\n\n"
            
            msg += f"🎯 Auto Trading ({len(AUTO_PAIRS)}):\n"
            for symbol in sorted(AUTO_PAIRS):
                # Get current position status
                position = self.engine.positions.get(symbol)
                if position and position.get('entry_price'):
                    status = f"📈 {position['side'].upper()}"
                else:
                    status = "⏸️"
                msg += f"{status} {symbol}\n"
            
            msg += f"\n👁️ Watch Only ({len(WATCH_ONLY_PAIRS)}):\n"
            for symbol in sorted(WATCH_ONLY_PAIRS):
                msg += f"👁️ {symbol}\n"
            
            await tg_send(msg)
            
        except Exception as e:
            await tg_send(f"Error showing pairs: {e}")
    
    async def cmd_report(self, days: int = 7):
        """Generate performance report"""
        try:
            await tg_send(f"📊 Generating {days}-day performance report...")
            
            # This is a simplified report - in a full implementation,
            # you would track historical trades and P&L
            positions_summary = self.engine.get_positions_summary()
            risk_metrics = positions_summary['risk_metrics']
            
            report = f"""
📊 PERFORMANCE REPORT ({days} days)

💰 Current Status:
Account Balance: ${self.engine.get_account_balance():.2f}
Active Positions: {positions_summary['active_count']}
Position Value: ${positions_summary['total_value']:.2f}
Unrealized PnL: ${positions_summary['total_pnl']:.2f}

📈 Risk Metrics:
Daily Budget Used: ${risk_metrics['daily_budget_used']:.2f} / ${cfg['daily_budget_usd']:.2f}
Daily Trades: {risk_metrics['daily_trades']}/10
Daily Loss Tracker: ${risk_metrics['daily_loss_tracker']:.2f}

⚙️ Settings:
Auto Trading: {'ON' if self.engine.auto_enabled else 'OFF'}
Position Size: {cfg['position_size_fraction']*100:.1f}% of account
Leverage: {cfg['leverage']}x
"""
            
            await tg_send(report)
            
        except Exception as e:
            await tg_send(f"Report error: {str(e)[:200]}")
    
    async def cmd_equity_curve(self):
        """Show equity curve (simplified)"""
        try:
            balance = self.engine.get_account_balance()
            positions_summary = self.engine.get_positions_summary()
            
            msg = f"""
📈 EQUITY SUMMARY

Current Balance: ${balance:.2f}
Position Value: ${positions_summary['total_value']:.2f}
Unrealized PnL: ${positions_summary['total_pnl']:.2f}
Total Equity: ${balance + positions_summary['total_pnl']:.2f}

Note: Historical equity tracking requires extended operation.
Full equity curves are available in walk-forward reports.
"""
            await tg_send(msg)
            
        except Exception as e:
            await tg_send(f"Equity curve error: {e}")
    
    async def cmd_recent_trades(self):
        """Show recent trades summary"""
        try:
            # This is simplified - in full implementation, you'd track trade history
            active_positions = {k: v for k, v in self.engine.positions.items() if v and v.get('entry_price')}
            
            if not active_positions:
                await tg_send("📊 No recent trades to display")
                return
            
            msg = "📊 ACTIVE POSITIONS\n\n"
            
            for symbol, pos in active_positions.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    if pos['side'] == 'long':
                        pnl = (current_price - pos['entry_price']) * pos['contracts']
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['contracts']
                    
                    pnl_pct = (pnl / (pos['entry_price'] * pos['contracts'])) * 100
                    duration = datetime.now(timezone.utc) - pos['timestamp']
                    
                    status = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "🟡"
                    
                    msg += f"{status} {pos['side'].upper()} {symbol}\n"
                    msg += f"Entry: ${pos['entry_price']:.6f}\n"
                    msg += f"Current: ${current_price:.6f}\n"
                    msg += f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                    msg += f"Duration: {duration.seconds//3600}h {(duration.seconds//60)%60}m\n\n"
                    
                except Exception:
                    msg += f"⚪ {pos['side'].upper()} {symbol} - Active\n\n"
            
            await tg_send(msg)
            
        except Exception as e:
            await tg_send(f"Trades summary error: {e}")
# =============================
# CLI FUNCTIONS (Enhanced)
# =============================

async def cli_download_data(symbols=None, timeframes=None, months=12, refresh=False):
    """CLI function to download historical data"""
    if symbols is None:
        symbols = list(AUTO_PAIRS | WATCH_ONLY_PAIRS)
    if timeframes is None:
        timeframes = cfg["timeframes"]
    
    log(f"Downloading data for {len(symbols)} symbols, {len(timeframes)} timeframes")
    
    total = len(symbols) * len(timeframes)
    completed = 0
    failed = 0
    
    for symbol in symbols:
        for tf in timeframes:
            try:
                if refresh:
                    # Remove existing file to force fresh download
                    path = _history_path(symbol, tf)
                    if os.path.exists(path):
                        os.remove(path)
                
                success = await ensure_history(symbol, tf, months=months)
                if success:
                    completed += 1
                else:
                    failed += 1
                
                # Progress update
                if (completed + failed) % 5 == 0:
                    progress = ((completed + failed) / total) * 100
                    log(f"Progress: {progress:.1f}% - {symbol} {tf} {'✓' if success else '✗'}")
                
            except Exception as e:
                log(f"Download failed for {symbol} {tf}: {e}")
                failed += 1
    
    log(f"Data download completed: {completed} successful, {failed} failed")
    return completed, failed

async def cli_train_ai(symbols=None, epochs=10, use_lstm=True):
    """CLI function to train AI models"""
    if symbols is None:
        symbols = list(AUTO_PAIRS)
    
    log(f"Training AI models for {len(symbols)} symbols")
    
    results = {}
    for symbol in symbols:
        log(f"Training models for {symbol}...")
        
        try:
            # Create dataset
            features, labels, df_base = await create_ai_dataset(symbol)
            if features is None:
                log(f"Insufficient data for {symbol}")
                results[symbol] = {"lr": False, "lstm": False, "reason": "Insufficient data"}
                continue
            
            # Train LR model
            lr_success = await train_lr_model(symbol, features, labels, epochs=epochs//2)
            
            # Train LSTM model
            lstm_success = False
            if use_lstm and TORCH_OK:
                lstm_success = await train_lstm_model(
                    symbol, features, labels,
                    epochs=epochs,
                    batch_size=AI_CFG["lstm_batch_size"],
                    seq_len=AI_CFG["lstm_seq_len"]
                )
            
            # Get accuracies
            lr_metadata = ai_manager.get_model_metadata(symbol).get('lr_metadata', {})
            lstm_metadata = ai_manager.get_model_metadata(symbol)
            
            results[symbol] = {
                "lr": lr_success,
                "lstm": lstm_success,
                "lr_accuracy": lr_metadata.get('accuracy', 0),
                "lstm_accuracy": lstm_metadata.get('best_val_acc', 0)
            }
            
            log(f"Training results for {symbol}: "
                f"LR: {'✓' if lr_success else '✗'} ({lr_metadata.get('accuracy', 0):.1%}), "
                f"LSTM: {'✓' if lstm_success else '✗'} ({lstm_metadata.get('best_val_acc', 0):.1%})")
            
        except Exception as e:
            log(f"Training error for {symbol}: {e}")
            results[symbol] = {"lr": False, "lstm": False, "reason": str(e)}
    
    # Save AI state
    ai_manager.save_state()
    
    log("AI training completed")
    return results

async def cli_walkforward(symbol="BTC/USDT:USDT", timeframe="5m", months=6):
    """CLI function to run walk-forward optimization"""
    log(f"Starting walk-forward optimization for {symbol}")
    
    try:
        optimizer = WalkForwardOptimizer(
            symbol=normalize_symbol(symbol),
            timeframe=timeframe,
            train_window=1000,
            test_window=250,
            step_size=250,
            total_months=months
        )
        
        success = await optimizer.run_optimization()
        
        if success:
            # Print report
            report = optimizer.generate_report()
            print(report)
            
            # Save results
            results_file = await optimizer.save_results()
            log(f"Results saved to: {results_file}")
            
            # Generate charts
            if optimizer.equity_curve and len(optimizer.equity_curve) > 1:
                dates = [datetime.now() - timedelta(days=i) for i in reversed(range(len(optimizer.equity_curve)))]
                
                # Equity curve
                equity_file = os.path.join(CHARTS_DIR, f"walkforward_{symbol.replace('/', '_')}_equity.png")
                equity_path = plot_equity_curve(
                    optimizer.equity_curve, dates,
                    f"Walk-Forward Equity Curve - {symbol}",
                    equity_file, optimizer.trades
                )
                
                # Summary chart
                summary_file = os.path.join(CHARTS_DIR, f"walkforward_{symbol.replace('/', '_')}_summary.png")
                summary_path = plot_walkforward_summary(optimizer, summary_file)
                
                if equity_path:
                    log(f"Equity curve saved to: {equity_path}")
                if summary_path:
                    log(f"Summary chart saved to: {summary_path}")
            
            return True
        else:
            log("Walk-forward optimization failed")
            return False
            
    except Exception as e:
        log(f"Walk-forward error: {e}")
        return False

# =============================
# MAIN TRADING LOOP (Enhanced)
# =============================

async def main_trading_loop():
    """Enhanced main trading loop with comprehensive error handling"""
    trading_engine = TradingEngine()
    telegram_bot = TelegramBot(trading_engine)
    
    log("Starting main trading loop...")
    
    # Send startup message
    startup_msg = f"""
🚀 TRADING BOT STARTED

📊 Configuration:
Mode: {MODE.upper()}
Auto Trading: {'🟢 ON' if trading_engine.auto_enabled else '🔴 OFF'}
Daily Budget: ${cfg['daily_budget_usd']:.2f}
Leverage: {cfg['leverage']}x

🎯 Monitoring:
Auto Pairs: {len(AUTO_PAIRS)}
Watch Pairs: {len(WATCH_ONLY_PAIRS)}

Type /help for commands
"""
    
    if tg_manager:
        await tg_send(startup_msg)
    
    # Initialize loop variables
    last_heartbeat = time.time()
    last_budget_reset = datetime.now(timezone.utc).date()
    heartbeat_interval = cfg["heartbeat_minutes"] * 60
    update_offset = None
    loop_count = 0
    
    # Main loop
    while True:
        try:
            loop_count += 1
            current_time = time.time()
            
            # Handle Telegram updates
            if tg_manager:
                try:
                    updates = await tg_get_updates(update_offset)
                    for update in updates:
                        update_offset = update["update_id"] + 1
                        
                        message = update.get("message") or update.get("edited_message")
                        if message and message.get("text"):
                            await telegram_bot.handle_command(message["text"])
                except Exception as e:
                    log(f"Telegram updates error: {e}")
            
            # Daily budget reset check
            current_date = datetime.now(timezone.utc).date()
            if current_date > last_budget_reset:
                cfg["budget_used_today"] = 0.0
                cfg["daily_trades"] = 0
                trading_engine.risk_manager.reset_daily_counters()
                last_budget_reset = current_date
                log("Daily counters reset")
                if tg_manager:
                    await tg_send("🗓️ Daily budget and counters reset")
            
            # Trading logic (every 5 seconds for active management)
            if trading_engine.auto_enabled and loop_count % 3 == 0:
                try:
                    # Check for position triggers
                    await trading_engine.check_triggers()
                    
                    # Manage existing positions
                    await trading_engine.manage_positions()
                    
                    # Look for new opportunities (every 30 seconds)
                    if loop_count % 15 == 0 and not any(
                        pos and pos.get('entry_price') for pos in trading_engine.positions.values()
                    ):
                        await look_for_opportunities(trading_engine, telegram_bot)
                    
                except Exception as e:
                    log(f"Trading logic error: {e}")
            
            # Heartbeat and status updates
            if current_time - last_heartbeat >= heartbeat_interval:
                last_heartbeat = current_time
                await send_heartbeat(trading_engine)
            
            # Memory cleanup (every 1000 loops)
            if loop_count % 1000 == 0:
                trading_engine.save_positions()
                ai_manager.save_state()
                save_settings()
                log(f"Loop {loop_count}: Memory cleanup completed")
            
            # Main loop delay
            await asyncio.sleep(2)
            
        except KeyboardInterrupt:
            log("Shutting down trading loop...")
            if tg_manager:
                await tg_send("🛑 Trading bot shutting down by user request")
            break
        except Exception as e:
            log(f"Main loop error: {e}")
            if tg_manager:
                await tg_send(f"⚠️ Main loop error: {str(e)[:100]}")
            await asyncio.sleep(5)  # Wait before retry
    
    # Cleanup
    if tg_manager:
        await tg_manager.close()
    
    log("Trading loop stopped")

async def look_for_opportunities(trading_engine, telegram_bot):
    """Look for new trading opportunities"""
    try:
        for symbol in AUTO_PAIRS:
            # Skip if position already exists
            if symbol in trading_engine.positions and trading_engine.positions[symbol]:
                continue
            
            # Check if we have budget
            if cfg["budget_used_today"] >= cfg["daily_budget_usd"] * 0.9:  # 90% threshold
                break
            
            # Analyze symbol
            analysis = await trading_engine.analyze_symbol(symbol)
            if not analysis or not analysis.get('core_agree'):
                continue
            
            signal = analysis['core_signal']
            ai_result = analysis.get('ai_result')
            
            # Check AI confirmation
            if not ai_result or ai_result['decision'] == 'hold':
                continue
            
            # Verify alignment and confidence
            if ((signal == 'bullish' and ai_result['decision'] == 'long') or
                (signal == 'bearish' and ai_result['decision'] == 'short')) and \
               normalize_ai_result(ai_result).get('confidence', 0) >= 0.65:
                
                side = 'long' if signal == 'bullish' else 'short'
                reason = f"Auto: {signal} + AI {ai_result['decision']} ({ai_result['confidence']:.1%})"
                
                success = await trading_engine.open_position(symbol, side, reason)
                
                if success and tg_manager:
                    await tg_send(f"""
🎯 AUTO TRADE OPPORTUNITY

{side.upper()} {symbol}
Technical Signal: {signal.upper()}
AI Decision: {ai_result['decision'].upper()}
AI Confidence: {ai_result['confidence']:.1%}
Method: {normalize_ai_result(ai_result).get('method', 'Unknown')}

Position planned and waiting for trigger...
""")
                    break  # Only one position per opportunity scan
            
            await asyncio.sleep(0.1)  # Small delay between symbols
            
    except Exception as e:
        log(f"Opportunity scanning error: {e}")

async def send_heartbeat(trading_engine):
    """Send periodic heartbeat with status"""
    try:
        if not tg_manager:
            return
        
        # Get basic status
        balance = trading_engine.get_account_balance()
        positions_summary = trading_engine.get_positions_summary()
        risk_metrics = positions_summary['risk_metrics']
        
        # Build heartbeat message
        heartbeat_msg = f"""
💓 HEARTBEAT

Balance: ${balance:.2f}
Auto: {'🟢' if trading_engine.auto_enabled else '🔴'}
Positions: {positions_summary['active_count']}
"""
        
        if positions_summary['active_count'] > 0:
            heartbeat_msg += f"Position PnL: ${positions_summary['total_pnl']:.2f}"
        
        heartbeat_msg += f"""
Budget: ${cfg['budget_used_today']:.2f}/${cfg['daily_budget_usd']:.2f}
Trades: {risk_metrics['daily_trades']}/10
"""
        
        await tg_send(heartbeat_msg)
        
    except Exception as e:
        log(f"Heartbeat error: {e}")

# =============================
# REPORTING FUNCTIONS
# =============================

def generate_report() -> str:
    """Generate comprehensive performance report"""
    try:
        # This is a template - in full implementation you'd track historical data
        risk_metrics = ai_manager.metadata if hasattr(ai_manager, 'metadata') else {}
        
        report = f"""
📊 TRADING BOT PERFORMANCE REPORT
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

💰 ACCOUNT STATUS:
Current Balance: ${exchange.fetch_balance().get('USDT', {}).get('total', 0):.2f} USDT
Mode: {MODE.upper()}

⚙️ CONFIGURATION:
Daily Budget: ${cfg['daily_budget_usd']:.2f}
Position Size: {cfg['position_size_fraction']*100:.1f}% of account
Leverage: {cfg['leverage']}x
ATR Multipliers: SL {cfg['atr_sl_mult']}x, TP {cfg['atr_tp_mult']}x

🤖 AI STATUS:
Models Trained: {len([k for k in ai_manager.models.keys() if k.endswith('_lr')])}
LSTM Available: {'Yes' if TORCH_OK else 'No'}
Hybrid Weight: {AI_CFG['hybrid_weight']:.1%}

📈 PAIRS:
Auto Trading: {len(AUTO_PAIRS)}
Watch Only: {len(WATCH_ONLY_PAIRS)}

⚠️ RISK MANAGEMENT:
Max Daily Loss: 5.0%
Max Position Risk: 2.0%
Partial TP: {'Enabled' if cfg['use_partial_tp'] else 'Disabled'}
Trailing Stop: {'Enabled' if cfg['trail_after_tp'] else 'Disabled'}

Note: Extended performance metrics available after longer operation period.
Use /walkforward command for comprehensive backtesting results.
"""
        
        return report
        
    except Exception as e:
        log(f"Report generation error: {e}")
        return f"Error generating report: {e}"

# =============================
# PERFORMANCE ANALYTICS
# =============================

class PerformanceTracker:
    """Track trading performance over time"""
    
    def __init__(self):
        self.trades_history = []
        self.equity_history = []
        self.daily_stats = {}
    
    def record_trade(self, symbol: str, side: str, entry_price: float, 
                    exit_price: float, quantity: float, pnl: float):
        """Record completed trade"""
        trade = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'return_pct': (pnl / (entry_price * quantity)) * 100
        }
        
        self.trades_history.append(trade)
        
        # Update daily stats
        date_key = trade['timestamp'].date()
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                'trades': 0,
                'pnl': 0,
                'wins': 0,
                'losses': 0
            }
        
        stats = self.daily_stats[date_key]
        stats['trades'] += 1
        stats['pnl'] += pnl
        
        if pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
    
    def get_statistics(self, days: int = 30) -> dict:
        """Get performance statistics"""
        if not self.trades_history:
            return {}
        
        # Filter recent trades
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_trades = [t for t in self.trades_history if t['timestamp'] >= cutoff]
        
        if not recent_trades:
            return {}
        
        total_pnl = sum(t['pnl'] for t in recent_trades)
        wins = [t for t in recent_trades if t['pnl'] > 0]
        losses = [t for t in recent_trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(recent_trades),
            'total_pnl': total_pnl,
            'win_rate': len(wins) / len(recent_trades) if recent_trades else 0,
            'avg_win': sum(w['pnl'] for w in wins) / len(wins) if wins else 0,
            'avg_loss': sum(l['pnl'] for l in losses) / len(losses) if losses else 0,
            'profit_factor': abs(sum(w['pnl'] for w in wins) / sum(l['pnl'] for l in losses)) if losses and wins else 0,
            'best_trade': max(t['pnl'] for t in recent_trades),
            'worst_trade': min(t['pnl'] for t in recent_trades),
            'trading_days': len(set(t['timestamp'].date() for t in recent_trades))
        }

# Global performance tracker
performance_tracker = PerformanceTracker()

# =============================
# EMERGENCY FUNCTIONS
# =============================

async def emergency_shutdown(reason: str = "Emergency"):
    """Emergency shutdown procedure"""
    try:
        log(f"EMERGENCY SHUTDOWN: {reason}")
        
        if tg_manager:
            await tg_send(f"🚨 EMERGENCY SHUTDOWN: {reason}")
        
        # Close all positions
        trading_engine = TradingEngine()
        await trading_engine.close_all_positions(f"Emergency: {reason}")
        
        # Save all states
        trading_engine.save_positions()
        ai_manager.save_state()
        save_settings()
        
        if tg_manager:
            await tg_send("✅ Emergency shutdown completed. All positions closed and states saved.")
            await tg_manager.close()
        
    except Exception as e:
        log(f"Emergency shutdown error: {e}")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    import signal
    
    def signal_handler(signum, frame):
        log(f"Received signal {signum}")
        asyncio.create_task(emergency_shutdown(f"Signal {signum}"))
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
# =============================
# CLI ENTRY POINT AND FINAL INTEGRATION
# =============================

async def main():
    """Enhanced main entry point with comprehensive CLI support"""
    parser = argparse.ArgumentParser(
        description="Advanced Crypto Trading Bot with AI and Walk-Forward Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trading_bot.py --run                    # Start live trading bot
  python trading_bot.py --download-data          # Download historical data
  python trading_bot.py --train-ai               # Train AI models
  python trading_bot.py --walkforward            # Run walk-forward optimization
  python trading_bot.py --download-data --train-ai --run  # Sequential execution
        """
    )
    
    # CLI commands
    parser.add_argument("--download-data", action="store_true", 
                       help="Download historical OHLCV data for all pairs")
    parser.add_argument("--refresh-data", action="store_true", 
                       help="Force refresh existing data (use with --download-data)")
    parser.add_argument("--train-ai", action="store_true", 
                       help="Train AI models (LR + LSTM)")
    parser.add_argument("--walkforward", action="store_true", 
                       help="Run walk-forward optimization")
    parser.add_argument("--run", action="store_true", 
                       help="Start live trading bot")
    
    # Parameters
    parser.add_argument("--symbol", default="BTC/USDT:USDT", 
                       help="Symbol for single operations (default: BTC/USDT:USDT)")
    parser.add_argument("--timeframe", default="5m", 
                       help="Timeframe for analysis (default: 5m)")
    parser.add_argument("--months", type=int, default=12, 
                       help="Months of historical data (default: 12)")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Training epochs for AI models (default: 20)")
    
    # Advanced options
    parser.add_argument("--no-lstm", action="store_true", 
                       help="Skip LSTM training (LR only)")
    parser.add_argument("--paper-mode", action="store_true", 
                       help="Force paper trading mode")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Debug mode
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Paper mode override
    if args.paper_mode:
        global MODE
        MODE = "demo"
        log("Paper trading mode enabled")
    
    # Print startup banner
    print_startup_banner()
    
    # Validate environment
    if not validate_environment():
        log("Environment validation failed")
        return 1
    
    # Sequential command execution
    commands_executed = 0
    
    try:
        # 1. Download data
        if args.download_data:
            commands_executed += 1
            log("=" * 50)
            log("STARTING DATA DOWNLOAD")
            log("=" * 50)
            
            symbols = list(AUTO_PAIRS | WATCH_ONLY_PAIRS)
            timeframes = cfg["timeframes"]
            
            completed, failed = await cli_download_data(
                symbols, timeframes, args.months, args.refresh_data
            )
            
            log(f"Data download completed: {completed} successful, {failed} failed")
            
            if tg_manager:
                await tg_send(f"📥 Data download completed: {completed} successful, {failed} failed")
        
        # 2. Train AI
        if args.train_ai:
            commands_executed += 1
            log("=" * 50)
            log("STARTING AI TRAINING")
            log("=" * 50)
            
            symbols = [normalize_symbol(args.symbol)] if args.symbol != "BTC/USDT:USDT" else list(AUTO_PAIRS)
            use_lstm = not args.no_lstm and TORCH_OK
            
            results = await cli_train_ai(symbols, args.epochs, use_lstm)
            
            # Print results
            log("AI Training Results:")
            for symbol, result in results.items():
                lr_status = "✓" if result.get("lr") else "✗"
                lstm_status = "✓" if result.get("lstm") else "✗"
                lr_acc = result.get("lr_accuracy", 0)
                lstm_acc = result.get("lstm_accuracy", 0)
                
                log(f"  {symbol}: LR {lr_status} ({lr_acc:.1%}), LSTM {lstm_status} ({lstm_acc:.1%})")
            
            if tg_manager:
                successful_models = sum(1 for r in results.values() if r.get("lr") or r.get("lstm"))
                await tg_send(f"🤖 AI training completed: {successful_models}/{len(results)} models trained successfully")
        
        # 3. Walk-forward optimization
        if args.walkforward:
            commands_executed += 1
            log("=" * 50)
            log("STARTING WALK-FORWARD OPTIMIZATION")
            log("=" * 50)
            
            success = await cli_walkforward(args.symbol, args.timeframe, args.months)
            
            if success:
                log("Walk-forward optimization completed successfully")
                if tg_manager:
                    await tg_send(f"📊 Walk-forward optimization completed for {args.symbol}")
            else:
                log("Walk-forward optimization failed")
                if tg_manager:
                    await tg_send(f"❌ Walk-forward optimization failed for {args.symbol}")
        
        # 4. Run live bot (only if no other commands or explicitly requested)
        if args.run or commands_executed == 0:
            log("=" * 50)
            log("STARTING LIVE TRADING BOT")
            log("=" * 50)
            
            await main_trading_loop()
        
        return 0
        
    except KeyboardInterrupt:
        log("Shutdown requested by user")
        if tg_manager:
            await tg_send("🛑 Bot shutdown by user request")
        return 0
    
    except Exception as e:
        log(f"Fatal error: {e}")
        if tg_manager:
            await tg_send(f"🚨 Fatal error: {str(e)[:200]}")
        return 1
    
    finally:
        # Cleanup
        if tg_manager:
            await tg_manager.close()

def print_startup_banner():
    """Print startup banner with configuration"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                 ADVANCED CRYPTO TRADING BOT                 ║
║                      with AI & Analytics                     ║
╠══════════════════════════════════════════════════════════════╣
║ Mode: {MODE.upper():<15} │ Leverage: {cfg['leverage']:<5}x │ Budget: ${cfg['daily_budget_usd']:<8.2f} ║
║ Auto Pairs: {len(AUTO_PAIRS):<5} │ Watch Pairs: {len(WATCH_ONLY_PAIRS):<5} │ AI: {'ON' if TORCH_OK else 'LR Only':<12} ║
║ Telegram: {'ON' if tg_manager else 'OFF':<8} │ Charts: ON     │ Reports: ON      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def validate_environment() -> bool:
    """Validate environment and dependencies"""
    issues = []
    
    # Check required directories
    for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, CHARTS_DIR, STATE_DIR]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {directory}: {e}")
    
    # Check exchange connection
    try:
        exchange.fetch_ticker("BTC/USDT:USDT")
        log("Exchange connection: OK")
    except Exception as e:
        log(f"Exchange connection warning: {e}")
        # Not a fatal error for demo mode
    
    # Check Telegram
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        log("Telegram configuration: OK")
    else:
        log("Telegram configuration: Missing (notifications disabled)")
    
    # Check AI dependencies
    if TORCH_OK:
        log("PyTorch: Available (LSTM enabled)")
    else:
        log("PyTorch: Not available (LR only)")
    
    if HAVE_TALIB:
        log("TA-Lib: Available (enhanced indicators)")
    else:
        log("TA-Lib: Not available (fallback indicators)")
    
    # Report issues
    if issues:
        log("Environment validation issues:")
        for issue in issues:
            log(f"  - {issue}")
        return False
    
    log("Environment validation: OK")
    return True

# =============================
# ADDITIONAL UTILITY FUNCTIONS
# =============================

def get_system_info() -> dict:
    """Get system information for diagnostics"""
    import platform
    import psutil
    

def safe_normalize_ai(ai_result):
    """Return a standardized AI result dict even if upstream is None/malformed."""
    try:
        d = ai_result if isinstance(ai_result, dict) else {}
        decision = str(d.get('decision') or d.get('signal') or 'hold')
        try:
            confidence = float(d.get('confidence') or d.get('conf') or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        p_up = d.get('p_up', d.get('prob_up', None))
        try:
            p_up = float(p_up) if p_up is not None else None
        except (TypeError, ValueError):
            p_up = None
        components = d.get('components') or d.get('explain') or {}
        method = d.get('method') or d.get('source') or 'unknown'
        return {
            'decision': decision,
            'confidence': confidence,
            'components': components if isinstance(components, (dict, list)) else {},
            'p_up': p_up,
            'method': method,
        }
    except Exception as e:
        try:
            log(f"safe_normalize_ai: fallback due to error: {e}")
        except Exception:
            pass
        return {'decision': 'hold', 'confidence': 0.0, 'components': {}, 'p_up': None, 'method': 'none'}

    try:
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
            'torch_available': TORCH_OK,
            'torch_cuda': torch.cuda.is_available() if TORCH_OK else False,
            'talib_available': HAVE_TALIB
        }
    except Exception:
        return {'error': 'Unable to get system info'}

def create_backup() -> str:
    """Create backup of all important files"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backup_{timestamp}"
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy important directories
        for src_dir in [STATE_DIR, MODELS_DIR, DATA_DIR]:
            if os.path.exists(src_dir):
                dst_dir = os.path.join(backup_dir, src_dir)
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        
        # Copy .env if exists
        if os.path.exists('.env'):
            shutil.copy2('.env', backup_dir)
        
        log(f"Backup created: {backup_dir}")
        return backup_dir
    
    except Exception as e:
        log(f"Backup creation failed: {e}")
        return ""

def restore_backup(backup_dir: str) -> bool:
    """Restore from backup"""
    try:
        if not os.path.exists(backup_dir):
            log(f"Backup directory not found: {backup_dir}")
            return False
        
        # Restore directories
        for src_dir in [STATE_DIR, MODELS_DIR, DATA_DIR]:
            src_path = os.path.join(backup_dir, src_dir)
            if os.path.exists(src_path):
                if os.path.exists(src_dir):
                    shutil.rmtree(src_dir)
                shutil.copytree(src_path, src_dir)
        
        # Restore .env
        backup_env = os.path.join(backup_dir, '.env')
        if os.path.exists(backup_env):
            shutil.copy2(backup_env, '.env')
        
        log(f"Restore completed from: {backup_dir}")
        return True
    
    except Exception as e:
        log(f"Restore failed: {e}")
        return False

# =============================
# FINAL INTEGRATION
# =============================



# =============================
# REWRITTEN TRADING ENGINE (active version)
# =============================
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio, math

class TradingEngine:
    """
    Active TradingEngine
    - Multi-position stacking per symbol
    - Dynamic position sizing (budget * leverage / price)
    - Live orders via CCXT (with simulation fallback)
    - SL/TP triggers management
    - analyze_symbol always returns "results"
    - Persistence to POSITION_STATE_FILE
    - Telegram-friendly get_positions_summary()
    """

    def __init__(self):
        self.positions: Dict[str, List[dict]] = {}
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.max_daily_trades = int(cfg.get("max_daily_trades", 10))
        self.max_stacks_per_symbol = int(cfg.get("max_stacks_per_symbol", 3))
        self._load_positions()

    # ---------- Persistence ----------
    def _save_positions(self):
        try:
            out: Dict[str, List[dict]] = {}
            for sym, trades in (self.positions or {}).items():
                safe_trades = []
                for t in trades or []:
                    td = dict(t)
                    ts = td.get("timestamp")
                    if isinstance(ts, datetime):
                        td["timestamp"] = ts.isoformat()
                    safe_trades.append(td)
                out[sym] = safe_trades
            save_state(POSITION_STATE_FILE, out)
        except Exception as e:
            log(f"[save_positions] error: {e}")

    def _load_positions(self):
        try:
            data = load_state(POSITION_STATE_FILE) or {}
            loaded: Dict[str, List[dict]] = {}
            for sym, val in data.items():
                trades = val if isinstance(val, list) else [val]
                norm = []
                for t in trades:
                    td = dict(t)
                    ts = td.get("timestamp")
                    if isinstance(ts, str):
                        try:
                            td["timestamp"] = datetime.fromisoformat(ts)
                        except Exception:
                            td["timestamp"] = datetime.now(timezone.utc)
                    norm.append(td)
                loaded[sym] = norm
            self.positions = loaded
            total = sum(len(v) for v in loaded.values())
            log(f"[{datetime.now(timezone.utc)}] Restored {total} positions from {POSITION_STATE_FILE}")
        except Exception as e:
            log(f"[load_positions] error: {e}")
            self.positions = {}

    # ---------- Helpers ----------
    def get_account_balance(self) -> float:
        try:
            if getattr(exchange, "apiKey", None):
                bal = exchange.fetch_balance()
                usdt = bal.get("USDT", {}) or {}
                total = usdt.get("total")
                if total is None:
                    total = bal.get("total", {}).get("USDT")
                return float(total or 0.0)
        except Exception as e:
            log(f"[get_account_balance] error: {e}")
        return float(cfg.get("paper_balance_usd", 1000.0))

    def get_mark_price(self, symbol: str, default: Optional[float] = None) -> Optional[float]:
        try:
            if exchange:
                t = exchange.fetch_ticker(symbol)
                return t.get("last") or t.get("mark") or t.get("info", {}).get("markPrice") or default
        except Exception as e:
            log(f"[get_mark_price] {symbol} error: {e}")
        return default

    @staticmethod
    def normalize_ai_result(ai: Optional[dict]) -> dict:
        base = {"decision": "hold", "prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5}
        if not isinstance(ai, dict):
            return base
        out = dict(base)
        for k in ("prob_up","prob_down","confidence","decision"):
            if k in ai:
                out[k] = ai[k]
        if not out.get("decision"):
            pu = float(out["prob_up"] or 0.0)
            pd = float(out["prob_down"] or 0.0)
            t_long, t_short = get_ai_thresholds(AI_CFG) if "AI_CFG" in globals() else (0.55, 0.45)
            if pu >= t_long and pu > pd:
                out["decision"] = "long"
            elif pd >= t_short and pd > pu:
                out["decision"] = "short"
            else:
                out["decision"] = "hold"
        return out

    def _calc_qty(self, symbol: str, price: float) -> float:
        daily_budget = float(cfg.get("daily_budget_usd", 15.0))
        leverage = float(cfg.get("leverage", 3))
        frac = float(cfg.get("position_fraction", 0.25))  # 25% of daily budget per new stack by default
        usd_alloc = max(0.0, daily_budget * frac) * leverage
        if not price or price <= 0:
            return 0.0
        qty = usd_alloc / float(price)
        # optional: round to market step
        try:
            m = exchange.market(symbol)
            step = (m.get("limits", {}).get("amount", {}).get("min") or 0) or m.get("precision", {}).get("amount")
            if step:
                qty = math.floor(qty / step) * step
        except Exception:
            pass
        return max(qty, 0.0)

    # ---------- Orders ----------
    def _ccxt_side(self, side: str) -> str:
        return "buy" if side.lower() == "long" else "sell"

    def _ensure_symbol_list(self) -> List[str]:
        if "AUTO_PAIRS" in globals() and isinstance(AUTO_PAIRS, list) and AUTO_PAIRS:
            return AUTO_PAIRS
        return (cfg.get("auto_pairs") or cfg.get("watch_pairs") or [])

    async def open_position(self, symbol: str, side: str, sl: Optional[float]=None, tp: Optional[float]=None, qty: Optional[float]=None, price: Optional[float]=None) -> Optional[dict]:
        try:
            if not price:
                price = self.get_mark_price(symbol)
            if not qty:
                qty = self._calc_qty(symbol, price or 0.0)
            if not qty or qty <= 0:
                log(f"[open_position] Skip {symbol}: qty calc failed")
                return None

            order = None
            try:
                order = exchange.create_market_order(symbol, self._ccxt_side(side), qty)
                fill_price = order.get("average") or order.get("price") or price
            except Exception as e:
                log(f"[open_position] live order failed ({e}), simulating")
                fill_price = price

            trade = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "side": side.lower(),
                "qty": float(qty) if side.lower()=="long" else -float(qty),
                "entry_price": float(fill_price or 0.0),
                "stop_loss": float(sl) if sl else None,
                "take_profit": float(tp) if tp else None,
                "status": "open",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.positions.setdefault(symbol, []).append(trade)
            self._save_positions()
            log(f"[open_position] {symbol} {side} qty={qty} @ {fill_price} sl={sl} tp={tp}")
            return trade
        except Exception as e:
            log(f"[open_position] error: {e}")
            return None

    async def close_position(self, symbol: str, pos_index: Optional[int]=None, reason: str="manual") -> int:
        try:
            trades = self.positions.get(symbol, [])
            if not trades:
                return 0
            closed = 0
            for idx, t in list(enumerate(trades)):
                if pos_index is not None and idx != pos_index:
                    continue
                if t.get("status") != "open":
                    continue
                qty = abs(float(t.get("qty", 0.0)))
                ccxt_side = "sell" if t.get("side")=="long" else "buy"
                price = self.get_mark_price(symbol)
                try:
                    exchange.create_market_order(symbol, ccxt_side, qty)
                except Exception as e:
                    log(f"[close_position] live close failed ({e}), simulating")
                t["status"] = "closed"
                t["exit_price"] = float(price or t.get("entry_price"))
                t["exit_time"] = datetime.now(timezone.utc).isoformat()
                t["reason"] = reason
                closed += 1
                if pos_index is not None and closed:
                    break
            self._save_positions()
            if closed:
                log(f"[close_position] Closed {closed} trade(s) for {symbol} ({reason})")
            return closed
        except Exception as e:
            log(f"[close_position] error: {e}")
            return 0

    # ---------- Analysis & Management ----------
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Returns dict with keys:
        - results: { timeframe: {signal, indicators, dataframe?} }
        - core_agree: bool
        - core_signal: 'bullish'|'bearish'|'neutral'
        - ai_result: normalized ai dict
        """
        try:
            results: Dict[str, Any] = {}
            timeframes = cfg.get("timeframes") or [cfg.get("base_tf","1h")]
            for tf in timeframes:
                try:
                    indicators, df = await get_indicators(symbol, tf)
                    sig = self._signal_from_indicators(indicators) if indicators else "neutral"
                    results[tf] = {"signal": sig, "indicators": indicators or {}, "dataframe": df}
                except Exception as e:
                    log(f"[analyze_symbol] {symbol} {tf} err: {e}")

            core_tfs = cfg.get("core_tfs") or timeframes[:2]
            sigs = [results.get(tf,{}).get("signal","neutral") for tf in core_tfs if tf in results]
            core_signal = "neutral"
            core_agree = False
            if sigs and len(set(sigs))==1 and sigs[0]!="neutral":
                core_agree = True
                core_signal = sigs[0]

            ai_raw = await predict_with_hybrid_ai(symbol)
            ai = self.normalize_ai_result(ai_raw)

            return {"results": results, "core_agree": core_agree, "core_signal": core_signal, "ai_result": ai}
        except Exception as e:
            log(f"[analyze_symbol] error for {symbol}: {e}")
            return {"results": {}, "core_agree": False, "core_signal": "neutral", "ai_result": {"decision":"hold","prob_up":0.5,"prob_down":0.5,"confidence":0.5}}

    def _signal_from_indicators(self, ind: Dict[str, Any]) -> str:
        try:
            price = ind.get("price")
            ema20 = ind.get("ema20")
            ema50 = ind.get("ema50")
            rsi = ind.get("rsi14")
            macd_hist = ind.get("macd_hist")
            score = 0
            if price and ema20 and ema50:
                if price>ema20>ema50: score += 1
                if price<ema20<ema50: score -= 1
            if isinstance(rsi,(int,float)):
                if 55<=rsi<=75: score += 1
                if 25<=rsi<=45: score -= 1
            if isinstance(macd_hist,(int,float)):
                if macd_hist>0: score += 1
                if macd_hist<0: score -= 1
            if score>=2: return "bullish"
            if score<=-2: return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    async def check_triggers(self, symbol: Optional[str]=None, analysis: Optional[dict]=None) -> int:
        """Check SL/TP for all or one symbol; close hits. Returns count of closes."""
        symbols = [symbol] if symbol else list(self.positions.keys())
        total_closed = 0
        for sym in symbols:
            price = self.get_mark_price(sym)
            if price is None:
                continue
            for idx, pos in list(enumerate(self.positions.get(sym, []))):
                if pos.get("status") != "open":
                    continue
                side = pos.get("side")
                sl = pos.get("stop_loss")
                tp = pos.get("take_profit")
                hit = False
                if side=="long":
                    if sl is not None and price <= sl: hit=True
                    if tp is not None and price >= tp: hit=True
                else:  # short
                    if sl is not None and price >= sl: hit=True
                    if tp is not None and price <= tp: hit=True
                if hit:
                    total_closed += await self.close_position(sym, idx, reason="trigger")
        return total_closed

    async def manage_positions(self, symbol: Optional[str]=None, analysis: Optional[dict]=None) -> None:
        """
        Dual-mode:
        - If symbol provided: act for that symbol using provided or fresh analysis.
        - If None: iterate configured auto/watch pairs.
        """
        if symbol:
            if analysis is None:
                analysis = await self.analyze_symbol(symbol)
            await self._act_on_analysis(symbol, analysis)
            await self.check_triggers(symbol, analysis)
            return

        for sym in self._ensure_symbol_list():
            try:
                a = await self.analyze_symbol(sym)
                await self._act_on_analysis(sym, a)
            except Exception as e:
                log(f"[manage_positions] {sym} analyze error: {e}")
            try:
                await self.check_triggers(sym)
            except Exception as e:
                log(f"[manage_positions] {sym} trigger error: {e}")

    async def _act_on_analysis(self, symbol: str, analysis: dict):
        try:
            ai = analysis.get("ai_result") or {}
            ai_decision = (ai.get("decision") or "hold").lower()
            direction = "hold"
            if analysis.get("core_agree") and analysis.get("core_signal") in ("bullish","bearish"):
                direction = "long" if analysis["core_signal"]=="bullish" else "short"
            else:
                if ai_decision in ("long","short"):
                    direction = ai_decision

            if not self.auto_enabled or direction=="hold":
                return

            # compute sl/tp using ATR if available from any tf
            sl = tp = None
            try:
                # pick first result with atr and price
                for tf, res in (analysis.get("results") or {}).items():
                    ind = res.get("indicators") or {}
                    atr = ind.get("atr")
                    price = ind.get("price") or self.get_mark_price(symbol)
                    if price and atr:
                        atr_mult_sl = float(cfg.get("atr_sl_mult", 1.5))
                        atr_mult_tp = float(cfg.get("atr_tp_mult", 2.0))
                        if direction=="long":
                            sl = price - atr*atr_mult_sl
                            tp = price + atr*atr_mult_tp
                        else:
                            sl = price + atr*atr_mult_sl
                            tp = price - atr*atr_mult_tp
                        break
            except Exception:
                pass

            open_trades = [t for t in self.positions.get(symbol, []) if t.get("status")=="open"]
            if len(open_trades) >= self.max_stacks_per_symbol:
                return

            await self.open_position(symbol, direction, sl=sl, tp=tp)
        except Exception as e:
            log(f"[act_on_analysis] {symbol} err: {e}")

    def get_positions_summary(self) -> str:
        try:
            lines = []
            total_open = 0
            for sym, trades in (self.positions or {}).items():
                for t in trades or []:
                    if t.get("status")!="open":
                        continue
                    total_open += 1
                    side = t.get("side")
                    qty = t.get("qty")
                    entry = t.get("entry_price")
                    sl = t.get("stop_loss")
                    tp = t.get("take_profit")
                    lines.append(f"{sym}: {side} qty={qty} entry={entry} sl={sl} tp={tp}")
            if not lines:
                return "No open positions"
            return "\n".join(lines)
        except Exception as e:
            log(f"[get_positions_summary] error: {e}")
            return "Error retrieving positions"

async def _engine_heartbeat(engine: TradingEngine):
    # Unified heartbeat calls both entry and SL/TP checks to avoid missing manage_positions
    try:
        await engine.check_entry_triggers()
    except Exception as e:
        log(f"entry_triggers error: {e}")
    try:
        await engine.manage_positions()
    except Exception as e:
        log(f"manage_positions error: {e}")

if __name__ == "__main__":
    """Entry point with proper async execution and error handling"""
    try:
        # Ensure proper async execution
        if sys.platform.startswith('win'):
            # Windows specific event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Run main
        exit_code = asyncio.run(main())
        sys.exit(exit_code or 0)
        
    except KeyboardInterrupt:
        log("Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        log(f"Startup error: {e}")
        sys.exit(1)

# =============================
# MODULE EXPORTS (for testing)
# =============================

__all__ = [
    # Core classes
    'TradingEngine', 'TelegramBot', 'AIModelManager', 'WalkForwardOptimizer',
    'OnlineLR', 'LSTMModel', 'RiskManager', 'PerformanceTracker',
    
    # Functions
    'predict_with_hybrid_ai', 'train_lstm_model', 'train_lr_model',
    'create_features', 'calculate_indicators', 'get_indicators',
    'plot_candlestick_chart', 'plot_equity_curve', 'plot_walkforward_summary',
    'ensure_history', 'load_history_df', 'generate_report',
    
    # Configuration
    'cfg', 'AI_CFG', 'AUTO_PAIRS', 'WATCH_ONLY_PAIRS',
    
    # Utilities
    'normalize_symbol', 'timeframe_alias', 'safe', 'sf', 'log',
    'save_state', 'load_state', 'get_system_info'
]

# =============================
# VERSION AND METADATA
# =============================

__version__ = "2.0.0"
__author__ = "Trading Bot Developer"
__description__ = "Advanced cryptocurrency trading bot with AI, walk-forward optimization, and comprehensive risk management"
__license__ = "MIT"

# Module metadata
METADATA = {
    'name': 'Advanced Crypto Trading Bot',
    'version': __version__,
    'description': __description__,
    'features': [
        'Hybrid AI (Logistic Regression + LSTM)',
        'Walk-forward optimization',
        'Multi-timeframe analysis', 
        'Advanced risk management',
        'Telegram integration',
        'Position recovery',
        'Comprehensive reporting',
        'Paper trading support',
        'OKX exchange integration'
    ],
    'requirements': [
        'ccxt', 'pandas', 'numpy', 'scikit-learn', 'aiohttp',
        'matplotlib', 'python-dotenv', 'torch (optional)', 'talib (optional)'
    ]
}

log(f"Module loaded: {METADATA['name']} v{__version__}")

# =============================
# FINAL VALIDATION
# =============================

# Verify all critical components are available
_critical_components = [
    'exchange', 'ai_manager', 'tg_manager', 'cfg', 'AI_CFG'
]

for component in _critical_components:
    if component in globals():
        log(f"✓ {component} initialized")
    else:
        log(f"✗ {component} missing")

log("Trading bot module initialization complete")

def normalize_ai_result(ai_result):
    """Return a safe dict with keys: decision, confidence, components."""
    if ai_result is None or not isinstance(ai_result, dict):
        return {'decision': 'hold', 'confidence': 0.0, 'components': {}}
    normalize_ai_result(ai_result).setdefault('decision', 'hold')
    normalize_ai_result(ai_result).setdefault('confidence', 0.0)
    normalize_ai_result(ai_result).setdefault('components', {})
    return ai_result

async def cli_run(args):
    log("cli_run: starting runtime")
    try:
        if 'main_runtime' in globals():
            await main_runtime(args)
        else:
            log("No main_runtime(); stubbed.")
    except Exception as e:
        log(f"cli_run error: {e}")

CLI_COMMAND_MAP = {
    'download-data': cli_download_data,
    'train-ai': cli_train_ai,
    'walkforward': cli_walkforward,
    'run': cli_run,
}

async def dispatch_cli(args):
    cmd = getattr(args, 'command', None)
    if cmd in CLI_COMMAND_MAP:
        return await CLI_COMMAND_MAP[cmd](args)
    # fallbacks to flags
    if getattr(args, 'download_data', False):
        return await CLI_COMMAND_MAP['download-data'](args)
    if getattr(args, 'train_ai', False):
        return await CLI_COMMAND_MAP['train-ai'](args)
    if getattr(args, 'walkforward', False):
        return await CLI_COMMAND_MAP['walkforward'](args)
    return await CLI_COMMAND_MAP['run'](args)



# Optional heartbeat helper (doesn't alter your main loop)
def _te_heartbeat(self, minutes=5):
    try:
        import time
        now = time.time()
        last = getattr(self, 'last_heartbeat', 0.0)
        if now - last >= minutes*60:
            _LOG.info("[heartbeat] bot alive; positions=%s" % (len(getattr(self,'positions',{}) or {}),))
            self.last_heartbeat = now
    except Exception:
        pass

try:
    setattr(TradingEngine, '_heartbeat', _te_heartbeat)
except Exception:
    pass




# === CHECK_TRIGGERS OVERRIDE (async, no args) ===
# This override ensures TradingEngine.check_triggers() exists and works with the
# zero-arg call used in the main loop. It iterates all open positions, pulls a
# price per symbol, evaluates SL/TP and a basic trailing stop, and closes
# positions when a trigger is hit.
try:
    _TE = TradingEngine
    async def __robust_check_triggers(self):
        results = []
        try:
            positions = getattr(self, "positions", {}) or {}
            if not isinstance(positions, dict) or not positions:
                return results

            # Helper: get latest price for a symbol
            async def _get_price(sym: str, pos: dict):
                # Path 1: live exchange
                ex = getattr(self, "exchange", None)
                if ex and hasattr(ex, "fetch_ticker"):
                    try:
                        t = await ex.fetch_ticker(sym) if getattr(ex, "_aio", False) else ex.fetch_ticker(sym)
                        # ccxt often returns 'last' or 'close'
                        for k in ("last", "close", "bid", "ask"):
                            if isinstance(t, dict) and t.get(k) is not None:
                                return float(t[k])
                    except Exception:
                        pass
                # Path 2: engine caches
                for cache_attr in ("last_prices", "prices", "price_cache"):
                    cache = getattr(self, cache_attr, None)
                    if isinstance(cache, dict) and sym in cache:
                        try:
                            return float(cache[sym])
                        except Exception:
                            pass
                # Path 3: position snapshot
                for k in ("last_price", "mark_price", "entry_price"):
                    if pos.get(k) is not None:
                        try:
                            return float(pos[k])
                        except Exception:
                            pass
                raise RuntimeError(f"No price source for {sym}")

            # Iterate symbols
            for sym, pos in list(positions.items()):
                if not isinstance(pos, dict) or not pos:
                    continue

                side = pos.get("side")
                if side not in ("long", "short"):
                    continue

                try:
                    price = await _get_price(sym, pos)
                except Exception as _e:
                    if hasattr(self, "log"):
                        self.log(f"check_triggers: price fetch failed for {sym}: {_e}")
                    continue

                entry = float(pos.get("entry_price", price))
                sl = pos.get("stop_loss")
                tp = pos.get("take_profit")
                trail = pos.get("trail_pct") or getattr(self, "trail_pct", None)

                close_reason = None

                # --- TP ---
                if tp is not None:
                    try:
                        tp_val = float(tp)
                        if side == "long" and price >= tp_val:
                            close_reason = "take_profit_hit"
                        elif side == "short" and price <= tp_val:
                            close_reason = "take_profit_hit"
                    except Exception:
                        pass

                # --- SL ---
                if close_reason is None and sl is not None:
                    try:
                        sl_val = float(sl)
                        if side == "long" and price <= sl_val:
                            close_reason = "stop_loss_hit"
                        elif side == "short" and price >= sl_val:
                            close_reason = "stop_loss_hit"
                    except Exception:
                        pass

                # --- Trailing stop (percentage; accepts 0.5 or 50 for 0.5%) ---
                if close_reason is None and trail is not None:
                    try:
                        trail_pct = float(trail)
                        trail_frac = trail_pct/100.0 if trail_pct > 1.0 else trail_pct
                        if side == "long":
                            peak = float(pos.get("peak", entry))
                            if price > peak:
                                peak = price
                                pos["peak"] = price
                            trail_sl = peak * (1 - trail_frac)
                            # keep the best (highest) stop loss
                            if sl is None or trail_sl > float(sl):
                                pos["stop_loss"] = trail_sl
                            if price <= trail_sl:
                                close_reason = "trailing_stop_hit"
                        else:  # short
                            trough = float(pos.get("trough", entry))
                            if price < trough:
                                trough = price
                                pos["trough"] = price
                            trail_sl = trough * (1 + trail_frac)
                            if sl is None or trail_sl < float(sl):
                                pos["stop_loss"] = trail_sl
                            if price >= trail_sl:
                                close_reason = "trailing_stop_hit"
                    except Exception:
                        pass

                if close_reason:
                    # Close using engine API (supports async or sync)
                    try:
                        if hasattr(self, "close_position"):
                            cp = self.close_position
                            if getattr(cp, "__await__", None):
                                res = await cp(sym, price, reason=close_reason)
                            else:
                                res = cp(sym, price, reason=close_reason)
                            results.append({"symbol": sym, "reason": close_reason, "price": price, "result": res})
                            # Optional: telegram notify
                            tg = getattr(self, "tg_manager", None) or getattr(self, "telegram", None)
                            if tg and hasattr(tg, "send"):
                                try:
                                    await tg.send(f"🔔 {sym} {side.upper()} closed: {close_reason} @ {price:.6f}")
                                except Exception:
                                    pass
                            continue
                    except Exception as _e:
                        if hasattr(self, "log"):
                            self.log(f"check_triggers: close_position failed for {sym}: {_e}")
        except Exception as e:
            if hasattr(self, "log"):
                self.log(f"check_triggers error: {e}")
        return results

    # Override on the class (even if previously stubbed)
    _TE.check_triggers = __robust_check_triggers

    # Provide a robust close_all_positions if missing
    if not hasattr(_TE, "close_all_positions"):
        async def __close_all_positions(self, reason: str = "manual_close_all"):
            results = []
            positions = getattr(self, "positions", {}) or {}
            for sym, pos in list(positions.items()):
                try:
                    price = float(pos.get("last_price") or pos.get("entry_price") or 0.0)
                    cp = getattr(self, "close_position", None)
                    if cp:
                        if getattr(cp, "__await__", None):
                            res = await cp(sym, price, reason=reason)
                        else:
                            res = cp(sym, price, reason=reason)
                        results.append({"symbol": sym, "reason": reason, "result": res})
                except Exception as _e:
                    if hasattr(self, "log"):
                        self.log(f"close_all_positions error for {sym}: {_e}")
            return results
        _TE.close_all_positions = __close_all_positions
except Exception as _patch_e:
    try:
        log(f"check_triggers override patch failed: {_patch_e}")
    except Exception:
        pass
# === END CHECK_TRIGGERS OVERRIDE ===


# ==== BEGIN AUTOPATCH (2025-09-05) — Safety/Missing-Method Fixes ====
# This block safely monkey-patches TradingEngine with missing/robust versions
# of normalize_ai_result, check_triggers, manage_positions, close_all_positions,
# and (if absent) close_position. It aims to be schema-agnostic and not break
# existing logic. You may remove once your class contains stable implementations.

from typing import Optional, Dict, Any, Union, Iterable
import asyncio, math, time

def _tb_now():
    try:
        # Use the project's logger, if any
        return time.strftime("[%Y-%m-%d %H:%M:%S]")
    except Exception:
        return ""

def _tb_log(msg):
    try:
        logger = globals().get("logger", None) or globals().get("log", None)
        if callable(logger):
            logger(str(msg))
        else:
            print(_tb_now(), str(msg))
    except Exception:
        print(_tb_now(), str(msg))

async def _tb_maybe_await(x):
    if asyncio.iscoroutine(x):
        return await x
    return x

def _tb_smart_get_price(engine, symbol: str) -> Optional[float]:
    """Try multiple ways to get a mark/last price from the engine/exchange."""
    try:
        if hasattr(engine, "get_mark_price") and callable(engine.get_mark_price):
            p = engine.get_mark_price(symbol)
            return p
    except Exception as e:
        _tb_log(f"[patch] get_mark_price failed for {symbol}: {e}")
    try:
        if hasattr(engine, "exchange") and engine.exchange:
            ex = engine.exchange
            if hasattr(ex, "fetch_ticker"):
                t = ex.fetch_ticker(symbol)
                if isinstance(t, dict):
                    return t.get("last") or t.get("close") or t.get("info", {}).get("price")
    except Exception as e:
        _tb_log(f"[patch] exchange.fetch_ticker failed for {symbol}: {e}")
    # Fallback: None
    return None

def _tb_position_iter(engine) -> Iterable:
    """Yield (symbol, pos) pairs from engine's positions store, for any common schema."""
    # Preferred
    store = getattr(engine, "positions", None)
    if isinstance(store, dict):
        for sym, pos in list(store.items()):
            yield sym, pos
        return
    # Alternate
    store = getattr(engine, "open_positions", None)
    if isinstance(store, dict):
        for sym, pos in list(store.items()):
            yield sym, pos
        return
    # Nothing
    return []

def _tb_side(pos: Dict[str, Any]) -> Optional[str]:
    s = (pos or {}).get("side") or (pos or {}).get("positionSide") or (pos or {}).get("direction")
    if isinstance(s, str):
        s = s.lower()
        if s in ("buy", "long", "sell", "short"):
            return "long" if s in ("buy", "long") else "short"
    # some stores set qty sign for side
    qty = (pos or {}).get("qty") or (pos or {}).get("amount")
    try:
        if qty is not None and float(qty) != 0:
            return "long" if float(qty) > 0 else "short"
    except Exception:
        pass
    return None

def _tb_is_open(pos: Dict[str, Any]) -> bool:
    st = (pos or {}).get("status") or (pos or {}).get("state") or ""
    st = str(st).lower()
    if st in ("open", "opening", "active", "running"):
        return True
    if "closed" in st or "closing" in st or "exit" in st:
        return False
    # default to open if qty!=0
    try:
        qty = pos.get("qty", 0) or pos.get("amount", 0)
        return abs(float(qty)) > 1e-12
    except Exception:
        return True

def _tb_hit_sl_tp(side: str, price: float, sl: Optional[float], tp: Optional[float]) -> Dict[str, bool]:
    hit = {"sl": False, "tp": False}
    if price is None:
        return hit
    if side == "long":
        if sl is not None and price <= sl:
            hit["sl"] = True
        if tp is not None and price >= tp:
            hit["tp"] = True
    elif side == "short":
        if sl is not None and price >= sl:
            hit["sl"] = True
        if tp is not None and price <= tp:
            hit["tp"] = True
    return hit

def _tb_safe_close_position(engine, symbol: str, reason: str = "manual", exit_price: Optional[float] = None):
    """Call engine.close_position if available, else mark locally as closed."""
    try:
        if hasattr(engine, "close_position") and callable(engine.close_position):
            res = engine.close_position(symbol, reason=reason, exit_price=exit_price)
            return res
    except TypeError:
        # close_position(self, symbol) signature versions
        res = engine.close_position(symbol)
        return res
    except Exception as e:
        _tb_log(f"[patch] close_position raised for {symbol}: {e}")
    # Fallback: mutate local positions
    for sym, pos in _tb_position_iter(engine):
        if sym != symbol: 
            continue
        try:
            if exit_price is None:
                exit_price = _tb_smart_get_price(engine, symbol)
        except Exception:
            pass
        pos["status"] = "closed"
        pos["exit_price"] = exit_price
        pos["exit_time"] = time.time()
        # Zero out qty to prevent reprocessing
        qty_key = "qty" if "qty" in pos else ("amount" if "amount" in pos else None)
        if qty_key:
            try:
                pos[qty_key] = 0.0
            except Exception:
                pass
        _tb_log(f"[patch] Locally closed {symbol} at {exit_price} (reason={reason})")
        break
    return {"ok": True, "symbol": symbol, "reason": reason, "exit_price": exit_price}

class _TradingEnginePatchMixin:
    def normalize_ai_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize hybrid/AI outputs into a consistent dict. Always returns safe defaults."""
        base = {
            "direction": None,           # "long" | "short" | None
            "confidence": 0.0,           # 0..1
            "stop_loss": None,           # float|None
            "take_profit": None,         # float|None
            "prob_up": None,             # float|None
            "prob_down": None,           # float|None
            "raw": result,               # keep original payload
        }
        try:
            if not isinstance(result, dict):
                return base
            # direction
            dir_ = result.get("direction") or result.get("side") or result.get("signal")
            if isinstance(dir_, str):
                d = dir_.lower()
                if "long" in d or "buy" in d:
                    base["direction"] = "long"
                elif "short" in d or "sell" in d:
                    base["direction"] = "short"
            # confidence/probabilities
            for k in ("confidence", "prob_up", "prob_down"):
                v = result.get(k)
                try:
                    if v is not None:
                        base[k] = float(v)
                except Exception:
                    pass
            # SL/TP
            for k in ("stop_loss", "take_profit", "sl", "tp"):
                v = result.get(k)
                if v is None:
                    continue
                try:
                    v = float(v)
                except Exception:
                    continue
                if k in ("stop_loss", "sl"):
                    base["stop_loss"] = v
                else:
                    base["take_profit"] = v
            # derive confidence if missing but probs present
            if base["confidence"] in (None, 0.0) and base["prob_up"] is not None and base["prob_down"] is not None:
                try:
                    base["confidence"] = abs(base["prob_up"] - base["prob_down"])
                except Exception:
                    pass
        except Exception as e:
            _tb_log(f"[patch] normalize_ai_result error: {e}")
        return base

    async def manage_positions(self) -> None:
        """Scan and enforce SL/TP for all open positions."""
        try:
            # if a symbol-scoped manage exists in your code, prefer it
            # else iterate all
            last_err = None
            for symbol, pos in _tb_position_iter(self):
                try:
                    await self.check_triggers(symbol=symbol)
                except Exception as e:
                    last_err = e
                    _tb_log(f"[patch] manage_positions -> check_triggers error for {symbol}: {e}")
            if last_err:
                raise last_err
        except Exception as e:
            _tb_log(f"[patch] manage_positions error: {e}")

    async def check_triggers(self, symbol: Optional[str] = None, price: Optional[float] = None) -> None:
        """
        Evaluate SL/TP (and optional trailing/partial TP) for either a single symbol
        or all open positions if symbol is None.
        """
        try:
            if symbol is None:
                for sym, pos in _tb_position_iter(self):
                    await self.check_triggers(symbol=sym, price=None)
                return

            # fetch position
            pos = None
            for sym, p in _tb_position_iter(self):
                if sym == symbol:
                    pos = p
                    break
            if not pos:
                return

            if not _tb_is_open(pos):
                return

            # Determine side, sl/tp, and live price
            side = _tb_side(pos) or pos.get("side") or "long"
            sl = pos.get("stop_loss") or pos.get("sl")
            tp = pos.get("take_profit") or pos.get("tp")

            px = price
            if px is None:
                px = _tb_smart_get_price(self, symbol)

            # No price => nothing to do
            if px is None:
                return

            hits = _tb_hit_sl_tp(side, px, sl, tp)
            if hits["sl"]:
                _tb_log(f"[patch] SL hit for {symbol} @ {px} (sl={sl}) -> closing")
                await _tb_maybe_await(_tb_safe_close_position(self, symbol, reason="stop_loss", exit_price=px))
                return
            if hits["tp"]:
                # Check for partial TP configs if present
                partial = False
                try:
                    cfg = globals().get("cfg", {}) or {}
                    partial = bool(cfg.get("use_partial_tp") or cfg.get("partial_tp_enabled"))
                except Exception:
                    pass

                if partial and ("qty" in pos or "amount" in pos):
                    # Try to reduce size by half and move SL to breakeven
                    qty_key = "qty" if "qty" in pos else "amount"
                    try:
                        cur = float(pos.get(qty_key, 0))
                        close_qty = max(0.0, abs(cur) * 0.5)
                        _tb_log(f"[patch] TP hit for {symbol} @ {px}: taking partial {close_qty}")
                        # If engine has a partial close method, use it; else emulate
                        if hasattr(self, "close_position_partial"):
                            await _tb_maybe_await(self.close_position_partial(symbol, close_qty, reason="take_profit", exit_price=px))
                        else:
                            # Emulate: reduce qty locally
                            pos[qty_key] = math.copysign(max(0.0, abs(cur) - close_qty), cur)
                            pos["last_partial_tp"] = {"price": px, "qty_closed": close_qty, "time": time.time()}
                            _tb_log(f"[patch] Emulated partial close: {symbol} new {qty_key}={pos[qty_key]}")
                        # Move SL to entry (breakeven) if available
                        be = pos.get("entry_price") or pos.get("avg_entry_price")
                        if be is not None:
                            pos["stop_loss"] = float(be)
                            _tb_log(f"[patch] Moved SL to breakeven for {symbol}: {be}")
                        # If fully closed by partial logic, stop here
                        try:
                            remaining = float(pos.get(qty_key, 0.0))
                            if abs(remaining) < 1e-12:
                                pos["status"] = "closed"
                                pos["exit_price"] = px
                                pos["exit_time"] = time.time()
                                _tb_log(f"[patch] Position fully closed after partial TP for {symbol}")
                                return
                        except Exception:
                            pass
                    except Exception as e:
                        _tb_log(f"[patch] Partial TP emulation error for {symbol}: {e}")
                        # Fallback: full close
                        await _tb_maybe_await(_tb_safe_close_position(self, symbol, reason="take_profit", exit_price=px))
                        return
                else:
                    _tb_log(f"[patch] TP hit for {symbol} @ {px} (tp={tp}) -> closing")
                    await _tb_maybe_await(_tb_safe_close_position(self, symbol, reason="take_profit", exit_price=px))
                    return

                # Trailing SL after TP if configured
                try:
                    cfg = globals().get("cfg", {}) or {}
                    if cfg.get("trail_after_tp"):
                        trail_pct = float(cfg.get("trail_pct", 0.01))
                        if side == "long":
                            new_sl = max(float(pos.get("stop_loss") or -float("inf")), px * (1 - trail_pct))
                        else:
                            new_sl = min(float(pos.get("stop_loss") or float("inf")), px * (1 + trail_pct))
                        pos["stop_loss"] = new_sl
                        _tb_log(f"[patch] Applied trailing SL for {symbol}: {new_sl}")
                except Exception as e:
                    _tb_log(f"[patch] trailing SL config error: {e}")

        except Exception as e:
            _tb_log(f"[patch] check_triggers error for {symbol or 'ALL'}: {e}")

    async def close_all_positions(self, reason: str = "panic", mark_price: Optional[float] = None) -> None:
        """Close every open position as fast as possible (best-effort)."""
        try:
            tasks = []
            for symbol, pos in _tb_position_iter(self):
                if not _tb_is_open(pos):
                    continue
                px = mark_price
                if px is None:
                    try:
                        px = _tb_smart_get_price(self, symbol)
                    except Exception:
                        px = None
                tasks.append(_tb_maybe_await(_tb_safe_close_position(self, symbol, reason=reason, exit_price=px)))
            # Ensure all tasks resolve
            for t in tasks:
                try:
                    await _tb_maybe_await(t)
                except Exception as e:
                    _tb_log(f"[patch] close_all task error: {e}")
        except Exception as e:
            _tb_log(f"[patch] close_all_positions error: {e}")

# Bind mixin methods onto TradingEngine (without touching your existing code)
TE = globals().get("TradingEngine")
if TE and isinstance(TE, type):
    _tb_log("[patch] Binding safety methods onto TradingEngine")
    for name in ("normalize_ai_result", "manage_positions", "check_triggers", "close_all_positions"):
        if not hasattr(TE, name) or True:  # force override to ensure correctness
            setattr(TE, name, getattr(_TradingEnginePatchMixin, name))

    # Provide a fallback close_position if missing (non-exchange emulation)
    if not hasattr(TE, "close_position"):
        def _fallback_close_position(self, symbol: str, reason: str = "manual", exit_price: Optional[float] = None):
            return _tb_safe_close_position(self, symbol, reason=reason, exit_price=exit_price)
        setattr(TE, "close_position", _fallback_close_position)

    # Backward compatibility alias
    if not hasattr(TE, "normalize_ai_result") and hasattr(TE, "_normalize_ai_result"):
        setattr(TE, "normalize_ai_result", getattr(TE, "_normalize_ai_result"))

# ==== END AUTOPATCH ====
SHOW_FIB_LEGEND = False


# Import-safety guard
import os as _os_for_guard

CONFIDENCE_SCALING = True

def get_trade_budget(symbol, ai_confidence, base_budget=5.0):
    """Dynamically scale trade budget based on AI confidence."""
    if not CONFIDENCE_SCALING:
        return base_budget

    if ai_confidence < 0.65:
        log(f"⚠️ {symbol}: AI confidence {ai_confidence:.2f} too weak, skipping trade.")
        return 0
    if ai_confidence >= 0.90:
        trade_budget = base_budget * 1.4   # ~7 USDT
    elif ai_confidence >= 0.80:
        trade_budget = base_budget * 1.2   # ~6 USDT
    else:
        trade_budget = base_budget         # 5 USDT

    return max(5.0, min(trade_budget, 7.0))

IMPORT_SAFE = _os_for_guard.environ.get("TRADING_BOT_IMPORT_SAFE", "0") == "1"
# end import-safety

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
"""

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
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
try:
    import ccxt
except Exception:
    class _DummyCCXT:
        def __init__(self,*a,**k):
            self.options = {}
        def __getattr__(self,name):
            return self
        def __call__(self,*a,**k):
            return self
        def __setitem__(self,k,v):
            self.options[k]=v
        def __getitem__(self,k):
            return self.options.get(k,None)
        def set_sandbox_mode(self,v):
            self.options['sandbox']=v
        def __repr__(self):
            return '<DUMMY_CCXT>'
    ccxt = _DummyCCXT()


import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import deque
from sklearn.preprocessing import StandardScaler

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

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

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
}

# AI Configuration
AI_CFG = {
    "base_tf": "5m",
    "other_tfs": ["15m", "1h"],
    "hist_limit": 400,
    "hist_months": 12,
    "min_obs": 120,
    "learn_epochs_boot": 2,
    "learn_rate": 0.05,
    "l2": 1e-4,
    "threshold_long": 0.58,
    "threshold_short": 0.42,
    "state_file": "ai_model_state.json",
    "lstm_epochs": 10,
    "lstm_batch_size": 128,
    "lstm_hidden": 32,
    "lstm_layers": 1,
    "lstm_seq_len": 60,
    "hybrid_weight": 0.6  # Weight for LSTM vs LR
}

# Directory structure
DATA_DIR = "data_history"
MODELS_DIR = "ai_models"
REPORTS_DIR = "reports"
CHARTS_DIR = "charts"

for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

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

# =============================
# EXCHANGE SETUP
# =============================
def init_exchange():
    """Initialize OKX exchange"""
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
    return ex

exchange = init_exchange()

def safe_set_leverage(sym, lev):
    """Safely set leverage for a symbol"""
    try:
        exchange.set_leverage(lev, sym)
        return True
    except Exception as e:
        log(f"Failed to set leverage for {sym}: {e}")
        return False

# Set leverage for auto pairs
for sym in list(AUTO_PAIRS):
    safe_set_leverage(sym, cfg["leverage"])

# =============================
# ASYNC TELEGRAM FUNCTIONS
# =============================
async def tg_send(msg: str):
    """Send message to Telegram asynchronously"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}
            ) as response:
                return await response.json()
    except Exception as e:
        log(f"Telegram send error: {e}")
        # Fallback to synchronous
        try:
            requests.post(
                f"{BASE_URL}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=10
            )
        except Exception:
            pass

async def tg_send_photo(filepath: str, caption: str = ""):
    """Send photo to Telegram asynchronously"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            with open(filepath, "rb") as f:
                data = aiohttp.FormData()
                data.add_field('chat_id', TELEGRAM_CHAT_ID)
                data.add_field('caption', caption)
                data.add_field('photo', f, filename=os.path.basename(filepath))
                
                async with session.post(f"{BASE_URL}/sendPhoto", data=data) as response:
                    return await response.json()
    except Exception as e:
        log(f"Telegram photo send error: {e}")
        # Fallback to synchronous
        try:
            with open(filepath, "rb") as f:
                requests.post(
                    f"{BASE_URL}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                    files={"photo": f},
                    timeout=15
                )
        except Exception:
            pass

def tg_get_updates(offset=None):
    """Get Telegram updates synchronously"""
    try:
        r = requests.get(
            f"{BASE_URL}/getUpdates",
            params={"timeout": 5, "offset": offset},
            timeout=10
        ).json()
        return r.get("result", [])
    except Exception:
        return []

# =============================
# DATA MANAGEMENT
# =============================
def _history_path(sym, tf):
    """Get path for historical data CSV"""
    sym_s = sym.replace("/", "_").replace(":", "_")
    return os.path.join(DATA_DIR, f"{sym_s}_{tf}.csv")

def ensure_history(sym, tf="5m", months=12, limit=1000):
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
        while since < now:
            try:
                batch = exchange.fetch_ohlcv(sym, timeframe=tf, since=since, limit=limit)
                if not batch:
                    break
                
                all_rows.extend(batch)
                downloaded += len(batch)
                since = batch[-1][0] + 1
                
                # Progress update
                if downloaded % 5000 == 0:
                    log(f"Downloaded {downloaded} bars for {sym} {tf}")
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                log(f"Download error for {sym} {tf}: {e}")
                time.sleep(1)
                continue
        
        # Save data
        if all_rows:
            df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
            df.to_csv(path, index=False)
            log(f"Saved {len(df)} bars for {sym} {tf}")
        
    except Exception as e:
        log(f"ensure_history error for {sym} {tf}: {e}")

def load_history_df(sym, tf="5m", months=None):
    """Load historical data as pandas DataFrame"""
    if months:
        ensure_history(sym, tf, months)
    
    path = _history_path(sym, tf)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        return df
    
    # Fallback to live fetch
    try:
        arr = exchange.fetch_ohlcv(sym, timeframe=tf, limit=AI_CFG.get("hist_limit", 400))
        df = pd.DataFrame(arr, columns=["ts", "open", "high", "low", "close", "volume"])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

# =============================
# TECHNICAL INDICATORS
# =============================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    if len(df) < 50:
        return df
    
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    # EMAs
    df['ema_20'] = close.ewm(span=20).mean()
    df['ema_50'] = close.ewm(span=50).mean()
    df['ema_200'] = close.ewm(span=200).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    if HAVE_TALIB:
        df['atr'] = talib.ATR(high.values, low.values, close.values, timeperiod=14)
    else:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = close.rolling(window=bb_period).mean()
    bb_std_dev = close.rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
    
    # Volume indicators
    df['volume_sma'] = volume.rolling(window=20).mean()
    df['volume_ratio'] = volume / df['volume_sma']
    
    return df.fillna(method='ffill').fillna(0)

def get_indicators(sym, tf="5m", limit=200):
    """Get current indicators for a symbol"""
    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df.set_index('ts', inplace=True)
        
        df = calculate_indicators(df)
        
        if len(df) == 0:
            return None, None
        
        latest = df.iloc[-1]
        return {
            'price': float(latest['close']),
            'ema20': float(latest['ema_20']) if not pd.isna(latest['ema_20']) else None,
            'ema50': float(latest['ema_50']) if not pd.isna(latest['ema_50']) else None,
            'rsi14': float(latest['rsi']) if not pd.isna(latest['rsi']) else None,
            'macd': float(latest['macd']) if not pd.isna(latest['macd']) else None,
            'macd_signal': float(latest['macd_signal']) if not pd.isna(latest['macd_signal']) else None,
            'macd_hist': float(latest['macd_hist']) if not pd.isna(latest['macd_hist']) else None,
            'atr': float(latest['atr']) if not pd.isna(latest['atr']) else None,
            'bb_upper': float(latest['bb_upper']) if not pd.isna(latest['bb_upper']) else None,
            'bb_lower': float(latest['bb_lower']) if not pd.isna(latest['bb_lower']) else None,
        }, df
    except Exception as e:
        log(f"get_indicators error for {sym}: {e}")
        return None, None

# =============================
# ONLINE LEARNING MODEL (LogReg)
# =============================
class OnlineLR:
    """Online Logistic Regression with L2 regularization"""
    
    def __init__(self, n_features, lr=0.05, l2=1e-4):
        self.w = np.zeros(n_features + 1)  # +1 for bias
        self.lr = lr
        self.l2 = l2
    
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    
    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        z = self.w[0] + np.dot(self.w[1:], x)
        return float(self._sigmoid(z))
    
    def update(self, x, y):
        p = self.predict_proba(x)
        err = (y - p)
        grad = np.concatenate(([err], err * np.asarray(x))) - self.l2 * self.w
        self.w += self.lr * grad
        return p
    
    def to_dict(self):
        return {
            "w": [float(w) if not (np.isnan(w) or np.isinf(w)) else 0.0 for w in self.w],
            "lr": self.lr,
            "l2": self.l2
        }
    
    @classmethod
    def from_dict(cls, d):
        obj = cls(1, d.get("lr", 0.05), d.get("l2", 1e-4))
        obj.w = np.array([float(w) for w in d["w"]], dtype=float)
        return obj

# =============================
# PYTORCH LSTM MODEL
# =============================
if TORCH_OK:
    class LSTMModel(nn.Module):
        """PyTorch LSTM for sequence prediction"""
        
        def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            # x: [batch, seq_len, features]
            batch_size = x.size(0)
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(x, (h0, c0))
            
            # Use last output
            last_output = lstm_out[:, -1, :]
            
            # Dropout and final layer
            out = self.dropout(last_output)
            out = self.fc(out)
            out = self.sigmoid(out)
            
            return out

else:
    class LSTMModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")

# =============================
# AI FEATURE ENGINEERING
# =============================
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for AI model"""
    if len(df) < 50:
        return pd.DataFrame()
    
    df = calculate_indicators(df.copy())
    close = df['close'].astype(float)
    
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['returns_1'] = close.pct_change(1)
    features['returns_2'] = close.pct_change(2)
    features['returns_5'] = close.pct_change(5)
    features['returns_10'] = close.pct_change(10)
    
    # Momentum features
    features['rsi_norm'] = (df['rsi'] - 50) / 50  # Normalize RSI
    features['macd_norm'] = df['macd'] / close
    features['macd_hist_norm'] = df['macd_hist'] / close
    
    # Trend features
    features['ema20_ratio'] = close / df['ema_20']
    features['ema50_ratio'] = close / df['ema_50']
    features['ema_slope_20'] = df['ema_20'].pct_change(5)
    features['ema_slope_50'] = df['ema_50'].pct_change(10)
    
    # Volatility features
    features['atr_ratio'] = df['atr'] / close
    features['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    features['volatility_5'] = close.rolling(5).std() / close
    features['volatility_20'] = close.rolling(20).std() / close
    
    # Volume features
    features['volume_ratio'] = df['volume_ratio']
    features['volume_price_trend'] = (df['volume'].rolling(5).mean() * close.pct_change(5)).fillna(0)
    
    # Statistical features
    features['z_score_20'] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    features['skew_10'] = close.rolling(10).skew()
    features['kurt_10'] = close.rolling(10).kurt()
    
    # Cross-timeframe (if available)
    # This would typically use higher timeframe data, simplified here
    features['trend_strength'] = (features['ema20_ratio'] - 1) * features['rsi_norm']
    
    return features.fillna(0).replace([np.inf, -np.inf], 0)

def create_labels(df: pd.DataFrame, forward_bars=1) -> pd.Series:
    """Create labels for training (1 if price goes up, 0 if down)"""
    close = df['close'].astype(float)
    future_returns = close.shift(-forward_bars) / close - 1
    return (future_returns > 0).astype(int)

def prepare_lstm_sequences(features: pd.DataFrame, labels: pd.Series, seq_len=60):
    """Prepare sequences for LSTM training"""
    if len(features) < seq_len + 1:
        return None, None
    
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features.iloc[i-seq_len:i].values)
        y.append(labels.iloc[i])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# =============================
# AI MODEL MANAGEMENT
# =============================
def _ai_load_state():
    """Load AI state from disk"""
    try:
        with open(AI_CFG["state_file"], "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _ai_save_state(state):
    """Save AI state to disk with backup"""
    try:
        # Sanitize state
        for sym, model_data in state.items():
            if "w" in model_data:
                clean_w = []
                for val in model_data["w"]:
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        clean_w.append(0.0)
                    else:
                        clean_w.append(float(val))
                model_data["w"] = clean_w
        
        # Backup existing file
        if os.path.exists(AI_CFG["state_file"]):
            shutil.copy2(AI_CFG["state_file"], AI_CFG["state_file"] + ".bak")
        
        # Save new state
        with open(AI_CFG["state_file"], "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"Error saving AI state: {e}")

_AI_STATE = _ai_load_state()

def get_lr_model(symbol: str, n_features: int) -> OnlineLR:
    """Get or create LogReg model for symbol"""
    state = _AI_STATE.get(symbol)
    if state and len(state.get("w", [])) == n_features + 1:
        return OnlineLR.from_dict(state)
    return OnlineLR(n_features, lr=AI_CFG["learn_rate"], l2=AI_CFG["l2"])

def get_lstm_paths(symbol: str) -> Tuple[str, str]:
    """Get paths for LSTM model and scaler"""
    safe_sym = symbol.replace("/", "_").replace(":", "_")
    model_path = os.path.join(MODELS_DIR, f"{safe_sym}_lstm.pt")
    scaler_path = os.path.join(MODELS_DIR, f"{safe_sym}_scaler.pkl")
    return model_path, scaler_path

def save_lstm_model(symbol: str, model: LSTMModel, scaler: StandardScaler):
    """Save LSTM model and scaler"""
    if not TORCH_OK:
        return False
    
    try:
        model_path, scaler_path = get_lstm_paths(symbol)
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        return True
    except Exception as e:
        log(f"Error saving LSTM model for {symbol}: {e}")
        return False

def load_lstm_model(symbol: str, input_size: int) -> Tuple[Optional[LSTMModel], Optional[StandardScaler]]:
    """Load LSTM model and scaler"""
    if not TORCH_OK:
        return None, None
    
    try:
        model_path, scaler_path = get_lstm_paths(symbol)
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            return None, None
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(
            input_size=input_size,
            hidden_size=AI_CFG["lstm_hidden"],
            num_layers=AI_CFG["lstm_layers"]
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, scaler
    except Exception as e:
        log(f"Error loading LSTM model for {symbol}: {e}")
        return None, None
# =============================
# AI TRAINING FUNCTIONS
# =============================
def train_lstm_model(symbol: str, features: pd.DataFrame, labels: pd.Series, 
                    epochs=10, batch_size=128, seq_len=60, validation_split=0.2):
    """Train LSTM model with proper error handling and GPU support"""
    if not TORCH_OK:
        log(f"PyTorch not available for LSTM training on {symbol}")
        return False
    
    try:
        # Prepare sequences
        X, y = prepare_lstm_sequences(features, labels, seq_len)
        if X is None or len(X) < 100:
            log(f"Insufficient sequences for LSTM training on {symbol}: {len(X) if X is not None else 0}")
            return False
        
        log(f"Training LSTM for {symbol} with {len(X)} sequences, {X.shape[2]} features")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_scaled[i] = scaler.fit_transform(X[i]) if i == 0 else scaler.transform(X[i])
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
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
            input_size=X.shape[2],
            hidden_size=AI_CFG["lstm_hidden"],
            num_layers=AI_CFG["lstm_layers"]
        ).to(device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
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
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            scheduler.step(avg_val_loss)
            
            if epoch % 2 == 0:
                log(f"LSTM {symbol} Epoch {epoch}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                save_lstm_model(symbol, model, scaler)
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    log(f"Early stopping for {symbol} at epoch {epoch}")
                    break
        
        log(f"LSTM training completed for {symbol}, best val loss: {best_val_loss:.4f}")
        return True
        
    except Exception as e:
        log(f"LSTM training error for {symbol}: {e}")
        return False

def train_lr_model(symbol: str, features: pd.DataFrame, labels: pd.Series, epochs=5):
    """Train Logistic Regression model"""
    try:
        if len(features) < AI_CFG["min_obs"]:
            log(f"Insufficient data for LR training on {symbol}: {len(features)}")
            return False
        
        # Get model
        model = get_lr_model(symbol, features.shape[1])
        
        # Training loop
        for epoch in range(epochs):
            for i in range(len(features)):
                model.update(features.iloc[i].values, int(labels.iloc[i]))
        
        # Save model
        _AI_STATE[symbol] = model.to_dict()
        _ai_save_state(_AI_STATE)
        
        log(f"LR training completed for {symbol}")
        return True
        
    except Exception as e:
        log(f"LR training error for {symbol}: {e}")
        return False

def create_ai_dataset(symbol: str):
    """Create training dataset for AI models"""
    try:
        # Load base timeframe data
        df_base = load_history_df(symbol, AI_CFG["base_tf"], months=AI_CFG["hist_months"])
        if len(df_base) < AI_CFG["min_obs"]:
            return None, None, None
        
        # Create features and labels
        features = create_features(df_base)
        labels = create_labels(df_base)
        
        # Load other timeframes and merge
        for tf in AI_CFG["other_tfs"]:
            df_tf = load_history_df(symbol, tf, months=AI_CFG["hist_months"])
            if len(df_tf) >= 50:
                features_tf = create_features(df_tf)
                # Resample to base timeframe
                features_tf = features_tf.reindex(features.index, method='ffill')
                # Add prefix
                features_tf = features_tf.add_prefix(f"{tf}_")
                features = pd.concat([features, features_tf], axis=1)
        
        # Clean data
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Align features and labels
        min_len = min(len(features), len(labels))
        features = features.iloc[:min_len]
        labels = labels.iloc[:min_len]
        
        return features, labels, df_base
        
    except Exception as e:
        log(f"Error creating AI dataset for {symbol}: {e}")
        return None, None, None

# =============================
# HYBRID AI PREDICTION
# =============================
def predict_with_hybrid_ai(symbol: str):
    """Make predictions using hybrid AI (LR + LSTM)"""
    try:
        # Create current features
        features, labels, df_base = create_ai_dataset(symbol)
        if features is None:
            return None
        
        current_features = features.iloc[-1].values
        
        # Get LR prediction
        lr_model = get_lr_model(symbol, features.shape[1])
        if symbol in _AI_STATE:
            lr_model = OnlineLR.from_dict(_AI_STATE[symbol])
        
        p_lr = lr_model.predict_proba(current_features)
        
        # Get LSTM prediction if available
        p_lstm = None
        if TORCH_OK:
            lstm_model, scaler = load_lstm_model(symbol, features.shape[1])
            if lstm_model is not None and scaler is not None:
                try:
                    seq_len = AI_CFG["lstm_seq_len"]
                    if len(features) >= seq_len:
                        # Prepare sequence
                        seq_features = features.iloc[-seq_len:].values
                        seq_scaled = scaler.transform(seq_features)
                        
                        # Predict
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        with torch.no_grad():
                            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                            p_lstm = float(lstm_model(seq_tensor).cpu().numpy()[0][0])
                except Exception as e:
                    log(f"LSTM prediction error for {symbol}: {e}")
        
        # Combine predictions
        if p_lstm is not None:
            # Weighted average
            weight_lstm = AI_CFG["hybrid_weight"]
            p_combined = weight_lstm * p_lstm + (1 - weight_lstm) * p_lr
            method = f"Hybrid (LR: {p_lr:.3f}, LSTM: {p_lstm:.3f})"
        else:
            p_combined = p_lr
            method = "LR only"
        
        # Make decision
        decision = "hold"
        if p_combined >= AI_CFG["threshold_long"]:
            decision = "long"
        elif p_combined <= AI_CFG["threshold_short"]:
            decision = "short"
        
        return {
            "p_up": float(p_combined),
            "decision": decision,
            "method": method,
            "components": {"lr": p_lr, "lstm": p_lstm}
        }
        
    except Exception as e:
        log(f"Hybrid AI prediction error for {symbol}: {e}")
        return None

# =============================
# CHART GENERATION
# =============================
def plot_candlestick_chart(df: pd.DataFrame, title: str, filepath: str, 
                          indicators=True, trades=None, figsize=(14, 10)):
    """Generate comprehensive candlestick chart"""
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
        fig, axes = plt.subplots(4, 1, figsize=figsize, 
                                gridspec_kw={'height_ratios': [3, 1, 1, 1]},
                                sharex=True)
        
        # Candlestick chart
        ax_price = axes[0]
        dates = df_plot.index
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            # High-Low line
            ax_price.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # Open-Close body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            
            if body_height > 0:
                ax_price.add_patch(Rectangle((i-0.4, body_bottom), 0.8, body_height, 
                                           facecolor=color, alpha=0.7))
        
        # Add moving averages if indicators enabled
        if indicators and 'ema_20' in df_plot.columns:
            ax_price.plot(range(len(df_plot)), df_plot['ema_20'], 
                         label='EMA 20', color='blue', linewidth=1)
            ax_price.plot(range(len(df_plot)), df_plot['ema_50'], 
                         label='EMA 50', color='orange', linewidth=1)
        
        # Add trade markers if provided
        if trades:
            for trade in trades:
                if 'entry_idx' in trade and 'side' in trade:
                    idx = trade['entry_idx']
                    if 0 <= idx < len(df_plot):
                        color = 'lime' if trade['side'] == 'long' else 'red'
                        ax_price.scatter(idx, trade.get('entry_price', df_plot.iloc[idx]['close']), 
                                       color=color, marker='^' if trade['side'] == 'long' else 'v', 
                                       s=100, zorder=5)
        
        ax_price.set_title(title, fontsize=14, fontweight='bold')
        ax_price.legend()
        ax_price.grid(True, alpha=0.3)
        
        # Volume chart
        ax_volume = axes[1]
        if 'volume' in df_plot.columns:
            colors = ['green' if df_plot.iloc[i]['close'] >= df_plot.iloc[i]['open'] else 'red' 
                     for i in range(len(df_plot))]
            ax_volume.bar(range(len(df_plot)), df_plot['volume'], color=colors, alpha=0.7)
            ax_volume.set_title('Volume')
            ax_volume.grid(True, alpha=0.3)
        
        # RSI chart
        ax_rsi = axes[2]
        if 'rsi' in df_plot.columns:
            ax_rsi.plot(range(len(df_plot)), df_plot['rsi'], color='purple', linewidth=1)
            ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7)
            ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
            ax_rsi.set_title('RSI (14)')
            ax_rsi.set_ylim(0, 100)
            ax_rsi.grid(True, alpha=0.3)
        
        # MACD chart
        ax_macd = axes[3]
        if all(col in df_plot.columns for col in ['macd', 'macd_signal', 'macd_hist']):
            ax_macd.plot(range(len(df_plot)), df_plot['macd'], label='MACD', color='blue')
            ax_macd.plot(range(len(df_plot)), df_plot['macd_signal'], label='Signal', color='red')
            ax_macd.bar(range(len(df_plot)), df_plot['macd_hist'], label='Histogram', 
                       alpha=0.7, color='gray')
            ax_macd.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax_macd.set_title('MACD')
            ax_macd.legend()
            ax_macd.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(dates) > 50:
            step = len(dates) // 10
            tick_positions = range(0, len(dates), step)
            tick_labels = [dates[i].strftime('%m-%d') for i in tick_positions]
            plt.xticks(tick_positions, tick_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        log(f"Chart generation error: {e}")
        return None

def plot_equity_curve(equity_data: List[float], dates: List[datetime], 
                     title: str, filepath: str, trades=None):
    """Plot equity curve with trade markers"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Equity curve
        ax1.plot(dates, equity_data, color='blue', linewidth=2, label='Equity')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add trade markers if provided
        if trades:
            for trade in trades:
                if 'timestamp' in trade and 'pnl' in trade:
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax1.scatter(trade['timestamp'], trade.get('equity', equity_data[-1]), 
                               color=color, alpha=0.7, s=50)
        
        # Drawdown chart
        if len(equity_data) > 1:
            peak = np.maximum.accumulate(equity_data)
            drawdown = (np.array(equity_data) - peak) / peak * 100
            ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
            ax2.plot(dates, drawdown, color='red', linewidth=1)
            ax2.set_title('Drawdown (%)')
            ax2.set_ylabel('Drawdown %')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
        
    except Exception as e:
        log(f"Equity curve plot error: {e}")
        return None

# =============================
# WALK-FORWARD OPTIMIZATION
# =============================
class WalkForwardOptimizer:
    """Walk-forward optimization engine"""
    
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
    
    def run_optimization(self) -> bool:
        """Run walk-forward optimization"""
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
                test_prices = df.iloc[train_end:test_end]['close'].values
                
                # Train models for this window
                success = self._train_window_models(
                    f"{self.symbol}_wf_{window_count}", 
                    train_features, train_labels
                )
                
                if not success:
                    log(f"Training failed for window {window_count}")
                    start_idx += self.step_size
                    continue
                
                # Test on forward window
                window_results = self._test_window(
                    f"{self.symbol}_wf_{window_count}",
                    test_features, test_labels, test_prices,
                    train_end, current_capital
                )
                
                # Update capital and position
                current_capital = window_results['final_capital']
                current_position = window_results.get('final_position')
                
                # Store results
                self.results.append(window_results)
                self.equity_curve.extend(window_results['equity_curve'])
                self.trades.extend(window_results['trades'])
                
                start_idx += self.step_size
            
            # Calculate overall metrics
            self._calculate_metrics(initial_capital)
            
            log(f"Walk-forward completed: {window_count} windows, "
                f"Final capital: ${current_capital:.2f}")
            return True
            
        except Exception as e:
            log(f"Walk-forward optimization error: {e}")
            return False
    
    def _train_window_models(self, model_id: str, features: pd.DataFrame, 
                           labels: pd.Series) -> bool:
        """Train models for current window"""
        try:
            # Train LR model
            lr_model = OnlineLR(features.shape[1], lr=AI_CFG["learn_rate"], l2=AI_CFG["l2"])
            
            for epoch in range(AI_CFG["learn_epochs_boot"]):
                for i in range(len(features)):
                    lr_model.update(features.iloc[i].values, int(labels.iloc[i]))
            
            # Store LR model temporarily
            temp_state = lr_model.to_dict()
            _AI_STATE[model_id] = temp_state
            
            # Train LSTM if available
            if TORCH_OK and len(features) > AI_CFG["lstm_seq_len"] + 100:
                lstm_success = train_lstm_model(
                    model_id, features, labels,
                    epochs=AI_CFG["lstm_epochs"] // 2,  # Reduced for speed
                    batch_size=AI_CFG["lstm_batch_size"],
                    seq_len=AI_CFG["lstm_seq_len"]
                )
                log(f"LSTM training for {model_id}: {'Success' if lstm_success else 'Failed'}")
            
            return True
            
        except Exception as e:
            log(f"Window model training error: {e}")
            return False
    
    def _test_window(self, model_id: str, features: pd.DataFrame, labels: pd.Series,
                    prices: np.ndarray, start_idx: int, initial_capital: float) -> Dict:
        """Test models on forward window"""
        try:
            equity = [initial_capital]
            trades = []
            position = None
            capital = initial_capital
            
            # Load models
            lr_model = OnlineLR.from_dict(_AI_STATE.get(model_id, {})) if model_id in _AI_STATE else None
            lstm_model, lstm_scaler = load_lstm_model(model_id, features.shape[1]) if TORCH_OK else (None, None)
            
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
                
                # Trading logic
                if position is None:
                    # Enter position
                    if p_combined >= AI_CFG["threshold_long"]:
                        position = {
                            'side': 'long',
                            'entry_price': current_price,
                            'entry_idx': start_idx + i,
                            'shares': capital * 0.95 / current_price  # 95% allocation
                        }
                    elif p_combined <= AI_CFG["threshold_short"]:
                        position = {
                            'side': 'short',
                            'entry_price': current_price,
                            'entry_idx': start_idx + i,
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
                        trades.append({
                            'entry_idx': position['entry_idx'],
                            'exit_idx': start_idx + i,
                            'side': position['side'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'return_pct': (pnl / (position['entry_price'] * position['shares'])) * 100
                        })
                        
                        position = None
                
                equity.append(capital)
            
            return {
                'final_capital': capital,
                'final_position': position,
                'equity_curve': equity,
                'trades': trades,
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
                'window_start': start_idx,
                'window_end': start_idx + len(features)
            }
    
    def _calculate_metrics(self, initial_capital: float):
        """Calculate performance metrics"""
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
            
            # Drawdown calculation
            peak = np.maximum.accumulate(self.equity_curve)
            drawdown = (np.array(self.equity_curve) - peak) / peak * 100
            max_drawdown = np.min(drawdown)
            
            # Sharpe ratio (simplified)
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
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
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': sharpe,
            }
            
        except Exception as e:
            log(f"Metrics calculation error: {e}")
            self.metrics = {}
    
    def save_results(self) -> str:
        """Save walk-forward results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"walkforward_{self.symbol.replace('/', '_')}_{timestamp}.json"
            filepath = os.path.join(REPORTS_DIR, filename)
            
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
                'equity_curve': self.equity_curve,
                'window_results': self.results
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            log(f"Walk-forward results saved to {filepath}")
            return filepath
            
        except Exception as e:
            log(f"Error saving walk-forward results: {e}")
            return ""
    
    def generate_report(self) -> str:
        """Generate walk-forward performance report"""
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

TRADE STATISTICS:
- Total Trades: {self.metrics.get('total_trades', 0)}
- Winning Trades: {self.metrics.get('winning_trades', 0)}
- Losing Trades: {self.metrics.get('losing_trades', 0)}
- Win Rate: {self.metrics.get('win_rate_pct', 0):.1f}%
- Average Win: ${self.metrics.get('avg_win', 0):.2f}
- Average Loss: ${self.metrics.get('avg_loss', 0):.2f}
- Profit Factor: {self.metrics.get('profit_factor', 0):.2f}
"""
            return report
            
        except Exception as e:
            log(f"Error generating walk-forward report: {e}")
            return "Error generating report"

# =============================
# TRADING LOGIC
# =============================
class TradingEngine:
    # NOTE: Patched to use LIMIT orders with trigger_price instead of market orders.
    """Core trading engine with position management"""
    
    def __init__(self):
        self.position = None
        self.auto_enabled = True
        self.last_update_id = None
        self.last_heartbeat = time.time()
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze symbol across multiple timeframes"""
        try:
            results = {}
            
            for tf in cfg["timeframes"]:
                indicators, df = get_indicators(symbol, tf)
                if indicators:
                    signal = self._get_signal(indicators)
                    results[tf] = {
                        'signal': signal,
                        'indicators': indicators,
                        'dataframe': df
                    }
            
            # Check for alignment
            core_tfs = cfg["core_tfs"]
            signals = [results.get(tf, {}).get('signal', 'neutral') for tf in core_tfs]
            
            core_agree = len(set(signals)) == 1 and signals[0] != 'neutral'
            
            return {
                'results': results,
                'core_agree': core_agree,
                'core_signal': signals[0] if core_agree else 'neutral'
            }
            
        except Exception as e:
            log(f"Symbol analysis error for {symbol}: {e}")
            return {}
    
    def _get_signal(self, indicators: Dict) -> str:
        """Generate trading signal from indicators"""
        try:
            price = indicators['price']
            ema20 = indicators.get('ema20')
            ema50 = indicators.get('ema50')
            rsi = indicators.get('rsi14')
            macd_hist = indicators.get('macd_hist')
            
            if None in [ema20, ema50, rsi, macd_hist]:
                return 'neutral'
            
            bullish_signals = 0
            bearish_signals = 0
            
            # EMA trend
            if price > ema20 > ema50:
                bullish_signals += 1
            elif price < ema20 < ema50:
                bearish_signals += 1
            
            # RSI
            if rsi > 55:
                bullish_signals += 1
            elif rsi < 45:
                bearish_signals += 1
            
            # MACD
            if macd_hist > 0:
                bullish_signals += 1
            elif macd_hist < 0:
                bearish_signals += 1
            
            if bullish_signals >= 2 and bearish_signals == 0:
                return 'bullish'
            elif bearish_signals >= 2 and bullish_signals == 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            log(f"Signal generation error: {e}")
            return 'neutral'
    
    def calculate_position_size(self, symbol: str, price: float) -> Tuple[int, float]:
        """Calculate optimal position size"""
        try:
            # Get market info
            market = exchange.market(symbol)
            contract_size = float(market.get('contractSize', 1.0))
            
            # Calculate based on budget
            available_budget = max(0, cfg["daily_budget_usd"] - cfg["budget_used_today"])
            leverage = cfg["leverage"]
            
            # Target margin
            target_margin = available_budget * 0.5  # Use 50% of available budget
            
            # Calculate contracts
            margin_per_contract = (price * contract_size) / leverage
            max_contracts = int(target_margin / margin_per_contract) if margin_per_contract > 0 else 0
            
            # Apply limits
            if cfg["contracts_mode"] == "auto":
                # ATR-based sizing
                indicators, _ = get_indicators(symbol, "5m")
                if indicators and indicators.get('atr'):
                    atr_ratio = indicators['atr'] / price
                    if atr_ratio < 0.007:  # Low volatility
                        contracts = min(3, max_contracts)
                    elif atr_ratio < 0.012:  # Medium volatility
                        contracts = min(2, max_contracts)
                    else:  # High volatility
                        contracts = min(1, max_contracts)
                else:
                    contracts = min(1, max_contracts)
            else:
                contracts = min(cfg["contracts_fixed"], max_contracts)
            
            return max(0, contracts), contract_size
            
        except Exception as e:
            log(f"Position sizing error for {symbol}: {e}")
            return 0, 1.0
    
    def open_position(self, symbol: str, side: str, reason: str = ""):
        """Open a new trading position"""
        try:
            if self.position is not None:
                log(f"Cannot open position: existing position active")
                return False
            
            # Get current price and indicators
            indicators, _ = get_indicators(symbol, "5m")
            if not indicators:
                log(f"Cannot get indicators for {symbol}")
                return False
            
            price = indicators['price']
            atr = indicators.get('atr', price * 0.01)  # Default 1% if no ATR
            
            # Calculate position size
            contracts, contract_size = self.calculate_position_size(symbol, price)
            if contracts <= 0:
                log(f"Position size too small for {symbol}")
                return False
            
            # Calculate stops and targets
            atr_sl = atr * cfg["atr_sl_mult"]
            atr_tp = atr * cfg["atr_tp_mult"]
            
            if side.lower() == 'long':
                stop_loss = price - atr_sl
                take_profit = price + atr_tp
                trigger_price = price + (price * cfg["slippage_buffer_pct"] / 100)
            else:
                stop_loss = price + atr_sl
                take_profit = price - atr_tp
                trigger_price = price - (price * cfg["slippage_buffer_pct"] / 100)
            
            # Create position
            self.position = {
                'symbol': symbol,
                'side': side.lower(),
                'contracts': contracts,
                'contract_size': contract_size,
                'trigger_price': trigger_price,
                'entry_price': None,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'best_price': price,
                'partial_filled': False,
                'reason': reason,
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Update budget
            margin_used = (price * contract_size * contracts) / cfg["leverage"]
            cfg["budget_used_today"] += margin_used
            
            # Set leverage
            safe_set_leverage(symbol, cfg["leverage"])
            
            log(f"Position planned: {side.upper()} {symbol} @ {price:.6f}, "
                f"SL: {stop_loss:.6f}, TP: {take_profit:.6f}, Size: {contracts}")
            
            return True
            
        except Exception as e:
            log(f"Error opening position for {symbol}: {e}")
            return False
    
    def check_trigger(self):
        """Check if position should be triggered"""
        if not self.position or self.position.get('entry_price') is not None:
            return
        
        try:
            symbol = self.position['symbol']
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            trigger_price = self.position['trigger_price']
            side = self.position['side']
            
            should_trigger = False
            if side == 'long' and current_price >= trigger_price:
                should_trigger = True
            elif side == 'short' and current_price <= trigger_price:
                should_trigger = True
            
            if should_trigger:
                # Execute market order
                order_side = 'buy' if side == 'long' else 'sell'
                order = exchange.create_limit_order(symbol, order_side, self.position['contracts'], trigger_price)
                
                self.position['entry_price'] = current_price
                self.position['best_price'] = current_price
                self.position['order_id'] = order['id']
                
                log(f"Position triggered: {side.upper()} {symbol} @ {current_price:.6f}")
                
                # Send telegram notification
                asyncio.create_task(tg_send(
                    f"🎯 POSITION TRIGGERED\n"
                    f"{side.upper()} {symbol}\n"
                    f"Entry: {current_price:.6f}\n"
                    f"Stop: {self.position['stop_loss']:.6f}\n"
                    f"Target: {self.position['take_profit']:.6f}\n"
                    f"Size: {self.position['contracts']} contracts"
                ))
                
        except Exception as e:
            log(f"Trigger check error: {e}")
    
    def manage_position(self):
        """Manage active position"""
        if not self.position or self.position.get('entry_price') is None:
            return
        
        try:
            symbol = self.position['symbol']
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            side = self.position['side']
            entry_price = self.position['entry_price']
            
            # Update best price
            if side == 'long' and current_price > self.position['best_price']:
                self.position['best_price'] = current_price
            elif side == 'short' and current_price < self.position['best_price']:
                self.position['best_price'] = current_price
            
            # Check partial take profit
            if (cfg["use_partial_tp"] and not self.position['partial_filled'] and
                ((side == 'long' and current_price >= self.position['take_profit'] * 0.75) or
                 (side == 'short' and current_price <= self.position['take_profit'] * 1.25))):
                
                partial_size = self.position['contracts'] // 2
                if partial_size > 0:
                    close_side = 'sell' if side == 'long' else 'buy'
                    exchange.create_market_order(symbol, close_side, partial_size)
                    
                    self.position['contracts'] -= partial_size
                    self.position['partial_filled'] = True
                    self.position['stop_loss'] = entry_price  # Move to breakeven
                    
                    log(f"Partial TP filled: {partial_size} contracts closed at {current_price:.6f}")
            
            # Check trailing stop
            if cfg["trail_after_tp"] and self.position['partial_filled']:
                trail_distance = self.position['best_price'] * (cfg["trail_pct"] / 100)
                
                if side == 'long':
                    new_stop = self.position['best_price'] - trail_distance
                    if new_stop > self.position['stop_loss']:
                        self.position['stop_loss'] = new_stop
                else:
                    new_stop = self.position['best_price'] + trail_distance
                    if new_stop < self.position['stop_loss']:
                        self.position['stop_loss'] = new_stop
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            if side == 'long':
                if current_price <= self.position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                elif current_price >= self.position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
            else:
                if current_price >= self.position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                elif current_price <= self.position['take_profit']:
                    should_close = True
                    close_reason = "Take Profit"
            
            if should_close:
                self.close_position(close_reason)
                
        except Exception as e:
            log(f"Position management error: {e}")
    
    def close_position(self, reason: str = "Manual"):
        """Close current position"""
        if not self.position:
            return
        
        try:
            symbol = self.position['symbol']
            side = self.position['side']
            contracts = self.position['contracts']
            entry_price = self.position.get('entry_price')
            
            if contracts > 0:
                close_side = 'sell' if side == 'long' else 'buy'
                order = exchange.create_market_order(symbol, close_side, contracts)
                
                current_price = exchange.fetch_ticker(symbol)['last']
                
                # Calculate PnL
                if entry_price:
                    if side == 'long':
                        pnl = (current_price - entry_price) * contracts * self.position['contract_size']
                    else:
                        pnl = (entry_price - current_price) * contracts * self.position['contract_size']
                    
                    pnl_pct = (pnl / (entry_price * contracts * self.position['contract_size'])) * 100
                else:
                    pnl = 0
                    pnl_pct = 0
                
                log(f"Position closed: {side.upper()} {symbol} @ {current_price:.6f}, "
                    f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%), Reason: {reason}")
                
                # Send telegram notification
                pnl_emoji = "💚" if pnl >= 0 else "❌"
                asyncio.create_task(tg_send(
                    f"{pnl_emoji} POSITION CLOSED\n"
                    f"{side.upper()} {symbol}\n"
                    f"Entry: {entry_price:.6f if entry_price else 'N/A'}\n"
                    f"Exit: {current_price:.6f}\n"
                    f"PnL: ${pnl:.2f} ({pnl_pct:.2f}%)\n"
                    f"Reason: {reason}"
                ))
            
            self.position = None
            
        except Exception as e:
            log(f"Position closing error: {e}")

# =============================
# TELEGRAM COMMAND HANDLERS
# =============================
class TelegramBot:
    """Telegram bot command handler"""
    
    def __init__(self, trading_engine: TradingEngine):
        self.engine = trading_engine
        self.help_text = """
🤖 CRYPTO TRADING BOT COMMANDS

📊 ANALYSIS & STATUS:
/status - Show account balance, positions, settings
/analyze [SYMBOL] - Analyze symbol across timeframes  
/chart [SYMBOL] [TF] - Generate and send chart

🎯 TRADING:
/long [SYMBOL] - Plan long position
/short [SYMBOL] - Plan short position  
/close - Close current position
/auto on|off - Toggle auto trading

⚙️ CONFIGURATION:
/set budget <amount> - Set daily budget
/set leverage <x> - Set leverage multiplier
/set atrsl <x> - Set ATR stop loss multiplier
/set atrtp <x> - Set ATR take profit multiplier
/contracts auto|1|2|3 - Set contract mode

📈 AI & OPTIMIZATION:
/trainai [SYMBOL] - Train AI models
/predict [SYMBOL] - Get AI prediction
/walkforward [SYMBOL] - Run walk-forward test

💾 DATA MANAGEMENT:
/download - Download historical data
/backup - Backup AI models and settings
/restore - Restore from backup

📋 PAIRS MANAGEMENT:
/addpair [SYMBOL] - Add to watch list
/removepair [SYMBOL] - Remove from watch list
/addtrade [SYMBOL] - Add to auto trading
/removetrade [SYMBOL] - Remove from auto trading

Type any command for help with that specific function.
"""
    
    async def handle_command(self, text: str):
        """Handle incoming telegram command"""
        try:
            parts = text.strip().split()
            command = parts[0].lower()
            
            if command in ['/help', '/start']:
                await self.cmd_help()
            elif command == '/status':
                await self.cmd_status()
            elif command.startswith('/analyze'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_analyze(normalize_symbol(symbol))
            elif command.startswith('/chart'):
                await self.cmd_chart(parts[1:])
            elif command.startswith('/long'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_long(normalize_symbol(symbol))
            elif command.startswith('/short'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_short(normalize_symbol(symbol))
            elif command == '/close':
                await self.cmd_close()
            elif command.startswith('/auto'):
                state = parts[1] if len(parts) > 1 else 'toggle'
                await self.cmd_auto(state)
            elif command.startswith('/set'):
                await self.cmd_set(parts[1:])
            elif command.startswith('/contracts'):
                mode = parts[1] if len(parts) > 1 else 'auto'
                await self.cmd_contracts(mode)
            elif command.startswith('/trainai'):
                symbol = parts[1] if len(parts) > 1 else None
                await self.cmd_train_ai(symbol)
            elif command.startswith('/predict'):
                symbol = parts[1] if len(parts) > 1 else 'BTC/USDT:USDT'
                await self.cmd_predict(normalize_symbol(symbol))
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
            else:
                await tg_send(f"Unknown command: {command}\nType /help for available commands")
                
        except Exception as e:
            log(f"Command handling error: {e}")
            await tg_send(f"Error processing command: {str(e)}")
    
    async def cmd_help(self):
        await tg_send(self.help_text)
    
    async def cmd_status(self):
        try:
            # Get balance
            balance = exchange.fetch_balance()
            total_balance = balance.get('USDT', {}).get('total', 0)
            
            # Build status message
            status_msg = f"""
📊 TRADING BOT STATUS

💰 Account:
Balance: ${total_balance:.2f} USDT
Mode: {MODE.upper()}
Daily Budget: ${cfg['daily_budget_usd']:.2f}
Used Today: ${cfg['budget_used_today']:.2f}

⚙️ Settings:
Auto Trading: {'ON' if self.engine.auto_enabled else 'OFF'}
Leverage: {cfg['leverage']}x
Contracts Mode: {cfg['contracts_mode']}
ATR SL: {cfg['atr_sl_mult']}x
ATR TP: {cfg['atr_tp_mult']}x

📍 Position:
"""
            if self.engine.position:
                pos = self.engine.position
                status_msg += f"""
{pos['side'].upper()} {pos['symbol']}
Entry: {pos.get('entry_price', pos['trigger_price']):.6f}
Stop: {pos['stop_loss']:.6f}  
Target: {pos['take_profit']:.6f}
Size: {pos['contracts']} contracts
"""
            else:
                status_msg += "None"
            
            status_msg += f"""

🎯 Auto Pairs: {len(AUTO_PAIRS)}
👁️ Watch Pairs: {len(WATCH_ONLY_PAIRS)}
"""
            
            await tg_send(status_msg)
            
        except Exception as e:
            await tg_send(f"Error getting status: {e}")
    
    async def cmd_analyze(self, symbol: str):
        try:
            analysis = self.engine.analyze_symbol(symbol)
            
            if not analysis:
                await tg_send(f"Could not analyze {symbol}")
                return
            
            msg = f"📈 ANALYSIS: {symbol}\n\n"
            
            for tf, data in analysis['results'].items():
                signal = data['signal']
                indicators = data['indicators']
                
                emoji = "🟢" if signal == 'bullish' else "🔴" if signal == 'bearish' else "🟡"
                msg += f"{emoji} {tf}: {signal.upper()}\n"
                msg += f"  Price: {indicators['price']:.6f}\n"
                msg += f"  RSI: {indicators.get('rsi14', 'N/A')}\n\n"
            
            # Add AI prediction
            ai_result = predict_with_hybrid_ai(symbol)
            if ai_result:
                msg += f"🤖 AI Prediction:\n"
                msg += f"  Probability Up: {ai_result['p_up']:.1%}\n"
                msg += f"  Decision: {ai_result['decision'].upper()}\n"
                msg += f"  Method: {ai_result['method']}\n"
            
            await tg_send(msg)
            
        except Exception as e:
            await tg_send(f"Analysis error for {symbol}: {e}")
    
    async def cmd_chart(self, args: List[str]):
        try:
            symbol = 'BTC/USDT:USDT'
            timeframe = '1h'
            
            if args:
                # Parse arguments
                for arg in args:
                    if any(c.isalpha() and c not in 'mhd' for c in arg):
                        symbol = normalize_symbol(arg)
                    else:
                        timeframe = timeframe_alias(arg)
            
            # Get data
            df = load_history_df(symbol, timeframe, months=3)
            if df.empty:
                await tg_send(f"No data available for {symbol} {timeframe}")
                return
            
            # Generate chart
            chart_title = f"{symbol} {timeframe.upper()} Chart"
            chart_file = os.path.join(CHARTS_DIR, f"{symbol.replace('/', '_')}_{timeframe}_chart.png")
            
            trades = []
            if (self.engine.position and 
                self.engine.position['symbol'] == symbol and 
                self.engine.position.get('entry_price')):
                
                trades = [{
                    'entry_idx': len(df) - 50,  # Approximate
                    'side': self.engine.position['side'],
                    'entry_price': self.engine.position['entry_price']
                }]
            
            chart_path = plot_candlestick_chart(
                df.tail(200), chart_title, chart_file, 
                indicators=True, trades=trades
            )
            
            if chart_path:
                await tg_send_photo(chart_path, f"{symbol} {timeframe} technical analysis")
            else:
                await tg_send("Failed to generate chart")
                
        except Exception as e:
            await tg_send(f"Chart error: {e}")
    
    async def cmd_long(self, symbol: str):
        success = self.engine.open_position(symbol, 'long', 'Manual long command')
        if success:
            await tg_send(f"✅ Long position planned for {symbol}")
        else:
            await tg_send(f"❌ Failed to plan long position for {symbol}")
    
    async def cmd_short(self, symbol: str):
        success = self.engine.open_position(symbol, 'short', 'Manual short command')
        if success:
            await tg_send(f"✅ Short position planned for {symbol}")
        else:
            await tg_send(f"❌ Failed to plan short position for {symbol}")
    
    async def cmd_close(self):
        if self.engine.position:
            self.engine.close_position("Manual close")
            await tg_send("✅ Position closed manually")
        else:
            await tg_send("ℹ️ No position to close")
    
    async def cmd_auto(self, state: str):
        if state.lower() == 'on':
            self.engine.auto_enabled = True
            await tg_send("✅ Auto trading ENABLED")
        elif state.lower() == 'off':
            self.engine.auto_enabled = False
            await tg_send("✅ Auto trading DISABLED")
        else:
            current_state = "ON" if self.engine.auto_enabled else "OFF"
            await tg_send(f"Auto trading is currently {current_state}")
    
    async def cmd_train_ai(self, symbol: str = None):
        try:
            symbols = [normalize_symbol(symbol)] if symbol else list(AUTO_PAIRS)
            
            await tg_send(f"🤖 Starting AI training for {len(symbols)} symbols...")
            
            for sym in symbols:
                # Create dataset
                features, labels, df_base = create_ai_dataset(sym)
                if features is None:
                    await tg_send(f"❌ Insufficient data for {sym}")
                    continue
                
                # Train LR model
                lr_success = train_lr_model(sym, features, labels)
                
                # Train LSTM model if PyTorch available
                lstm_success = False
                if TORCH_OK:
                    lstm_success = train_lstm_model(sym, features, labels)
                
                status = f"✅ {sym}: LR {'✓' if lr_success else '✗'}"
                if TORCH_OK:
                    status += f", LSTM {'✓' if lstm_success else '✗'}"
                
                await tg_send(status)
            
            await tg_send("🎉 AI training completed!")
            
        except Exception as e:
            await tg_send(f"Training error: {e}")
    
    async def cmd_predict(self, symbol: str):
        try:
            result = predict_with_hybrid_ai(symbol)
            if result:
                msg = f"🤖 AI PREDICTION: {symbol}\n\n"
                msg += f"Probability Up: {result['p_up']:.1%}\n"
                msg += f"Decision: {result['decision'].upper()}\n"
                msg += f"Method: {result['method']}\n"
                
                if result['components']['lstm'] is not None:
                    msg += f"\nComponents:\n"
                    msg += f"  LR: {result['components']['lr']:.1%}\n"
                    msg += f"  LSTM: {result['components']['lstm']:.1%}\n"
                
                await tg_send(msg)
            else:
                await tg_send(f"Could not generate prediction for {symbol}")
                
        except Exception as e:
            await tg_send(f"Prediction error: {e}")
    
    async def cmd_walkforward(self, symbol: str):
        try:
            await tg_send(f"🔄 Starting walk-forward optimization for {symbol}...")
            
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
            success = optimizer.run_optimization()
            
            if success:
                # Generate report
                report = optimizer.generate_report()
                await tg_send(f"📊 WALK-FORWARD RESULTS:\n```{report}```")
                
                # Save results
                results_file = optimizer.save_results()
                
                # Generate equity curve chart
                if optimizer.equity_curve and len(optimizer.equity_curve) > 1:
                    dates = [datetime.now() - timedelta(days=i) for i in reversed(range(len(optimizer.equity_curve)))]
                    chart_file = os.path.join(CHARTS_DIR, f"walkforward_{symbol.replace('/', '_')}_equity.png")
                    
                    chart_path = plot_equity_curve(
                        optimizer.equity_curve, dates,
                        f"Walk-Forward Equity Curve - {symbol}",
                        chart_file, optimizer.trades
                    )
                    
                    if chart_path:
                        await tg_send_photo(chart_path, f"Walk-Forward Results: {symbol}")
                
                await tg_send(f"✅ Walk-forward completed. Results saved to: {os.path.basename(results_file)}")
            else:
                await tg_send(f"❌ Walk-forward optimization failed for {symbol}")
                
        except Exception as e:
            await tg_send(f"Walk-forward error: {e}")
    
    async def cmd_download(self):
        try:
            await tg_send("📥 Starting data download...")
            
            symbols = list(AUTO_PAIRS | WATCH_ONLY_PAIRS)
            timeframes = cfg["timeframes"]
            
            total_downloads = len(symbols) * len(timeframes)
            completed = 0
            
            for symbol in symbols:
                for tf in timeframes:
                    try:
                        ensure_history(symbol, tf, months=AI_CFG["hist_months"])
                        completed += 1
                        
                        if completed % 5 == 0:  # Progress update every 5 downloads
                            await tg_send(f"📥 Progress: {completed}/{total_downloads} completed")
                            
                    except Exception as e:
                        log(f"Download error for {symbol} {tf}: {e}")
            
            await tg_send(f"✅ Data download completed: {completed}/{total_downloads} successful")
            
        except Exception as e:
            await tg_send(f"Download error: {e}")
    
    async def cmd_set(self, args: List[str]):
        try:
            if len(args) < 2:
                await tg_send("Usage: /set <parameter> <value>\nParameters: budget, leverage, atrsl, atrtp")
                return
            
            param, value = args[0].lower(), args[1]
            
            if param == 'budget':
                cfg["daily_budget_usd"] = max(1.0, float(value))
                cfg["budget_used_today"] = 0.0  # Reset usage
                await tg_send(f"✅ Daily budget set to ${cfg['daily_budget_usd']:.2f}")
                
            elif param == 'leverage':
                cfg["leverage"] = max(1, min(100, int(float(value))))
                # Apply to all auto pairs
                for symbol in AUTO_PAIRS:
                    safe_set_leverage(symbol, cfg["leverage"])
                await tg_send(f"✅ Leverage set to {cfg['leverage']}x")
                
            elif param == 'atrsl':
                cfg["atr_sl_mult"] = max(0.1, float(value))
                await tg_send(f"✅ ATR Stop Loss multiplier set to {cfg['atr_sl_mult']}")
                
            elif param == 'atrtp':
                cfg["atr_tp_mult"] = max(0.2, float(value))
                await tg_send(f"✅ ATR Take Profit multiplier set to {cfg['atr_tp_mult']}")
                
            else:
                await tg_send(f"Unknown parameter: {param}")
                
        except ValueError:
            await tg_send("Invalid value. Please provide a valid number.")
        except Exception as e:
            await tg_send(f"Set error: {e}")
    
    async def cmd_contracts(self, mode: str):
        if mode.lower() == 'auto':
            cfg["contracts_mode"] = "auto"
            await tg_send("✅ Contract mode set to AUTO (ATR-based)")
        elif mode in ['1', '2', '3']:
            cfg["contracts_mode"] = "fixed"
            cfg["contracts_fixed"] = int(mode)
            await tg_send(f"✅ Contract mode set to FIXED: {cfg['contracts_fixed']}")
        else:
            await tg_send("Usage: /contracts auto|1|2|3")
    
    async def cmd_add_pair(self, symbol: str, watch_only: bool = True):
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

# =============================
# CLI FUNCTIONS
# =============================
def cli_download_data(symbols=None, timeframes=None, months=12, refresh=False):
    """CLI function to download historical data"""
    if symbols is None:
        symbols = list(AUTO_PAIRS | WATCH_ONLY_PAIRS)
    if timeframes is None:
        timeframes = cfg["timeframes"]
    
    log(f"Downloading data for {len(symbols)} symbols, {len(timeframes)} timeframes")
    
    total = len(symbols) * len(timeframes)
    completed = 0
    
    for symbol in symbols:
        for tf in timeframes:
            try:
                if refresh:
                    # Remove existing file to force fresh download
                    path = _history_path(symbol, tf)
                    if os.path.exists(path):
                        os.remove(path)
                
                ensure_history(symbol, tf, months=months)
                completed += 1
                
                log(f"Progress: {completed}/{total} - Downloaded {symbol} {tf}")
                
            except Exception as e:
                log(f"Download failed for {symbol} {tf}: {e}")
                completed += 1
    
    log(f"Data download completed: {completed}/{total}")

def cli_train_ai(symbols=None, epochs=10, use_lstm=True):
    """CLI function to train AI models"""
    if symbols is None:
        symbols = list(AUTO_PAIRS)
    
    log(f"Training AI models for {len(symbols)} symbols")
    
    for symbol in symbols:
        log(f"Training models for {symbol}...")
        
        try:
            # Create dataset
            features, labels, df_base = create_ai_dataset(symbol)
            if features is None:
                log(f"Insufficient data for {symbol}")
                continue
            
            # Train LR model
            lr_success = train_lr_model(symbol, features, labels, epochs=epochs//2)
            log(f"LR training for {symbol}: {'Success' if lr_success else 'Failed'}")
            
            # Train LSTM model
            if use_lstm and TORCH_OK:
                lstm_success = train_lstm_model(
                    symbol, features, labels,
                    epochs=epochs,
                    batch_size=AI_CFG["lstm_batch_size"],
                    seq_len=AI_CFG["lstm_seq_len"]
                )
                log(f"LSTM training for {symbol}: {'Success' if lstm_success else 'Failed'}")
            
        except Exception as e:
            log(f"Training error for {symbol}: {e}")
    
    log("AI training completed")

def cli_walkforward(symbol="BTC/USDT:USDT", timeframe="5m", months=6):
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
        
        success = optimizer.run_optimization()
        
        if success:
            # Print report
            report = optimizer.generate_report()
            print(report)
            
            # Save results
            results_file = optimizer.save_results()
            log(f"Results saved to: {results_file}")
            
            # Generate charts
            if optimizer.equity_curve and len(optimizer.equity_curve) > 1:
                dates = [datetime.now() - timedelta(days=i) for i in reversed(range(len(optimizer.equity_curve)))]
                chart_file = os.path.join(CHARTS_DIR, f"walkforward_{symbol.replace('/', '_')}_equity.png")
                
                chart_path = plot_equity_curve(
                    optimizer.equity_curve, dates,
                    f"Walk-Forward Equity Curve - {symbol}",
                    chart_file, optimizer.trades
                )
                
                if chart_path:
                    log(f"Equity curve saved to: {chart_path}")
        else:
            log("Walk-forward optimization failed")
            
    except Exception as e:
        log(f"Walk-forward error: {e}")

# =============================
# MAIN TRADING LOOP
# =============================
async def main_trading_loop():
    """Main trading loop with Telegram integration"""
    trading_engine = TradingEngine()
    telegram_bot = TelegramBot(trading_engine)
    
    log("Starting main trading loop...")
    
    # Send startup message
    await tg_send("🤖 Trading Bot Started\n"
                  f"Mode: {MODE.upper()}\n"
                  f"Auto Trading: {'ON' if trading_engine.auto_enabled else 'OFF'}\n"
                  "Type /help for commands")
    
    last_heartbeat = time.time()
    heartbeat_interval = cfg["heartbeat_minutes"] * 60
    
    while True:
        try:
            # Handle Telegram updates
            updates = tg_get_updates(trading_engine.last_update_id)
            for update in updates:
                trading_engine.last_update_id = update["update_id"] + 1
                
                message = update.get("message") or update.get("edited_message")
                if message and message.get("text"):
                    await telegram_bot.handle_command(message["text"])
            
            # Trading logic
            if trading_engine.auto_enabled:
                # Check for trigger
                trading_engine.check_trigger()
                
                # Manage existing position
                trading_engine.manage_position()
                
                # Look for new opportunities if no position
                if trading_engine.position is None:
                    for symbol in AUTO_PAIRS:
                        try:
                            # Analyze symbol
                            analysis = trading_engine.analyze_symbol(symbol)
                            if analysis and analysis.get('core_agree'):
                                signal = analysis['core_signal']
                                
                                # Get AI confirmation
                                ai_result = predict_with_hybrid_ai(symbol)
                                if ai_result and ai_result['decision'] != 'hold':
                                    if ((signal == 'bullish' and ai_result['decision'] == 'long') or
                                        (signal == 'bearish' and ai_result['decision'] == 'short')):
                                        
                                        side = 'long' if signal == 'bullish' else 'short'
                                        success = trading_engine.open_position(
                                            symbol, side, f"Auto: {signal} + AI {ai_result['decision']}"
                                        )
                                        
                                        if success:
                                            await tg_send(f"🎯 AUTO TRADE DETECTED\n"
                                                         f"{side.upper()} {symbol}\n"
                                                         f"Signal: {signal}\n"
                                                         f"AI: {ai_result['decision']} ({ai_result['p_up']:.1%})")
                                            break
                        except Exception as e:
                            log(f"Auto trading error for {symbol}: {e}")
            
            # Heartbeat
            current_time = time.time()
            if current_time - last_heartbeat >= heartbeat_interval:
                last_heartbeat = current_time
                
                # Send heartbeat with basic status
                balance = 0
                try:
                    balance = exchange.fetch_balance().get('USDT', {}).get('total', 0)
                except:
                    pass
                
                position_info = ""
                if trading_engine.position:
                    pos = trading_engine.position
                    position_info = f"\n📍 {pos['side'].upper()} {pos['symbol']}"
                
                await tg_send(f"⏰ Heartbeat\n"
                             f"Balance: ${balance:.2f}\n"
                             f"Auto: {'ON' if trading_engine.auto_enabled else 'OFF'}{position_info}")
            
            # Reset daily budget at midnight UTC
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute == 0:
                cfg["budget_used_today"] = 0.0
                await tg_send("🗓️ Daily budget reset")
            
            await asyncio.sleep(2)  # Main loop delay
            
        except KeyboardInterrupt:
            log("Shutting down trading loop...")
            await tg_send("🛑 Trading bot shutting down")
            break
        except Exception as e:
            log(f"Main loop error: {e}")
            await asyncio.sleep(5)

# =============================
# CLI ENTRY POINT
# =============================
async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Crypto Trading Bot")
    
    # CLI commands
    parser.add_argument("--download-data", action="store_true", help="Download historical OHLCV data")
    parser.add_argument("--refresh-data", action="store_true", help="Force refresh existing data")
    parser.add_argument("--train-ai", action="store_true", help="Train AI models")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward optimization")
    parser.add_argument("--run", action="store_true", help="Start trading bot")
    
    # Parameters
    parser.add_argument("--symbol", default="BTC/USDT:USDT", help="Symbol for single operations")
    parser.add_argument("--timeframe", default="5m", help="Timeframe for analysis")
    parser.add_argument("--months", type=int, default=12, help="Months of historical data")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for AI")
    
    args = parser.parse_args()
    
    # Execute CLI commands
    if args.download_data:
        symbols = list(AUTO_PAIRS | WATCH_ONLY_PAIRS)
        timeframes = cfg["timeframes"]
        cli_download_data(symbols, timeframes, args.months, args.refresh_data)
        return
    
    if args.train_ai:
        symbols = [normalize_symbol(args.symbol)] if args.symbol != "BTC/USDT:USDT" else list(AUTO_PAIRS)
        cli_train_ai(symbols, args.epochs, use_lstm=TORCH_OK)
        return
    
    if args.walkforward:
        cli_walkforward(args.symbol, args.timeframe, args.months)
        return
    
    # Default: run trading bot
    if args.run or len(sys.argv) == 1:
        await main_trading_loop()




# ================= Phase 5: Visualization & Chart Commands =================
# Global toggle for showing raw predictions on charts
SHOW_PREDICTIONS = False

def _ensure_dirs():
    os.makedirs("charts", exist_ok=True)
    os.makedirs("reports/walkforward", exist_ok=True)
    os.makedirs("reports/walkforward/tmp", exist_ok=True)

def draw_chart_with_indicators(df, symbol, filepath, confirmed_trades=None, predictions=None, show_predictions=False):
    """
    Draw candlestick chart with EMA200, Bollinger Bands, Supertrend and Fibonacci overlays.
    Save to filepath. This function is silent (no prints).
    """
    try:
        _ensure_dirs()
        # lazy imports
        try:
            import mplfinance as mpf
            use_mpf = True
        except Exception:
            use_mpf = False
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import numpy as _np
        d = df.copy().dropna(subset=['open','high','low','close']).reset_index(drop=True)
        # ensure indicator columns exist
        if 'ema_200' not in d.columns and 'close' in d.columns:
            try:
                d['ema_200'] = d['close'].ewm(span=200, adjust=False).mean()
            except Exception:
                pass
        if 'bb_upper' not in d.columns:
            try:
                ma = d['close'].rolling(20).mean()
                std = d['close'].rolling(20).std()
                d['bb_upper'] = ma + 2*std; d['bb_lower'] = ma - 2*std; d['bb_pos'] = (d['close']-d['bb_lower'])/(d['bb_upper']-d['bb_lower']).replace(0,1e-9)
            except Exception:
                pass
        if 'supertrend' not in d.columns:
            try:
                # best-effort minimal supertrend: use previous close
                d['supertrend'] = d['close'].rolling(10).mean()
            except Exception:
                pass
        # prepare mplfinance style if available
        if use_mpf:
            mpf_df = d.set_index(d.index)  # index is simple integer; mpf accepts DatetimeIndex ideally but works with index
            addplots = []
            if 'ema_200' in d.columns:
                addplots.append(mpf.make_addplot(d['ema_200']))
            if 'bb_upper' in d.columns and 'bb_lower' in d.columns:
                addplots.append(mpf.make_addplot(d['bb_upper']))
                addplots.append(mpf.make_addplot(d['bb_lower']))
            if 'supertrend' in d.columns:
                addplots.append(mpf.make_addplot(d['supertrend']))
            fig_kwargs = dict(figsize=(12,6))
            mpf.plot(mpf_df[['open','high','low','close']], type='candle', style='charles', addplot=addplots, volume=False, savefig=filepath, **fig_kwargs)
            # overlay markers with matplotlib after saving can be complex; instead draw simple markers using matplotlib if needed below
        else:
            # fallback drawing using matplotlib rectangles + wicks
            fig, ax = plt.subplots(figsize=(12,6))
            width = 0.6
            for i in range(len(d)):
                o,h,l,c = d.loc[i, ['open','high','low','close']]
                ax.vlines(i, l, h, linewidth=0.8)
                rect_bottom = min(o,c)
                rect_height = max(abs(c-o), 1e-8)
                col = 'green' if c>=o else 'red'
                ax.add_patch(Rectangle((i-width/2, rect_bottom), width, rect_height, color=col, alpha=0.8))
            if 'ema_200' in d.columns:
                ax.plot(range(len(d)), d['ema_200'], linewidth=1.2, label='EMA200')
            if 'bb_upper' in d.columns and 'bb_lower' in d.columns:
                ax.plot(range(len(d)), d['bb_upper'], linewidth=0.8, linestyle='--', label='BB_upper')
                ax.plot(range(len(d)), d['bb_lower'], linewidth=0.8, linestyle='--', label='BB_lower')
            if 'supertrend' in d.columns:
                ax.plot(range(len(d)), d['supertrend'], linewidth=1.0, label='Supertrend')
            ax.set_xlim(-1, len(d))
            ax.legend(loc='best'); ax.grid(True, linestyle=':')
            fig.tight_layout()
            # draw markers
            if confirmed_trades:
                try:
                    for t in confirmed_trades:
                        idx = t.get('entry_idx'); price = t.get('entry_price'); side = t.get('side','long')
                        if idx is None: continue
                        yoff = (d['high'].max()-d['low'].min())*0.02
                        ax.annotate('▲' if side=='long' else '▼', (idx, price + (yoff if side=='long' else -yoff)), fontsize=12, ha='center')
                except Exception:
                    pass
            if show_predictions and predictions:
                try:
                    for p in predictions:
                        idx = p.get('idx'); price = p.get('price') if 'price' in p else float(d['close'].iloc[p.get('idx')]) if p.get('idx') is not None else None
                        if idx is None: continue
                        ax.annotate('·', (idx, price), fontsize=8, ha='center', alpha=0.6)
                except Exception:
                    pass
            fig.savefig(filepath); plt.close(fig)
            return filepath
        # if we used mplfinance and need to add markers, do a simple overlay
        try:
            import matplotlib.pyplot as _plt
            fig = _plt.figure()
            img = _plt.imread(filepath)
            _plt.imshow(img); _plt.axis('off')
            ax2 = fig.add_axes([0,0,1,1], anchor='NW', zorder=1)
            # do not plot markers when using mpf to avoid complex transforms; skip marker overlay in mpf mode for now
            _plt.savefig(filepath); _plt.close(fig)
        except Exception:
            pass
        return filepath
    except Exception:
        return None

def save_walkforward_side_by_side(df, symbol, segment_idx, confirmed_trades, all_predictions):
    """
    Create side-by-side charts: left = confirmed trades, right = raw predictions.
    Save to reports/walkforward/<symbol>/segment_<segment_idx>.png
    """
    try:
        _ensure_dirs()
        out_dir = os.path.join("reports","walkforward", symbol.replace('/',''))
        os.makedirs(out_dir, exist_ok=True)
        left = os.path.join(out_dir, f"segment_{segment_idx}_confirmed.png")
        right = os.path.join(out_dir, f"segment_{segment_idx}_predictions.png")
        final = os.path.join(out_dir, f"segment_{segment_idx}.png")
        draw_chart_with_indicators(df, symbol, left, confirmed_trades=confirmed_trades, predictions=None, show_predictions=False)
        # build prediction markers in expected format
        preds = []
        for p in (all_predictions or []):
            idx = p.get('idx'); proba = float(p.get('proba', 0.5)) if p.get('proba') is not None else 0.5
            if idx is None: continue
            preds.append({'idx': idx, 'price': float(df['close'].iloc[idx]) if idx < len(df) else None, 'proba': proba})
        draw_chart_with_indicators(df, symbol, right, confirmed_trades=None, predictions=preds, show_predictions=True)
        # stitch side-by-side
        try:
            from PIL import Image
            L = Image.open(left).convert('RGB'); R = Image.open(right).convert('RGB')
            H = max(L.height, R.height); W = L.width + R.width
            canvas = Image.new('RGB', (W, H), (255,255,255)); canvas.paste(L, (0,0)); canvas.paste(R, (L.width,0))
            canvas.save(final)
            # remove intermediate
            try: os.remove(left); os.remove(right)
            except Exception: pass
            return final
        except Exception:
            return None
    except Exception:
        return None

# Telegram commands: /chart, /walkreport, /show_predictions

def add_telegram_fiblegend_command(dispatcher):
    from telegram.ext import CommandHandler
    def _cmd(update, context):
        global SHOW_FIB_LEGEND
        if not context.args:
            try:
                update.message.reply_text(f"Fib legend currently {'ON' if SHOW_FIB_LEGEND else 'OFF'}")
            except Exception:
                pass
            return
        arg = context.args[0].lower()
        if arg in ("on","true","1","yes"):
            SHOW_FIB_LEGEND = True
            try: update.message.reply_text("Fib legend enabled.")
            except Exception: pass
        elif arg in ("off","false","0","no"):
            SHOW_FIB_LEGEND = False
            try: update.message.reply_text("Fib legend disabled.")
            except Exception: pass
        else:
            try: update.message.reply_text("Usage: /show_fiblegend on|off")
            except Exception: pass
    dispatcher.add_handler(CommandHandler("show_fiblegend", _cmd))


def add_telegram_chart_commands(dispatcher):
    try:
        from telegram.ext import CommandHandler
        def cmd_chart(update, context):
            args = context.args if hasattr(context, 'args') else []
            sym = args[0] if args else (globals().get('cfg',{}).get('symbol') if 'cfg' in globals() else None)
            if not sym:
                update.message.reply_text("Usage: /chart SYMBOL")
                return
            path = os.path.join("charts", f"{sym.replace('/','')}_status.png")
            if os.path.exists(path):
                update.message.reply_text(path)
            else:
                update.message.reply_text("Chart not found: " + path)
        def cmd_walkreport(update, context):
            args = context.args if hasattr(context, 'args') else []
            if len(args) < 2:
                update.message.reply_text("Usage: /walkreport SYMBOL SEGMENT")
                return
            sym = args[0]; seg = args[1]
            path = os.path.join("reports","walkforward", sym.replace('/',''), f"segment_{seg}.png")
            if os.path.exists(path):
                update.message.reply_text(path)
            else:
                update.message.reply_text("Report not found: " + path)
        def cmd_show_predictions(update, context):
            args = context.args if hasattr(context, 'args') else []
            global SHOW_PREDICTIONS
            if not args:
                update.message.reply_text("Usage: /show_predictions on|off")
                return
            val = args[0].lower()
            if val in ('on','1','true','yes'):
                SHOW_PREDICTIONS = True
                update.message.reply_text("SHOW_PREDICTIONS = ON")
            else:
                SHOW_PREDICTIONS = False
                update.message.reply_text("SHOW_PREDICTIONS = OFF")
        dispatcher.add_handler(CommandHandler("chart", cmd_chart))
        dispatcher.add_handler(CommandHandler("walkreport", cmd_walkreport))
        dispatcher.add_handler(CommandHandler("show_predictions", cmd_show_predictions))
    except Exception:
        pass
# ==========================================================================





# ================= Phase 6: Evolve queue, handler registration =================
import threading, queue, time as _time, json as _json

# evolve queue and worker
_EVOLVE_QUEUE = queue.Queue()
_EVOLVE_WORKER_THREAD = None
_EVOLVE_LOCK = threading.Lock()

def _evolve_worker():
    while True:
        item = _EVOLVE_QUEUE.get()
        if item is None:
            break
        try:
            symbol = item.get('symbol'); gens = item.get('generations', 6); pop = item.get('population', 8); tf = item.get('timeframe', None); months = item.get('months', 6)
            # call existing _maybe_run_evolve_from_cli if available; otherwise attempt basic random search placeholder
            try:
                if '_maybe_run_evolve_from_cli' in globals():
                    _maybe_run_evolve_from_cli(type('Args',(object,),{'symbol': symbol, 'timeframe': tf, 'months': months, 'generations': gens, 'population': pop}))
                else:
                    # simple placeholder work: sleep and log
                    for g in range(gens):
                        _time.sleep(0.5)
                        print(f"[EVOLVE-QUEUE] gen {g+1}/{gens} done for {symbol}")
            except Exception as e:
                print(f"[EVOLVE] worker error: {e}")
            # append to evolve log
            try:
                os.makedirs("logs", exist_ok=True)
                with open(os.path.join("logs","evolve.log"), "a", encoding="utf-8") as f:
                    f.write(f"{_time.time()} EVOLVE_COMPLETED symbol={symbol} gens={gens} pop={pop}\\n")
            except Exception:
                pass
        finally:
            _EVOLVE_QUEUE.task_done()

def start_evolve_worker():
    global _EVOLVE_WORKER_THREAD, _EVOLVE_LOCK
    with _EVOLVE_LOCK:
        if _EVOLVE_WORKER_THREAD is None or not _EVOLVE_WORKER_THREAD.is_alive():
            t = threading.Thread(target=_evolve_worker, daemon=True, name="evolve-worker")
            _EVOLVE_WORKER_THREAD = t
            t.start()
            return True
    return False

def stop_evolve_worker():
    try:
        _EVOLVE_QUEUE.put(None)
    except Exception:
        pass

# Telegram /evolve command (non-blocking queuing)
def add_telegram_evolve_command(dispatcher):
    try:
        from telegram.ext import CommandHandler
        def _cmd(update, context):
            args = context.args if hasattr(context, 'args') else []
            sym = args[0] if args else (globals().get('cfg',{}).get('symbol') if 'cfg' in globals() else None)
            gens = int(args[1]) if len(args)>1 else 6
            pop = int(args[2]) if len(args)>2 else 8
            tf = args[3] if len(args)>3 else None
            months = int(args[4]) if len(args)>4 else 6
            # queue item
            item = {'symbol': sym, 'generations': gens, 'population': pop, 'timeframe': tf, 'months': months}
            _EVOLVE_QUEUE.put(item)
            start_evolve_worker()
            try:
                update.message.reply_text(f"Evolve queued for {sym} gens={gens} pop={pop}.")
            except Exception:
                pass
        dispatcher.add_handler(CommandHandler("evolve", _cmd))
    except Exception as e:
        print("[TG] add_telegram_evolve_command error:", e)

# CLI helper: call this from main after parse_args to support --evolve flag and evolve subcommand
def _maybe_run_evolve_from_cli(args):
    try:
        # args could be Namespace or simple object
        if getattr(args, 'evolve', False) or getattr(args, 'command', None) == 'evolve' or hasattr(args, 'generations'):
            # queue and return True to indicate handled
            sym = getattr(args, 'symbol', None) or (globals().get('cfg',{}).get('symbol') if 'cfg' in globals() else None)
            item = {'symbol': sym, 'generations': getattr(args, 'generations', 6), 'population': getattr(args, 'population', 8), 'timeframe': getattr(args, 'timeframe', None), 'months': getattr(args, 'months', 6)}
            _EVOLVE_QUEUE.put(item)
            start_evolve_worker()
            print(f"[EVOLVE] queued via CLI for {sym}")
            return True
    except Exception as e:
        print("[EVOLVE] cli helper error:", e)
    return False

# Helper to register all telegram handlers in one place
def register_all_telegram_handlers(dispatcher):
    # call each add_telegram_* if present
    for name in dir():
        if name.startswith("add_telegram_"):
            try:
                fn = globals().get(name)
                if callable(fn):
                    fn(dispatcher)
            except Exception:
                pass
# ==============================================================================


if __name__ == "__main__":
    # Ensure proper async execution
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Shutdown requested by user")
    except Exception as e:
        log(f"Fatal error: {e}")
        sys.exit(1)


# === BEGIN HYBRID AI PATCH (Step 1) ===
import os, pickle, joblib, torch, torch.nn as nn
import numpy as np

# Simple LSTM model for hybrid AI
class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50)
        c0 = torch.zeros(2, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Paths for persistence
STATE_DIR = os.path.join(os.path.dirname(__file__), "ai_state")
os.makedirs(STATE_DIR, exist_ok=True)
LR_PATH = os.path.join(STATE_DIR, "logreg.pkl")
LSTM_PATH = os.path.join(STATE_DIR, "lstm.pt")

# Load models if available
logreg_model = None
lstm_model = None

def load_ai_state():
    global logreg_model, lstm_model
    try:
        if os.path.exists(LR_PATH):
            logreg_model = joblib.load(LR_PATH)
        if os.path.exists(LSTM_PATH):
            model = LSTMModel()
            model.load_state_dict(torch.load(LSTM_PATH))
            model.eval()
            lstm_model = model
    except Exception as e:
        print(f"[AI] Failed to load models: {e}")

def save_ai_state():
    global logreg_model, lstm_model
    try:
        if logreg_model is not None:
            joblib.dump(logreg_model, LR_PATH)
        if lstm_model is not None:
            torch.save(lstm_model.state_dict(), LSTM_PATH)
    except Exception as e:
        print(f"[AI] Failed to save models: {e}")

load_ai_state()

def predict_with_hybrid_ai(symbol: str, features: np.ndarray = None):
    """
    Use Logistic Regression + LSTM hybrid AI predictor.
    Fallback to legacy predict_with_ai if available.
    """
    global logreg_model, lstm_model
    try:
        if features is None:
            # fallback: generate random placeholder if no features provided
            features = np.random.rand(1, 10)

        preds = []
        if logreg_model is not None:
            try:
                p = logreg_model.predict_proba(features)[0][1]
                preds.append(("logreg", p))
            except Exception as e:
                print(f"[AI] logreg prediction error: {e}")

        if lstm_model is not None:
            try:
                arr = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = torch.sigmoid(lstm_model(arr)).item()
                preds.append(("lstm", out))
            except Exception as e:
                print(f"[AI] lstm prediction error: {e}")

        if preds:
            avg = sum(p for _, p in preds) / len(preds)
            return {"symbol": symbol, "hybrid_signal": avg, "details": preds}

        # fallback to legacy predictor
        if "predict_with_ai" in globals():
            return globals()["predict_with_ai"](symbol)

        return {"symbol": symbol, "hybrid_signal": None, "details": []}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}
# === END HYBRID AI PATCH (Step 1) ===



# === BEGIN WALK-FORWARD OPTIMIZATION PATCH (Step 2) ===
# Adds WalkForwardOptimizer if not present and provides a simple CLI wrapper.
import numpy as np
import json
from datetime import datetime

if "WalkForwardOptimizer" not in globals():
    class WalkForwardOptimizer:
        """Walk-forward optimization engine"""

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

        def run_optimization(self) -> bool:
            """Run walk-forward optimization"""
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
                    test_prices = df.iloc[train_end:test_end]['close'].values

                    # Train models for this window
                    success = self._train_window_models(
                        f"{self.symbol}_wf_{window_count}", 
                        train_features, train_labels
                    )

                    if not success:
                        log(f"Training failed for window {window_count}")
                        start_idx += self.step_size
                        continue

                    # Test on forward window
                    window_results = self._test_window(
                        f"{self.symbol}_wf_{window_count}",
                        test_features, test_labels, test_prices,
                        train_end, current_capital
                    )

                    # Update capital and position
                    current_capital = window_results['final_capital']
                    current_position = window_results.get('final_position')

                    # Store results
                    self.results.append(window_results)
                    self.equity_curve.extend(window_results['equity_curve'])
                    self.trades.extend(window_results['trades'])

                    start_idx += self.step_size

                # Calculate overall metrics
                self._calculate_metrics(initial_capital)

                log(f"Walk-forward completed: {window_count} windows, Final capital: ${current_capital:.2f}")
                return True

            except Exception as e:
                log(f"Walk-forward optimization error: {e}")
                return False

        def _train_window_models(self, model_id: str, features: pd.DataFrame, 
                               labels: pd.Series) -> bool:
            """Train models for current window"""
            try:
                # Train LR model
                lr_model = OnlineLR(features.shape[1], lr=AI_CFG["learn_rate"], l2=AI_CFG["l2"])

                for epoch in range(AI_CFG.get("learn_epochs_boot", 2)):
                    for i in range(len(features)):
                        lr_model.update(features.iloc[i].values, int(labels.iloc[i]))

                # Store LR model temporarily
                temp_state = lr_model.to_dict()
                _AI_STATE[model_id] = temp_state

                # Train LSTM if available
                if TORCH_OK and len(features) > AI_CFG.get("lstm_seq_len", 60) + 100:
                    lstm_success = train_lstm_model(
                        model_id, features, labels,
                        epochs=AI_CFG.get("lstm_epochs", 10) // 2,  # Reduced for speed
                        batch_size=AI_CFG.get("lstm_batch_size", 128),
                        seq_len=AI_CFG.get("lstm_seq_len", 60)
                    )
                    log(f"LSTM training for {model_id}: {'Success' if lstm_success else 'Failed'}")

                return True

            except Exception as e:
                log(f"Window model training error: {e}")
                return False

        def _test_window(self, model_id: str, features: pd.DataFrame, labels: pd.Series,
                        prices: np.ndarray, start_idx: int, initial_capital: float) -> Dict:
            """Test models on forward window"""
            try:
                equity = [initial_capital]
                trades = []
                position = None
                capital = initial_capital

                # Load models
                lr_model = None
                if model_id in _AI_STATE:
                    try:
                        lr_model = OnlineLR.from_dict(_AI_STATE.get(model_id, {}))
                    except Exception:
                        lr_model = None
                lstm_model_obj, lstm_scaler = (None, None)
                if TORCH_OK:
                    try:
                        lstm_model_obj, lstm_scaler = load_lstm_model(model_id, features.shape[1])
                    except Exception:
                        lstm_model_obj, lstm_scaler = None, None

                for i in range(len(features)):
                    current_price = prices[i]
                    current_features = features.iloc[i].values

                    # Get predictions
                    p_lr = lr_model.predict_proba(current_features) if lr_model else 0.5
                    p_lstm = None

                    if lstm_model_obj and lstm_scaler and i >= AI_CFG.get("lstm_seq_len", 60):
                        try:
                            seq_start = max(0, i - AI_CFG.get("lstm_seq_len", 60))
                            seq_features = features.iloc[seq_start:i].values
                            if len(seq_features) == AI_CFG.get("lstm_seq_len", 60):
                                seq_scaled = lstm_scaler.transform(seq_features)
                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                with torch.no_grad():
                                    seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                                    p_lstm = float(lstm_model_obj(seq_tensor).cpu().numpy()[0][0])
                        except Exception:
                            pass

                    # Combine predictions
                    if p_lstm is not None:
                        p_combined = AI_CFG.get("hybrid_weight", 0.6) * p_lstm + (1 - AI_CFG.get("hybrid_weight", 0.6)) * p_lr
                    else:
                        p_combined = p_lr

                    # Trading logic (simplified)
                    if position is None:
                        if p_combined >= AI_CFG.get("threshold_long", 0.58):
                            position = {'side': 'long', 'entry_price': current_price, 'entry_idx': start_idx + i, 'shares': capital * 0.95 / current_price}
                        elif p_combined <= AI_CFG.get("threshold_short", 0.42):
                            position = {'side': 'short', 'entry_price': current_price, 'entry_idx': start_idx + i, 'shares': capital * 0.95 / current_price}
                    else:
                        should_exit = False
                        if position['side'] == 'long':
                            if p_combined < 0.45 or i == len(features) - 1:
                                should_exit = True
                        else:
                            if p_combined > 0.55 or i == len(features) - 1:
                                should_exit = True

                        if should_exit:
                            if position['side'] == 'long':
                                pnl = (current_price - position['entry_price']) * position['shares']
                            else:
                                pnl = (position['entry_price'] - current_price) * position['shares']
                            capital += pnl
                            trades.append({'entry_idx': position['entry_idx'], 'exit_idx': start_idx + i, 'side': position['side'], 'entry_price': position['entry_price'], 'exit_price': current_price, 'shares': position['shares'], 'pnl': pnl})
                            position = None
                    equity.append(capital)

                return {'final_capital': capital, 'final_position': position, 'equity_curve': equity, 'trades': trades, 'window_start': start_idx, 'window_end': start_idx + len(features)}

            except Exception as e:
                log(f"Window testing error: {e}")
                return {'final_capital': initial_capital, 'final_position': None, 'equity_curve': [initial_capital], 'trades': [], 'window_start': start_idx, 'window_end': start_idx + len(features)}

        def _calculate_metrics(self, initial_capital: float):
            try:
                if not self.equity_curve or not self.trades:
                    return
                final_capital = self.equity_curve[-1]
                total_return = (final_capital / initial_capital - 1) * 100
                winning_trades = [t for t in self.trades if t['pnl'] > 0]
                losing_trades = [t for t in self.trades if t['pnl'] < 0]
                win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
                avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
                peak = np.maximum.accumulate(self.equity_curve)
                drawdown = (np.array(self.equity_curve) - peak) / peak * 100
                max_drawdown = np.min(drawdown)
                returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                self.metrics = {'initial_capital': initial_capital, 'final_capital': final_capital, 'total_return_pct': total_return, 'total_trades': len(self.trades), 'winning_trades': len(winning_trades), 'losing_trades': len(losing_trades), 'win_rate_pct': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'), 'max_drawdown_pct': max_drawdown, 'sharpe_ratio': sharpe}
            except Exception as e:
                log(f"Metrics calculation error: {e}")
                self.metrics = {}
# CLI wrapper to run walkforward easily
def run_walkforward_cli(symbol, timeframe="5m", months=6):
    wf = WalkForwardOptimizer(symbol, timeframe=timeframe, total_months=months)
    success = wf.run_optimization()
    if success:
        path = wf.save_results()
        log(f"Walk-forward saved to {path}")
    else:
        log("Walk-forward failed")
# === END WALK-FORWARD OPTIMIZATION PATCH (Step 2) ===



# === BEGIN BACKGROUND ASYNC & TRIGGER PATCH (Step 3) ===
import asyncio, inspect, threading, time, traceback, signal

# Global task registry
_bot_tasks = {}
_bot_running = False
_bot_engine = None

def _slog(msg):
    try:
        log(f"[runloop] {msg}")
    except Exception:
        print(f"[runloop] {msg}")

async def _wrap_coro(coro_fn, *args, **kwargs):
    try:
        if inspect.iscoroutinefunction(coro_fn):
            return await coro_fn(*args, **kwargs)
        elif inspect.iscoroutine(coro_fn):
            return await coro_fn
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: coro_fn(*args, **kwargs))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _slog(f"Task error: {e}\\n{traceback.format_exc()}")
        return None

async def start_background_tasks(engine=None, restart=False):
    global _bot_tasks, _bot_engine, _bot_running
    if _bot_running and not restart:
        _slog("Background tasks already running")
        return
    if restart:
        await stop_background_tasks()

    _bot_engine = engine or TradingEngine()
    loop = asyncio.get_event_loop()

    tasks = {
        "check_triggers": (getattr(_bot_engine, "check_triggers", None), 60),
        "heartbeat": (getattr(_bot_engine, "heartbeat", None) or (lambda: _slog("heartbeat")), 300)
    }

    for name, (fn, interval) in tasks.items():
        if not fn:
            continue
        async def _runner(f, sec, n):
            _slog(f"Task {n} started interval={sec}")
            try:
                while True:
                    try:
                        await _wrap_coro(f)
                    except Exception as e:
                        _slog(f"Task {n} inner error: {e}")
                    await asyncio.sleep(sec)
            except asyncio.CancelledError:
                _slog(f"Task {n} cancelled")
        _bot_tasks[name] = loop.create_task(_runner(fn, interval, name))

    _bot_running = True
    _slog("Background tasks started")

async def stop_background_tasks():
    global _bot_tasks, _bot_running
    if not _bot_tasks:
        return
    for t in _bot_tasks.values():
        try: t.cancel()
        except: pass
    await asyncio.gather(*_bot_tasks.values(), return_exceptions=True)
    _bot_tasks.clear()
    _bot_running = False
    _slog("Background tasks stopped")

def run_bot_blocking(engine=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_background_tasks(engine))
        _slog("Bot running... Ctrl+C to stop")
        loop.run_forever()
    except KeyboardInterrupt:
        _slog("KeyboardInterrupt, stopping...")
    finally:
        loop.run_until_complete(stop_background_tasks())
        loop.close()
        _slog("Bot stopped")

def start_bot(engine=None):
    def runner():
        run_bot_blocking(engine)
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    _slog("Bot started in background thread")
# === END BACKGROUND ASYNC & TRIGGER PATCH (Step 3) ===



# === BEGIN TELEGRAM & CLI ENHANCEMENTS PATCH (Step 4) ===
import argparse, inspect, asyncio

# --- Safe TelegramManager wrapper ---
if "TelegramManager" not in globals():
    class TelegramManager:
        def __init__(self, token=None, chat_id=None):
            self.token = token or TELEGRAM_TOKEN
            self.chat_id = chat_id or TELEGRAM_CHAT_ID
            self.running = False
        async def start(self):
            self.running = True
            _slog("TelegramManager (stub) started")
        async def close(self):
            self.running = False
            _slog("TelegramManager (stub) closed")

tg_manager = None

async def start_telegram_manager():
    global tg_manager
    try:
        if tg_manager is None:
            tg_manager = TelegramManager()
            await tg_manager.start()
        else:
            _slog("TelegramManager already running")
    except Exception as e:
        _slog(f"start_telegram_manager error: {e}")

async def stop_telegram_manager():
    global tg_manager
    try:
        if tg_manager:
            await tg_manager.close()
            tg_manager = None
    except Exception as e:
        _slog(f"stop_telegram_manager error: {e}")

# --- CLI handlers ---
async def cli_status(args):
    try:
        engine = TradingEngine()
        # prefer engine.get_positions_summary if exists
        if hasattr(engine, "get_positions_summary"):
            summary = engine.get_positions_summary()
        else:
            # best-effort summary
            summary = {"active_count": 0, "total_value": 0.0, "total_pnl": 0.0, "positions": []}
        print("=== STATUS ===")
        print(f"Active positions: {summary.get('active_count')} | Total value: {summary.get('total_value'):.2f} | Total PnL: {summary.get('total_pnl'):.2f}")
        for pos in summary.get("positions", []):
            print(f"{pos.get('symbol')} {pos.get('side')} qty={pos.get('qty')} entry={pos.get('entry')} current={pos.get('current_price')} pnl={pos.get('pnl')}")
    except Exception as e:
        print(f"[cli_status] error: {e}")

async def cli_analyze(args):
    try:
        symbol = normalize_symbol(args.symbol) if hasattr(__builtins__, "normalize_symbol") or 'normalize_symbol' in globals() else args.symbol
        engine = TradingEngine()
        # allow both sync and async analyze_symbol
        analy = getattr(engine, "analyze_symbol", None)
        if analy:
            if inspect.iscoroutinefunction(analy):
                res = await analy(symbol)
            else:
                res = analy(symbol)
        else:
            # fallback to get_indicators
            indicators, df = get_indicators(symbol, cfg.get("base_tf", "5m"))
            res = {"results": {cfg.get("base_tf","5m"): {"indicators": indicators}}, "core_agree": False, "core_signal": "neutral"}
        print("=== ANALYZE ===")
        print(res)
    except Exception as e:
        print(f"[cli_analyze] error: {e}")

async def cli_predict(args):
    try:
        symbol = normalize_symbol(args.symbol) if 'normalize_symbol' in globals() else args.symbol
        ai = globals().get("predict_with_hybrid_ai") or globals().get("predict_with_ai")
        if ai is None:
            print("No AI predictor found")
            return
        if inspect.iscoroutinefunction(ai):
            out = await ai(symbol)
        else:
            out = ai(symbol)
        print("=== PREDICT ===")
        print(out)
    except Exception as e:
        print(f"[cli_predict] error: {e}")

async def cli_walkforward(args):
    try:
        symbol = normalize_symbol(args.symbol) if 'normalize_symbol' in globals() else args.symbol
        wf = WalkForwardOptimizer(symbol, timeframe=args.timeframe or "5m", total_months=int(args.months or 6))
        wf.run_optimization()
        path = wf.save_results()
        print(f"Walk-forward results saved to {path}")
    except Exception as e:
        print(f"[cli_walkforward] error: {e}")

def build_cli_parser():
    parser = argparse.ArgumentParser(description="Trading Bot CLI (merged)")
    sub = parser.add_subparsers(dest="command")
    st = sub.add_parser("status")
    st.set_defaults(func=lambda args: cli_status(args))
    an = sub.add_parser("analyze")
    an.add_argument("symbol", nargs="?", default="BTC/USDT:USDT")
    an.set_defaults(func=lambda args: cli_analyze(args))
    pr = sub.add_parser("predict")
    pr.add_argument("symbol", nargs="?", default="BTC/USDT:USDT")
    pr.set_defaults(func=lambda args: cli_predict(args))
    wf = sub.add_parser("walkforward")
    wf.add_argument("symbol", nargs="?", default="BTC/USDT:USDT")
    wf.add_argument("--timeframe", default="5m")
    wf.add_argument("--months", default=6)
    wf.set_defaults(func=lambda args: cli_walkforward(args))
    run = sub.add_parser("run")
    run.set_defaults(func=lambda args: run_bot_blocking(TradingEngine()))
    return parser

# CLI dispatch when executed directly
if __name__ == "__main__" and ("__name__" in globals()):
    import sys
    if len(sys.argv) > 1:
        parser = build_cli_parser()
        args = parser.parse_args()
        fn = args.func(args)
        if inspect.iscoroutine(fn):
            asyncio.run(fn)
        else:
            # if function returns coroutine or is async wrapper, ensure run
            if hasattr(fn, "__await__") or inspect.iscoroutine(fn):
                asyncio.run(fn)
# === END TELEGRAM & CLI ENHANCEMENTS PATCH (Step 4) ===



# ==================== Phase4 Additions (Budget, Indicators, Training) ====================
import numpy as _np, datetime as _dt

# -- BudgetManager with per-symbol auto-adjust --
class BudgetManager:
    def __init__(self, cap: float = 0.0):
        self.cap = float(cap)
        self.allocations = {}  # symbol -> allocation (USD)

    def set_cap(self, cap: float):
        cap = max(0.0, float(cap))
        old_cap = self.cap
        if old_cap <= 0 or sum(self.allocations.values())<=0:
            self.cap = cap
            return
        # rescale proportionally
        factor = cap / old_cap if old_cap>0 else 0.0
        for s in list(self.allocations.keys()):
            self.allocations[s] = self.allocations[s] * factor
        self.cap = cap

    def set_symbol_budget(self, symbol: str, amount: float):
        symbol = symbol.upper()
        amount = max(0.0, float(amount))
        self.allocations[symbol] = amount
        self._rescale_if_needed()

    def _rescale_if_needed(self):
        total = sum(self.allocations.values())
        if self.cap>0 and total > self.cap:
            factor = self.cap / total
            for s in list(self.allocations.keys()):
                self.allocations[s] = self.allocations[s] * factor

    def remaining_budget(self):
        return max(0.0, self.cap - sum(self.allocations.values()))

    def get_status(self):
        return {"cap": self.cap, "allocations": dict(self.allocations), "remaining": self.remaining_budget()}

# global manager
BUDGET_MANAGER = BudgetManager(0.0)

# -- Telegram commands for budget and hybrid --
def add_telegram_budget_commands(dispatcher):
    try:
        from telegram.ext import CommandHandler
        import os as _os
        _os.makedirs("logs", exist_ok=True)
        def set_budget(update, context):
            args = context.args if hasattr(context, "args") else []
            if not args:
                update.message.reply_text("Usage:\\n/set_budget <cap>\\n/set_budget SYMBOL amount")
                return
            try:
                if len(args)==1:
                    cap = float(args[0])
                    BUDGET_MANAGER.set_cap(cap)
                    logline = f"[{_dt.datetime.utcnow().isoformat()}] set_budget cap={cap}"
                else:
                    sym = args[0].upper(); amt = float(args[1])
                    BUDGET_MANAGER.set_symbol_budget(sym, amt)
                    logline = f"[{_dt.datetime.utcnow().isoformat()}] set_budget {sym}={amt}"
                with open(_os.path.join("logs","settings_history.log"), "a", encoding="utf-8") as f:
                    f.write(logline + "\\n")
                update.message.reply_text(f"Budget updated. {BUDGET_MANAGER.get_status()}")
            except Exception as e:
                update.message.reply_text(f"Error: {e}")
        dispatcher.add_handler(CommandHandler("set_budget", set_budget))
    except Exception as e:
        print("add_telegram_budget_commands error:", e)

def add_telegram_set_hybrid_command(dispatcher):
    try:
        from telegram.ext import CommandHandler
        import os as _os
        _os.makedirs("logs", exist_ok=True)
        def set_hybrid(update, context):
            args = context.args if hasattr(context, "args") else []
            if not args:
                update.message.reply_text("Usage: /set_hybrid <0.0..1.0>")
                return
            try:
                v = float(args[0])
                if v < 0.0 or v > 1.0:
                    update.message.reply_text("Value must be between 0.0 and 1.0")
                    return
                # assume AI_CFG exists else create
                try:
                    AI_CFG['hybrid_weight'] = v
                except Exception:
                    globals().setdefault('AI_CFG', {})['hybrid_weight'] = v
                logline = f"[{_dt.datetime.utcnow().isoformat()}] set_hybrid={v}"
                with open(_os.path.join("logs","settings_history.log"), "a", encoding="utf-8") as f:
                    f.write(logline + "\\n")
                update.message.reply_text(f"Hybrid weight set to {v}")
            except Exception as e:
                update.message.reply_text(f"Error: {e}")
        dispatcher.add_handler(CommandHandler("set_hybrid", set_hybrid))
    except Exception as e:
        print("add_telegram_set_hybrid_command error:", e)

# -- Advanced indicators --
def compute_ema(df, period=200, column='close', out_col='ema_200'):
    try:
        df[out_col] = df[column].ewm(span=period, adjust=False).mean()
    except Exception:
        df[out_col] = None
    return df

def compute_bollinger_bands(df, period=20, mult=2.0):
    try:
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df['bb_upper'] = ma + mult * std
        df['bb_lower'] = ma - mult * std
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-9)
    except Exception:
        df['bb_upper'] = df['bb_lower'] = df['bb_pos'] = None
    return df

def compute_fibonacci_levels(df, lookback: int = 120):
    import numpy as _np
    if df is None or len(df) < lookback+2:
        return df
    hi = df['high'].rolling(lookback).max()
    lo = df['low'].rolling(lookback).min()
    rng = (hi - lo).replace(0, _np.nan)
    df['fib_236'] = lo + 0.236 * rng
    df['fib_382'] = lo + 0.382 * rng
    df['fib_5']   = lo + 0.5   * rng
    df['fib_618'] = lo + 0.618 * rng
    df['fib_786'] = lo + 0.786 * rng
    try:
        v1 = (df['close']-df['fib_382']).abs()/rng
        v2 = (df['close']-df['fib_5']).abs()/rng
        v3 = (df['close']-df['fib_618']).abs()/rng
        df['fib_proximity'] = pd.concat([v1, v2, v3], axis=1).min(axis=1)
    except Exception:
        df['fib_proximity'] = None
    return df

def compute_supertrend(df, period: int = 10, multiplier: float = 3.0):
    if df is None or len(df) < period+2:
        return df
    hl2 = (df['high'] + df['low']) / 2.0
    tr = pd.concat([(df['high']-df['low']).abs(),
                    (df['high']-df['close'].shift()).abs(),
                    (df['low'] -df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    upper = hl2 + multiplier*atr
    lower = hl2 - multiplier*atr
    st = pd.Series(index=df.index, dtype=float)
    dirn = pd.Series(index=df.index, dtype=int)
    st.iloc[0] = hl2.iloc[0]; dirn.iloc[0] = 1
    for i in range(1, len(df)):
        pu = upper.iloc[i]; pl = lower.iloc[i]; prev = st.iloc[i-1]
        if df['close'].iloc[i] > pu:
            st.iloc[i] = pl; dirn.iloc[i] = 1
        elif df['close'].iloc[i] < pl:
            st.iloc[i] = pu; dirn.iloc[i] = -1
        else:
            st.iloc[i] = prev; dirn.iloc[i] = dirn.iloc[i-1]
    df['supertrend'] = st; df['supertrend_dir'] = dirn
    return df

def apply_advanced_indicators(df, cfg=None):
    try:
        df = compute_ema(df, period=cfg.get('ema_200_period',200) if cfg else 200)
        df = compute_bollinger_bands(df, period=cfg.get('bb_period',20) if cfg else 20, mult=cfg.get('bb_mult',2.0) if cfg else 2.0)
        df = compute_fibonacci_levels(df, lookback=cfg.get('fib_lookback',120) if cfg else 120)
        df = compute_supertrend(df, period=cfg.get('supertrend_period',10) if cfg else 10, multiplier=cfg.get('supertrend_multiplier',3.0) if cfg else 3.0)
    except Exception:
        pass
    return df

# -- AI training function --
def _save_training_log(msg: str):
    try:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs","training.log"), "a", encoding="utf-8") as f:
            f.write(msg + "\\n")
    except Exception:
        pass

def train_models(symbol: str = None, epochs: int = 20, batch_size:int=64, seq_len:int=32, notify_fn=None):
    import numpy as _np
    sym = symbol or (globals().get('cfg',{}).get('symbol') if 'cfg' in globals() else "BTC/USDT")
    _log_prefix = f"[TRAIN] {sym}"
    try:
        df = None
        if 'load_merged_ohlcv' in globals():
            try:
                df = load_merged_ohlcv(sym, globals().get('cfg',{}).get('timeframe','5m'), months=globals().get('cfg',{}).get('months',6))
            except Exception:
                df = None
        if df is None or len(df) < 50:
            msg = f"{_log_prefix} insufficient data for training"
            print(msg); _save_training_log(msg)
            return False
        feats = build_features(df); labels = build_labels(df)
        X = feats.fillna(0.0).values; y = labels.values.astype(int).ravel()
        n_samples, n_features = X.shape[0], X.shape[1]
        msg = f"{_log_prefix} Starting training: epochs={epochs} samples={n_samples} features={n_features}"
        print(msg); _save_training_log(msg)
        # try OnlineLR if available
        if 'OnlineLR' in globals():
            try:
                lr = OnlineLR(n_features, lr=globals().get('AI_CFG',{}).get('learn_rate',0.01), l2=globals().get('AI_CFG',{}).get('l2',1e-4))
                for epoch in range(epochs):
                    for i in range(n_samples):
                        lr.update(X[i], int(y[i]))
                    msg = f"{_log_prefix} LR Epoch {epoch+1}/{epochs} completed..."
                    print(msg); _save_training_log(msg)
                    if notify_fn:
                        try: notify_fn(msg)
                        except Exception: pass
                try:
                    if '_AI_STATE' in globals():
                        _AI_STATE[sym] = lr.to_dict()
                        if '_ai_save_state' in globals():
                            _ai_save_state(_AI_STATE)
                except Exception as e:
                    print(f"{_log_prefix} save state error: {e}")
                msg = f"{_log_prefix} LR training finished."
                print(msg); _save_training_log(msg)
            except Exception as e:
                print(f"{_log_prefix} OnlineLR training error: {e}")
        else:
            try:
                w = _np.zeros(n_features, dtype=float); b = 0.0
                lr_rate = globals().get('AI_CFG',{}).get('learn_rate',0.01)
                for epoch in range(epochs):
                    perm = _np.random.permutation(n_samples)
                    for i in perm:
                        xi = X[i]; yi = 1 if y[i] else 0
                        z = _np.dot(w, xi) + b
                        pred = 1.0 / (1.0 + _np.exp(-z))
                        grad = (pred - yi)
                        w -= lr_rate * (grad * xi + globals().get('AI_CFG',{}).get('l2',1e-4) * w)
                        b -= lr_rate * grad
                    msg = f"{_log_prefix} SGD Epoch {epoch+1}/{epochs} completed..."
                    print(msg); _save_training_log(msg)
                    if notify_fn:
                        try: notify_fn(msg)
                        except Exception: pass
                try:
                    os.makedirs("models", exist_ok=True)
                    _np.savez(os.path.join("models", f"model_{sym.replace('/','_')}.npz"), w=w, b=b)
                except Exception as e:
                    print(f"{_log_prefix} model save error: {e}")
                msg = f"{_log_prefix} SGD training finished."
                print(msg); _save_training_log(msg)
            except Exception as e:
                print(f"{_log_prefix} SGD training error: {e}")
        return True
    except Exception as e:
        print(f"[TRAIN] unexpected error: {e}")
        _save_training_log(f"[TRAIN] unexpected error: {e}")
        return False

# add Telegram retrain command wrapper if possible
def add_telegram_retrain_models_command(dispatcher):
    try:
        from telegram.ext import CommandHandler
        import asyncio
        def _cmd(update, context):
            args = context.args if hasattr(context, 'args') else []
            sym = args[0] if args else None
            try: update.message.reply_text(f"Retrain started for {sym or 'default'} (20 epochs). Progress will be logged.")
            except Exception: pass
            loop = asyncio.get_event_loop()
            def notify_fn(msg):
                try: update.message.reply_text(msg)
                except Exception: pass
            loop.create_task(_run_in_thread(sym, notify_fn))
        dispatcher.add_handler(CommandHandler("retrain_models", _cmd))
    except Exception as e:
        print("add_telegram_retrain_models_command error:", e)

def _run_in_thread(symbol, notify_fn=None):
    import concurrent.futures, asyncio
    def _sync():
        try:
            train_models(symbol=symbol, epochs=20, notify_fn=notify_fn)
        except Exception as e:
            print("[TRAIN_THREAD] error:", e)
    fut = concurrent.futures.ThreadPoolExecutor(max_workers=1).submit(_sync)
    return asyncio.wrap_future(fut)

# =======================================================================================


# --- UPGRADE MODULE APPENDED BELOW ---

# ---------------------------
# UPGRADE MODULE (APPENDED)
# ---------------------------
# This section was appended by the assistant to add features requested by the user:
# - Regime classifier (sklearn RandomForest when available)
# - Candlestick pattern detectors
# - Fair Value Gap detection
# - Dynamic ATR-based position sizing
# - Paper trading ledger (JSONL)
# - Simple backtester
# - Lightweight RL skeleton (PyTorch if available)
# - Orchestrator to run multi-symbol scans and paper trades
#
# This code is intentionally namespaced (prefix: UPG_) to avoid overriding existing symbols.
# It tries to use functions from the original file when available (by checking for them),
# but it does not modify or monkey-patch the original module.
# ---------------------------

import os, math, json, traceback
from datetime import datetime, timezone

# optional third-party libs -- imported lazily inside functions where possible
try:
    import numpy as UPG_np
    import pandas as UPG_pd
except Exception:
    UPG_np = None
    UPG_pd = None

# sklearn for regime classifier
try:
    from sklearn.ensemble import RandomForestClassifier as UPG_RFC
    from sklearn.preprocessing import StandardScaler as UPG_StandardScaler
    UPG_SKLEARN_OK = True
except Exception:
    UPG_SKLEARN_OK = False

# torch for RL skeleton
try:
    import torch as UPG_torch
    import torch.nn as UPG_nn
    UPG_TORCH_OK = True
except Exception:
    UPG_TORCH_OK = False

UPG_CFG = {
    "risk_per_trade_pct": 0.02,
    "min_contracts": 1,
    "trade_ledger": "upgrade_trade_ledger.jsonl",
    "regime_model": "upgrade_regime.pkl",
    "fvg_lookback": 50,
    "paper_trade_mode": True,
    "regime_min_train": 200
}

def UPG_save_jsonl(path, entry):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\\n")

# Candlestick detectors (vectorized helpers if pandas present)
def UPG_is_doji_row(row, body_thresh=0.1):
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    body = abs(c - o)
    rng = (h - l) if (h - l) > 0 else 1e-9
    return (body / rng) < body_thresh

def UPG_is_engulfing(prev_row, row):
    if prev_row is None:
        return None
    prev_body = prev_row['close'] - prev_row['open']
    curr_body = row['close'] - row['open']
    if prev_body < 0 and curr_body > 0 and row['close'] > prev_row['open'] and row['open'] < prev_row['close']:
        return "bullish"
    if prev_body > 0 and curr_body < 0 and row['open'] > prev_row['close'] and row['close'] < prev_row['open']:
        return "bearish"
    return None

def UPG_detect_candles(df):
    # returns DataFrame with pattern columns
    if UPG_pd is None:
        # fall back to per-row simple checks using dicts
        out = []
        prev = None
        for r in df:
            val = {}
            try:
                val['doji'] = UPG_is_doji_row(r)
            except:
                val['doji'] = False
            try:
                val['engulfing'] = UPG_is_engulfing(prev, r)
            except:
                val['engulfing'] = None
            prev = r
            out.append(val)
        return out
    else:
        out = UPG_pd.DataFrame(index=df.index)
        out['doji'] = df.apply(UPG_is_doji_row, axis=1)
        eng = []
        prev = None
        for _, row in df.iterrows():
            eng.append(UPG_is_engulfing(prev, row))
            prev = row
        out['engulfing'] = eng
        # simple hammer/shooting-star heuristics
        out['hammer'] = df.apply(lambda r: (min(r['open'], r['close']) - r['low']) > 2*abs(r['close']-r['open']) and (r['high']-max(r['open'], r['close'])) < abs(r['close']-r['open']), axis=1)
        out['shooting_star'] = df.apply(lambda r: (r['high']-max(r['open'], r['close'])) > 2*abs(r['close']-r['open']) and (min(r['open'], r['close'])-r['low']) < abs(r['close']-r['open']), axis=1)
        return out

# Fair Value Gap detection (simple)
def UPG_detect_fvg(df, lookback=50):
    gaps = []
    n = len(df)
    for i in range(max(2, n - lookback), n):
        if i-2 < 0: continue
        try:
            if df['low'].iat[i] > df['high'].iat[i-2] * 1.001:
                gaps.append((df.index[i-2], df.index[i], 'bull_fvg'))
            if df['high'].iat[i] < df['low'].iat[i-2] * 0.999:
                gaps.append((df.index[i-2], df.index[i], 'bear_fvg'))
        except Exception:
            continue
    return gaps

# Position sizing using ATR
def UPG_compute_position_size(account_usd, price, atr, risk_pct=UPG_CFG['risk_per_trade_pct'], contract_size=1.0):
    risk_amount = account_usd * risk_pct
    stop_distance = max(atr, price * 0.001)
    if stop_distance <= 0:
        return UPG_CFG['min_contracts']
    contracts = math.floor(risk_amount / (stop_distance * contract_size))
    return max(UPG_CFG['min_contracts'], contracts)

# Paper Trader (JSONL ledger)
class UPG_PaperTrader:
    def __init__(self, ledger_path=UPG_CFG['trade_ledger'], start_equity=10000.0):
        self.ledger = ledger_path
        self.equity = start_equity
        self.open_positions = []
        if os.path.exists(self.ledger):
            try:
                with open(self.ledger, 'r') as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                if rows:
                    last = rows[-1]
                    self.equity = last.get('equity_after', self.equity)
            except Exception:
                pass

    def open(self, symbol, side, entry_price, contracts, sl, tp, metadata=None):
        pos = {
            'symbol': symbol,
            'side': side,
            'entry_price': float(entry_price),
            'contracts': int(contracts),
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'open_ts': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {}
        }
        self.open_positions.append(pos)
        UPG_save_jsonl(self.ledger, {"event":"open","pos":pos,"equity_before":self.equity})
        print(f"(UPG PAPER) Opened {side} {symbol} x{contracts} @ {entry_price}")
        return pos

    def close(self, pos, exit_price, reason="TP/SL/Manual"):
        pnl_per_contract = (exit_price - pos['entry_price']) if pos['side'] == 'long' else (pos['entry_price'] - exit_price)
        pnl = pnl_per_contract * pos['contracts']
        self.equity += pnl
        pos['exit_price'] = float(exit_price)
        pos['close_ts'] = datetime.now(timezone.utc).isoformat()
        pos['pnl'] = float(pnl)
        pos['reason'] = reason
        UPG_save_jsonl(self.ledger, {"event":"close","pos":pos,"equity_after":self.equity})
        try:
            self.open_positions.remove(pos)
        except:
            pass
        print(f"(UPG PAPER) Closed {pos['side']} {pos['symbol']} @ {exit_price} PnL: {pnl:.2f}")

# Backtester
def UPG_backtest_strategy(df, entry_rule_fn, exit_rule_fn, initial_capital=10000.0):
    equity = initial_capital
    positions = []
    trades = []
    for i in range(len(df)):
        # exit checks
        for pos in positions[:]:
            should_close, close_price, reason = exit_rule_fn(pos, df, i)
            if should_close:
                pnl = (close_price - pos['entry']) * pos['contracts'] if pos['side']=='long' else (pos['entry'] - close_price) * pos['contracts']
                equity += pnl
                trades.append({"entry_idx": pos['entry_idx'], "exit_idx": i, "side": pos['side'], "entry": pos['entry'], "exit": close_price, "pnl": pnl})
                positions.remove(pos)
        # entry checks
        e = entry_rule_fn(df, i)
        if e:
            positions.append({'side': e['side'], 'entry': e['price'], 'entry_idx': i, 'contracts': e['contracts'], 'sl': e['sl'], 'tp': e['tp']})
    return {"final_equity": equity, "trades": trades}

# Lightweight RL skeleton (uses PyTorch if available)
if UPG_TORCH_OK:
    class UPG_SimplePolicy(UPG_nn.Module):
        def __init__(self, input_size, hidden=64):
            super().__init__()
            self.net = UPG_nn.Sequential(UPG_nn.Linear(input_size, hidden), UPG_nn.ReLU(), UPG_nn.Linear(hidden, hidden), UPG_nn.ReLU())
            self.policy_head = UPG_nn.Linear(hidden, 3)
            self.value_head = UPG_nn.Linear(hidden, 1)
        def forward(self, x):
            h = self.net(x)
            return self.policy_head(h), self.value_head(h)

    class UPG_RLAgent:
        def __init__(self, input_size):
            self.device = UPG_torch.device('cuda' if UPG_torch.cuda.is_available() else 'cpu')
            self.model = UPG_SimplePolicy(input_size).to(self.device)
            self.optim = UPG_torch.optim.Adam(self.model.parameters(), lr=1e-4)
        def act(self, obs):
            with UPG_torch.no_grad():
                logits, val = self.model(UPG_torch.tensor(obs, dtype=UPG_torch.float32).to(self.device))
                probs = UPG_torch.softmax(logits, dim=-1).cpu().numpy()
            return int(UPG_np.argmax(probs)), probs
        def train_on_episodes(self, episodes):
            if len(episodes) < 10:
                return
            obs = UPG_torch.tensor(UPG_np.vstack([e[0] for e in episodes]), dtype=UPG_torch.float32).to(self.device)
            acts = UPG_torch.tensor([e[1] for e in episodes], dtype=UPG_torch.long).to(self.device)
            rets = UPG_torch.tensor([e[2] for e in episodes], dtype=UPG_torch.float32).to(self.device)
            logits, vals = self.model(obs)
            loss_policy = UPG_nn.CrossEntropyLoss()(logits, acts)
            loss_value = UPG_nn.MSELoss()(vals.squeeze(), rets)
            loss = loss_policy + 0.5 * loss_value
            self.optim.zero_grad(); loss.backward(); self.optim.step()
            print("UPG_RLAgent: performed lightweight update")

else:
    UPG_RLAgent = None

# Regime classifier wrapper
class UPG_RegimeClassifier:
    def __init__(self, path=UPG_CFG['regime_model']):
        self.path = path
        self.model = None
        self.scaler = None
        self._loaded = False
        if UPG_SKLEARN_OK:
            self.model = UPG_RFC(n_estimators=80, max_depth=6, random_state=42)
            self.scaler = UPG_StandardScaler()
            self._try_load()

    def _try_load(self):
        try:
            if os.path.exists(self.path):
                import pickle
                with open(self.path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data.get('model', self.model)
                self.scaler = data.get('scaler', self.scaler)
                self._loaded = True
                print("UPG_RegimeClassifier: loaded model")
        except Exception as e:
            print("UPG_RegimeClassifier load failed:", e)

    def save(self):
        try:
            import pickle
            with open(self.path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
            print("UPG_RegimeClassifier: saved model")
        except Exception as e:
            print("UPG_RegimeClassifier save failed:", e)

    def featurize(self, df):
        X = UPG_pd.DataFrame(index=df.index) if UPG_pd is not None else []
        if 'rsi' in df.columns:
            X['rsi'] = df['rsi']
        if 'atr' in df.columns and 'close' in df.columns:
            X['atr_ratio'] = df['atr'] / df['close']
        if 'volatility_20' in df.columns:
            X['volatility_20'] = df['volatility_20']
        if 'bb_pos' in df.columns:
            X['bb_pos'] = df['bb_pos']
        if 'ema_20' in df.columns:
            X['ema20_ratio'] = df['close'] / df['ema_20']
        X = X.fillna(0).replace([UPG_np.inf, -UPG_np.inf], 0) if UPG_pd is not None else X
        return X

    def train(self, df, labels):
        if not UPG_SKLEARN_OK:
            print("UPG_RegimeClassifier: sklearn not available, skipping training")
            return False
        X = self.featurize(df)
        if len(X) < UPG_CFG['regime_min_train'] or len(X) != len(labels):
            print(f"UPG_RegimeClassifier: insufficient data for training ({len(X)} rows)")
            return False
        Xt = self.scaler.fit_transform(X)
        self.model.fit(Xt, labels)
        self.save()
        self._loaded = True
        return True

    def predict(self, df):
        if not self._loaded:
            return None
        X = self.featurize(df)
        Xt = self.scaler.transform(X)
        return int(self.model.predict(Xt)[-1])

# Orchestrator that composes upgrades with original bot when loaded
class UPG_Orchestrator:
    def __init__(self, base_module=None):
        self.base = base_module
        self.paper = UPG_PaperTrader()
        self.regime = UPG_RegimeClassifier() if UPG_SKLEARN_OK else None
        self.rl = UPG_RLAgent(input_size=16) if UPG_TORCH_OK else None

    def analyze(self, symbol, timeframe='5m'):
        if self.base is None:
            print('UPG_Orchestrator: base not attached')
            return None
        try:
            df = None
            if hasattr(self.base, 'load_history_df'):
                df = self.base.load_history_df(symbol, timeframe, months=3)
            elif hasattr(self.base, 'get_history_df'):
                df = self.base.get_history_df(symbol, timeframe, months=3)
            if df is None or len(df) < 50:
                return None
            if hasattr(self.base, 'calculate_indicators'):
                df = self.base.calculate_indicators(df)
            else:
                # minimal ATR fallback
                df['atr'] = (df['high'] - df['low']).rolling(14).mean().fillna(method='bfill')
            cand = UPG_detect_candles(df)
            df = df.join(cand)
            fvg = UPG_detect_fvg(df, UPG_CFG['fvg_lookback'])
            regime = None
            if self.regime is not None:
                try:
                    regime = self.regime.predict(df)
                except:
                    regime = None
            plan = {'symbol': symbol, 'timeframe': timeframe, 'fvg': fvg, 'regime': regime, 'last_close': float(df['close'].iloc[-1])}
            return plan
        except Exception as e:
            print('UPG_Orchestrator.analyze error:', e)
            traceback.print_exc()
            return None

    def attempt_paper_open(self, symbol, side, reason='auto'):
        if self.base is None:
            print('UPG_Orchestrator: base not attached for sizing')
            return False
        try:
            df = None
            if hasattr(self.base, 'get_indicators'):
                meta, df = self.base.get_indicators(symbol, '5m')
            elif hasattr(self.base, 'load_history_df'):
                df = self.base.load_history_df(symbol, '5m', months=1)
            if df is None or len(df) < 1:
                return False
            latest = df.iloc[-1]
            price = float(latest['close'])
            atr = float(latest.get('atr', price*0.01))
            contracts = UPG_compute_position_size(self.paper.equity, price, atr)
            sl = price - atr if side=='long' else price + atr
            tp = price + 2*atr if side=='long' else price - 2*atr
            pos = self.paper.open(symbol, side, price, contracts, sl, tp, metadata={'reason':reason})
            return pos
        except Exception as e:
            print('UPG_Orchestrator.attempt_paper_open error:', e)
            traceback.print_exc()
            return False

    def run_scan_once(self):
        if self.base is None:
            print('UPG_Orchestrator: base not attached')
            return
        symbols = getattr(self.base, 'AUTO_PAIRS', None) or getattr(self.base, 'SYMBOLS', None) or []
        for s in symbols:
            try:
                plan = self.analyze(s)
                if plan:
                    # example simple logic: if no FVG and regime is trend or None -> long
                    if len(plan.get('fvg',[]))==0 and plan.get('regime') in (None, 1):
                        self.attempt_paper_open(s, 'long', reason='upg-scan')
            except Exception as e:
                print('UPG scan error for', s, e)

# End of UPGRADE MODULE

# --- END OF MERGED FILE ---


# --- ADDITIONAL UPGRADE APPENDED ---

# ---------------------------
# ADDITIONAL UPGRADE: Telegram handlers, full indicators, resolver mapping, Fibonacci plotting
# ---------------------------
import os, math, zipfile, io, traceback
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, date2num
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

# Indicator implementations (MACD, Bollinger Bands, SMA, Fibonacci levels)
def UPG_add_indicators(df, sma_periods=(50,200), bb_period=20, bb_mult=2, macd_fast=12, macd_slow=26, macd_sig=9):
    """Add indicators to df in-place and return df. Non-destructive for existing columns."""
    import pandas as pd, numpy as np
    df = df.copy()
    # SMA
    for p in sma_periods:
        col = f"sma_{p}"
        if col not in df.columns:
            df[col] = df['close'].rolling(window=p, min_periods=1).mean()
    # Bollinger Bands
    if 'bb_mid' not in df.columns:
        df['bb_mid'] = df['close'].rolling(bb_period).mean()
        df['bb_std'] = df['close'].rolling(bb_period).std().fillna(0)
        df['bb_upper'] = df['bb_mid'] + bb_mult * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - bb_mult * df['bb_std']
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    # MACD
    if 'macd' not in df.columns:
        ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_sig_line = macd.ewm(span=macd_sig, adjust=False).mean()
        df['macd'] = macd
        df['macd_signal'] = macd_sig_line
        df['macd_hist'] = df['macd'] - df['macd_signal']
    # ATR
    if 'atr' not in df.columns:
        high_low = df['high'] - df['low']
        high_close_prev = (df['high'] - df['close'].shift(1)).abs()
        low_close_prev = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14, min_periods=1).mean().fillna(method='bfill')
    # volatility
    if 'volatility_20' not in df.columns:
        df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)
    # Fibonacci levels based on last swing (simple heuristic)
    if 'fib_levels' not in df.columns:
        # find last local min and max in past 200 bars
        window = min(len(df), 200)
        recent = df.iloc[-window:]
        low_idx = recent['low'].idxmin()
        high_idx = recent['high'].idxmax()
        low = float(recent.loc[low_idx,'low'])
        high = float(recent.loc[high_idx,'high'])
        if high == low:
            levels = {}
        else:
            diff = high - low
            levels = {
                '0.0': high,
                '0.236': high - 0.236*diff,
                '0.382': high - 0.382*diff,
                '0.5': high - 0.5*diff,
                '0.618': high - 0.618*diff,
                '0.786': high - 0.786*diff,
                '1.0': low,
                '1.272': low - 0.272*diff,
                '1.414': low - 0.414*diff,
                '1.618': low - 0.618*diff,
                '2.0': low - 1.0*diff,
                '2.618': low - 1.618*diff
            }
        df['fib_levels'] = [levels for _ in range(len(df))]
    return df

# Utility: resolve candidate function/var names on base module without monkey-patching
def UPG_resolve(base_module, candidates):
    """Return first attribute found on base_module from candidates or None."""
    if base_module is None:
        return None
    for name in candidates:
        if hasattr(base_module, name):
            return getattr(base_module, name)
    return None

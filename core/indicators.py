import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.clip(0, 100)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"].fillna(0)).cumsum()


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    money_flow = tp * df["Volume"].fillna(0)
    pos = money_flow.where(tp.diff() > 0, 0.0)
    neg = money_flow.where(tp.diff() < 0, 0.0)
    pmf = pos.rolling(n).sum()
    nmf = neg.abs().rolling(n).sum()
    ratio = pmf / nmf.replace(0, np.nan)
    out = 100 - (100 / (1 + ratio))
    return out.clip(0, 100)


def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return lower, mid, upper


def pivot_levels(df: pd.DataFrame, lookback: int = 60):
    recent = df.tail(lookback)
    return float(recent["Low"].min()), float(recent["High"].max())

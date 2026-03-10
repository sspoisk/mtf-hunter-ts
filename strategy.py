"""
MTF Hunter — стратегия Multi-Timeframe Pullback (без изменений от V2)
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    k = 2.0 / (period + 1)
    for i, v in enumerate(arr):
        if np.isnan(v):
            continue
        if i == 0 or np.isnan(out[i - 1]):
            out[i] = v
        else:
            out[i] = v * k + out[i - 1] * (1 - k)
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.full(len(close), np.nan)
    avg_l = np.full(len(close), np.nan)
    if period < len(gain):
        avg_g[period] = gain[:period].mean()
        avg_l[period] = loss[:period].mean()
        for i in range(period + 1, len(close)):
            avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i - 1]) / period
            avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
    return 100 - 100 / (1 + rs)


def check_signal(candles_1h: list, cfg: dict) -> dict | None:
    if len(candles_1h) < 100:
        return None

    arr     = np.array(candles_1h, dtype=float)
    close   = arr[:, 4]
    high    = arr[:, 2]
    low     = arr[:, 3]
    n       = len(close)

    ema_period = cfg.get('ema_period', 30)
    rsi_period = cfg.get('rsi_period', 14)
    rsi_lo     = cfg.get('rsi_lo', 40)
    rsi_hi     = cfg.get('rsi_hi', 60)
    sl_bars    = cfg.get('sl_bars', 5)
    rr         = cfg.get('rr', 2.5)

    groups = n // 4
    if groups < ema_period + 5:
        return None

    close_4h = np.array([arr[g*4:(g+1)*4, 4][-1] for g in range(groups)])
    ema_4h   = _ema(close_4h, ema_period)

    i = n - 2
    c4h_idx = i // 4
    if c4h_idx >= len(ema_4h) or np.isnan(ema_4h[c4h_idx]):
        return None

    trend_up = close_4h[c4h_idx] > ema_4h[c4h_idx]

    rsi_arr = _rsi(close, rsi_period)
    rsi_val = rsi_arr[i]
    if np.isnan(rsi_val):
        return None

    current_price = close[i]
    if current_price == 0:
        return None

    direction = None
    if trend_up and rsi_val < rsi_lo:
        direction = 'LONG'
    elif not trend_up and rsi_val > rsi_hi:
        direction = 'SHORT'

    if direction is None:
        return None

    sl_window_lo = low[max(0, i - sl_bars):i + 1].min()
    sl_window_hi = high[max(0, i - sl_bars):i + 1].max()

    if direction == 'LONG':
        sl = sl_window_lo
        if sl >= current_price:
            return None
        dist = current_price - sl
        tp   = current_price + rr * dist
    else:
        sl = sl_window_hi
        if sl <= current_price:
            return None
        dist = sl - current_price
        tp   = current_price - rr * dist

    sl_pct = abs(current_price - sl) / current_price * 100
    tp_pct = abs(tp - current_price) / current_price * 100

    return {
        'direction': direction,
        'price':     round(current_price, 8),
        'sl':        round(sl, 8),
        'tp':        round(tp, 8),
        'sl_pct':    round(sl_pct, 2),
        'tp_pct':    round(tp_pct, 2),
        'rsi':       round(rsi_val, 1),
        'ema_4h':    round(ema_4h[c4h_idx], 8),
        'trend':     'UP' if trend_up else 'DOWN',
        'rr':        rr,
    }

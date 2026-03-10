#!/usr/bin/env python3
"""
MTF Hunter TS — Backtest
Сравнивает V2 (фиксированный TP/SL) vs Trailing Stop на исторических данных OKX.

Использование:
  python backtest.py [--symbols N] [--bars N]
  python backtest.py --symbols 15 --bars 800

Параметры TS для перебора:
  Activation: с какого % прибыли активируется трейлинг
  Distance:   отступ трейлинга от пика (%)
"""

import ccxt
import numpy as np
import time
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ── Встроенные функции стратегии (чтобы не зависеть от imports) ──────────────

def _ema(arr, period):
    out = np.full(len(arr), np.nan)
    k = 2.0 / (period + 1)
    for i, v in enumerate(arr):
        if np.isnan(v): continue
        if i == 0 or np.isnan(out[i-1]): out[i] = v
        else: out[i] = v * k + out[i-1] * (1 - k)
    return out

def _rsi(close, period=14):
    delta = np.diff(close)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = np.full(len(close), np.nan)
    avg_l = np.full(len(close), np.nan)
    if period < len(gain):
        avg_g[period] = gain[:period].mean()
        avg_l[period] = loss[:period].mean()
        for i in range(period + 1, len(close)):
            avg_g[i] = (avg_g[i-1] * (period-1) + gain[i-1]) / period
            avg_l[i] = (avg_l[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
    return 100 - 100 / (1 + rs)

def check_signal_bt(candles, cfg):
    """Возвращает {direction, price, sl, tp} или None."""
    if len(candles) < 120: return None
    arr   = np.array(candles, dtype=float)
    close = arr[:, 4]; high = arr[:, 2]; low = arr[:, 3]; n = len(close)

    ep = cfg['ema_period']; rp = cfg['rsi_period']
    rlo = cfg['rsi_lo'];    rhi = cfg['rsi_hi']
    slb = cfg['sl_bars'];   rr  = cfg['rr']

    groups = n // 4
    if groups < ep + 5: return None
    close_4h = np.array([arr[g*4:(g+1)*4, 4][-1] for g in range(groups)])
    ema_4h   = _ema(close_4h, ep)

    i = n - 2; c4h = i // 4
    if c4h >= len(ema_4h) or np.isnan(ema_4h[c4h]): return None
    trend_up = close_4h[c4h] > ema_4h[c4h]

    rsi_arr = _rsi(close, rp); rsi_val = rsi_arr[i]
    if np.isnan(rsi_val): return None

    price = close[i]
    if price == 0: return None

    direction = None
    if trend_up and rsi_val < rlo:    direction = 'LONG'
    elif not trend_up and rsi_val > rhi: direction = 'SHORT'
    if direction is None: return None

    sl_lo = low[max(0, i-slb):i+1].min()
    sl_hi = high[max(0, i-slb):i+1].max()

    if direction == 'LONG':
        sl = sl_lo
        if sl >= price: return None
        tp = price + rr * (price - sl)
    else:
        sl = sl_hi
        if sl <= price: return None
        tp = price - rr * (sl - price)

    return {'direction': direction, 'price': price, 'sl': sl, 'tp': tp}


# ── Симуляция сделок ──────────────────────────────────────────────────────────

COMMISSION = 0.08   # %
SIZE_USDT  = 100
MAX_BARS   = 48     # максимальное время удержания (1h bars)

def pnl_usdt(entry, exit_p, side):
    if side == 'LONG': pct = (exit_p - entry) / entry * 100
    else:              pct = (entry - exit_p) / entry * 100
    return round(SIZE_USDT * (pct - COMMISSION) / 100, 4)


def simulate_v2(candles, entry_i, sig):
    """V2: фиксированные SL и TP."""
    entry = sig['price']; sl = sig['sl']; tp = sig['tp']; side = sig['direction']
    end_i = min(entry_i + MAX_BARS, len(candles) - 1)

    for j in range(entry_i + 1, end_i + 1):
        high = candles[j][2]; low = candles[j][3]
        if side == 'LONG':
            if low  <= sl: return pnl_usdt(entry, sl, side), 'SL', j - entry_i
            if high >= tp: return pnl_usdt(entry, tp, side), 'TP', j - entry_i
        else:
            if high >= sl: return pnl_usdt(entry, sl, side), 'SL', j - entry_i
            if low  <= tp: return pnl_usdt(entry, tp, side), 'TP', j - entry_i

    close = candles[end_i][4]
    return pnl_usdt(entry, close, side), 'TIME', end_i - entry_i


def simulate_ts(candles, entry_i, sig, act_pct, dist_pct):
    """Trailing Stop: активация при act_pct прибыли, отступ dist_pct от пика."""
    entry = sig['price']; sl = sig['sl']; side = sig['direction']
    end_i = min(entry_i + MAX_BARS, len(candles) - 1)

    ts_active = False
    ts_peak   = entry    # лучшая цена с момента входа
    ts_sl     = 0.0      # текущий уровень трейлинга

    for j in range(entry_i + 1, end_i + 1):
        high = candles[j][2]; low = candles[j][3]

        if side == 'LONG':
            # ① Сначала проверяем оригинальный SL (до активации трейлинга)
            if not ts_active and low <= sl:
                return pnl_usdt(entry, sl, side), 'SL', j - entry_i

            # ② Обновляем пик по HIGH
            if high > ts_peak: ts_peak = high

            # ③ Проверяем активацию
            if not ts_active:
                profit_pct = (ts_peak - entry) / entry * 100
                if profit_pct >= act_pct:
                    ts_active = True
                    ts_sl = ts_peak * (1 - dist_pct / 100)

            # ④ Если трейлинг активен — обновляем уровень, проверяем срабатывание
            if ts_active:
                new_sl = ts_peak * (1 - dist_pct / 100)
                if new_sl > ts_sl: ts_sl = new_sl
                if low <= ts_sl:
                    return pnl_usdt(entry, ts_sl, side), 'TS', j - entry_i

        else:  # SHORT
            if not ts_active and high >= sl:
                return pnl_usdt(entry, sl, side), 'SL', j - entry_i

            if low < ts_peak or ts_peak == entry: ts_peak = low

            if not ts_active:
                profit_pct = (entry - ts_peak) / entry * 100
                if profit_pct >= act_pct:
                    ts_active = True
                    ts_sl = ts_peak * (1 + dist_pct / 100)

            if ts_active:
                new_sl = ts_peak * (1 + dist_pct / 100)
                if new_sl < ts_sl or ts_sl == 0: ts_sl = new_sl
                if high >= ts_sl:
                    return pnl_usdt(entry, ts_sl, side), 'TS', j - entry_i

    close = candles[end_i][4]
    return pnl_usdt(entry, close, side), 'TIME', end_i - entry_i


# ── Статистика ────────────────────────────────────────────────────────────────

def calc_stats(results):
    """results: [(pnl, reason, bars), ...]"""
    if not results:
        return {'n': 0, 'wr': 0, 'pnl': 0, 'avg': 0, 'sl': 0, 'tp_or_ts': 0, 'time': 0, 'max_dd': 0}
    wins  = [r for r in results if r[0] > 0]
    sl_c  = sum(1 for r in results if r[1] == 'SL')
    tp_c  = sum(1 for r in results if r[1] in ('TP', 'TS'))
    tim_c = sum(1 for r in results if r[1] == 'TIME')
    total_pnl = sum(r[0] for r in results)

    # Max drawdown (peak-to-trough cumulative PnL)
    cum = 0; peak = 0; max_dd = 0
    for r in results:
        cum += r[0]
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    return {
        'n':         len(results),
        'wr':        round(len(wins) / len(results) * 100, 1),
        'pnl':       round(total_pnl, 2),
        'avg':       round(total_pnl / len(results), 3),
        'sl':        sl_c,
        'tp_or_ts':  tp_c,
        'time':      tim_c,
        'max_dd':    round(max_dd, 2),
    }


# ── Главная функция ───────────────────────────────────────────────────────────

TS_ACTIVATIONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TS_DISTANCES   = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

STRATEGY_CFG = {
    'ema_period': 30,
    'rsi_period': 14,
    'rsi_lo':     40,
    'rsi_hi':     60,
    'sl_bars':    5,
    'rr':         2.5,
}

EXCLUDE_BASES = {
    'BTC', 'ETH', 'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'FDUSD',
    'USDE', 'PYUSD', 'XAUT', 'PAXG', 'LTC', 'BNB', 'FIL', 'SOL',
    'AR', 'COMP', 'APT', 'TRUMP',
}


def main():
    parser = argparse.ArgumentParser(description='MTF Hunter TS Backtest')
    parser.add_argument('--symbols', type=int, default=20,  help='Кол-во символов (default 20)')
    parser.add_argument('--bars',    type=int, default=1000, help='Кол-во свечей на символ (default 1000)')
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  MTF Hunter TS — Backtest")
    print(f"  Символов: {args.symbols} | Свечей: {args.bars} (~{args.bars//24} дней)")
    print(f"  Стратегия: EMA{STRATEGY_CFG['ema_period']} · RSI{STRATEGY_CFG['rsi_lo']}/{STRATEGY_CFG['rsi_hi']} · SL_bars={STRATEGY_CFG['sl_bars']} · RR={STRATEGY_CFG['rr']}")
    print(f"{'='*65}\n")

    # ── Подключение к OKX ────────────────────────────────────────────────────
    exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    print("► Получаем список символов...")
    tickers = exchange.fetch_tickers()
    pairs = []
    for sym, t in tickers.items():
        if not sym.endswith('/USDT:USDT'): continue
        base = sym.split('/')[0]
        if base in EXCLUDE_BASES: continue
        vol = t.get('quoteVolume') or 0
        if vol == 0: vol = (t.get('baseVolume') or 0) * (t.get('last') or 0)
        if vol > 0: pairs.append((sym, vol))

    pairs.sort(key=lambda x: -x[1])
    symbols = [p[0] for p in pairs[:args.symbols]]
    print(f"► Выбрано символов: {len(symbols)}")
    print(f"  {', '.join(s.split('/')[0] for s in symbols)}\n")

    # ── Загрузка данных ──────────────────────────────────────────────────────
    print(f"► Загружаем данные ({args.bars} свечей 1h)...")
    all_candles = {}
    for sym in symbols:
        try:
            c = exchange.fetch_ohlcv(sym, '1h', limit=args.bars)
            if len(c) >= 150:
                all_candles[sym] = c
                print(f"  {sym.split('/')[0]:10s} {len(c)} свечей", flush=True)
            time.sleep(0.12)
        except Exception as e:
            print(f"  {sym.split('/')[0]:10s} ОШИБКА: {e}")

    print(f"\n► Загружено: {len(all_candles)} символов\n")

    # ── Поиск сигналов и симуляция ───────────────────────────────────────────
    print("► Сканируем сигналы и симулируем сделки...")

    v2_results  = []
    ts_results  = {(a, d): [] for a in TS_ACTIVATIONS for d in TS_DISTANCES}
    signals_by_sym = {}

    for sym, candles in all_candles.items():
        sym_signals = 0
        n = len(candles)
        prev_signal_bar = -1  # не открываем позицию если предыдущая ещё открыта

        for i in range(150, n - 1):
            # Один сигнал за раз (как в боте)
            if prev_signal_bar >= 0 and (i - prev_signal_bar) < MAX_BARS:
                continue

            sig = check_signal_bt(candles[:i+1], STRATEGY_CFG)
            if sig is None:
                continue

            # Симуляция V2
            pnl_v2, reason_v2, bars_v2 = simulate_v2(candles, i, sig)
            v2_results.append((pnl_v2, reason_v2, bars_v2))

            # Симуляция TS для каждой комбинации параметров
            for act in TS_ACTIVATIONS:
                for dist in TS_DISTANCES:
                    pnl_ts, reason_ts, bars_ts = simulate_ts(candles, i, sig, act, dist)
                    ts_results[(act, dist)].append((pnl_ts, reason_ts, bars_ts))

            prev_signal_bar = i
            sym_signals += 1

        signals_by_sym[sym.split('/')[0]] = sym_signals

    total_signals = len(v2_results)
    print(f"► Найдено сигналов: {total_signals}\n")

    if not total_signals:
        print("Сигналов не найдено. Попробуйте увеличить --bars или --symbols.")
        return

    # Распределение по символам
    print("► Сигналы по символам:")
    for sym, cnt in sorted(signals_by_sym.items(), key=lambda x: -x[1]):
        if cnt > 0: print(f"  {sym:10s} {cnt}")
    print()

    # ── Период данных ────────────────────────────────────────────────────────
    all_ts = [c[0] for clist in all_candles.values() for c in clist]
    if all_ts:
        date_from = datetime.utcfromtimestamp(min(all_ts)/1000).strftime('%Y-%m-%d')
        date_to   = datetime.utcfromtimestamp(max(all_ts)/1000).strftime('%Y-%m-%d')
        print(f"  Период данных: {date_from} → {date_to}\n")

    # ── Результаты ───────────────────────────────────────────────────────────
    v2 = calc_stats(v2_results)
    print(f"{'='*65}")
    print(f"  V2 BASELINE (фиксированный SL + TP  RR={STRATEGY_CFG['rr']})")
    print(f"{'='*65}")
    print(f"  Трейдов:  {v2['n']:4d}  |  WR: {v2['wr']:5.1f}%  |  PnL: {v2['pnl']:+.2f}$")
    print(f"  Avg:  {v2['avg']:+.3f}$  |  SL: {v2['sl']}  TP: {v2['tp_or_ts']}  TIME: {v2['time']}")
    print(f"  Max Drawdown: -{v2['max_dd']:.2f}$")
    print()

    # TS результаты — собираем все и сортируем по PnL
    ts_stats_list = []
    for (act, dist), results in ts_results.items():
        s = calc_stats(results)
        s['act'] = act; s['dist'] = dist
        ts_stats_list.append(s)

    ts_stats_list.sort(key=lambda x: -x['pnl'])

    print(f"{'='*65}")
    print(f"  TRAILING STOP — ВСЕ КОМБИНАЦИИ (сортировка по PnL)")
    print(f"{'='*65}")
    header = f"{'Act%':>5} {'Dist%':>5} {'Trades':>7} {'WR%':>6} {'PnL':>9} {'Avg':>7} {'SL':>4} {'TS':>4} {'TIME':>5} {'MaxDD':>8}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for s in ts_stats_list:
        pnl_color = '+' if s['pnl'] >= 0 else ''
        print(f"  {s['act']:>5.2f} {s['dist']:>5.2f} "
              f"{s['n']:>7d} {s['wr']:>6.1f}% "
              f"{pnl_color}{s['pnl']:>8.2f}$ "
              f"{s['avg']:>+7.3f}$ "
              f"{s['sl']:>4d} {s['tp_or_ts']:>4d} {s['time']:>5d} "
              f"-{s['max_dd']:>7.2f}$")

    # ── Топ-5 ────────────────────────────────────────────────────────────────
    best = ts_stats_list[:5]
    print(f"\n{'='*65}")
    print(f"  ТОП-5 ЛУЧШИХ КОМБИНАЦИЙ TS (по PnL)")
    print(f"{'='*65}")
    for rank, s in enumerate(best, 1):
        vs_v2 = s['pnl'] - v2['pnl']
        print(f"  #{rank}  Act={s['act']:.2f}%  Dist={s['dist']:.2f}%  "
              f"PnL={s['pnl']:+.2f}$  WR={s['wr']:.1f}%  vs V2: {vs_v2:+.2f}$")

    # ── Рекомендация ─────────────────────────────────────────────────────────
    best1 = best[0]
    print(f"\n{'='*65}")
    print(f"  РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ")
    print(f"{'='*65}")
    print(f"  TS Activation: {best1['act']:.2f}%")
    print(f"  TS Distance:   {best1['dist']:.2f}%")
    print(f"  Ожидаемый PnL: {best1['pnl']:+.2f}$ vs V2: {v2['pnl']:+.2f}$")
    print(f"  Улучшение:     {best1['pnl'] - v2['pnl']:+.2f}$ "
          f"({(best1['pnl'] - v2['pnl']) / max(abs(v2['pnl']), 1) * 100:+.1f}%)")
    print(f"\n  Конфигурация для config.json:")
    print(f'  "trailing_stop": {{')
    print(f'    "activation_pct": {best1["act"]},')
    print(f'    "distance_pct": {best1["dist"]}')
    print(f'  }}')
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()

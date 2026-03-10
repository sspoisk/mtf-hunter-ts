"""
MTF Hunter TS — управление PAPER позициями с Trailing Stop
"""

import uuid
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import db

logger = logging.getLogger(__name__)
TZ = timedelta(hours=2)


def now_str() -> str:
    return (datetime.utcnow() + TZ).strftime('%Y-%m-%d %H:%M:%S')


@dataclass
class Position:
    id: str
    symbol: str
    side: str           # LONG / SHORT
    entry_price: float
    current_price: float
    stop_loss: float    # оригинальный SL (до активации трейлинга)
    take_profit: float  # не используется в TS режиме (для совместимости UI)
    size_usdt: float
    sl_pct: float
    tp_pct: float
    rsi_at_entry: float
    trend_at_entry: str
    # Trailing Stop поля
    ts_active: bool  = False   # активирован ли трейлинг
    ts_peak: float   = 0.0     # лучшая цена с момента входа
    ts_sl: float     = 0.0     # текущий уровень трейлинг-стопа
    ts_activation_pct: float = 1.5   # % прибыли для активации (из конфига)
    ts_distance_pct: float   = 1.0   # отступ трейлинга от пика (%)
    # Метаданные
    opened_at: str  = ''
    closed_at: str  = ''
    close_reason: str = ''
    status: str     = 'OPEN'
    pnl_usdt: float = 0.0
    pnl_pct: float  = 0.0

    def calc_pnl(self) -> tuple[float, float]:
        if self.side == 'LONG':
            pct = (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            pct = (self.entry_price - self.current_price) / self.entry_price * 100
        pct -= 0.08  # комиссия
        return round(self.size_usdt * pct / 100, 4), round(pct, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['pnl_usdt'], d['pnl_pct'] = self.calc_pnl()
        # Показываем ts_sl или оригинальный SL в зависимости от состояния
        d['display_sl'] = self.ts_sl if self.ts_active else self.stop_loss
        return d


class MTFTrader:
    def __init__(self, config: dict):
        self.config      = config
        self.positions: Dict[str, Position] = {}
        self.closed:    List[dict]          = []
        self.lock       = threading.Lock()
        self._update_from_config(config)

    def _update_from_config(self, cfg: dict):
        t = cfg.get('trading', {})
        s = cfg.get('strategy', {})
        ts = cfg.get('trailing_stop', {})
        self.pos_size        = t.get('position_size', 100)
        self.max_pos         = t.get('max_positions', 10)
        self.max_hold_h      = s.get('max_hold_hours', 48)
        self.ts_activation   = ts.get('activation_pct', 1.5)
        self.ts_distance     = ts.get('distance_pct', 1.0)

    # ── Открытие позиции ─────────────────────────────────────────────────────

    def open_position(self, symbol: str, signal: dict) -> Optional[Position]:
        with self.lock:
            for p in self.positions.values():
                if p.symbol == symbol and p.status == 'OPEN':
                    return None
            if len(self.positions) >= self.max_pos:
                return None

            pos = Position(
                id                = str(uuid.uuid4())[:8],
                symbol            = symbol,
                side              = signal['direction'],
                entry_price       = signal['price'],
                current_price     = signal['price'],
                stop_loss         = signal['sl'],
                take_profit       = signal.get('tp', 0),
                size_usdt         = self.pos_size,
                sl_pct            = signal['sl_pct'],
                tp_pct            = signal.get('tp_pct', 0),
                rsi_at_entry      = signal['rsi'],
                trend_at_entry    = signal['trend'],
                ts_active         = False,
                ts_peak           = signal['price'],
                ts_sl             = 0.0,
                ts_activation_pct = self.ts_activation,
                ts_distance_pct   = self.ts_distance,
                opened_at         = now_str(),
                status            = 'OPEN',
            )
            self.positions[pos.id] = pos
            db.save_position_open(pos)
            logger.info(f"[OPEN] {symbol} {pos.side} @ {pos.entry_price} "
                        f"SL={pos.stop_loss} TS_act={pos.ts_activation_pct}% TS_dist={pos.ts_distance_pct}%")
            return pos

    # ── Обновление цен + проверка SL/TS/TIME ─────────────────────────────────

    def update_prices(self, prices: dict) -> list:
        """Возвращает список событий TS (для push SSE)."""
        ts_events = []
        with self.lock:
            to_close = []
            for pid, pos in self.positions.items():
                sym = pos.symbol.replace('/USDT:USDT', '/USDT').replace(':USDT', '')
                price = (prices.get(pos.symbol) or
                         prices.get(sym) or
                         prices.get(pos.symbol.split('/')[0]))
                if not price:
                    continue
                pos.current_price = price

                # TIME limit
                opened = datetime.strptime(pos.opened_at, '%Y-%m-%d %H:%M:%S')
                age_h  = (datetime.utcnow() + TZ - opened).total_seconds() / 3600

                reason = None

                if pos.side == 'LONG':
                    # Обновляем пик
                    if price > pos.ts_peak:
                        pos.ts_peak = price

                    if not pos.ts_active:
                        # До активации: проверяем оригинальный SL
                        if price <= pos.stop_loss:
                            reason = 'SL'
                        else:
                            # Проверяем активацию трейлинга
                            profit_pct = (pos.ts_peak - pos.entry_price) / pos.entry_price * 100
                            if profit_pct >= pos.ts_activation_pct:
                                pos.ts_active = True
                                pos.ts_sl = pos.ts_peak * (1 - pos.ts_distance_pct / 100)
                                db.save_position_update_ts(pos)
                                ts_events.append({'pid': pid, 'symbol': pos.symbol,
                                                  'profit_pct': round(profit_pct, 2),
                                                  'ts_sl': round(pos.ts_sl, 8)})
                                logger.info(f"[TS ACTIVATED] {pos.symbol} profit={profit_pct:.2f}% "
                                            f"ts_sl={pos.ts_sl:.6f}")
                    else:
                        # Трейлинг активен: обновляем уровень
                        new_ts_sl = pos.ts_peak * (1 - pos.ts_distance_pct / 100)
                        if new_ts_sl > pos.ts_sl:
                            pos.ts_sl = new_ts_sl
                            db.save_position_update_ts(pos)
                        # Проверяем срабатывание
                        if price <= pos.ts_sl:
                            reason = 'TS'

                else:  # SHORT
                    # Обновляем пик (минимум для SHORT)
                    if price < pos.ts_peak or pos.ts_peak == pos.entry_price:
                        pos.ts_peak = price

                    if not pos.ts_active:
                        if price >= pos.stop_loss:
                            reason = 'SL'
                        else:
                            profit_pct = (pos.entry_price - pos.ts_peak) / pos.entry_price * 100
                            if profit_pct >= pos.ts_activation_pct:
                                pos.ts_active = True
                                pos.ts_sl = pos.ts_peak * (1 + pos.ts_distance_pct / 100)
                                db.save_position_update_ts(pos)
                                ts_events.append({'pid': pid, 'symbol': pos.symbol,
                                                  'profit_pct': round(profit_pct, 2),
                                                  'ts_sl': round(pos.ts_sl, 8)})
                                logger.info(f"[TS ACTIVATED] {pos.symbol} profit={profit_pct:.2f}% "
                                            f"ts_sl={pos.ts_sl:.6f}")
                    else:
                        new_ts_sl = pos.ts_peak * (1 + pos.ts_distance_pct / 100)
                        if new_ts_sl < pos.ts_sl or pos.ts_sl == 0:
                            pos.ts_sl = new_ts_sl
                            db.save_position_update_ts(pos)
                        if price >= pos.ts_sl:
                            reason = 'TS'

                if reason is None and age_h >= self.max_hold_h:
                    reason = 'TIME'

                if reason:
                    to_close.append((pid, reason, price))

            for pid, reason, price in to_close:
                self._close(pid, reason, price)

        return ts_events

    def _close(self, pid: str, reason: str, price: float):
        pos = self.positions.pop(pid, None)
        if not pos:
            return
        pos.status        = 'CLOSED'
        pos.close_reason  = reason
        pos.closed_at     = now_str()
        pos.current_price = price
        pos.pnl_usdt, pos.pnl_pct = pos.calc_pnl()
        self.closed.append(pos.to_dict())
        db.save_position_close(pos)
        ts_info = f" [TS peak={pos.ts_peak:.6f}]" if pos.ts_active else ""
        logger.info(f"[CLOSE] {pos.symbol} {pos.side} reason={reason} "
                    f"pnl={pos.pnl_usdt:+.2f}${ts_info}")

    # ── Данные для UI ────────────────────────────────────────────────────────

    def get_open_positions(self) -> List[dict]:
        with self.lock:
            return [p.to_dict() for p in self.positions.values()]

    def get_closed_positions(self, limit: int = 50) -> List[dict]:
        with self.lock:
            return self.closed[-limit:]

    def get_stats(self) -> dict:
        with self.lock:
            closed = self.closed
            if not closed:
                return {'total': 0, 'wins': 0, 'losses': 0,
                        'wr': 0, 'total_pnl': 0, 'open': len(self.positions)}
            wins      = [t for t in closed if t['pnl_usdt'] > 0]
            losses    = [t for t in closed if t['pnl_usdt'] <= 0]
            total_pnl = sum(t['pnl_usdt'] for t in closed)
            ts_wins   = sum(1 for t in closed if t.get('close_reason') == 'TS' and t['pnl_usdt'] > 0)
            return {
                'total':     len(closed),
                'wins':      len(wins),
                'losses':    len(losses),
                'wr':        round(len(wins) / len(closed) * 100, 1),
                'total_pnl': round(total_pnl, 2),
                'open':      len(self.positions),
                'ts_wins':   ts_wins,
            }

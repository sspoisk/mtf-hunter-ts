"""
MTF Hunter TS — SQLite persistence (V2 + trailing stop fields)
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).parent / 'data' / 'mtf_hunter.db'
TZ = timedelta(hours=2)


def now_str():
    return (datetime.utcnow() + TZ).strftime('%Y-%m-%d %H:%M:%S')


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id             TEXT PRIMARY KEY,
            symbol         TEXT NOT NULL,
            side           TEXT NOT NULL,
            entry_price    REAL,
            exit_price     REAL,
            stop_loss      REAL,
            take_profit    REAL,
            size_usdt      REAL,
            sl_pct         REAL,
            tp_pct         REAL,
            rsi_at_entry   REAL,
            trend_at_entry TEXT,
            status         TEXT DEFAULT 'OPEN',
            close_reason   TEXT,
            pnl_usdt       REAL,
            pnl_pct        REAL,
            opened_at      TEXT,
            closed_at      TEXT,
            ts_active      INTEGER DEFAULT 0,
            ts_peak        REAL    DEFAULT 0,
            ts_sl          REAL    DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS signals (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT NOT NULL,
            direction TEXT,
            price     REAL,
            sl        REAL,
            tp        REAL,
            sl_pct    REAL,
            tp_pct    REAL,
            rsi       REAL,
            trend     TEXT,
            rr        REAL,
            found_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            symbol     TEXT,
            data       TEXT,
            created_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(status);
        CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_sig_found  ON signals(found_at);
        CREATE INDEX IF NOT EXISTS idx_evt_type   ON events(event_type);
        """)

    # Добавляем новые колонки если DB уже существовала (миграция)
    with get_conn() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
        for col, defn in [
            ('ts_active', 'INTEGER DEFAULT 0'),
            ('ts_peak',   'REAL DEFAULT 0'),
            ('ts_sl',     'REAL DEFAULT 0'),
        ]:
            if col not in cols:
                conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {defn}")

    logger.info(f"[DB] Initialized: {DB_PATH}")


# ── Positions ──────────────────────────────────────────────────────────────────

def save_position_open(pos):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO positions
              (id, symbol, side, entry_price, stop_loss, take_profit,
               size_usdt, sl_pct, tp_pct, rsi_at_entry, trend_at_entry,
               status, opened_at, ts_active, ts_peak, ts_sl)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pos.id, pos.symbol, pos.side, pos.entry_price,
            pos.stop_loss, pos.take_profit, pos.size_usdt,
            pos.sl_pct, pos.tp_pct, pos.rsi_at_entry, pos.trend_at_entry,
            'OPEN', pos.opened_at,
            int(pos.ts_active), pos.ts_peak, pos.ts_sl,
        ))
    log_event('position_open', pos.symbol, {
        'id': pos.id, 'side': pos.side, 'price': pos.entry_price,
        'sl': pos.stop_loss,
    })


def save_position_update_ts(pos):
    """Обновляем TS поля в реальном времени."""
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE positions SET ts_active=?, ts_peak=?, ts_sl=? WHERE id=?",
                (int(pos.ts_active), pos.ts_peak, pos.ts_sl, pos.id)
            )
    except Exception as e:
        logger.debug(f"[DB] save_position_update_ts: {e}")


def save_position_close(pos):
    pnl_usdt, pnl_pct = pos.calc_pnl()
    with get_conn() as conn:
        conn.execute("""
            UPDATE positions SET
              status=?, close_reason=?, exit_price=?,
              pnl_usdt=?, pnl_pct=?, closed_at=?,
              ts_active=?, ts_peak=?, ts_sl=?
            WHERE id=?
        """, (
            'CLOSED', pos.close_reason, pos.current_price,
            round(pnl_usdt, 4), round(pnl_pct, 3), pos.closed_at,
            int(pos.ts_active), pos.ts_peak, pos.ts_sl,
            pos.id,
        ))
    log_event('position_close', pos.symbol, {
        'id': pos.id, 'reason': pos.close_reason,
        'pnl': round(pnl_usdt, 2), 'ts_active': pos.ts_active,
    })


def load_open_positions():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY opened_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_closed_positions(limit=100, symbol=None):
    with get_conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='CLOSED' AND symbol=? "
                "ORDER BY closed_at DESC LIMIT ?", (symbol, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='CLOSED' "
                "ORDER BY closed_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_stats():
    with get_conn() as conn:
        row = conn.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins,
              SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) as losses,
              SUM(pnl_usdt) as total_pnl,
              AVG(pnl_usdt) as avg_pnl
            FROM positions WHERE status='CLOSED'
        """).fetchone()
    d = dict(row)
    total = d['total'] or 0
    wins  = d['wins']  or 0
    return {
        'total':     total,
        'wins':      wins,
        'losses':    d['losses'] or 0,
        'wr':        round(wins / total * 100, 1) if total else 0,
        'total_pnl': round(d['total_pnl'] or 0, 2),
        'avg_pnl':   round(d['avg_pnl']   or 0, 2),
    }


# ── Signals ───────────────────────────────────────────────────────────────────

def save_signals(signals: list):
    if not signals: return
    with get_conn() as conn:
        conn.executemany("""
            INSERT INTO signals
              (symbol, direction, price, sl, tp, sl_pct, tp_pct, rsi, trend, rr, found_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, [(
            s['symbol'], s['direction'], s['price'],
            s['sl'], s['tp'], s['sl_pct'], s['tp_pct'],
            s['rsi'], s['trend'], s['rr'], s['found_at'],
        ) for s in signals])


def get_signals_history(limit=200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY found_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Events ────────────────────────────────────────────────────────────────────

def log_event(event_type: str, symbol: str = None, data: dict = None):
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO events (event_type, symbol, data, created_at) VALUES (?,?,?,?)",
                (event_type, symbol, json.dumps(data or {}), now_str())
            )
    except Exception as e:
        logger.debug(f"[DB] log_event error: {e}")


def get_events(limit=100, event_type=None):
    with get_conn() as conn:
        if event_type:
            rows = conn.execute(
                "SELECT * FROM events WHERE event_type=? ORDER BY created_at DESC LIMIT ?",
                (event_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]

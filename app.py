"""
MTF Hunter TS — Trailing Stop Bot
Port: 8102

Стратегия: 4h EMA(30) тренд + 1h RSI откат → LONG/SHORT + Trailing Stop
"""

import os, json, time, logging, threading
from datetime import datetime, timedelta
from pathlib import Path

def _load_dotenv(path='.env'):
    if not os.path.exists(path): return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k, v = line.split('=', 1)
        if not os.environ.get(k.strip()): os.environ[k.strip()] = v.strip()
_load_dotenv()

import ccxt
from flask import Flask, render_template, jsonify, request, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from strategy import check_signal
from trader import MTFTrader
import db

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.FileHandler('logs/bot.log', encoding='utf-8')],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / 'config.json'

def load_config() -> dict:
    return json.loads(CFG_PATH.read_text())

cfg = load_config()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

@app.after_request
def no_cache_html(r):
    if r.content_type and 'text/html' in r.content_type:
        r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return r

# ── Global state ──────────────────────────────────────────────────────────────
trader   = MTFTrader(cfg)
exchange = None
signals: list   = []
watchlist: list = []
scan_lock       = threading.Lock()
last_scan_at    = None
sse_clients: list = []
sse_lock        = threading.Lock()

TZ = timedelta(hours=2)
def now_str():
    return (datetime.utcnow() + TZ).strftime('%Y-%m-%d %H:%M:%S')

# ── Exchange ──────────────────────────────────────────────────────────────────
def init_exchange():
    global exchange
    exchange = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    logger.info("[EXCHANGE] OKX connected")

def fetch_candles(symbol: str, timeframe: str = '1h', limit: int = 220) -> list:
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        logger.debug(f"fetch_candles {symbol}: {e}")
        return []

def fetch_price(symbol: str) -> float | None:
    try:
        t = exchange.fetch_ticker(symbol)
        return t.get('last') or t.get('close')
    except Exception:
        return None

# ── SSE push ──────────────────────────────────────────────────────────────────
def push_event(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    with sse_lock:
        dead = []
        for q in sse_clients:
            try: q.put_nowait(msg)
            except Exception: dead.append(q)
        for q in dead: sse_clients.remove(q)

# ── Blacklist helpers ─────────────────────────────────────────────────────────
def get_blacklist() -> list:
    return cfg.get('blacklist_symbols', [])

def is_blacklisted(symbol: str) -> bool:
    base = symbol.split('/')[0]
    bl = get_blacklist()
    return symbol in bl or base in bl

# ── Auto symbol discovery ─────────────────────────────────────────────────────
def get_symbols(cfg: dict) -> list:
    auto_cfg = cfg.get('auto_symbols', {})
    if not auto_cfg.get('enabled'):
        return cfg.get('symbols', [])

    top_n         = auto_cfg.get('top_n', 50)
    min_vol       = auto_cfg.get('min_volume_usdt', 5_000_000)
    exclude_bases = set(auto_cfg.get('exclude_bases', []))
    blacklist     = set(get_blacklist())

    try:
        tickers = exchange.fetch_tickers()
        pairs = []
        for sym, t in tickers.items():
            if not sym.endswith('/USDT:USDT'): continue
            base = sym.split('/')[0]
            if base in exclude_bases: continue
            if sym in blacklist or base in blacklist: continue
            vol = (t.get('quoteVolume') or 0)
            if vol == 0:
                vol = (t.get('baseVolume') or 0) * (t.get('last') or 0)
            if vol < min_vol: continue
            pairs.append((sym, vol))
        pairs.sort(key=lambda x: -x[1])
        result = [p[0] for p in pairs[:top_n]]
        logger.info(f"[AUTO] Выбрано {len(result)} пар (top_n={top_n}, min_vol={min_vol/1e6:.0f}M)")
        return result
    except Exception as e:
        logger.error(f"[AUTO] Ошибка: {e}")
        return cfg.get('symbols', [])

# ── Scanner ───────────────────────────────────────────────────────────────────
def scan_loop():
    global signals, last_scan_at
    logger.info("[SCANNER] Started")
    while True:
        try:
            _do_scan()
        except Exception as e:
            logger.error(f"[SCANNER] Error: {e}", exc_info=True)
        interval = cfg.get('strategy', {}).get('scan_interval_min', 5) * 60
        time.sleep(interval)


def _do_scan():
    global signals, watchlist, last_scan_at, cfg
    cfg = load_config()
    strategy_cfg = cfg.get('strategy', {})
    symbols = get_symbols(cfg)
    rsi_lo = strategy_cfg.get('rsi_lo', 40)
    rsi_hi = strategy_cfg.get('rsi_hi', 60)

    logger.info(f"[SCAN] Scanning {len(symbols)} symbols...")
    found = []
    wl    = []
    prices = {}

    for sym in symbols:
        candles = fetch_candles(sym, '1h', 220)
        if len(candles) < 100: continue

        sig = check_signal(candles, strategy_cfg)
        price = candles[-1][4] if candles else None
        if price: prices[sym] = price

        rsi_val = sig['rsi']   if sig else None
        trend   = sig['trend'] if sig else None
        if not sig:
            try:
                from strategy import _rsi as _s_rsi, _ema as _s_ema
                import numpy as np
                arr   = np.array(candles, dtype=float)
                close = arr[:, 4]; n = len(close)
                r_arr = _s_rsi(close, strategy_cfg.get('rsi_period', 14))
                rsi_val = round(float(r_arr[n-2]), 1) if n > 14 else None
                groups  = n // 4
                c4h     = np.array([arr[g*4:(g+1)*4,4][-1] for g in range(groups)])
                ema4    = _s_ema(c4h, strategy_cfg.get('ema_period', 30))
                idx4    = (n-2) // 4
                if idx4 < len(ema4) and not (ema4[idx4] != ema4[idx4]):
                    trend = 'UP' if c4h[idx4] > ema4[idx4] else 'DOWN'
            except Exception:
                pass

        near = None
        if rsi_val is not None and trend is not None:
            if trend == 'UP':
                dist = round(rsi_val - rsi_lo, 1)
                near = f"RSI -{dist} до LONG" if dist > 0 else "LONG READY"
            else:
                dist = round(rsi_hi - rsi_val, 1)
                near = f"RSI +{dist} до SHORT" if dist > 0 else "SHORT READY"

        wl.append({
            'symbol': sym,
            'name':   sym.split('/')[0],
            'price':  round(price, 6) if price else None,
            'rsi':    rsi_val,
            'trend':  trend,
            'near':   near,
            'signal': sig['direction'] if sig else None,
        })

        if sig:
            sig_data = {
                'symbol':    sym,
                'name':      sym.split('/')[0],
                'direction': sig['direction'],
                'price':     sig['price'],
                'sl':        sig['sl'],
                'tp':        sig['tp'],
                'sl_pct':    sig['sl_pct'],
                'tp_pct':    sig['tp_pct'],
                'rsi':       sig['rsi'],
                'trend':     sig['trend'],
                'rr':        sig['rr'],
                'found_at':  now_str(),
            }
            found.append(sig_data)
            logger.info(f"  SIGNAL: {sym} {sig['direction']} RSI={sig['rsi']:.1f}")
            if cfg.get('auto_trade', True):
                pos = trader.open_position(sym, sig)
                if pos:
                    logger.info(f"  AUTO-OPEN: {sym} {sig['direction']} id={pos.id}")
                    push_event({'type': 'position_opened', 'symbol': sym})

        time.sleep(0.15)

    with scan_lock:
        signals   = found
        watchlist = sorted(wl, key=lambda x: (
            0 if x['signal'] else
            1 if x['near'] and 'READY' in x['near'] else 2,
            float(x['rsi']) if x['rsi'] else 50
        ))
        last_scan_at = now_str()

    trader.update_prices(prices)
    if found:
        db.save_signals(found)
    db.log_event('scan', data={'symbols': len(symbols), 'signals': len(found)})
    logger.info(f"[SCAN] Done. Signals: {len(found)}")
    push_event({'type': 'scan_done', 'count': len(found), 'at': last_scan_at})


# ── Price updater ─────────────────────────────────────────────────────────────
def price_loop():
    while True:
        time.sleep(30)
        try:
            open_pos = trader.get_open_positions()
            if not open_pos: continue
            syms = list({p['symbol'] for p in open_pos})
            prices = {}
            for sym in syms:
                p = fetch_price(sym)
                if p: prices[sym] = p
                time.sleep(0.1)
            if prices:
                ts_events = trader.update_prices(prices)
                push_event({'type': 'positions_update'})
                for ev in ts_events:
                    push_event({'type': 'ts_activated', **ev})
        except Exception as e:
            logger.debug(f"[PRICES] {e}")


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    ts_cfg = cfg.get('trailing_stop', {})
    return jsonify({
        'running':      True,
        'last_scan':    last_scan_at,
        'signals':      len(signals),
        'open_pos':     len(trader.get_open_positions()),
        'mode':         cfg.get('trade_mode', 'PAPER'),
        'ts_activation': ts_cfg.get('activation_pct', 1.5),
        'ts_distance':   ts_cfg.get('distance_pct', 1.0),
    })


@app.route('/api/signals')
def api_signals():
    return jsonify(signals)


@app.route('/api/positions')
def api_positions():
    return jsonify({
        'open':   trader.get_open_positions(),
        'closed': trader.get_closed_positions(50),
        'stats':  trader.get_stats(),
    })


@app.route('/api/open', methods=['POST'])
def api_open():
    data = request.get_json() or {}
    sym  = data.get('symbol')
    sig  = data.get('signal')
    if not sym or not sig:
        return jsonify({'error': 'symbol and signal required'}), 400
    if is_blacklisted(sym):
        return jsonify({'error': f'{sym.split("/")[0]} в чёрном списке'}), 400
    pos = trader.open_position(sym, sig)
    if pos is None:
        return jsonify({'error': 'Cannot open: duplicate or max positions reached'}), 400
    push_event({'type': 'position_opened', 'symbol': sym})
    return jsonify({'ok': True, 'id': pos.id})


@app.route('/api/close/<pid>', methods=['POST'])
def api_close(pid):
    with trader.lock:
        pos = trader.positions.get(pid)
        if not pos:
            return jsonify({'error': 'not found'}), 404
        price = fetch_price(pos.symbol) or pos.current_price
        trader._close(pid, 'MANUAL', price)
    push_event({'type': 'position_closed', 'id': pid})
    return jsonify({'ok': True})


@app.route('/api/scan', methods=['POST'])
def api_scan():
    threading.Thread(target=_do_scan, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Scan started'})


@app.route('/api/config', methods=['GET'])
def api_config_get():
    return jsonify(cfg)


@app.route('/api/config', methods=['POST'])
def api_config_post():
    global cfg
    data = request.get_json() or {}
    current = load_config()
    for section in ('strategy', 'trading', 'trailing_stop', 'auto_symbols'):
        if section in data:
            current.setdefault(section, {}).update(data[section])
    for key in ('trade_mode', 'debug', 'auto_trade'):
        if key in data:
            current[key] = data[key]
    CFG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    cfg = current
    trader.config = current
    trader._update_from_config(current)
    return jsonify({'ok': True})


@app.route('/api/history')
def api_history():
    limit  = int(request.args.get('limit', 100))
    symbol = request.args.get('symbol')
    return jsonify({
        'positions': db.get_closed_positions(limit, symbol),
        'signals':   db.get_signals_history(limit),
        'stats':     db.get_stats(),
        'events':    db.get_events(50),
    })


@app.route('/api/watchlist')
def api_watchlist():
    return jsonify(watchlist)


@app.route('/api/candles')
def api_candles():
    sym  = request.args.get('symbol', '')
    tf   = request.args.get('tf', '1h')
    lim  = int(request.args.get('limit', 200))
    if not sym: return jsonify([])
    raw = fetch_candles(sym, tf, lim)
    result = [{'time': int(c[0]/1000), 'open': c[1], 'high': c[2],
               'low': c[3], 'close': c[4], 'volume': c[5]} for c in raw]
    return jsonify(result)


# ── Blacklist API ─────────────────────────────────────────────────────────────

@app.route('/api/blacklist', methods=['GET'])
def api_blacklist_get():
    return jsonify({'blacklist': get_blacklist()})


@app.route('/api/blacklist', methods=['POST'])
def api_blacklist_post():
    """Установить чёрный список. Body: {symbols: [...]}"""
    global cfg
    data = request.get_json() or {}
    symbols = data.get('symbols', [])
    # Нормализуем: принимаем и 'DASH' и 'DASH/USDT:USDT'
    normalized = [s.split('/')[0].upper() for s in symbols if s]
    current = load_config()
    current['blacklist_symbols'] = normalized
    CFG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    cfg = current
    logger.info(f"[BLACKLIST] Updated: {normalized}")
    return jsonify({'ok': True, 'blacklist': normalized})


@app.route('/api/blacklist/add', methods=['POST'])
def api_blacklist_add():
    """Добавить символ в ЧС. Body: {symbol: 'DASH'}"""
    global cfg
    data = request.get_json() or {}
    sym = data.get('symbol', '').split('/')[0].upper()
    if not sym:
        return jsonify({'error': 'symbol required'}), 400
    current = load_config()
    bl = current.get('blacklist_symbols', [])
    if sym not in bl:
        bl.append(sym)
        current['blacklist_symbols'] = bl
        CFG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
        cfg = current
    logger.info(f"[BLACKLIST] Added: {sym}")
    return jsonify({'ok': True, 'blacklist': bl})


@app.route('/api/blacklist/remove', methods=['POST'])
def api_blacklist_remove():
    """Убрать символ из ЧС. Body: {symbol: 'DASH'}"""
    global cfg
    data = request.get_json() or {}
    sym = data.get('symbol', '').split('/')[0].upper()
    current = load_config()
    bl = current.get('blacklist_symbols', [])
    bl = [s for s in bl if s != sym]
    current['blacklist_symbols'] = bl
    CFG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    cfg = current
    logger.info(f"[BLACKLIST] Removed: {sym}")
    return jsonify({'ok': True, 'blacklist': bl})


@app.route('/api/blacklist/reset', methods=['POST'])
def api_blacklist_reset():
    """Очистить чёрный список."""
    global cfg
    current = load_config()
    current['blacklist_symbols'] = []
    CFG_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    cfg = current
    logger.info("[BLACKLIST] Cleared")
    return jsonify({'ok': True, 'blacklist': []})


@app.route('/stream')
def stream():
    import queue
    # Лимит одновременных SSE соединений (защита от накопления)
    with sse_lock:
        if len(sse_clients) >= 10:
            # Выбрасываем самые старые соединения
            old = sse_clients[:3]
            for q in old:
                sse_clients.remove(q)

    q = queue.Queue(maxsize=20)
    with sse_lock:
        sse_clients.append(q)

    def generate():
        yield "data: {\"type\":\"connected\"}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except Exception:
                    yield "data: {\"type\":\"ping\"}\n\n"
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Main ──────────────────────────────────────────────────────────────────────
def restore_positions():
    from trader import Position
    rows = db.load_open_positions()
    for r in rows:
        pos = Position(
            id                = r['id'],
            symbol            = r['symbol'],
            side              = r['side'],
            entry_price       = r['entry_price'],
            current_price     = r['entry_price'],
            stop_loss         = r['stop_loss'],
            take_profit       = r.get('take_profit') or 0,
            size_usdt         = r['size_usdt'],
            sl_pct            = r.get('sl_pct') or 0,
            tp_pct            = r.get('tp_pct') or 0,
            rsi_at_entry      = r.get('rsi_at_entry') or 0,
            trend_at_entry    = r.get('trend_at_entry') or '',
            ts_active         = bool(r.get('ts_active', 0)),
            ts_peak           = r.get('ts_peak') or r['entry_price'],
            ts_sl             = r.get('ts_sl') or 0,
            ts_activation_pct = trader.ts_activation,
            ts_distance_pct   = trader.ts_distance,
            opened_at         = r.get('opened_at') or '',
            status            = 'OPEN',
        )
        trader.positions[pos.id] = pos
    if rows:
        logger.info(f"[DB] Restored {len(rows)} open positions")


def main():
    db.init_db()
    init_exchange()
    restore_positions()

    threading.Thread(target=scan_loop,  daemon=True, name='scanner').start()
    threading.Thread(target=price_loop, daemon=True, name='prices').start()

    port = cfg.get('port', 8102)
    logger.info(f"[MTF HUNTER TS] Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()

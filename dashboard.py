import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime

st.set_page_config(layout="wide", page_title="Desk Bravo - Live Terminal")

BASE_URL = "https://web-production-919a1.up.railway.app"
USERNAME  = "desk_bravo"
PASSWORD  = "bravo123"

# ──────────────────────────────────────────
# API
# ──────────────────────────────────────────
def get_token():
    try:
        r = requests.post(f"{BASE_URL}/api/auth/login",
                          json={"username": USERNAME, "password": PASSWORD}, timeout=5)
        if r.status_code == 200:
            return r.json().get("token")
    except Exception:
        pass
    return None

def fetch(path, token, method="GET", **kwargs):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        fn = requests.get if method == "GET" else requests.post
        r  = fn(f"{BASE_URL}{path}", headers=headers, timeout=5, **kwargs)
        return r.json()
    except Exception:
        return {} if "orders" not in path else []

def trades_to_ohlcv(trades, freq='30s'):
    """Convierte lista de trades en barras OHLCV resampleadas."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df['price']    = pd.to_numeric(df['price'])
    df['quantity'] = pd.to_numeric(df['quantity'])
    df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True).dt.tz_localize(None)
    df = df.sort_values('createdAt').set_index('createdAt')
    ohlcv = df['price'].resample(freq).ohlc()
    ohlcv['volume'] = df['quantity'].resample(freq).sum()
    ohlcv = ohlcv.dropna(subset=['open'])
    ohlcv = ohlcv.reset_index().rename(columns={'createdAt': 'timestamp'})
    return ohlcv

def get_all_data(token):
    # Chart: trades de la sesión actual (precio real)
    trades_raw = fetch("/api/trades?limit=500", token)
    if not isinstance(trades_raw, list):
        trades_raw = []
    df = trades_to_ohlcv(trades_raw, freq='30s')

    # Histórico solo para métricas de riesgo
    hist_raw = requests.get(f"{BASE_URL}/api/historical?limit=10000", timeout=10).json().get("bars", [])
    df_hist  = pd.DataFrame(hist_raw)
    if not df_hist.empty:
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df_hist[c] = pd.to_numeric(df_hist[c], errors='coerce')

    book    = fetch("/api/book",            token)
    status  = fetch("/api/status",          token)
    pos     = fetch("/api/positions",       token)
    pnl     = fetch("/api/pnl",             token)
    orders  = fetch("/api/orders?status=FILLED&limit=500", token)
    if not isinstance(orders, list):
        orders = []
    return df, df_hist, book, status, pos, pnl, orders

# ──────────────────────────────────────────
# INDICADORES
# ──────────────────────────────────────────
def add_indicators(df, book):
    if df.empty:
        return df, 0.0

    # Precio en vivo desde el book
    live_price = float(df['close'].iloc[-1])
    if book.get('bids') and book.get('asks'):
        best_bid = max(b['price'] for b in book['bids'])
        best_ask = min(a['price'] for a in book['asks'])
        live_price = round((best_bid + best_ask) / 2, 4)

    # Actualizar el último bar con el precio en vivo (mismo timestamp, sin proyección)
    df.iloc[-1, df.columns.get_loc('close')] = live_price
    df.iloc[-1, df.columns.get_loc('high')]  = max(float(df.iloc[-1]['high']), live_price)
    df.iloc[-1, df.columns.get_loc('low')]   = min(float(df.iloc[-1]['low']),  live_price)

    # MAs
    df['SMA10'] = df['close'].rolling(10).mean()
    df['SMA30'] = df['close'].rolling(30).mean()

    # MACD (12, 26, 9)
    ema12        = df['close'].ewm(span=12, adjust=False).mean()
    ema26        = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDh']  = df['MACD'] - df['Signal']   # histograma

    # RSI 14
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    return df, live_price

# ──────────────────────────────────────────
# MÉTRICAS DE RIESGO
# ──────────────────────────────────────────
def calc_risk_metrics(df):
    """
    Calcula métricas de riesgo sobre la serie de precios histórica.
    Barras de 1 minuto → factor de anualización = sqrt(252 * 390).
    """
    closes = df['close'].dropna()
    if len(closes) < 30:
        return {}

    returns = closes.pct_change().dropna()
    ann     = (252 * 390) ** 0.5          # ~313.5 — barras de 1 min

    # Retorno y volatilidad anualizados
    mean_ret  = returns.mean()
    std_ret   = returns.std()
    vol_ann   = std_ret * ann * 100        # en %

    # Sharpe (asumimos rf ≈ 0 para el simulador)
    sharpe = (mean_ret / std_ret * ann) if std_ret > 0 else float('nan')

    # Sortino (solo desviación a la baja)
    downside = returns[returns < 0]
    sortino_std = downside.std()
    sortino = (mean_ret / sortino_std * ann) if sortino_std > 0 else float('nan')

    # Max Drawdown sobre la curva de precios
    cum     = closes / closes.iloc[0]            # índice normalizado
    rolling_max = cum.cummax()
    dd_series   = (cum - rolling_max) / rolling_max
    max_dd  = dd_series.min() * 100              # en %

    # Calmar = retorno anualizado / |max drawdown|
    ret_ann = mean_ret * ann * 100               # en %
    calmar  = (ret_ann / abs(max_dd)) if max_dd != 0 else float('nan')

    # VaR 95% y CVaR 95% (Expected Shortfall)
    var_95  = returns.quantile(0.05) * 100       # en %
    cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100

    # Beta del precio respecto a su propia media móvil (proxy de market beta)
    # Aquí no hay benchmark externo, usamos correlación serial como proxy de momentum
    autocorr = returns.autocorr(lag=1)

    return {
        "Volatilidad anual.": f"{vol_ann:.2f}%",
        "Sharpe Ratio":       f"{sharpe:.3f}",
        "Sortino Ratio":      f"{sortino:.3f}",
        "Calmar Ratio":       f"{calmar:.3f}",
        "Max Drawdown":       f"{max_dd:.2f}%",
        "VaR 95% (1 min)":   f"{var_95:.4f}%",
        "CVaR 95%":          f"{cvar_95:.4f}%",
        "Retorno anual.":     f"{ret_ann:.2f}%",
        "Autocorr. ret.":     f"{autocorr:.4f}",
        "N barras":           str(len(closes)),
    }, dd_series, closes

# ──────────────────────────────────────────
# WIN RATE  (FIFO: BUY → SELL round trips)
# ──────────────────────────────────────────
def calc_win_rate(orders):
    filled = sorted(
        [o for o in orders if o.get('status') == 'FILLED'],
        key=lambda o: o['createdAt']
    )
    buy_queue = []
    wins = losses = 0

    for o in filled:
        price = float(o['price'])
        qty   = int(o['quantity'])
        if o['side'] == 'BUY':
            buy_queue.append({'price': price, 'qty': qty})
        elif o['side'] == 'SELL':
            remaining = qty
            while remaining > 0 and buy_queue:
                b = buy_queue[0]
                matched = min(b['qty'], remaining)
                if price > b['price']:
                    wins += 1
                else:
                    losses += 1
                b['qty']  -= matched
                remaining -= matched
                if b['qty'] == 0:
                    buy_queue.pop(0)

    total = wins + losses
    rate  = wins / total if total else None
    return wins, losses, total, rate

# ──────────────────────────────────────────
# CACHÉ DE TOKEN
# ──────────────────────────────────────────
if 'token' not in st.session_state:
    st.session_state['token'] = get_token()

token = st.session_state['token']
if not token:
    st.error("Error de autenticación.")
    st.stop()

# ──────────────────────────────────────────
# DATOS
# ──────────────────────────────────────────
df_raw, df_hist, book, status, pos, pnl_data, orders = get_all_data(token)

# Re-auth si token expiró
if not pos and not pnl_data:
    st.session_state['token'] = get_token()
    token = st.session_state['token']
    df_raw, df_hist, book, status, pos, pnl_data, orders = get_all_data(token)

df, current_price = add_indicators(df_raw, book)
wins, losses, total_trades, win_rate = calc_win_rate(orders)
risk_result = calc_risk_metrics(df_hist)
risk_metrics, dd_series, closes_series = risk_result if risk_result else ({}, None, None)

# ──────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────
session_status = status.get('status', 'OFFLINE')
icon = '🟢' if session_status == 'LIVE' else ('🟡' if session_status == 'PAUSED' else '🔴')
time_rem = status.get('timeRemaining', '—')
st.title(f"📈 Desk Bravo  |  {icon} {session_status}  |  ⏱ {time_rem}")

# ──────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────
total_pnl      = pnl_data.get('totalPnl', 0)
realized_pnl   = pnl_data.get('realizedPnl', 0)
unrealized_pnl = pnl_data.get('unrealizedPnl', 0)
quantity       = pos.get('quantity', 0)
avg_entry      = pos.get('avgEntryPrice', 0) or 0
avail_cash     = pos.get('availableCash', 0)

spread = 0.0
if book.get('bids') and book.get('asks'):
    spread = round(
        min(a['price'] for a in book['asks']) - max(b['price'] for b in book['bids']), 2
    )

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total PnL",     f"${total_pnl:,.2f}",
          delta=f"R: ${realized_pnl:,.2f}")
k2.metric("PnL No Real.",  f"${unrealized_pnl:,.2f}")
k3.metric("Cash",          f"${avail_cash:,.0f}")
k4.metric("Posición",      f"{quantity} acc",
          delta=f"Entry ${avg_entry:.2f}" if avg_entry else None)
k5.metric("Precio (Mid)",  f"${current_price:.2f}",
          delta=f"Spread ${spread:.2f}" )
wr_label = f"{win_rate*100:.0f}%  ({wins}W / {losses}L)" if win_rate is not None else "Sin trades"
k6.metric("Win Rate", wr_label)

st.divider()

# ──────────────────────────────────────────
# GRÁFICOS
# ──────────────────────────────────────────
col_chart, col_book = st.columns([3, 1])

with col_chart:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.80, 0.20],
        subplot_titles=["Precio  (SMA 10 / SMA 30)", "Volumen"]
    )

    # ── Fila 1: Velas + MAs ──
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'],   close=df['close'],
        name='Precio',
        increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
        decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['SMA10'],
        mode='lines', line=dict(color='#00e5ff', width=1.4),
        name='SMA 10'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['SMA30'],
        mode='lines', line=dict(color='#ff4081', width=1.4),
        name='SMA 30'
    ), row=1, col=1)

    # ── Fila 2: Volumen ──
    colors_vol = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['timestamp'], y=df['volume'],
        marker_color=colors_vol, name='Volumen', showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=860,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=10, l=60, r=20),
        legend=dict(orientation='h', y=1.03, x=0, font=dict(size=11)),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    # Fijar el rango X al último timestamp real (sin proyección)
    fig.update_xaxes(range=[df['timestamp'].iloc[0], df['timestamp'].iloc[-1]])

    st.plotly_chart(fig, use_container_width=True)

with col_book:
    # ── Order Book ──
    bids_df = pd.DataFrame(book.get('bids', []))
    asks_df = pd.DataFrame(book.get('asks', []))

    fig_book = go.Figure()
    if not bids_df.empty:
        bids_sorted = bids_df.sort_values('price', ascending=True)
        fig_book.add_trace(go.Bar(
            x=bids_sorted['totalQuantity'],
            y=bids_sorted['price'].astype(str),
            orientation='h',
            name='Bids',
            marker_color='rgba(38,166,154,0.7)'
        ))
    if not asks_df.empty:
        asks_sorted = asks_df.sort_values('price', ascending=True)
        fig_book.add_trace(go.Bar(
            x=asks_sorted['totalQuantity'],
            y=asks_sorted['price'].astype(str),
            orientation='h',
            name='Asks',
            marker_color='rgba(239,83,80,0.7)'
        ))

    fig_book.update_layout(
        template='plotly_dark',
        height=350,
        title='Order Book',
        barmode='overlay',
        margin=dict(t=50, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation='h', y=1.08),
        yaxis=dict(title='Precio', tickfont=dict(size=10)),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117'
    )
    st.plotly_chart(fig_book, use_container_width=True)

    # ── Resumen de cuenta ──
    st.markdown("**Resumen de cuenta**")
    data_rows = {
        "Cantidad":       f"{quantity} acc",
        "Entrada prom.":  f"${avg_entry:.2f}" if avg_entry else "—",
        "PnL Realizado":  f"${realized_pnl:,.2f}",
        "PnL No Real.":   f"${unrealized_pnl:,.2f}",
        "Cash disp.":     f"${avail_cash:,.0f}",
        "Trades totales": str(total_trades),
        "Wins / Losses":  f"{wins} / {losses}",
        "Win Rate":       f"{win_rate*100:.1f}%" if win_rate is not None else "—",
        "MM Uptime":      f"{pos.get('mmUptimePct', 0):.1f}%",
    }
    st.dataframe(
        pd.DataFrame(data_rows.items(), columns=["Campo", "Valor"]),
        use_container_width=True, hide_index=True
    )

# ──────────────────────────────────────────
# SECCIÓN DE RIESGO Y VOLATILIDAD
# ──────────────────────────────────────────
st.divider()
st.subheader("Métricas de Riesgo & Volatilidad")

if risk_metrics:
    # ── KPIs de riesgo ──
    r1, r2, r3, r4, r5, r6, r7, r8 = st.columns(8)
    cols_risk = [r1, r2, r3, r4, r5, r6, r7, r8]
    keys_order = [
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Max Drawdown", "Volatilidad anual.", "VaR 95% (1 min)",
        "CVaR 95%", "Retorno anual."
    ]
    for col, key in zip(cols_risk, keys_order):
        val = risk_metrics[key]
        # Color negativo si el valor es negativo
        num = float(val.replace('%','').replace('nan','0')) if val != 'nan' else 0
        delta_color = "inverse" if key == "Max Drawdown" else "normal"
        col.metric(key, val)

    # ── Gráficos de riesgo ──
    gc1, gc2 = st.columns(2)

    with gc1:
        # Drawdown Chart
        if dd_series is not None:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=df_raw['timestamp'],
                y=dd_series.values * 100,
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(239,83,80,0.15)',
                line=dict(color='#ef5350', width=1),
                name='Drawdown %'
            ))
            fig_dd.update_layout(
                template='plotly_dark',
                title='Drawdown histórico (%)',
                height=280,
                margin=dict(t=40, b=10, l=10, r=10),
                yaxis=dict(ticksuffix='%'),
                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117'
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    with gc2:
        # Distribución de retornos + VaR
        if closes_series is not None:
            rets = closes_series.pct_change().dropna() * 100
            var_val = float(risk_metrics["VaR 95% (1 min)"].replace('%',''))
            cvar_val = float(risk_metrics["CVaR 95%"].replace('%',''))

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=rets,
                nbinsx=80,
                marker_color='rgba(0,229,255,0.5)',
                name='Retornos (%)',
            ))
            fig_dist.add_vline(
                x=var_val,
                line_dash='dash', line_color='#ff4081',
                annotation_text=f'VaR 95%: {var_val:.4f}%',
                annotation_position='top right'
            )
            fig_dist.add_vline(
                x=cvar_val,
                line_dash='dot', line_color='#ff6d00',
                annotation_text=f'CVaR: {cvar_val:.4f}%',
                annotation_position='top left'
            )
            fig_dist.update_layout(
                template='plotly_dark',
                title='Distribución de retornos (1 min)',
                height=280,
                margin=dict(t=40, b=10, l=10, r=10),
                xaxis=dict(ticksuffix='%'),
                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # ── Tabla completa ──
    with st.expander("Ver todas las métricas"):
        st.dataframe(
            pd.DataFrame(risk_metrics.items(), columns=["Métrica", "Valor"]),
            use_container_width=True, hide_index=True
        )
else:
    st.info("Datos insuficientes para calcular métricas de riesgo.")

# ──────────────────────────────────────────
# AUTO-REFRESH cada 3 segundos
# ──────────────────────────────────────────
time.sleep(3)
st.rerun()

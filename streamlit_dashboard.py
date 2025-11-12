#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Streamlit - Agent 7 Training Monitor
Affiche les métriques en temps réel depuis training_stats.json
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Agent 7 - Training Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constante de normalisation PnL
PNL_MULTIPLIER = 100

def normalize_pnl(pnl):
    """Normalise le PnL en divisant par 100 (bug environnement)"""
    return pnl / PNL_MULTIPLIER

def load_data():
    """Charge les données depuis training_stats.json"""
    json_path = Path(__file__).parent / "training_stats.json"

    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"❌ Erreur chargement JSON: {e}")
        return None

def calculate_metrics(data):
    """Calcule toutes les métriques depuis les données"""
    # Vérification robuste de la structure des données
    if not data:
        return None

    # Le JSON est un tableau de checkpoints, pas un objet avec 'history'
    if not isinstance(data, list):
        st.error(f"❌ Format de données invalide. Type attendu: list, Type reçu: {type(data)}")
        return None

    if len(data) == 0:
        st.warning("⚠️ Aucun checkpoint - training pas encore démarré")
        return None

    # Le dernier checkpoint
    latest = data[-1]

    # Récupération des trades uniques
    all_trades = []
    seen_trades = set()

    for checkpoint in data:
        for trade in checkpoint.get('trades', []):
            # Clé unique pour déduplication
            key = (
                trade.get('entry_price', 0),
                trade.get('exit_price', 0),
                trade.get('size', 0),
                trade.get('pnl', 0)
            )
            if key not in seen_trades:
                seen_trades.add(key)
                all_trades.append(trade)

    # Récupération des statistiques précalculées (DÉJÀ NORMALISÉES dans le JSON)
    trading_stats = latest.get('trading_stats', {})
    institutional_metrics = latest.get('institutional_metrics', {})

    # Total PnL depuis equity
    total_pnl = latest['equity'] - 100000
    pnl_method = "✅ Equity delta"

    # Récupération des trades avec normalisation PnL
    winning_trades = [t for t in all_trades if normalize_pnl(t.get('pnl', 0)) > 0]
    losing_trades = [t for t in all_trades if normalize_pnl(t.get('pnl', 0)) < 0]

    # Métriques de trading (du JSON, PAS de recalcul)
    total_trades = trading_stats.get('total_trades', len(all_trades))
    win_rate = trading_stats.get('win_rate', 0)
    profit_factor = trading_stats.get('profit_factor', 0)

    # Avg Win/Loss depuis JSON (DOIVENT être normalisés - multipliés par 100 dans le JSON)
    avg_win = trading_stats.get('avg_win', 0) / PNL_MULTIPLIER
    avg_loss = trading_stats.get('avg_loss', 0) / PNL_MULTIPLIER

    # Max Win/Loss calculés depuis les trades (normalisation appliquée)
    max_win = max([normalize_pnl(t['pnl']) for t in winning_trades], default=0)
    max_loss = min([normalize_pnl(t['pnl']) for t in losing_trades], default=0)

    # Max RR (meilleur trade / perte moyenne)
    max_rr = max_win / abs(avg_loss) if avg_loss < 0 else 0

    # ROI depuis JSON (déjà en %)
    roi = latest.get('roi_pct', 0)

    # Métriques institutionnelles
    sharpe = institutional_metrics.get('sharpe_ratio', 0)
    sortino = institutional_metrics.get('sortino_ratio', 0)
    calmar = institutional_metrics.get('calmar_ratio', 0)
    var_95 = institutional_metrics.get('var_95', 0) * 100  # Convertir en %
    cvar_95 = institutional_metrics.get('cvar_95', 0) * 100  # Convertir en %

    # Max Drawdown (déjà en pourcentage dans le JSON, NE PAS multiplier par 100)
    max_dd_pct = latest.get('max_drawdown_pct', 0)

    # Calcul Max DD en dollars (basé sur l'equity actuelle)
    current_equity = latest.get('equity', 100000)
    if max_dd_pct > 0:
        # Le peak equity est l'equity actuelle divisée par (1 - DD%)
        peak_equity = current_equity / (1 - max_dd_pct / 100)
        max_dd_dollar = peak_equity - current_equity
    else:
        max_dd_dollar = 0

    return {
        'timesteps': latest.get('timesteps', 0),
        'equity': latest['equity'],
        'total_pnl': total_pnl,
        'pnl_method': pnl_method,
        'roi': roi,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_dd_pct': max_dd_pct,
        'max_dd_dollar': max_dd_dollar,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        'max_rr': max_rr,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'all_trades': all_trades,
        'history': data  # data est déjà le tableau complet de checkpoints
    }

def create_equity_curve(history):
    """Crée la courbe d'équité"""
    timesteps = [h.get('timesteps', 0) for h in history]
    equity = [h.get('equity', 100000) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=equity,
        mode='lines',
        name='Equity',
        line=dict(color='#00D9FF', width=2)
    ))

    fig.add_hline(y=100000, line_dash="dash", line_color="gray", annotation_text="Initial Capital")

    fig.update_layout(
        title="Courbe d'Équité",
        xaxis_title="Timesteps",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )

    return fig

def create_drawdown_chart(history):
    """Crée le graphique de drawdown"""
    timesteps = [h.get('timesteps', 0) for h in history]
    # max_drawdown_pct est déjà en pourcentage dans le JSON, NE PAS multiplier
    dd_pct = [h.get('max_drawdown_pct', 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=dd_pct,
        mode='lines',
        name='Max DD %',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy'
    ))

    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="FTMO Limit (10%)")

    fig.update_layout(
        title="Maximum Drawdown",
        xaxis_title="Timesteps",
        yaxis_title="Max DD (%)",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )

    return fig

def create_sharpe_chart(history):
    """Crée le graphique du Sharpe Ratio"""
    timesteps = [h.get('timesteps', 0) for h in history]
    # Sharpe Ratio est dans institutional_metrics, pas directement dans le checkpoint
    sharpe = [h.get('institutional_metrics', {}).get('sharpe_ratio', 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=sharpe,
        mode='lines',
        name='Sharpe Ratio',
        line=dict(color='#51CF66', width=2)
    ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="yellow", annotation_text="Target (1.0)")
    fig.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Excellent (1.5)")

    fig.update_layout(
        title="Sharpe Ratio",
        xaxis_title="Timesteps",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )

    return fig

def create_pnl_distribution(trades):
    """Crée l'histogramme de distribution des PnL"""
    pnls = [normalize_pnl(t.get('pnl', 0)) for t in trades]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnls,
        nbinsx=50,
        marker=dict(
            color=pnls,
            colorscale='RdYlGn',
            cmin=-max(abs(min(pnls, default=0)), abs(max(pnls, default=0))),
            cmax=max(abs(min(pnls, default=0)), abs(max(pnls, default=0)))
        )
    ))

    fig.update_layout(
        title="Distribution des PnL par Trade",
        xaxis_title="PnL ($)",
        yaxis_title="Nombre de trades",
        template='plotly_dark',
        height=400
    )

    return fig

# Interface principale
st.title("📊 Agent 7 - Training Dashboard")
st.markdown("**Monitoring en temps réel - PPO Momentum Trader H1**")

# Sidebar avec contrôles
with st.sidebar:
    st.header("⚙️ Contrôles")

    auto_refresh = st.checkbox("Auto-refresh", value=True)

    if auto_refresh:
        refresh_interval = st.slider("Intervalle (secondes)", 5, 60, 10)

    st.markdown("---")
    st.markdown("### 📝 Informations")
    st.markdown(f"**Dernière mise à jour**: {datetime.now().strftime('%H:%M:%S')}")

    if st.button("🔄 Rafraîchir maintenant"):
        st.rerun()

# Chargement des données
data = load_data()

if data is None:
    st.warning("⚠️ Fichier training_stats.json non trouvé. Lancez d'abord le training.")
    st.info("📍 Chemin attendu: `C:\\Users\\lbye3\\Desktop\\GoldRL\\AGENT\\AGENT 7\\ENTRAINEMENT\\training_stats.json`")
    st.stop()

# Calcul des métriques
metrics = calculate_metrics(data)

if metrics is None:
    st.error("❌ Impossible de calculer les métriques (données vides)")
    st.stop()

# === SECTION 1: OVERVIEW ===
st.header("💰 Vue d'ensemble")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Timesteps",
        value=f"{metrics['timesteps']:,}",
        delta=f"{metrics['timesteps']/1_500_000*100:.1f}% de 1.5M"
    )

with col2:
    st.metric(
        label="Equity",
        value=f"${metrics['equity']:,.2f}",
        delta=f"${metrics['total_pnl']:,.2f}"
    )

with col3:
    st.metric(
        label="ROI",
        value=f"{metrics['roi']:.2f}%",
        delta="✅ Profitable" if metrics['roi'] > 0 else "❌ Perte"
    )

with col4:
    st.metric(
        label="Total Trades",
        value=f"{metrics['total_trades']:,}"
    )

st.info(f"**Méthode calcul Total PnL**: {metrics['pnl_method']}")

# === SECTION 2: PERFORMANCE ===
st.header("📈 Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Win Rate",
        value=f"{metrics['win_rate']:.1f}%",
        delta="✅ Bon" if metrics['win_rate'] > 50 else "⚠️ Faible"
    )

with col2:
    st.metric(
        label="Profit Factor",
        value=f"{metrics['profit_factor']:.2f}",
        delta="✅ Excellent" if metrics['profit_factor'] > 1.5 else "⚠️ Moyen"
    )

with col3:
    sharpe_value = metrics['sharpe']
    if sharpe_value > 1.5:
        sharpe_delta = "✅ Excellent"
    elif sharpe_value > 1.0:
        sharpe_delta = "✅ Bon"
    elif sharpe_value > 0:
        sharpe_delta = "⚠️ Faible"
    else:
        sharpe_delta = "❌ Négatif"

    st.metric(
        label="Sharpe Ratio",
        value=f"{sharpe_value:.2f}",
        delta=sharpe_delta
    )

with col4:
    st.metric(
        label="Max RR",
        value=f"{metrics['max_rr']:.2f}R"
    )

# === SECTION 3: RISK ===
st.header("⚠️ Risk Management")

col1, col2, col3, col4 = st.columns(4)

with col1:
    dd_status = "✅ OK" if metrics['max_dd_pct'] < 10 else "🚨 FTMO VIOLATION"
    st.metric(
        label="Max DD (%)",
        value=f"{metrics['max_dd_pct']:.2f}%",
        delta=dd_status
    )

with col2:
    st.metric(
        label="Max DD ($)",
        value=f"${metrics['max_dd_dollar']:,.2f}"
    )

with col3:
    st.metric(
        label="Avg Win",
        value=f"${metrics['avg_win']:.2f}"
    )

with col4:
    st.metric(
        label="Avg Loss",
        value=f"${metrics['avg_loss']:.2f}"
    )

# === SECTION 3.5: MÉTRIQUES INSTITUTIONNELLES ===
st.header("🏛️ Métriques Institutionnelles")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Sortino Ratio",
        value=f"{metrics['sortino']:.2f}",
        delta="✅ Bon" if metrics['sortino'] > 1.5 else "⚠️ Faible"
    )

with col2:
    st.metric(
        label="Calmar Ratio",
        value=f"{metrics['calmar']:.2f}",
        delta="✅ Bon" if metrics['calmar'] > 1.0 else "⚠️ Faible"
    )

with col3:
    st.metric(
        label="VaR 95%",
        value=f"{metrics['var_95']:.2f}%",
        delta="✅ OK" if metrics['var_95'] > -2.0 else "⚠️ Élevé"
    )

with col4:
    st.metric(
        label="CVaR 95%",
        value=f"{metrics['cvar_95']:.2f}%",
        delta="✅ OK" if metrics['cvar_95'] > -3.0 else "⚠️ Élevé"
    )

# === SECTION 4: GRAPHIQUES ===
st.header("📊 Graphiques")

# Ligne 1: Equity + Drawdown
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_equity_curve(metrics['history']), use_container_width=True)

with col2:
    st.plotly_chart(create_drawdown_chart(metrics['history']), use_container_width=True)

# Ligne 2: Sharpe + Distribution PnL
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(create_sharpe_chart(metrics['history']), use_container_width=True)

with col2:
    st.plotly_chart(create_pnl_distribution(metrics['all_trades']), use_container_width=True)

# === SECTION 5: TRADES DÉTAILS ===
st.header("🎯 Top Trades")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Top 10 Meilleurs Trades")
    best_trades = sorted(metrics['all_trades'], key=lambda t: normalize_pnl(t.get('pnl', 0)), reverse=True)[:10]

    best_df = pd.DataFrame([
        {
            'Entry': f"${t.get('entry_price', 0):.2f}",
            'Exit': f"${t.get('exit_price', 0):.2f}",
            'Size': t.get('size', 0),
            'PnL': f"${normalize_pnl(t.get('pnl', 0)):.2f}"
        }
        for t in best_trades
    ])
    st.dataframe(best_df, use_container_width=True)

with col2:
    st.subheader("❌ Top 10 Pires Trades")
    worst_trades = sorted(metrics['all_trades'], key=lambda t: normalize_pnl(t.get('pnl', 0)))[:10]

    worst_df = pd.DataFrame([
        {
            'Entry': f"${t.get('entry_price', 0):.2f}",
            'Exit': f"${t.get('exit_price', 0):.2f}",
            'Size': t.get('size', 0),
            'PnL': f"${normalize_pnl(t.get('pnl', 0)):.2f}"
        }
        for t in worst_trades
    ])
    st.dataframe(worst_df, use_container_width=True)

# === SECTION 6: STATISTIQUES DÉTAILLÉES ===
with st.expander("📊 Statistiques Détaillées"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Trades")
        st.markdown(f"- **Total**: {metrics['total_trades']:,}")
        st.markdown(f"- **Gagnants**: {metrics['winning_trades']:,} ({metrics['win_rate']:.1f}%)")
        st.markdown(f"- **Perdants**: {metrics['losing_trades']:,} ({100-metrics['win_rate']:.1f}%)")

    with col2:
        st.markdown("### PnL Extremes")
        st.markdown(f"- **Max Gain**: ${metrics['max_win']:.2f}")
        st.markdown(f"- **Max Perte**: ${metrics['max_loss']:.2f}")
        st.markdown(f"- **Total PnL**: ${metrics['total_pnl']:.2f}")

    with col3:
        st.markdown("### Risk Metrics")
        st.markdown(f"- **Sharpe**: {metrics['sharpe']:.2f}")
        st.markdown(f"- **Max DD**: {metrics['max_dd_pct']:.2f}%")
        st.markdown(f"- **Profit Factor**: {metrics['profit_factor']:.2f}")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

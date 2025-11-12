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

def load_top_features():
    """Charge les top 100 features depuis le fichier de configuration"""
    # Plusieurs chemins possibles pour trouver le fichier
    possible_paths = [
        Path(__file__).parent / "top100_features_agent7.txt",
        Path(__file__).parent.parent.parent.parent / "output" / "feature_selection" / "top100_features_agent7.txt",
        Path("C:/Users/lbye3/Desktop/GoldRL/output/feature_selection/top100_features_agent7.txt")
    ]

    for features_path in possible_paths:
        if features_path.exists():
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    # Filtrer les lignes de commentaires et lignes vides
                    features = [line.strip() for line in f.readlines()
                               if line.strip() and not line.strip().startswith('#')]
                return features
            except Exception as e:
                st.warning(f"⚠️ Erreur lecture features: {e}")
                return None

    return None

def calculate_streaks(trades):
    """Calcule les séquences (streaks) de gains/pertes consécutifs"""
    if not trades:
        return {
            'max_winning_streak': 0,
            'max_losing_streak': 0,
            'current_streak': 0,
            'current_streak_type': 'N/A'
        }

    max_win_streak = 0
    max_lose_streak = 0
    current_streak = 0
    current_type = None

    for trade in trades:
        pnl = normalize_pnl(trade.get('pnl', 0))

        if pnl > 0:  # Gain
            if current_type == 'win':
                current_streak += 1
            else:
                current_streak = 1
                current_type = 'win'
            max_win_streak = max(max_win_streak, current_streak)
        elif pnl < 0:  # Perte
            if current_type == 'loss':
                current_streak += 1
            else:
                current_streak = 1
                current_type = 'loss'
            max_lose_streak = max(max_lose_streak, current_streak)
        else:  # Break-even
            current_streak = 0
            current_type = None

    return {
        'max_winning_streak': max_win_streak,
        'max_losing_streak': max_lose_streak,
        'current_streak': abs(current_streak),
        'current_streak_type': '✅ Gains' if current_type == 'win' else ('❌ Pertes' if current_type == 'loss' else 'N/A')
    }

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

    # Max RR (meilleur gain / pire perte) - Ratio Risk/Reward réel
    # max_loss est négatif, donc on prend sa valeur absolue
    max_rr = max_win / abs(max_loss) if max_loss < 0 else 0

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

    # Calcul des streaks (séquences)
    streaks = calculate_streaks(all_trades)

    # Expectancy (gain moyen par trade)
    expectancy = (avg_win * (win_rate / 100)) - (abs(avg_loss) * ((100 - win_rate) / 100))

    # Recovery Factor (Total Profit / Max DD)
    recovery_factor = total_pnl / max_dd_dollar if max_dd_dollar > 0 else 0

    # Avg RR (Risk/Reward moyen)
    avg_rr = avg_win / abs(avg_loss) if avg_loss < 0 else 0

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
        'avg_rr': avg_rr,
        'expectancy': expectancy,
        'recovery_factor': recovery_factor,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'max_winning_streak': streaks['max_winning_streak'],
        'max_losing_streak': streaks['max_losing_streak'],
        'current_streak': streaks['current_streak'],
        'current_streak_type': streaks['current_streak_type'],
        'all_trades': all_trades,
        'history': data  # data est déjà le tableau complet de checkpoints
    }

def create_equity_curve(history):
    """Crée la courbe d'équité avec Balance (réalisé) et Equity (avec positions flottantes)"""
    # Trier les données par timesteps pour assurer une courbe propre
    sorted_history = sorted(history, key=lambda h: h.get('timesteps', 0))

    timesteps = [h.get('timesteps', 0) for h in sorted_history]
    balance = [h.get('balance', 100000) for h in sorted_history]
    equity = [h.get('equity', 100000) for h in sorted_history]

    fig = go.Figure()

    # Balance (positions fermées) - TRACER EN PREMIER (dessous si superposition)
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=balance,
        mode='lines+markers',
        name='✅ Balance (Réalisé)',
        line=dict(color='#00D9FF', width=4, dash='solid'),  # Cyan - trait PLEIN et PLUS ÉPAIS
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='<b>Timestep</b>: %{x:,}<br><b>Balance</b>: $%{y:,.2f}<br><i>(Positions fermées seulement)</i><extra></extra>',
        opacity=1.0
    ))

    # Equity (avec positions flottantes) - TRACER EN SECOND (dessus si superposition)
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=equity,
        mode='lines+markers',
        name='💰 Equity (Total)',
        line=dict(color='#00FF00', width=3, dash='dot'),  # Vert FLUO - trait POINTILLÉ pour distinction
        marker=dict(size=5, symbol='circle'),
        hovertemplate='<b>Timestep</b>: %{x:,}<br><b>Equity</b>: $%{y:,.2f}<br><i>(Balance + positions ouvertes)</i><extra></extra>',
        opacity=0.9  # Légère transparence pour voir Balance dessous
    ))

    fig.add_hline(y=100000, line_dash="dash", line_color="gray", annotation_text="Initial Capital ($100,000)", line_width=2)

    fig.update_layout(
        title={
            'text': "Courbe d'Équité - Balance Réalisée vs Equity Totale",
            'font': {'size': 18, 'color': 'white'}
        },
        xaxis_title="Timesteps",
        yaxis_title="Capital ($)",
        hovermode='closest',
        template='plotly_dark',
        height=450,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.15,  # AU-DESSUS du graphique (hors de la zone de tracé)
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=2,
            font=dict(size=13, color='white')
        )
    )

    return fig

def create_drawdown_chart(history):
    """Crée le graphique de drawdown (calculé depuis le peak equity, pas $100K initial)"""
    # Trier les données par timesteps
    sorted_history = sorted(history, key=lambda h: h.get('timesteps', 0))

    timesteps = [h.get('timesteps', 0) for h in sorted_history]
    # max_drawdown_pct est déjà en pourcentage dans le JSON, NE PAS multiplier
    dd_pct = [h.get('max_drawdown_pct', 0) for h in sorted_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=dd_pct,
        mode='lines+markers',
        name='Max DD %',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)',
        hovertemplate='<b>Timestep</b>: %{x:,}<br><b>Max DD</b>: %{y:.2f}%<br><i>(depuis le peak equity)</i><extra></extra>'
    ))

    fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="FTMO Limit (10%)", line_width=2)

    fig.update_layout(
        title="Maximum Drawdown (depuis Peak Equity)",
        xaxis_title="Timesteps",
        yaxis_title="Max DD (%)",
        hovermode='x unified',
        template='plotly_dark',
        height=450,
        showlegend=True,
        annotations=[
            dict(
                text="<i>⚠️ DD = (Peak - Current) / Peak, PAS depuis $100K initial</i>",
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(size=10, color='#FFD700')
            )
        ]
    )

    return fig

def create_sharpe_chart(history):
    """Crée le graphique du Sharpe Ratio"""
    # Trier les données par timesteps
    sorted_history = sorted(history, key=lambda h: h.get('timesteps', 0))

    timesteps = [h.get('timesteps', 0) for h in sorted_history]
    # Sharpe Ratio est dans institutional_metrics, pas directement dans le checkpoint
    sharpe = [h.get('institutional_metrics', {}).get('sharpe_ratio', 0) for h in sorted_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timesteps,
        y=sharpe,
        mode='lines+markers',
        name='Sharpe Ratio',
        line=dict(color='#51CF66', width=3),
        marker=dict(size=4)
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Zero", line_width=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="yellow", annotation_text="Target (1.0)", line_width=2)
    fig.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Excellent (1.5)", line_width=2)

    fig.update_layout(
        title="Sharpe Ratio Evolution",
        xaxis_title="Timesteps",
        yaxis_title="Sharpe Ratio",
        hovermode='x unified',
        template='plotly_dark',
        height=450,
        showlegend=True
    )

    return fig

def create_pnl_distribution(trades):
    """Crée l'histogramme de distribution des PnL (filtre les trades < $0.50 pour clarté)"""
    # Normaliser tous les PnL
    all_pnls = [normalize_pnl(t.get('pnl', 0)) for t in trades]

    # Filtrer les trades avec PnL insignifiant (< $0.50) - bruit d'exploration RL
    MIN_PNL_THRESHOLD = 0.50
    pnls = [p for p in all_pnls if abs(p) >= MIN_PNL_THRESHOLD]

    # Séparer gains et pertes pour coloration distincte
    gains = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    fig = go.Figure()

    # Histogram des pertes (ROUGE)
    if losses:
        fig.add_trace(go.Histogram(
            x=losses,
            name='❌ Pertes',
            marker=dict(color='#FF4444', opacity=0.7),
            nbinsx=30,
            hovertemplate='<b>PnL</b>: $%{x:.2f}<br><b>Trades</b>: %{y}<extra></extra>'
        ))

    # Histogram des gains (VERT)
    if gains:
        fig.add_trace(go.Histogram(
            x=gains,
            name='✅ Gains',
            marker=dict(color='#00FF7F', opacity=0.7),
            nbinsx=30,
            hovertemplate='<b>PnL</b>: $%{x:.2f}<br><b>Trades</b>: %{y}<extra></extra>'
        ))

    # Ligne verticale à 0
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=2, annotation_text="Break-even")

    fig.update_layout(
        title=f"Distribution des PnL par Trade (filtre > ${MIN_PNL_THRESHOLD:.2f})",
        xaxis_title="PnL ($)",
        yaxis_title="Nombre de trades",
        template='plotly_dark',
        height=400,
        barmode='overlay',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"<i>Total: {len(pnls)} trades (exclus {len(all_pnls) - len(pnls)} trades < ${MIN_PNL_THRESHOLD:.2f})</i>",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
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
    # Calcul du peak equity pour l'afficher
    current_equity = metrics['equity']
    if metrics['max_dd_pct'] > 0:
        peak_equity = current_equity / (1 - metrics['max_dd_pct'] / 100)
    else:
        peak_equity = current_equity

    dd_status = "✅ FTMO OK" if metrics['max_dd_pct'] < 10 else "🚨 FTMO VIOLATION"
    st.metric(
        label=f"Max DD % (Peak: ${peak_equity:,.0f})",
        value=f"{metrics['max_dd_pct']:.2f}%",
        delta=dd_status,
        help="⚠️ DD = (Peak - Current) / Peak * 100. Peak = point le plus haut atteint, PAS le capital initial $100K"
    )

with col2:
    st.metric(
        label="Max DD ($)",
        value=f"${metrics['max_dd_dollar']:,.2f}",
        help=f"Perte max en $ depuis le peak (${peak_equity:,.0f})"
    )

with col3:
    st.metric(
        label="Avg Win",
        value=f"${metrics['avg_win']:.2f}"
    )

with col4:
    st.metric(
        label="Avg Loss",
        value=f"$-{metrics['avg_loss']:.2f}"  # Afficher avec signe négatif
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

# === SECTION 6: STATISTIQUES DÉTAILLÉES COMPLÈTES (HEDGE FUND GRADE) ===
with st.expander("📊 Statistiques Détaillées Complètes", expanded=True):
    st.markdown("### 🎯 TRADING STATISTICS")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**📈 Trades Overview**")
        st.markdown(f"- **Total Trades**: {metrics['total_trades']:,}")
        st.markdown(f"- **✅ Gagnants**: {metrics['winning_trades']:,} ({metrics['win_rate']:.1f}%)")
        st.markdown(f"- **❌ Perdants**: {metrics['losing_trades']:,} ({100-metrics['win_rate']:.1f}%)")
        st.markdown(f"- **Win Rate**: {metrics['win_rate']:.1f}%")

    with col2:
        st.markdown("**💰 PnL Moyens**")
        st.markdown(f"- **Avg Win**: ${metrics['avg_win']:.2f}")
        st.markdown(f"- **Avg Loss**: $-{abs(metrics['avg_loss']):.2f}")
        st.markdown(f"- **Avg RR**: {metrics['avg_rr']:.2f}R")
        st.markdown(f"- **Expectancy**: ${metrics['expectancy']:.2f}/trade")

    with col3:
        st.markdown("**🎯 PnL Extremes**")
        st.markdown(f"- **Max Gain**: ${metrics['max_win']:.2f}")
        st.markdown(f"- **Max Perte**: ${metrics['max_loss']:.2f}")
        st.markdown(f"- **Max RR**: {metrics['max_rr']:.2f}R")
        st.markdown(f"- **Total PnL**: ${metrics['total_pnl']:.2f}")

    with col4:
        st.markdown("**🔥 Streaks (Séquences)**")
        st.markdown(f"- **Max Win Streak**: {metrics['max_winning_streak']} trades")
        st.markdown(f"- **Max Loss Streak**: {metrics['max_losing_streak']} trades")
        st.markdown(f"- **Current Streak**: {metrics['current_streak']} ({metrics['current_streak_type']})")
        st.markdown("")

    st.markdown("---")
    st.markdown("### 📊 RISK METRICS (INSTITUTIONAL GRADE)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**⚠️ Drawdown Metrics**")
        st.markdown(f"- **Max DD %**: {metrics['max_dd_pct']:.2f}%")
        st.markdown(f"- **Max DD $**: ${metrics['max_dd_dollar']:,.2f}")
        dd_status = "✅ FTMO OK" if metrics['max_dd_pct'] < 10 else "🚨 FTMO VIOLATION"
        st.markdown(f"- **FTMO Status**: {dd_status}")
        st.markdown(f"- **Recovery Factor**: {metrics['recovery_factor']:.2f}")

    with col2:
        st.markdown("**📈 Risk-Adjusted Returns**")
        st.markdown(f"- **Sharpe Ratio**: {metrics['sharpe']:.2f}")
        st.markdown(f"- **Sortino Ratio**: {metrics['sortino']:.2f}")
        st.markdown(f"- **Calmar Ratio**: {metrics['calmar']:.2f}")
        st.markdown(f"- **Profit Factor**: {metrics['profit_factor']:.2f}")

    with col3:
        st.markdown("**📉 Tail Risk (VaR)**")
        st.markdown(f"- **VaR 95%**: {metrics['var_95']:.2f}%")
        var_status = "✅ OK" if metrics['var_95'] > -2.0 else "⚠️ Élevé"
        st.markdown(f"- **VaR Status**: {var_status}")
        st.markdown(f"- **CVaR 95%**: {metrics['cvar_95']:.2f}%")
        cvar_status = "✅ OK" if metrics['cvar_95'] > -3.0 else "⚠️ Élevé"
        st.markdown(f"- **CVaR Status**: {cvar_status}")

    with col4:
        st.markdown("**💼 Performance Summary**")
        st.markdown(f"- **ROI**: {metrics['roi']:.2f}%")
        st.markdown(f"- **Total Profit**: ${metrics['total_pnl']:,.2f}")
        st.markdown(f"- **Equity**: ${metrics['equity']:,.2f}")
        st.markdown(f"- **Timesteps**: {metrics['timesteps']:,}")

# === SECTION 7: FEATURES ANALYSIS (SHAP-BASED) ===
st.header("🧠 Features Analysis - Agent 7 (PPO)")

def get_feature_emoji(feature):
    """Retourne emoji selon le type de feature"""
    if any(x in feature.lower() for x in ['cot', 'commitment']):
        return "📊"
    elif any(x in feature.lower() for x in ['macro', 'us_', 'fomc', 'cpi', 'nfp', 'score']):
        return "🏛️"
    elif any(x in feature.lower() for x in ['seasonal', 'seasonax', 'month', 'week']):
        return "📅"
    elif any(x in feature.lower() for x in ['corr', 'eurusd', 'usdjpy', 'dxy', 'audchf', 'usdchf']):
        return "🔗"
    elif any(x in feature.lower() for x in ['rsi', 'macd', 'adx', 'stoch', 'bb_', 'tsi', 'momentum']):
        return "📈"
    elif any(x in feature.lower() for x in ['volume', 'vol_', 'va_']):
        return "📊"
    elif any(x in feature.lower() for x in ['retail', 'long_pct', 'short_pct']):
        return "👥"
    else:
        return "🔹"

top_features = load_top_features()

if top_features:
    st.success(f"✅ **{len(top_features)} features** utilisées par l'agent RL (classement par importance SHAP)")
    st.info("**📌 Note**: Les features sont triées par importance - Les premières ont le PLUS d'impact, les dernières le MOINS.")

    # TOP 10 BEST FEATURES (les plus importantes)
    st.markdown("---")
    st.subheader("🏆 TOP 10 BEST FEATURES (Plus d'Impact)")

    col1, col2 = st.columns(2)

    with col1:
        for i, feature in enumerate(top_features[:5], 1):
            emoji = get_feature_emoji(feature)
            st.markdown(f"**#{i}** {emoji} `{feature}`")

    with col2:
        for i, feature in enumerate(top_features[5:10], 6):
            emoji = get_feature_emoji(feature)
            st.markdown(f"**#{i}** {emoji} `{feature}`")

    # TOP 10 WORST FEATURES (les moins importantes)
    st.markdown("---")
    st.subheader("⚠️ TOP 10 WORST FEATURES (Moins d'Impact)")

    if len(top_features) >= 10:
        col1, col2 = st.columns(2)

        worst_features = top_features[-10:]

        with col1:
            for i, feature in enumerate(worst_features[:5], len(top_features)-9):
                emoji = get_feature_emoji(feature)
                st.markdown(f"**#{i}** {emoji} `{feature}`")

        with col2:
            for i, feature in enumerate(worst_features[5:], len(top_features)-4):
                emoji = get_feature_emoji(feature)
                st.markdown(f"**#{i}** {emoji} `{feature}`")
    else:
        st.warning("Pas assez de features pour afficher le TOP 10 WORST")

    # TOUTES LES FEATURES (dans un expander)
    st.markdown("---")
    with st.expander(f"📋 TOUTES LES {len(top_features)} FEATURES (Cliquer pour développer)", expanded=False):
        num_cols = 3
        features_per_col = (len(top_features) + num_cols - 1) // num_cols

        cols = st.columns(num_cols)

        for idx, feature in enumerate(top_features):
            col_idx = idx // features_per_col
            if col_idx < num_cols:
                with cols[col_idx]:
                    emoji = get_feature_emoji(feature)
                    st.markdown(f"**#{idx+1}** {emoji} `{feature}`")

    # Légende des catégories
    with st.expander("📖 Légende des Catégories"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **📊 COT (Commitment of Traders)**
            - Positions institutionnelles (Gold, DXY)
            - Divergence comm/non-comm
            - Z-score, percentiles

            **🏛️ Macro Events**
            - FOMC, NFP, CPI, PPI
            - Taux, inflation, emploi
            - Scores économiques (emploi, inflation, taux, croissance)
            """)

        with col2:
            st.markdown("""
            **📅 Seasonality**
            - Strong/Best month (Seasonax)
            - Weekly bias (bullish/bearish)
            - Patterns saisonniers Gold

            **🔗 Correlations**
            - EURUSD, USDJPY, USDCHF, AUDCHF
            - DXY (Dollar Index)
            - Gold vs devises/indices
            """)

        with col3:
            st.markdown("""
            **📈 Technical Indicators**
            - RSI, MACD, ADX, Stochastic
            - Bollinger Bands, ATR, TSI
            - SMA, EMA (H1, M15, D1)
            - Momentum, Divergences

            **👥 Retail Sentiment**
            - Positions retail (DXY, Gold)
            - Contrarian signal
            """)

else:
    st.error("❌ **Fichier features non trouvé**")
    st.info("""
    **Chemins recherchés**:
    - `C:/Users/lbye3/Desktop/GoldRL/AGENT/AGENT 7/ENTRAINEMENT/top100_features_agent7.txt`
    - `C:/Users/lbye3/Desktop/GoldRL/output/feature_selection/top100_features_agent7.txt`

    **Action**: Créer le fichier avec la liste des features utilisées par l'agent.
    """)

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

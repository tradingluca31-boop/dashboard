"""
🏛️ INSTITUTIONAL RL TRADING DASHBOARD - UNIVERSAL CSV SUPPORT
================================================================
Dashboard Streamlit pour monitoring en temps réel des trainings RL Gold Trading
Support TOUS les fichiers CSV générés automatiquement

Version: 3.0 CSV Universal
Date: 2025-11-19
Author: Claude Code + User
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import glob
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RL Trading Dashboard - CSV Universal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom pour style institutionnel
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 3px solid #00d4ff;}
    .stMetric:hover {border-left: 3px solid #00ff88; transition: 0.3s;}
    h1, h2, h3 {color: #00d4ff;}
    .success-box {background-color: #00ff8844; padding: 10px; border-radius: 5px; border-left: 4px solid #00ff88;}
    .warning-box {background-color: #ffaa0044; padding: 10px; border-radius: 5px; border-left: 4px solid #ffaa00;}
    .error-box {background-color: #ff004444; padding: 10px; border-radius: 5px; border-left: 4px solid #ff0044;}
    .metric-card {background: linear-gradient(135deg, #1e2130 0%, #2d3250 100%); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS DE DÉTECTION ET CATÉGORISATION CSV
# ============================================================================

def detect_csv_type(df: pd.DataFrame, filename: str) -> str:
    """
    Détecte automatiquement le type de CSV basé sur ses colonnes

    Args:
        df: DataFrame pandas
        filename: Nom du fichier

    Returns:
        Type de CSV ('training_report', 'trades', 'checkpoints', 'metrics', 'unknown')
    """
    cols = set(df.columns.str.lower())

    # Training Report (détaillé)
    if {'timesteps', 'roi_pct', 'sharpe', 'sortino', 'calmar', 'equity', 'balance'}.issubset(cols):
        return 'training_report'

    # Trades Details
    if {'entry_price', 'exit_price', 'side', 'pnl', 'pnl_pct'}.issubset(cols):
        return 'trades'

    # Checkpoints Analysis
    if {'steps', 'file', 'equity', 'roi_pct', 'composite_score'}.issubset(cols):
        return 'checkpoints'

    # Quick Metrics (simple)
    if {'timestamp', 'timesteps', 'roi_pct', 'equity'}.issubset(cols):
        return 'metrics'

    # Backtest Results
    if {'agent', 'roi', 'sharpe_ratio', 'max_drawdown'}.issubset(cols):
        return 'backtest'

    # Feature Importance (SHAP)
    if 'feature' in cols and ('importance' in cols or 'shap_value' in cols):
        return 'features'

    # TensorBoard Exports
    if 'step' in cols and any(x in cols for x in ['value', 'loss', 'reward']):
        return 'tensorboard'

    return 'unknown'


def load_all_csvs(directory: str) -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    """
    Charge tous les CSV d'un répertoire et les catégorise

    Args:
        directory: Chemin du répertoire

    Returns:
        Dictionnaire {type: [(filename, dataframe), ...]}
    """
    csv_data = {
        'training_report': [],
        'trades': [],
        'checkpoints': [],
        'metrics': [],
        'backtest': [],
        'features': [],
        'tensorboard': [],
        'unknown': []
    }

    # Recherche récursive de tous les CSV
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            filename = Path(csv_file).name
            csv_type = detect_csv_type(df, filename)
            csv_data[csv_type].append((filename, df, csv_file))
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur chargement {Path(csv_file).name}: {str(e)}")
            continue

    return csv_data


def load_uploaded_csv(uploaded_file) -> Tuple[str, pd.DataFrame]:
    """
    Charge un CSV uploadé et détecte son type

    Args:
        uploaded_file: Fichier Streamlit uploadé

    Returns:
        (type, dataframe)
    """
    df = pd.read_csv(uploaded_file)
    csv_type = detect_csv_type(df, uploaded_file.name)
    return csv_type, df


# ============================================================================
# VISUALISATIONS PAR TYPE DE CSV
# ============================================================================

def plot_training_report(df: pd.DataFrame, title: str = "Training Report"):
    """Visualisation pour training_report.csv"""

    # Tri par timesteps
    df = df.sort_values('timesteps')

    # Création de sous-graphiques
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📈 Equity Curve", "📊 ROI %",
            "💎 Sharpe & Sortino Ratio", "📉 Max Drawdown %",
            "🎯 Win Rate & Profit Factor", "🔥 Diversity & Entropy"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Equity Curve
    fig.add_trace(go.Scatter(
        x=df['timesteps'], y=df['equity'],
        name='Equity', line=dict(color='#00d4ff', width=2),
        fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
    ), row=1, col=1)

    # 2. ROI %
    fig.add_trace(go.Scatter(
        x=df['timesteps'], y=df['roi_pct'],
        name='ROI %', line=dict(color='#00ff88', width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'
    ), row=1, col=2)

    # 3. Sharpe & Sortino
    fig.add_trace(go.Scatter(
        x=df['timesteps'], y=df['sharpe'],
        name='Sharpe', line=dict(color='#ff00ff', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['timesteps'], y=df['sortino'],
        name='Sortino', line=dict(color='#ffaa00', width=2, dash='dash')
    ), row=2, col=1)

    # 4. Max Drawdown
    fig.add_trace(go.Scatter(
        x=df['timesteps'], y=df['max_dd_pct'],
        name='Max DD %', line=dict(color='#ff0044', width=2),
        fill='tozeroy', fillcolor='rgba(255, 0, 68, 0.1)'
    ), row=2, col=2)
    # Ligne FTMO 10%
    fig.add_hline(y=10, line_dash="dash", line_color="red",
                  annotation_text="FTMO Limit 10%", row=2, col=2)

    # 5. Win Rate & Profit Factor
    if 'win_rate' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timesteps'], y=df['win_rate'],
            name='Win Rate', line=dict(color='#00ff88', width=2)
        ), row=3, col=1)
    if 'profit_factor' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timesteps'], y=df['profit_factor'],
            name='Profit Factor', line=dict(color='#00d4ff', width=2),
            yaxis='y2'
        ), row=3, col=1)

    # 6. Diversity & Entropy
    if 'diversity_score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timesteps'], y=df['diversity_score'],
            name='Diversity', line=dict(color='#ff00ff', width=2)
        ), row=3, col=2)
    if 'policy_entropy' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timesteps'], y=df['policy_entropy'],
            name='Entropy', line=dict(color='#ffaa00', width=2, dash='dash')
        ), row=3, col=2)

    fig.update_layout(
        title=f"<b>{title}</b>",
        height=1000,
        showlegend=True,
        template='plotly_dark',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Métriques clés
    col1, col2, col3, col4, col5 = st.columns(5)

    latest = df.iloc[-1]
    with col1:
        st.metric("🎯 ROI Final", f"{latest['roi_pct']:.2f}%")
    with col2:
        st.metric("💎 Sharpe Ratio", f"{latest['sharpe']:.2f}")
    with col3:
        st.metric("📉 Max DD", f"{latest['max_dd_pct']:.2f}%")
    with col4:
        if 'win_rate' in df.columns:
            st.metric("🎲 Win Rate", f"{latest['win_rate']:.1f}%")
    with col5:
        st.metric("📊 Total Trades", f"{int(latest['total_trades'])}")


def plot_trades_analysis(df: pd.DataFrame, title: str = "Trades Analysis"):
    """Visualisation pour trades CSV"""

    st.subheader(f"📊 {title}")

    # Conversion des timestamps
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])

    # Statistiques globales
    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    win_trades = len(df[df['pnl'] > 0])
    loss_trades = len(df[df['pnl'] < 0])
    win_rate = (win_trades / len(df) * 100) if len(df) > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total PnL", f"${total_pnl:,.2f}")
    with col2:
        st.metric("📊 Avg PnL/Trade", f"${avg_pnl:.2f}")
    with col3:
        st.metric("✅ Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric("📈 Total Trades", f"{len(df)}")

    # Graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "📈 Cumulative PnL", "📊 PnL Distribution",
            "⏱️ Trade Duration (bars)", "💹 Long vs Short Performance"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. Cumulative PnL
    df['cumulative_pnl'] = df['pnl'].cumsum()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['cumulative_pnl'],
        name='Cumulative PnL', line=dict(color='#00d4ff', width=2),
        fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)'
    ), row=1, col=1)

    # 2. PnL Distribution
    fig.add_trace(go.Histogram(
        x=df['pnl'], nbinsx=50,
        name='PnL Distribution',
        marker=dict(color='#00ff88', line=dict(color='white', width=1))
    ), row=1, col=2)

    # 3. Trade Duration
    if 'duration_bars' in df.columns:
        fig.add_trace(go.Histogram(
            x=df['duration_bars'], nbinsx=30,
            name='Duration (bars)',
            marker=dict(color='#ff00ff', line=dict(color='white', width=1))
        ), row=2, col=1)

    # 4. Long vs Short
    if 'side' in df.columns or 'direction' in df.columns:
        side_col = 'side' if 'side' in df.columns else 'direction'
        long_pnl = df[df[side_col].isin([1, 'long'])]['pnl'].sum()
        short_pnl = df[df[side_col].isin([-1, 'short'])]['pnl'].sum()

        fig.add_trace(go.Bar(
            x=['Long', 'Short'],
            y=[long_pnl, short_pnl],
            marker=dict(color=['#00ff88', '#ff0044']),
            name='PnL by Direction'
        ), row=2, col=2)

    fig.update_layout(
        height=800,
        showlegend=False,
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top 10 meilleurs/pires trades
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Top 10 Best Trades")
        best_trades = df.nlargest(10, 'pnl')[['entry_price', 'exit_price', 'pnl', 'pnl_pct']]
        st.dataframe(best_trades.style.format({
            'entry_price': '{:.2f}',
            'exit_price': '{:.2f}',
            'pnl': '${:.2f}',
            'pnl_pct': '{:.2%}'
        }), use_container_width=True)

    with col2:
        st.markdown("### 💀 Top 10 Worst Trades")
        worst_trades = df.nsmallest(10, 'pnl')[['entry_price', 'exit_price', 'pnl', 'pnl_pct']]
        st.dataframe(worst_trades.style.format({
            'entry_price': '{:.2f}',
            'exit_price': '{:.2f}',
            'pnl': '${:.2f}',
            'pnl_pct': '{:.2%}'
        }), use_container_width=True)


def plot_checkpoints_analysis(df: pd.DataFrame, title: str = "Checkpoints Analysis"):
    """Visualisation pour checkpoints_analysis.csv"""

    st.subheader(f"🔍 {title}")

    # Tri par steps
    df = df.sort_values('steps')

    # Graphique principal
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "📊 Composite Score Evolution", "💰 ROI % by Checkpoint",
            "💎 Sharpe Ratio", "📉 Max Drawdown %"
        ),
        vertical_spacing=0.15
    )

    # 1. Composite Score
    fig.add_trace(go.Scatter(
        x=df['steps'], y=df['composite_score'],
        name='Composite Score',
        line=dict(color='#00d4ff', width=3),
        mode='lines+markers'
    ), row=1, col=1)

    # 2. ROI
    colors = ['#00ff88' if x > 0 else '#ff0044' for x in df['roi_pct']]
    fig.add_trace(go.Bar(
        x=df['steps'], y=df['roi_pct'],
        name='ROI %',
        marker=dict(color=colors)
    ), row=1, col=2)

    # 3. Sharpe
    fig.add_trace(go.Scatter(
        x=df['steps'], y=df['sharpe'],
        name='Sharpe',
        line=dict(color='#ff00ff', width=2),
        mode='lines+markers'
    ), row=2, col=1)

    # 4. Max DD
    fig.add_trace(go.Scatter(
        x=df['steps'], y=df['max_dd_pct'],
        name='Max DD %',
        line=dict(color='#ff0044', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 68, 0.1)'
    ), row=2, col=2)
    fig.add_hline(y=10, line_dash="dash", line_color="red",
                  annotation_text="FTMO 10%", row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Meilleurs checkpoints
    st.markdown("### 🏆 Top 5 Checkpoints (by Composite Score)")
    best_checkpoints = df.nlargest(5, 'composite_score')[
        ['steps', 'file', 'roi_pct', 'sharpe', 'max_dd_pct', 'composite_score']
    ]
    st.dataframe(best_checkpoints.style.format({
        'roi_pct': '{:.2f}%',
        'sharpe': '{:.2f}',
        'max_dd_pct': '{:.2f}%',
        'composite_score': '{:.4f}'
    }), use_container_width=True)


def plot_quick_metrics(df: pd.DataFrame, title: str = "Quick Metrics"):
    """Visualisation pour metrics simples"""

    st.subheader(f"⚡ {title}")

    # Métriques
    col1, col2, col3 = st.columns(3)

    latest = df.iloc[-1] if len(df) > 0 else None

    if latest is not None:
        with col1:
            st.metric("🎯 ROI", f"{latest['roi_pct']:.2f}%")
        with col2:
            st.metric("💰 Equity", f"${latest['equity']:,.2f}")
        with col3:
            st.metric("📊 Trades", f"{int(latest['total_trades'])}")

    # Graphique simple
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timesteps'], y=df['equity'],
        name='Equity',
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.2)'
    ))

    fig.update_layout(
        title="Equity Curve",
        height=400,
        template='plotly_dark'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_backtest_comparison(df: pd.DataFrame, title: str = "Backtest Results"):
    """Visualisation pour backtest multi-agents"""

    st.subheader(f"🏆 {title}")

    # Radar chart pour comparaison
    if 'agent' in df.columns:
        categories = ['ROI', 'Sharpe', 'Win Rate', 'Profit Factor']

        fig = go.Figure()

        for _, row in df.iterrows():
            agent_name = row['agent']
            values = [
                row.get('roi', 0),
                row.get('sharpe_ratio', 0) * 10,  # Mise à l'échelle
                row.get('win_rate', 0),
                row.get('profit_factor', 0) * 10
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=agent_name
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            height=500,
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Tableau de comparaison
    st.dataframe(df, use_container_width=True)


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    """Application principale"""

    # Header
    st.title("🏛️ INSTITUTIONAL RL TRADING DASHBOARD")
    st.markdown("### 📊 Universal CSV Support - Auto-Detection")
    st.markdown("---")

    # ========================================================================
    # SIDEBAR - Options de chargement
    # ========================================================================

    st.sidebar.title("⚙️ Data Source")

    data_source = st.sidebar.radio(
        "Select data source:",
        ["📁 Auto-detect from folder", "📤 Upload CSV files", "🔗 Load from GitHub repo"]
    )

    csv_data = None

    # Option 1: Auto-détection depuis un dossier
    if data_source == "📁 Auto-detect from folder":
        st.sidebar.markdown("---")

        default_path = r"C:\Users\lbye3\Desktop\GoldRL\AGENT"
        folder_path = st.sidebar.text_input(
            "Folder path:",
            value=default_path,
            help="Chemin du dossier contenant les CSV (recherche récursive)"
        )

        if st.sidebar.button("🔍 Scan Folder", type="primary"):
            with st.spinner("Scanning for CSV files..."):
                csv_data = load_all_csvs(folder_path)

                # Compteur de fichiers trouvés
                total_csvs = sum(len(files) for files in csv_data.values())
                st.sidebar.success(f"✅ {total_csvs} CSV files found!")

                # Détails par type
                for csv_type, files in csv_data.items():
                    if len(files) > 0:
                        st.sidebar.info(f"**{csv_type.upper()}**: {len(files)} files")

    # Option 2: Upload manuel
    elif data_source == "📤 Upload CSV files":
        st.sidebar.markdown("---")

        uploaded_files = st.sidebar.file_uploader(
            "Upload CSV files:",
            type=['csv'],
            accept_multiple_files=True,
            help="Vous pouvez uploader plusieurs CSV à la fois"
        )

        if uploaded_files:
            csv_data = {
                'training_report': [],
                'trades': [],
                'checkpoints': [],
                'metrics': [],
                'backtest': [],
                'features': [],
                'tensorboard': [],
                'unknown': []
            }

            for uploaded_file in uploaded_files:
                csv_type, df = load_uploaded_csv(uploaded_file)
                csv_data[csv_type].append((uploaded_file.name, df, None))

            st.sidebar.success(f"✅ {len(uploaded_files)} files uploaded!")

    # Option 3: Load depuis GitHub (futur)
    else:
        st.sidebar.info("🚧 GitHub integration coming soon...")

    # ========================================================================
    # AFFICHAGE DES DONNÉES
    # ========================================================================

    if csv_data:

        # Tabs pour chaque type de CSV
        tab_names = []
        tab_contents = []

        for csv_type, files in csv_data.items():
            if len(files) > 0:
                tab_names.append(f"{csv_type.upper()} ({len(files)})")
                tab_contents.append((csv_type, files))

        if len(tab_names) > 0:
            tabs = st.tabs(tab_names)

            for tab, (csv_type, files) in zip(tabs, tab_contents):
                with tab:
                    # Sélecteur de fichier si plusieurs
                    if len(files) > 1:
                        selected_file = st.selectbox(
                            "Select file:",
                            options=range(len(files)),
                            format_func=lambda i: files[i][0]
                        )
                        filename, df, filepath = files[selected_file]
                    else:
                        filename, df, filepath = files[0]

                    st.markdown(f"**📄 File:** `{filename}`")
                    if filepath:
                        st.markdown(f"**📂 Path:** `{filepath}`")

                    st.markdown("---")

                    # Affichage selon le type
                    if csv_type == 'training_report':
                        plot_training_report(df, filename)

                    elif csv_type == 'trades':
                        plot_trades_analysis(df, filename)

                    elif csv_type == 'checkpoints':
                        plot_checkpoints_analysis(df, filename)

                    elif csv_type == 'metrics':
                        plot_quick_metrics(df, filename)

                    elif csv_type == 'backtest':
                        plot_backtest_comparison(df, filename)

                    elif csv_type == 'features':
                        st.subheader("🎯 Feature Importance")
                        fig = px.bar(
                            df.nlargest(20, df.columns[1]),
                            x=df.columns[1], y=df.columns[0],
                            orientation='h',
                            title="Top 20 Features"
                        )
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)

                    elif csv_type == 'tensorboard':
                        st.subheader("📈 TensorBoard Metrics")
                        fig = px.line(
                            df, x=df.columns[0], y=df.columns[1],
                            title=filename
                        )
                        fig.update_layout(template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)

                    else:  # unknown
                        st.warning("⚠️ Type de CSV non reconnu - Affichage brut")
                        st.dataframe(df, use_container_width=True)

                    # Option de téléchargement
                    st.markdown("---")
                    csv_export = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_export,
                        file_name=filename,
                        mime='text/csv'
                    )

    else:
        # Instructions si pas de données
        st.info("""
        ### 📖 Instructions

        **Option 1 - Auto-Detection:**
        1. Sélectionnez "Auto-detect from folder" dans la sidebar
        2. Entrez le chemin du dossier contenant vos CSV
        3. Cliquez sur "Scan Folder"

        **Option 2 - Upload Manuel:**
        1. Sélectionnez "Upload CSV files"
        2. Uploadez un ou plusieurs CSV
        3. Le dashboard les catégorise automatiquement

        **Types de CSV supportés:**
        - ✅ Training Reports (timesteps, roi, sharpe, equity...)
        - ✅ Trades Details (entry, exit, pnl, duration...)
        - ✅ Checkpoints Analysis (steps, composite score...)
        - ✅ Quick Metrics (simple stats)
        - ✅ Backtest Results (multi-agents comparison)
        - ✅ Feature Importance (SHAP values)
        - ✅ TensorBoard Exports (loss, rewards...)
        """)

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏛️ Institutional RL Trading Dashboard v3.0 | Universal CSV Support</p>
        <p>Built with Streamlit + Plotly | © 2025</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == "__main__":
    main()

# 📊 RL Trading Dashboard - Universal CSV Support

Dashboard Streamlit institutionnel pour monitoring en temps réel des trainings RL Gold Trading avec **support universel de tous les CSV**.

## 🚀 Version 3.0 - Universal CSV Support

**NEW**: Le dashboard détecte et affiche automatiquement **TOUS les types de CSV** générés par votre système de trading RL.

## ✨ Fonctionnalités Principales

### 🔍 Auto-Détection Intelligente
- ✅ **Scan récursif** - Détecte tous les CSV dans un répertoire
- ✅ **Catégorisation automatique** - Identifie le type de CSV par ses colonnes
- ✅ **Upload manuel** - Supporte l'upload direct de CSV
- ✅ **Multi-agents** - Agent 7, 8, 9, 11, Meta-Agent

### 📊 Types de CSV Supportés

| Type | Colonnes Clés | Visualisations |
|------|---------------|----------------|
| **Training Reports** | timesteps, roi_pct, sharpe, sortino, equity | Courbes complètes, métriques institutionnelles |
| **Trades Details** | entry_price, exit_price, pnl, duration | Cumulative PnL, distribution, best/worst trades |
| **Checkpoints Analysis** | steps, composite_score, roi, sharpe | Evolution scores, comparaison checkpoints |
| **Quick Metrics** | timesteps, roi, equity | Métriques rapides, equity curve |
| **Backtest Results** | agent, roi, sharpe_ratio, max_drawdown | Radar chart multi-agents, comparaison |
| **Feature Importance** | feature, importance/shap_value | Top 20 features, bar chart |
| **TensorBoard Exports** | step, value/loss/reward | Courbes temporelles |

### 📈 Visualisations Institutionnelles

**Training Reports**:
- Equity Curve avec remplissage
- ROI % progression
- Sharpe & Sortino Ratios
- Max Drawdown % (avec seuil FTMO 10%)
- Win Rate & Profit Factor
- Diversity Score & Policy Entropy

**Trades Analysis**:
- Cumulative PnL curve
- PnL Distribution (histogramme)
- Trade Duration analysis
- Long vs Short performance
- Top 10 best/worst trades

**Checkpoints Analysis**:
- Composite Score evolution
- ROI by checkpoint (bar chart)
- Sharpe Ratio progression
- Max Drawdown tracking
- Top 5 best checkpoints

### 🎯 Métriques Institutionnelles

- **Performance**: ROI, Total PnL, Win Rate, Profit Factor
- **Risk**: Sharpe, Sortino, Calmar, Max Drawdown, VaR, CVaR
- **Trading**: Total Trades, Avg Win/Loss, R-Multiple, Expectancy
- **FTMO**: Max DD < 10%, Daily DD < 5% monitoring

## 🚀 Installation Locale

### Prérequis
```bash
Python 3.8+
pip
```

### Installation

```bash
# Cloner le repository
git clone https://github.com/tradingluca31-boop/dashboard.git
cd dashboard

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run streamlit_dashboard.py
```

Dashboard accessible sur: **http://localhost:8501**

## 📖 Utilisation

### Option 1: Auto-Détection (Recommandé)

1. Lancer le dashboard
2. Sidebar → Sélectionner **"Auto-detect from folder"**
3. Entrer le chemin du dossier contenant vos CSV:
   ```
   C:\Users\lbye3\Desktop\GoldRL\AGENT
   ```
4. Cliquer sur **"Scan Folder"**
5. Le dashboard détecte et catégorise automatiquement tous les CSV

### Option 2: Upload Manuel

1. Sidebar → Sélectionner **"Upload CSV files"**
2. Drag & drop ou sélectionner vos CSV
3. Le dashboard catégorise automatiquement chaque fichier
4. Visualisations adaptées affichées instantanément

### Option 3: GitHub Integration (Coming Soon)

Chargement direct depuis un repo GitHub

## 📁 Structure du Projet

```
dashboard/
├── streamlit_dashboard.py          # Dashboard principal (v3.0 Universal CSV)
├── requirements.txt                # Dependencies (streamlit, plotly, pandas, numpy)
├── top100_features_agent7.txt      # Feature ranking Agent 7
├── create_training_zip.py          # ZIP packaging script
├── update_dashboard.bat            # Auto-update script (Windows)
├── README.md                       # Documentation
├── DEPLOIEMENT_STREAMLIT_CLOUD.md  # Deployment guide
└── utils/                          # Utility functions
```

## 🔧 Configuration

### Auto-Refresh

Le dashboard se rafraîchit automatiquement (optionnel):
- Intervalle configurable (10-60 secondes)
- Toggle ON/OFF dans la sidebar

### Chemins par Défaut

**Dossier principal**: `C:\Users\lbye3\Desktop\GoldRL\AGENT`

Contient:
```
AGENT/
├── AGENT 7/
│   ├── training/
│   ├── models/
│   └── ENTRAINEMENT/
│       ├── training_report.csv
│       ├── checkpoints_analysis.csv
│       └── FICHIER EXCEL CSV AGENT 7/
│           ├── smoke_test_trades_*.csv
│           └── smoke_test_metrics_*.csv
├── AGENT 8/
├── AGENT 9/
├── AGENT 11/
└── backtest_preview_*.csv
```

## 🎨 Interface

### Sidebar
- **Data Source Selection**: Auto-detect, Upload, GitHub
- **Folder Path Input**: Chemin du dossier à scanner
- **Scan Button**: Lance la détection
- **File Statistics**: Nombre de CSV trouvés par type

### Main Dashboard
- **Tabs par Type**: Un tab par catégorie de CSV
- **File Selector**: Si plusieurs CSV du même type
- **Visualizations**: Graphiques Plotly interactifs
- **Metrics Cards**: Cartes métriques clés
- **Data Tables**: Tableaux détaillés (best/worst trades, etc.)
- **Download Button**: Export CSV

## 🚨 Détection Automatique

Le dashboard identifie le type de CSV basé sur ses colonnes:

**Training Report** → `timesteps` + `roi_pct` + `sharpe` + `sortino` + `equity`
**Trades** → `entry_price` + `exit_price` + `pnl` + `side`
**Checkpoints** → `steps` + `file` + `composite_score`
**Metrics** → `timestamp` + `timesteps` + `roi_pct` + `equity`
**Backtest** → `agent` + `roi` + `sharpe_ratio` + `max_drawdown`
**Features** → `feature` + `importance` (ou `shap_value`)
**TensorBoard** → `step` + `value` (ou `loss`, `reward`)

Si aucune correspondance → Affichage brut du CSV

## 📊 Exemples de CSV Supportés

### Training Report CSV
```csv
timesteps,roi_pct,sharpe,sortino,calmar,max_dd_pct,total_trades,win_rate,profit_factor,equity,balance
970000,134.39,0.338,0.528,6.802,19.75,4511,49.84,0.927,234392.61,234110.61
```

### Trades CSV
```csv
entry_price,exit_price,side,size,pnl,pnl_pct,entry_time,exit_time,direction,duration_bars
1913.1,1921.79,-1,0.599,-522.30,-0.00525,2021-01-04 01:00:00,2021-01-04 04:00:00,short,3
```

### Checkpoints CSV
```csv
steps,file,equity,balance,roi_pct,sharpe,sortino,calmar,max_dd_pct,composite_score
45500,checkpoint_45500_steps,87371.86,87371.86,-12.62,-2.16,-2.26,-0.97,12.91,0.592
```

## 🎯 Cas d'Usage

### Scénario 1: Monitoring Training en Cours

```python
# Vos scripts de training génèrent automatiquement des CSV
# → Le dashboard les détecte et affiche en temps réel
python train_agent7.py  # Génère training_report.csv
```

### Scénario 2: Analyse Post-Training

```python
# Après training, analyser tous les checkpoints
dashboard.scan("C:/GoldRL/AGENT/AGENT 7/ENTRAINEMENT")
# → Affiche checkpoints_analysis.csv avec meilleurs modèles
```

### Scénario 3: Comparaison Multi-Agents

```python
# Charger backtest multi-agents
dashboard.upload(backtest_comparison.csv)
# → Radar chart + tableau comparatif agents 7,8,9,11,Meta
```

## 🔥 Nouveautés v3.0

- ✅ **Auto-detection récursive** de tous les CSV
- ✅ **7 types de CSV** supportés automatiquement
- ✅ **Catégorisation intelligente** par colonnes
- ✅ **Visualisations adaptées** par type
- ✅ **Upload multi-fichiers**
- ✅ **Export/Download** intégré
- ✅ **Style institutionnel** (dark theme, gradient cards)
- ✅ **Support multi-agents** (7, 8, 9, 11, Meta)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/tradingluca31-boop/dashboard/issues)
- **Docs Streamlit**: https://docs.streamlit.io
- **Docs Plotly**: https://plotly.com/python/

## 🏆 Performance Attendue

```
Agent 7 (PPO):      ROI ~12%, Sharpe ~1.2, DD ~8%
Agent 8 (SAC):      ROI ~8%,  Sharpe ~1.0, DD ~9%
Agent 9 (TD3):      ROI ~10%, Sharpe ~1.1, DD ~8%
Agent 11 (A2C):     ROI ~6%,  Sharpe ~0.9, DD ~7%
Meta-Agent (PPO):   ROI ~15-18%, Sharpe ~1.5, DD ~7%
```

## 📝 Changelog

### v3.0.0 (2025-11-19)
- ✨ Universal CSV Support
- 🔍 Auto-detection de tous les types de CSV
- 📊 7 types de visualisations différentes
- 🎨 Interface refonte complète
- ⚡ Performance optimisée

### v2.0.0 (2025-11-12)
- Agent 7 Dashboard avec JSON
- TensorBoard integration
- FTMO compliance monitoring

### v1.0.0 (2025-11-09)
- Version initiale

## 📄 License

Projet privé - Tous droits réservés

---

**🏛️ Institutional RL Trading Dashboard** | Multi-Agent Gold Trading System | Powered by Streamlit + Plotly

*Built with Claude Code - https://claude.com/claude-code*

*Last updated: 2025-11-19 | Version: 3.0 Universal CSV*

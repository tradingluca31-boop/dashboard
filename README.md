# 📊 Agent 7 - Training Dashboard

Dashboard en temps réel pour monitorer l'entraînement de l'Agent 7 (PPO Momentum Trader H1).

## 🚀 Démo Live

**Dashboard déployé**: https://VOTRE_URL.streamlit.app *(à mettre à jour après déploiement)*

## 📈 Fonctionnalités

- ✅ **Monitoring temps réel** - Equity, PnL, Drawdown, Sharpe Ratio
- ✅ **Auto-refresh** - Mise à jour automatique toutes les 10 secondes
- ✅ **Graphiques interactifs** - Courbes d'équité, histogrammes PnL
- ✅ **FTMO Compliance** - Vérification Max DD < 10%
- ✅ **Top Trades** - Meilleurs et pires trades détaillés
- ✅ **Statistiques complètes** - Win Rate, Profit Factor, Sharpe, etc.

## 🛠️ Technologies

- **Framework**: Streamlit
- **Graphiques**: Plotly
- **Data**: JSON (training_stats.json)
- **Hébergement**: Streamlit Cloud (gratuit)

## 📊 Métriques Affichées

### Performance
- ROI (%)
- Total PnL ($)
- Win Rate (%)
- Profit Factor
- Sharpe Ratio
- Max Risk/Reward Ratio

### Risk Management
- Max Drawdown (% et $)
- Average Win/Loss
- FTMO Compliance (Max DD < 10%)

### Trading
- Total Trades
- Winning vs Losing Trades
- Top 10 meilleurs/pires trades
- Distribution PnL (histogramme)

## 🚀 Installation Locale

### Prérequis
- Python 3.8+
- pip

### Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/agent7-dashboard.git
cd agent7-dashboard

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run streamlit_dashboard.py
```

Le dashboard sera accessible sur: http://localhost:8501

## 📁 Structure du Projet

```
agent7-dashboard/
├── streamlit_dashboard.py      # Dashboard principal
├── requirements.txt            # Dépendances Python
├── training_stats.json         # Données training (mis à jour régulièrement)
├── update_dashboard.bat        # Script Windows pour push automatique
├── README.md                   # Ce fichier
└── DEPLOIEMENT_STREAMLIT_CLOUD.md  # Guide déploiement complet
```

## 🔄 Mise à Jour du Dashboard

### Option 1: Script automatique (Windows)

```bash
update_dashboard.bat
```

### Option 2: Manuel

```bash
git add training_stats.json
git commit -m "Update training stats"
git push
```

Le dashboard Streamlit Cloud se mettra à jour automatiquement en ~30 secondes.

## 📊 Exemple de Visualisation

Le dashboard affiche en temps réel:

1. **Vue d'ensemble**
   - Timesteps actuels / 1.5M
   - Equity et ROI
   - Total trades

2. **Courbes temporelles**
   - Évolution de l'equity
   - Drawdown dans le temps
   - Sharpe Ratio

3. **Distributions**
   - Histogramme des PnL par trade
   - Top trades (meilleurs/pires)

## ⚙️ Configuration

### Auto-refresh

Par défaut, le dashboard se rafraîchit automatiquement toutes les 10 secondes.

Vous pouvez:
- ✅ Activer/désactiver via la sidebar
- ⚙️ Ajuster l'intervalle (5-60 secondes)

### PnL Normalization

⚠️ **Important**: Les PnL dans le JSON sont multipliés par ×100 (bug environnement).

Le dashboard applique automatiquement une division par 100 pour afficher les valeurs correctes.

## 🚨 Troubleshooting

### Dashboard ne trouve pas training_stats.json

Vérifiez que le fichier existe:
```
C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT\training_stats.json
```

### Métriques incorrectes

Le dashboard effectue une **double vérification** du Total PnL:
- Méthode 1: Somme de tous les trades
- Méthode 2: Equity delta (equity finale - 100,000)

Si différence > $100 → Affiche un avertissement

### Dashboard ne se met pas à jour

1. Vérifier que `training_stats.json` est bien pushé sur GitHub
2. Attendre 30-60 secondes
3. Hard refresh (Ctrl + Shift + R)
4. Si problème persiste: "Reboot app" sur Streamlit Cloud

## 📖 Documentation Complète

Voir [DEPLOIEMENT_STREAMLIT_CLOUD.md](DEPLOIEMENT_STREAMLIT_CLOUD.md) pour:
- Guide déploiement complet
- Configuration GitHub
- Workflow de mise à jour
- Troubleshooting avancé

## 🎯 Métriques Cibles

```
ROI:            > 12%
Sharpe Ratio:   > 1.2
Max Drawdown:   < 10% (FTMO compliance)
Win Rate:       > 50%
Profit Factor:  > 1.5
```

## 📞 Support

- **Documentation Streamlit**: https://docs.streamlit.io
- **Issues**: https://github.com/VOTRE_USERNAME/agent7-dashboard/issues
- **Discussions**: https://discuss.streamlit.io

## 📝 License

Projet privé - Tous droits réservés

---

**Agent 7** - PPO Momentum Trader H1 - Reinforcement Learning for Gold (XAUUSD)

*Dernière mise à jour: 2025-11-12*

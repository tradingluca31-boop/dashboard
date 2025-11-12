# 🚀 DÉPLOIEMENT DASHBOARD SUR STREAMLIT CLOUD

**Date**: 2025-11-12
**Objectif**: Héberger le dashboard Agent 7 sur Streamlit Cloud (gratuit)

---

## ✅ AVANTAGES STREAMLIT CLOUD

| Critère | Local (HTML) | Streamlit Cloud |
|---------|--------------|-----------------|
| **Installation** | ❌ Complexe (server.py + HTML) | ✅ Simple (1 fichier Python) |
| **Accès** | 🏠 Localhost seulement | 🌐 URL publique partout |
| **Mise à jour** | ❌ Manuel (F5) | ✅ Auto-refresh intégré |
| **Déploiement** | ❌ N/A | ✅ 1-click depuis GitHub |
| **Coût** | ✅ Gratuit | ✅ Gratuit |

---

## 📋 PRÉREQUIS

1. **Compte GitHub** (gratuit)
   - Créer sur https://github.com si vous n'en avez pas

2. **Compte Streamlit Cloud** (gratuit)
   - Créer sur https://share.streamlit.io
   - Se connecter avec votre compte GitHub

3. **Git installé** sur votre PC
   - Télécharger: https://git-scm.com/download/win

---

## 🚀 ÉTAPE 1: CRÉER LE REPOSITORY GITHUB

### Option A: Via GitHub Desktop (Plus facile)

1. **Télécharger GitHub Desktop**
   ```
   https://desktop.github.com
   ```

2. **Créer nouveau repository**
   - Cliquer "File" → "New Repository"
   - Nom: `agent7-dashboard`
   - Local path: `C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT`
   - Cliquer "Create Repository"

3. **Ajouter les fichiers**
   - Les fichiers seront automatiquement détectés:
     - `streamlit_dashboard.py`
     - `requirements.txt`
     - `training_stats.json` (important!)

4. **Commit et Push**
   - Sélectionner tous les fichiers
   - Message: "Initial commit - Agent 7 Dashboard"
   - Cliquer "Commit to main"
   - Cliquer "Publish repository" (en haut)
   - ⚠️ **ATTENTION**: Décocher "Keep this code private" si vous voulez que Streamlit Cloud puisse y accéder (OU garder privé si vous avez Streamlit Cloud Pro)

### Option B: Via ligne de commande (Plus rapide)

```bash
# 1. Naviguer vers le dossier
cd "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT"

# 2. Initialiser Git
git init

# 3. Ajouter les fichiers
git add streamlit_dashboard.py
git add requirements.txt
git add training_stats.json

# 4. Premier commit
git commit -m "Initial commit - Agent 7 Dashboard"

# 5. Créer le repository sur GitHub (via site web)
# Aller sur https://github.com/new
# Nom: agent7-dashboard
# Public ou Private (selon préférence)
# NE PAS initialiser avec README

# 6. Lier le repository local au remote
git remote add origin https://github.com/VOTRE_USERNAME/agent7-dashboard.git

# 7. Push
git branch -M main
git push -u origin main
```

---

## 🌐 ÉTAPE 2: DÉPLOYER SUR STREAMLIT CLOUD

### 1. Accéder à Streamlit Cloud

```
https://share.streamlit.io
```

- Cliquer "Sign in" avec votre compte GitHub
- Autoriser l'accès

### 2. Créer une nouvelle app

1. Cliquer "New app" (en haut à droite)

2. **Remplir le formulaire**:
   ```
   Repository:  VOTRE_USERNAME/agent7-dashboard
   Branch:      main
   Main file:   streamlit_dashboard.py
   ```

3. **Advanced settings** (optionnel):
   - Python version: 3.11
   - Secrets: (vide pour l'instant)

4. Cliquer "Deploy!"

### 3. Attendre le déploiement

- ⏱️ Durée: 2-5 minutes
- Vous verrez les logs en temps réel
- Quand c'est prêt: "Your app is live!"

### 4. Récupérer l'URL publique

```
https://VOTRE_USERNAME-agent7-dashboard-RANDOM.streamlit.app
```

**Exemple**:
```
https://lbye3-agent7-dashboard-abc123.streamlit.app
```

---

## 🔄 ÉTAPE 3: METTRE À JOUR LE DASHBOARD

**Problème**: Le training tourne en LOCAL, mais le dashboard est sur Streamlit Cloud.

### Solution 1: Push Manuel du JSON (Recommandé pour démarrer)

**Chaque fois que vous voulez mettre à jour** (par exemple après 50K steps):

```bash
cd "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT"

# Ajouter le JSON mis à jour
git add training_stats.json

# Commit
git commit -m "Update training stats - 100K steps"

# Push vers GitHub
git push

# Streamlit Cloud détectera automatiquement le changement et redémarrera l'app
```

**⏱️ Temps de mise à jour**: ~30 secondes après le push

### Solution 2: Automatisation avec script BAT (Avancé)

Créer un fichier `update_dashboard.bat`:

```batch
@echo off
cd "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT"
git add training_stats.json
git commit -m "Auto-update training stats - %date% %time%"
git push
echo ✅ Dashboard mis à jour sur Streamlit Cloud !
pause
```

**Usage**: Double-cliquer sur `update_dashboard.bat` après chaque checkpoint

### Solution 3: Auto-Push toutes les 10 minutes (Très avancé)

⚠️ **Non recommandé** car GitHub n'aime pas les push trop fréquents.

---

## 📊 WORKFLOW COMPLET

### Pendant le Training (Local)

```
1. Lancer training:
   cd "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT"
   python train_from_scratch.py

2. Training sauvegarde training_stats.json tous les 50K steps

3. Après chaque checkpoint important (100K, 200K, etc.):
   - Double-cliquer update_dashboard.bat
   - OU git add/commit/push manuellement

4. Attendre 30 secondes

5. Rafraîchir le dashboard Streamlit Cloud:
   https://VOTRE_URL.streamlit.app
```

### Monitoring (Cloud)

```
1. Ouvrir l'URL Streamlit Cloud dans votre navigateur

2. Le dashboard se met à jour automatiquement:
   - Auto-refresh activé par défaut (10 secondes)
   - Pas besoin de F5 manuel

3. Accessible depuis:
   - ✅ PC
   - ✅ Téléphone
   - ✅ Tablette
   - ✅ N'importe où avec internet
```

---

## 🛠️ STRUCTURE FINALE DU REPOSITORY GITHUB

```
agent7-dashboard/
├── streamlit_dashboard.py      # Dashboard principal
├── requirements.txt            # Dépendances Python
├── training_stats.json         # Données training (mis à jour régulièrement)
└── README.md                   # (optionnel) Documentation
```

---

## ⚙️ CONFIGURATION AVANCÉE

### Secrets (pour données sensibles)

Si vous voulez garder le repository **privé** mais partager le dashboard **publiquement**:

1. Sur Streamlit Cloud, aller dans "Settings" de votre app
2. Section "Secrets"
3. Ajouter vos secrets (API keys, etc.)
4. Accès dans le code:
   ```python
   import streamlit as st
   api_key = st.secrets["API_KEY"]
   ```

### Custom Domain (Optionnel)

Streamlit Cloud Pro permet un domaine custom:
```
https://dashboard.votresite.com
```

---

## 🚨 LIMITES STREAMLIT CLOUD (GRATUIT)

| Limite | Valeur |
|--------|--------|
| **Apps publiques** | Illimité |
| **Apps privées** | 1 |
| **Resources** | 1 GB RAM, 1 CPU |
| **Inactivité** | App dort après 7 jours sans visite |

**Solution**: Visiter l'URL au moins 1 fois par semaine pour garder l'app active.

---

## 🐛 TROUBLESHOOTING

### "Module not found: streamlit"

**Cause**: `requirements.txt` mal configuré
**Solution**: Vérifier que `requirements.txt` contient:
```
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.1.0
```

### "File not found: training_stats.json"

**Cause**: JSON pas pushé sur GitHub
**Solution**:
```bash
git add training_stats.json
git commit -m "Add training stats"
git push
```

### Dashboard ne se met pas à jour

**Cause**: Streamlit Cloud n'a pas détecté le changement
**Solution**:
1. Sur Streamlit Cloud, cliquer "Reboot app" (⋮ menu en haut à droite)
2. OU attendre 1-2 minutes

### "Your app is over capacity"

**Cause**: Trop de visiteurs simultanés (limite gratuite)
**Solution**: Passer à Streamlit Cloud Pro ($20/mois) ou limiter l'accès

---

## ✅ CHECKLIST DÉPLOIEMENT

- [ ] Compte GitHub créé
- [ ] Compte Streamlit Cloud créé (avec GitHub login)
- [ ] Git installé sur PC
- [ ] Repository `agent7-dashboard` créé sur GitHub
- [ ] Fichiers pushés:
  - [ ] `streamlit_dashboard.py`
  - [ ] `requirements.txt`
  - [ ] `training_stats.json`
- [ ] App déployée sur Streamlit Cloud
- [ ] URL publique fonctionnelle
- [ ] Auto-refresh activé
- [ ] Script `update_dashboard.bat` créé (optionnel)

---

## 📞 LIENS UTILES

- **Streamlit Cloud**: https://share.streamlit.io
- **Documentation Streamlit**: https://docs.streamlit.io
- **GitHub**: https://github.com
- **Support Streamlit**: https://discuss.streamlit.io

---

## 🎯 RÉSUMÉ 3 ÉTAPES

```
1. GITHUB
   Créer repository → Push streamlit_dashboard.py + requirements.txt + training_stats.json

2. STREAMLIT CLOUD
   New app → Sélectionner repository → Deploy

3. MISE À JOUR
   git add training_stats.json → git commit → git push
   (Dashboard se met à jour automatiquement en 30 secondes)
```

---

**🚀 AVANTAGE PRINCIPAL**: Accès à votre dashboard depuis n'importe où, sur n'importe quel appareil, avec une simple URL !

**📱 EXEMPLE D'USAGE**:
```
Training sur PC desktop → Push JSON → Consulter dashboard sur téléphone depuis le canapé
```

---

*Document créé le 2025-11-12*
*Agent 7 - Streamlit Cloud Deployment Guide*

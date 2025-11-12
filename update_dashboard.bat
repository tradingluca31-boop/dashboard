@echo off
chcp 65001 > nul
echo ================================================================================
echo MISE À JOUR DASHBOARD STREAMLIT CLOUD
echo ================================================================================
echo.

cd /d "C:\Users\lbye3\Desktop\GoldRL\AGENT\AGENT 7\ENTRAINEMENT"

echo [1/3] Vérification fichier training_stats.json...
if not exist "training_stats.json" (
    echo ❌ ERREUR: training_stats.json non trouvé !
    pause
    exit /b 1
)

echo ✅ Fichier trouvé
echo.

echo [2/3] Ajout des modifications à Git...
git add training_stats.json

if errorlevel 1 (
    echo ❌ ERREUR: Git add a échoué
    echo 💡 Vérifiez que Git est installé et que le repository est initialisé
    pause
    exit /b 1
)

echo.
echo [3/3] Commit et push vers GitHub...
git commit -m "Update training stats - %date% %time%"
git push

if errorlevel 1 (
    echo ⚠️ AVERTISSEMENT: Push a échoué
    echo 💡 Causes possibles:
    echo    - Pas de modifications depuis le dernier push
    echo    - Problème de connexion GitHub
    echo    - Repository remote non configuré
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✅ DASHBOARD MIS À JOUR SUR STREAMLIT CLOUD
echo ================================================================================
echo.
echo 📊 Votre dashboard sera mis à jour dans ~30 secondes
echo 🌐 Accédez-y sur: https://VOTRE_URL.streamlit.app
echo.
echo 💡 Le dashboard se rafraîchit automatiquement toutes les 10 secondes
echo.

pause

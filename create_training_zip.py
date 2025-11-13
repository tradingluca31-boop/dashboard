"""
Script pour créer un fichier ZIP contenant training_stats.json
Usage: python create_training_zip.py
"""

import zipfile
from pathlib import Path
from datetime import datetime

# Chemins
json_file = Path(__file__).parent / "training_stats.json"
output_zip = Path(__file__).parent / "training_stats.zip"

# Vérifier que le JSON existe
if not json_file.exists():
    print(f"[ERROR] Fichier non trouve: {json_file}")
    print(f"        Verifiez que training_stats.json est dans le meme dossier que ce script")
    exit(1)

# Créer le ZIP
print(f"[INFO] Creation du ZIP...")
print(f"       Source: {json_file.name} ({json_file.stat().st_size / 1024 / 1024:.2f} MB)")

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
    # Ajouter le JSON au ZIP (avec compression maximale)
    zipf.write(json_file, arcname="training_stats.json")

# Afficher les résultats
original_size = json_file.stat().st_size / 1024 / 1024
compressed_size = output_zip.stat().st_size / 1024 / 1024
ratio = (1 - compressed_size / original_size) * 100

print(f"\n[SUCCESS] ZIP cree avec succes!")
print(f"          Fichier: {output_zip.name}")
print(f"          Taille originale: {original_size:.2f} MB")
print(f"          Taille compressee: {compressed_size:.2f} MB")
print(f"          Compression: {ratio:.1f}%")
print(f"\n[NEXT] Vous pouvez maintenant uploader {output_zip.name} dans le dashboard Streamlit")

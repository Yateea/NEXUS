# ==============================================================
# NEXUS - Script de creation de l'arborescence (Windows PowerShell)
# A executer DEPUIS L'INTERIEUR du dossier NEXUS deja cree
# Usage : powershell -ExecutionPolicy Bypass -File setup_nexus_structure.ps1
# ==============================================================

Write-Host "Creation de l'arborescence NEXUS..." -ForegroundColor Cyan

# ---------- BACKEND ----------
New-Item -ItemType Directory -Force -Path "backend\app\api" | Out-Null
New-Item -ItemType Directory -Force -Path "backend\app\models" | Out-Null
New-Item -ItemType Directory -Force -Path "backend\app\schemas" | Out-Null
New-Item -ItemType Directory -Force -Path "backend\app\services" | Out-Null
New-Item -ItemType Directory -Force -Path "backend\tests" | Out-Null
New-Item -ItemType File -Force -Path "backend\app\main.py" | Out-Null
New-Item -ItemType File -Force -Path "backend\app\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "backend\app\api\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "backend\app\models\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "backend\app\schemas\__init__.py" | Out-Null
New-Item -ItemType File -Force -Path "backend\app\services\__init__.py" | Out-Null

# ---------- DATA (structure projet) ----------
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path "data\synthetic" | Out-Null

# ---------- DATA (structure detaillee - section 19 du cahier des charges) ----------
New-Item -ItemType Directory -Force -Path "data\external\m5" | Out-Null
New-Item -ItemType Directory -Force -Path "data\external\nasa\C-MAPSS" | Out-Null
New-Item -ItemType Directory -Force -Path "data\external\energy" | Out-Null
New-Item -ItemType Directory -Force -Path "data\external\transport\yellow_taxi" | Out-Null
New-Item -ItemType Directory -Force -Path "data\generated\dimensions" | Out-Null
New-Item -ItemType Directory -Force -Path "data\generated\facts" | Out-Null
New-Item -ItemType Directory -Force -Path "data\bronze" | Out-Null
New-Item -ItemType Directory -Force -Path "data\silver" | Out-Null
New-Item -ItemType Directory -Force -Path "data\gold" | Out-Null

# ---------- PIPELINE ----------
New-Item -ItemType Directory -Force -Path "pipeline\ingestion" | Out-Null
New-Item -ItemType Directory -Force -Path "pipeline\cleaning" | Out-Null
New-Item -ItemType Directory -Force -Path "pipeline\transformation" | Out-Null
New-Item -ItemType Directory -Force -Path "pipeline\features" | Out-Null
New-Item -ItemType Directory -Force -Path "pipeline\validation" | Out-Null
New-Item -ItemType File -Force -Path "pipeline\__init__.py" | Out-Null

# ---------- MACHINE LEARNING ----------
New-Item -ItemType Directory -Force -Path "ml\forecasting" | Out-Null
New-Item -ItemType Directory -Force -Path "ml\anomaly_detection" | Out-Null
New-Item -ItemType Directory -Force -Path "ml\predictive_maintenance" | Out-Null
New-Item -ItemType Directory -Force -Path "ml\delay_prediction" | Out-Null
New-Item -ItemType File -Force -Path "ml\__init__.py" | Out-Null

# ---------- POWER BI ----------
New-Item -ItemType Directory -Force -Path "powerbi" | Out-Null

# ---------- FRONTEND ----------
New-Item -ItemType Directory -Force -Path "frontend\react-app" | Out-Null

# ---------- DATABASE ----------
New-Item -ItemType Directory -Force -Path "database\schema" | Out-Null
New-Item -ItemType Directory -Force -Path "database\seeds" | Out-Null

# ---------- DOCKER ----------
New-Item -ItemType Directory -Force -Path "docker" | Out-Null

# ---------- NOTEBOOKS ----------
New-Item -ItemType Directory -Force -Path "notebooks" | Out-Null

# ---------- TESTS (globaux) ----------
New-Item -ItemType Directory -Force -Path "tests" | Out-Null

# ---------- FICHIERS RACINE ----------
New-Item -ItemType File -Force -Path ".env" | Out-Null
New-Item -ItemType File -Force -Path "requirements.txt" | Out-Null

# ---------- .gitignore ----------
@"
# Python
venv/
__pycache__/
*.pyc
.env

# Data (jamais versionnees)
data/raw/
data/external/
data/generated/
data/bronze/
data/silver/
data/gold/
*.parquet
*.csv

# Frontend
node_modules/
frontend/react-app/build/
frontend/react-app/dist/

# IDE / OS
.vscode/
.idea/
.DS_Store

# Docker
*.log
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# ---------- README ----------
@"
# NEXUS - Intelligent Operations & Decision Platform

Plateforme Data Analytics / Data Science / Data Engineering pour une entreprise
industrielle fictive (NEXUS Manufacturing), couvrant Supply Chain, Inventory,
Production, Energy, Predictive Maintenance, Transport et Demand.

Stack : Python, Polars, Dask, PostgreSQL, FastAPI, Power BI, React, Docker.

## Demarrage rapide
1. python -m venv venv
2. venv\Scripts\Activate.ps1
3. pip install -r requirements.txt
"@ | Out-File -FilePath "README.md" -Encoding utf8

Write-Host ""
Write-Host "Arborescence NEXUS creee avec succes." -ForegroundColor Green
Write-Host ""
Get-ChildItem -Directory -Recurse -Depth 2 | Select-Object FullName

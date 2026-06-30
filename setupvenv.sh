#!/bin/bash
# Setup virtual environment untuk ArduPilot Log Viewer
# Jalankan: bash setup_venv.sh

set -e

echo "Membuat virtual environment 'venv'..."
python3 -m venv venv

echo "Mengaktifkan venv dan install dependensi..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo ""
echo "Selesai! Cara menjalankan aplikasi:"
echo "  source venv/bin/activate"
echo "  python ardupilot_log_viewer.py"
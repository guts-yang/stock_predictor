#!/bin/bash
echo "Starting Stock Predictor Server..."
cd "$(dirname "$0")"
python3 backend/api/app.py

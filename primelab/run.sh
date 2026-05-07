#!/usr/bin/env bash

cd "$(dirname "$0")"

"/c/Users/Asus/AppData/Local/Python/pythoncore-3.14-64/python.exe" -m streamlit run app.py \
    --server.port 8501 \
    --server.headless true \
    --theme.base dark \
    --theme.backgroundColor "#080D1A" \
    --theme.secondaryBackgroundColor "#0F172A" \
    --theme.primaryColor "#00D4FF" \
    --theme.textColor "#E2E8F0"
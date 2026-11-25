#!/usr/bin/env bash
set -eu
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf "Run: .venv/bin/streamlit run streamlit_app.py\n"

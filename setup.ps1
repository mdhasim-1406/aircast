$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
& .venv\Scripts\python -m pip install -r requirements.txt
Write-Output "Run: .venv\Scripts\streamlit run streamlit_app.py"

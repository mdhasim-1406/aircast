# Start AirCast

From /home/mohamedhasim/ml/aircast:

    PYTHONPATH=src .venv/bin/streamlit run streamlit_app.py --server.port 8520
    PYTHONPATH=src .venv/bin/uvicorn api.main:app --port 8020

Dashboard http://localhost:8520, Swagger http://localhost:8020/docs. Streamlit is independent of FastAPI and MLflow.

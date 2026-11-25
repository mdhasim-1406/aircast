FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 AIRCAST_ROOT=/app
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY artifacts ./artifacts
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

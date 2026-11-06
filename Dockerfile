FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements-api.txt
COPY api ./api
COPY src ./src
COPY models/production_model.pkl ./models/production_model.pkl
COPY models/production_preprocessor.pkl ./models/production_preprocessor.pkl
EXPOSE 8000
CMD ["uvicorn","api.app:app","--host","0.0.0.0","--port","8000"]

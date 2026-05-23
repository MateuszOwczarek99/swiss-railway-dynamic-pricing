FROM python:3.9-slim

WORKDIR /app

# install system dependencies needed for some python packages
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY src/ ./src/
COPY models/ ./models/

# create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# expose api port
EXPOSE 8000

# health check for kubernetes
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# run the api server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

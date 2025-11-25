FROM python:3.11-slim

# Não gerar .pyc e logs sem buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências de sistema mínimas (OpenCV headless, Ultralytics, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia o código (main.py, templates/, models/, etc.)
COPY . /app

# Cloud Run usa essa porta
ENV PORT=8080
EXPOSE 8080

# Usa python -m uvicorn (menos chance de dar "uvicorn: command not found")
CMD sh -c "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"

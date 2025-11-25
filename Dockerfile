FROM python:3.11-slim

# Não gerar .pyc e logs sem buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências de sistema para numpy / opencv / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências Python
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação (main.py, templates/, models/, etc.)
COPY . /app

# Cloud Run *vai* injetar PORT (padrão 8080)
EXPOSE 8080

# Respeita a variável PORT do Cloud Run
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port \${PORT:-8080}"

# Imagem base leve com Python 3.11
FROM python:3.11-slim

# Evita .pyc e deixa logs sem buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências de sistema mínimas (OpenCV, Ultralytics, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho dentro do container
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

ENV PORT=8000

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"

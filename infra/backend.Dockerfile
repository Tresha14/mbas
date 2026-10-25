# MBAS backend + inference workers (CPU). Trained models are mounted at run time (see docker-compose.yml).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY ml ml
COPY recorder/__init__.py recorder/__init__.py
COPY recorder/backend recorder/backend
COPY recorder/questions.json recorder/questions.json
COPY tools tools
ENV PYTHONUNBUFFERED=1
EXPOSE 8020

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libopenblas-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt || true

COPY . .

CMD ["/bin/bash", "-lc", "echo 'Run make bootstrap && make run for sample pipeline. For UI use make ui.'"]

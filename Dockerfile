FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ffmpeg \
    gfortran \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .


RUN pip3 install --no-cache-dir torch torchaudio
RUN pip3 install --no-cache-dir -r requirements.txt


COPY app.py .


ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV HF_TOKEN=""

EXPOSE 8080

CMD ["python3", "app.py"]

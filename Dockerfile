# CPU image by default (small). For NVIDIA GPUs use docker-compose.gpu.yml, which swaps the base image.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
# 1 = install torch (CPU wheel) ourselves; the GPU base image already ships a CUDA build of torch
ARG INSTALL_TORCH=1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CHESSTWIN_DATA=/data
WORKDIR /app
RUN if [ "$INSTALL_TORCH" = "1" ]; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
# Stockfish powers the post-game analysis (installs to /usr/games, which chesstwin looks in)
RUN apt-get update && apt-get install -y --no-install-recommends stockfish && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt zstandard
COPY schess ./schess
COPY chesstwin ./chesstwin
COPY models ./models
COPY preprocess.py train.py analyze.py analyze_model.py ./
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/api/status')" || exit 1
CMD ["python", "-m", "chesstwin", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]

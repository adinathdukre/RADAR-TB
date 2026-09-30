FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libglib2.0-0 libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu118

COPY docker/requirements.txt requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY radar ./radar
COPY scripts/predict.py ./predict.py
COPY weights /opt/radar/weights
COPY checkpoints /opt/radar/checkpoints

ENV PYTHONPATH=/workspace \
    RADAR_WEIGHTS=/opt/radar/weights \
    RADAR_CACHE=/opt/radar/checkpoints \
    HF_HOME=/opt/radar/checkpoints/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN python -c "import torch; from radar.inference import Radar; Radar(torch.device('cpu'))"

ENTRYPOINT ["python", "predict.py", "--input", "/input", "--output", "/output"]

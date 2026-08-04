FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git latentsync
WORKDIR /workspace/latentsync

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod "huggingface_hub[cli]"

# Checkpoints se descargan en build time y quedan horneados en la imagen —
# evita descargarlos en cada cold start (son ~varios GB).
RUN huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints && \
    huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

COPY handler.py /workspace/latentsync/handler.py

CMD ["python", "-u", "handler.py"]

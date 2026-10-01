FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    patchelf \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Raster Vision 0.21.3 (instala sus propias versiones fijadas de torch, numpy,
# albumentations, onnxruntime-gpu, etc.)
RUN git clone --depth 1 --branch v0.21.3 https://github.com/azavea/raster-vision.git && \
    pip install --no-cache-dir ./raster-vision/rastervision_pipeline && \
    pip install --no-cache-dir ./raster-vision/rastervision_core && \
    pip install --no-cache-dir ./raster-vision/rastervision_pytorch_learner && \
    pip install --no-cache-dir ./raster-vision/rastervision_aws_s3 && \
    pip install --no-cache-dir ./raster-vision/rastervision_pytorch_backend

# Corrige "cannot enable executable stack as shared object requires" (WSL2/Docker Desktop)
RUN find /usr/local/lib/python3.10/site-packages/onnxruntime -name "*.so*" \
    -exec patchelf --clear-execstack {} \;

COPY config/ /app/config/
COPY src/ /app/src/

RUN mkdir -p /app/data/raw /app/data/annotations /app/data/outputs /app/models

CMD ["python3"]
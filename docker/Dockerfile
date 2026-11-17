# Headless, CPU-only, pinned environment for the vpipe segment -> render -> ink chain.
# Native build deps mirror ScrollPrize/villa .github/workflows/test-vesuvius.yml (ubuntu 24.04);
# VC3D CLI runtime libs follow ScrollPrize/villa#2153.
FROM docker.io/library/ubuntu:24.04

ARG VILLA_REF=2d3fb38d33a18b257230c0bd23986436c3d4722e
ARG VC3D_TAG=stable
ARG VC3D_ASSET=VC3D-f7756c7-2026-11-03-linux-x86_64.AppImage
ARG UV_VERSION=0.13.8

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PATH=/opt/uv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git ninja-build pkg-config cmake \
      libblosc-dev libblosc2-dev libboost-dev libceres-dev libconcurrentqueue-dev \
      libcurl4-openssl-dev libeigen3-dev liblz4-dev libopencv-contrib-dev libopencv-dev \
      libsuitesparse-dev libjpeg-dev libpng-dev libtiff-dev libzstd-dev nlohmann-json3-dev \
      qt6-base-dev zlib1g-dev libsqlite3-dev libssl-dev \
      libegl1 libglx0 libopengl0 libice6 libsm6 libx11-xcb1 libx11-6 libcom-err2 libexpat1 \
      libfontconfig1 libfreetype6 libgmp10 libgpg-error0 libharfbuzz0b libuuid1 libxcb1 \
      time \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/opt/uv/bin sh

# villa at a pinned commit (blobless clone keeps the image small)
# a stalled transfer aborts after 60 s below 1 KB/s and is retried, instead of hanging the build
RUN for i in 1 2 3 4 5; do \
      rm -rf /opt/villa; \
      git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone --filter=blob:none \
        https://github.com/ScrollPrize/villa.git /opt/villa && break; \
      echo "clone attempt $i failed"; sleep 10; \
    done \
 && git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 -C /opt/villa checkout -q "${VILLA_REF}" \
 && git -C /opt/villa rev-parse HEAD > /opt/villa.REF

WORKDIR /opt/villa/vesuvius
RUN uv sync --extra models --frozen \
 && uv run --no-sync python -c "import torch, vc_reader; print('torch', torch.__version__)"

# VC3D CLI tools, extracted (no FUSE in containers) and called through AppRun
RUN mkdir -p /opt/vc3d && cd /opt/vc3d \
 && curl -fL --retry 5 -o vc3d.AppImage "https://github.com/ScrollPrize/villa/releases/download/${VC3D_TAG}/${VC3D_ASSET}" \
 && sha256sum vc3d.AppImage | tee vc3d.AppImage.sha256 \
 && chmod +x vc3d.AppImage && ./vc3d.AppImage --appimage-extract >/dev/null && rm vc3d.AppImage \
 && printf '#!/bin/sh\nexec /opt/vc3d/squashfs-root/AppRun "$@"\n' > /usr/local/bin/vc3d && chmod +x /usr/local/bin/vc3d \
 && vc3d vc_render_tifxyz --help >/dev/null

COPY pyproject.toml /opt/vpipe/
COPY src /opt/vpipe/src
COPY scripts /opt/vpipe/scripts
RUN uv pip install --python /opt/villa/vesuvius/.venv/bin/python /opt/vpipe

ENV VIRTUAL_ENV=/opt/villa/vesuvius/.venv PATH=/opt/villa/vesuvius/.venv/bin:/opt/uv/bin:$PATH
WORKDIR /work

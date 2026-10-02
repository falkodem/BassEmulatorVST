FROM registry.k.avito.ru/avito/service-python/cuda/runtime/devel:3.11-cuda11.7.1-cudnn8.5.0.96

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        libsndfile1 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --upgrade \
        pip \
        setuptools \
        wheel

COPY docker/pesto-train/requirements.txt ./requirements.txt

RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 \
        torchaudio==2.5.1 \
    && python3 -m pip install --break-system-packages -r requirements.txt

COPY ml ./ml

ARG HOST_UID=11098
ARG HOST_GID=11098

RUN groupadd --gid "${HOST_GID}" trainloop \
    && useradd --uid "${HOST_UID}" --gid trainloop --create-home --shell /bin/bash trainloop

ENV HOME=/home/trainloop
USER trainloop

CMD ["bash"]

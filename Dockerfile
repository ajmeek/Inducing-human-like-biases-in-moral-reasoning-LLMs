FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN add-apt-repository universe && \
    apt update && \
    apt install python3.9

RUN apt install \
        -y \
        --no-install-recommends \
        apt-utils \
        binutils \
        build-essential \
        ca-certificates \
        curl \
        gcc \
        git \
        htop \
        less \
        libaio-dev \
        libxext6 \
        libx11-6 \
        libglib2.0-0 \
        libxrender1 \
        libxtst6 \
        libxi6 \
        locales \
        nano \
        ninja-build \
        python3-dev \
        python3-setuptools \
        python3-venv \
        python3-pip \
        screen \
        ssh \
        sudo \
        tmux \
        unzip \
        vim \
        datalad \
        wget && \
    python3 -m \
        pip install \
        --no-cache-dir \
        --upgrade \
        pip

RUN python3 -m \
    pip install \
    --no-cache-dir \
    transformers   \
    torch==1.13.1 \
    torchaudio  \
    torchvision  \
    pandas  \
    lightning[extra]  \
    bids  \
    nilearn  \
    nibabel  \
    sympy 

WORKDIR /src
COPY . .

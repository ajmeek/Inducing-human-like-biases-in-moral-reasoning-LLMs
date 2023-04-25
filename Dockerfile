FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt update
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
    wget

RUN python3 -m \
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
    sympy \
    datalad-installer

CMD datalad-installer --sudo ok datalad git-annex -m datalad/git-annex:release 

WORKDIR /src
COPY . .

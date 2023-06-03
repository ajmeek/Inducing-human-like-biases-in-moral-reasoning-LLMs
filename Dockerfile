FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# To avoid being asked re timezone by tzdata:
ENV TZ=Europe/London
ENV DEBIAN_FRONTEND=noninteractive  

RUN apt update && \
    apt install \
    -y \
    --no-install-recommends \
    apt-utils \
    binutils \
    build-essential \
    ca-certificates \
    curl \
    datalad \
    gcc \
    git \
    htop \
    less \
    libaio-dev \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    locales \
    nano \
    ninja-build \
    pipenv \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3 \
    screen \
    ssh \
    sudo \
    tmux \
    tzdata \
    unzip \
    vim \
    wget

RUN python3 -m pipenv --python $( which python3 ) install

WORKDIR /src
COPY . .

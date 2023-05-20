FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# To avoid being asked re timezone by tzdata:
ENV TZ=Europe/London
ENV DEBIAN_FRONTEND=noninteractive  

RUN apt update && \
    apt install \
    -y \
    --no-install-recommends \
    tzdata \
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
    python3.10 \
    python3-dev \
    python3-setuptools \
    python3-venv \
    python3-pip \
    pipenv \
    screen \
    ssh \
    sudo \
    tmux \
    unzip \
    vim \
    datalad \
    wget

RUN pipenv --python $( which python3.10 ) install

WORKDIR /src
COPY . .

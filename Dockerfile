FROM gcr.io/tpu-pytorch/xla:r2.0_3.8_tpuvm

# To avoid being asked re timezone by tzdata:
#ENV TZ=Europe/London
#ENV DEBIAN_FRONTEND=noninteractive  
#
#RUN apt update && \
#    apt upgrade -y && \
#    apt install software-properties-common -y && \
#    add-apt-repository ppa:deadsnakes/ppa
#
#RUN apt install \
#    -y \
#    --no-install-recommends \
#    apt-utils \
#    binutils \
#    build-essential \
#    ca-certificates \
#    curl \
#    datalad \
#    gcc \
#    git \
#    htop \
#    less \
#    libaio-dev \
#    libglib2.0-0 \
#    libx11-6 \
#    libxext6 \
#    libxi6 \
#    libxrender1 \
#    libxtst6 \
#    locales \
#    nano \
#    ninja-build \
#    pipenv \
#    python3-dev \
#    python3-pip \
#    python3-setuptools \
#    python3-venv \
#    python3.10 \
#    screen \
#    ssh \
#    sudo \
#    tmux \
#    tzdata \
#    unzip \
#    vim \
#    wget

RUN python3 -m pip install pipenv && pipenv --python $( which python3 ) install

WORKDIR /src
COPY . .

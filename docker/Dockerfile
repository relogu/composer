# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

######################
# Base Image Arguments
######################

# CUDA Version
# For a slim CPU-only image, leave the CUDA_VERSION argument blank -- e.g.
# ARG CUDA_VERSION=
ARG CUDA_VERSION=11.3.1

# Calculate the base image based on CUDA_VERSION
ARG BASE_IMAGE=${CUDA_VERSION:+"nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04"}
ARG BASE_IMAGE=${BASE_IMAGE:-"ubuntu:22.04"}

# The Python version to install
ARG PYTHON_VERSION=3.10

# The Pytorch Version to install
ARG PYTORCH_VERSION=1.13.1

# The Torchvision version to install.
# Reference https://github.com/pytorch/vision#installation to determine the Torchvision
# version that corresponds to the PyTorch version
ARG TORCHVISION_VERSION=0.14.1

# Version of the Mellanox Drivers to install (for InfiniBand support)
# Leave blank for no Mellanox Drivers
ARG MOFED_VERSION=5.5-1.0.3.2

# Version of EFA Drivers to install (for AWS Elastic Fabric Adapter support)
# Leave blank for no EFA Drivers
ARG AWS_OFI_NCCL_VERSION=v1.7.4-aws

# Upgrade certifi to resolve CVE-2022-23491
ARG CERTIFI_VERSION='>=2022.12.7'

# Upgrade ipython to resolve CVE-2023-24816
ARG IPYTHON_VERSION='>=8.10.0'

# Upgrade urllib to resolve CVE-2021-33503
ARG URLLIB3_VERSION='>=1.26.5,<2'

##########################
# Composer Image Arguments
##########################

# Build the composer image on the pytorch image
ARG COMPOSER_BASE=pytorch_stage

# The command that is passed to `pip install` -- e.g. `pip install "${COMPOSER_INSTALL_COMMAND}"`
ARG COMPOSER_INSTALL_COMMAND='mosaicml[all]'

#########################
# Build the PyTorch Image
#########################

FROM ${BASE_IMAGE} AS pytorch_stage
ARG DEBIAN_FRONTEND=noninteractive

#######################
# Set the shell to bash
#######################
SHELL ["/bin/bash", "-c"]

ARG CUDA_VERSION

# Remove a bad symlink from the base composer image
# If this file is present after the first command, kaniko
# won't be able to build the docker image.
RUN if [ -n "$CUDA_VERSION" ]; then \
        rm -f /usr/local/cuda-$(echo $CUDA_VERSION | cut -c -4)/cuda-$(echo $CUDA_VERSION | cut -c -4); \
    fi


# update repository keys
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN if [ -n "$CUDA_VERSION" ] ; then \
        rm -f /etc/apt/sources.list.d/cuda.list && \
        rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
        apt-get update &&  \
        apt-get install -y --no-install-recommends wget && \
        apt-get autoclean && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* \
        apt-key del 7fa2af80 && \
        mkdir -p /tmp/cuda-keyring && \
        wget -P /tmp/cuda-keyring https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
        dpkg -i /tmp/cuda-keyring/cuda-keyring_1.0-1_all.deb && \
        rm -rf /tmp/cuda-keyring ; \
    fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        wget \
        sudo \
        build-essential \
        software-properties-common \
        dirmngr \
        apt-utils \
        gpg-agent \
        openssh-client \
        # For PILLOW:
        zlib1g-dev \
        libtiff-dev \
        libfreetype6-dev \
        liblcms2-dev \
        tcl \
        libjpeg8-dev \
        less \
        libsnappy-dev \
        # For AWS EFA:
        autoconf \
        autotools-dev \
        automake \
        libtool \
        # Compressors
        bzip2 \
        gzip \
        lz4 \
        lzop \
        xz-utils \
        zstd \
        # Development tools
        tmux \
        htop && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

###############################
# Install latest version of git
###############################
RUN add-apt-repository ppa:git-core/ppa && \
    apt-get install -y --no-install-recommends \
        git && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

##############################
# Install NodeJS (for Pyright)
##############################
RUN \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

################
# Install Python
################
ARG PYTHON_VERSION

# Python 3.10 changes where packages are installed. Workaround until all packages support new format
ENV DEB_PYTHON_INSTALL_LAYOUT=deb

# Python 3.12 no longer supports distutils
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-apt \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} - && \
    pip${PYTHON_VERSION} install --no-cache-dir --upgrade pip 'setuptools<79.0.0'

#################
# Install Pytorch
#################
ARG PYTORCH_VERSION
ARG PYTORCH_NIGHTLY_URL
ARG PYTORCH_NIGHTLY_VERSION
ARG TORCHVISION_VERSION

# Set so supporting PyTorch packages such as Torchvision pin PyTorch version
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
ENV PYTORCH_NIGHTLY_URL=${PYTORCH_NIGHTLY_URL}
ENV PYTORCH_NIGHTLY_VERSION=${PYTORCH_NIGHTLY_VERSION}

RUN if [ -z "$PYTORCH_NIGHTLY_URL" ] ; then \
        CUDA_VERSION_TAG=$(python${PYTHON_VERSION} -c "print('cu' + ''.join('${CUDA_VERSION}'.split('.')[:2]) if '${CUDA_VERSION}' else 'cpu')") && \
        pip${PYTHON_VERSION} install --no-cache-dir --find-links https://download.pytorch.org/whl/torch/ \
            torch==${PYTORCH_VERSION}+${CUDA_VERSION_TAG} && \
        pip${PYTHON_VERSION} install --no-cache-dir --find-links https://download.pytorch.org/whl/torchvision/ \
            torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION_TAG} ; \
    else \
        pip${PYTHON_VERSION} install --no-cache-dir --pre --index-url ${PYTORCH_NIGHTLY_URL} \
            torch==${PYTORCH_VERSION}.${PYTORCH_NIGHTLY_VERSION} \
	        torchvision==${TORCHVISION_VERSION}.${PYTORCH_NIGHTLY_VERSION} ; \
    fi

#####################################
# Install EFA and AWS-OFI-NCCL plugin
#####################################

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH
ENV FI_EFA_USE_DEVICE_RDMA=1

RUN if [ -n "$AWS_OFI_NCCL_VERSION" ] ; then \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        hwloc \
        libhwloc-dev && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ; \
    fi

RUN if [ -n "$AWS_OFI_NCCL_VERSION" ] ; then \
        cd /tmp && \
        curl -OsS https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
        tar -xf /tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
        cd aws-efa-installer && \
        apt-get update && \
        ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
        rm -rf /tmp/aws-efa-installer* ; \
    fi

RUN if [ -n "$AWS_OFI_NCCL_VERSION" ] ; then \
        git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl && \
        cd /opt/aws-ofi-nccl && \
        git checkout ${AWS_OFI_NCCL_VERSION} && \
        ./autogen.sh && \
        ./configure --prefix=/opt/aws-ofi-nccl/install \
            --with-libfabric=/opt/amazon/efa/ \
            --with-cuda=/usr/local/cuda \
            --disable-tests \
            --enable-platform-aws && \
        make && make install ; \
    fi

###################################
# Mellanox OFED driver installation
###################################

ARG MOFED_VERSION

RUN if [ -n "$MOFED_VERSION" ] ; then \
        wget -qO - http://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | sudo apt-key add - && \
        wget -P /etc/apt/sources.list.d/ http://linux.mellanox.com/public/repo/mlnx_ofed/$MOFED_VERSION/ubuntu22.04/mellanox_mlnx_ofed.list && \
        apt-get update && \
        apt-get install -y mlnx-ofed-dpdk-upstream-libs-user-only && \
	pip uninstall -y netifaces && pip install netifaces==0.11.0 ; \
    fi

##########################
# Install Flash Attention
##########################
RUN if [ -n "$CUDA_VERSION" ] ; then \
        pip${PYTHON_VERSION} install --upgrade --no-cache-dir ninja==1.11.1 && \
        pip${PYTHON_VERSION} install --upgrade --no-cache-dir wheel && \
        pip${PYTHON_VERSION} install --upgrade --no-cache-dir --force-reinstall packaging==22.0 && \
        MAJOR_CUDA_VERSION=$(echo $CUDA_VERSION | cut -d. -f1) && \
        CUDA_STRING="cu${MAJOR_CUDA_VERSION}" && \
        PYTORCH_MAJOR=$(echo $PYTORCH_VERSION | cut -d. -f1) && \
        PYTORCH_MINOR=$(echo $PYTORCH_VERSION | cut -d. -f2) && \
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1) && \
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2) && \
        WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+${CUDA_STRING}torch${PYTORCH_MAJOR}.${PYTORCH_MINOR}cxx11abiFALSE-cp${PYTHON_MAJOR}${PYTHON_MINOR}-cp${PYTHON_MAJOR}${PYTHON_MINOR}-linux_x86_64.whl" && \
        echo "Installing Flash Attention from: $WHEEL_URL" && \
        pip${PYTHON_VERSION} install --no-cache-dir $WHEEL_URL || \
        (echo "Pre-built wheel not found, falling back to source installation" && \
         MAX_JOBS=1 pip${PYTHON_VERSION} install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1); \
        cd .. ; \
    fi

###############
# Install cmake
###############
RUN pip${PYTHON_VERSION} install --no-cache-dir cmake==3.26.3

###########################
# Install Pandoc Dependency
###########################
RUN pip${PYTHON_VERSION} install --no-cache-dir pandoc==2.3

################################
# Use the correct python version
################################

# Set the default python by creating our own folder and hacking the path
# We don't want to use upgrade-alternatives as that will break system packages

ARG COMPOSER_PYTHON_BIN=/composer-python

RUN mkdir -p ${COMPOSER_PYTHON_BIN} && \
    ln -s $(which python${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/python && \
    ln -s $(which python${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/python3 && \
    ln -s $(which python${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/python${PYTHON_VERSION} && \
    ln -s $(which pip${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/pip && \
    ln -s $(which pip${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/pip3 && \
    ln -s $(which pip${PYTHON_VERSION}) ${COMPOSER_PYTHON_BIN}/pip${PYTHON_VERSION} && \
    # Include this folder, and the local bin folder, on the path
    echo "export PATH=~/.local/bin:$COMPOSER_PYTHON_BIN:$PATH" >> /etc/profile && \
    echo "export PATH=~/.local/bin:$COMPOSER_PYTHON_BIN:$PATH" >> /etc/bash.bashrc && \
    echo "export PATH=~/.local/bin:$COMPOSER_PYTHON_BIN:$PATH" >> /etc/zshenv

# Ensure that non-interactive shells load /etc/profile
ENV BASH_ENV=/etc/profile

#########################
# Configure non-root user
#########################
RUN useradd -rm -d /home/mosaicml -s /bin/bash -u 1000 -U -s /bin/bash mosaicml && \
    usermod -a -G sudo mosaicml && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

#########################
# Upgrade apt packages
#########################
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#########################
# Upgrade pip packages
#########################
RUN pip install --no-cache-dir --upgrade \
        certifi${CERTIFI_VERSION} \
        ipython${IPYTHON_VERSION} \
        urllib3${URLLIB3_VERSION} \
        python-snappy

RUN apt-get remove -y python3-blinker
RUN pip install blinker

##################################################
# Override NVIDIA mistaken env var for 11.8 images
##################################################
ARG NVIDIA_REQUIRE_CUDA_OVERRIDE
ENV NVIDIA_REQUIRE_CUDA=${NVIDIA_REQUIRE_CUDA_OVERRIDE:-$NVIDIA_REQUIRE_CUDA}

################
# Composer Image
################

FROM ${COMPOSER_BASE} as composer_stage

ARG DEBIAN_FRONTEND=noninteractive

##################
# Install Composer
##################

ARG COMPOSER_INSTALL_COMMAND

RUN pip install "${COMPOSER_INSTALL_COMMAND}"

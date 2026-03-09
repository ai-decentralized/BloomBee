FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer="Yotta Labs"
LABEL repository="bloombee"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  git \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install -y python~=3.10.12 pip && \
    python -m pip install --no-cache-dir "torch>=1.12" && \
    conda clean --all -y && rm -rf ~/.cache/pip

VOLUME /cache
ENV BLOOMBEE_CACHE=/cache

COPY . bloombee/
RUN python -m pip install --no-cache-dir -e bloombee

WORKDIR /home/bloombee/
CMD bash

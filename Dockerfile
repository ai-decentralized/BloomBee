FROM nvcr.io/nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
LABEL maintainer="Yotta Labs"
LABEL repository="bloombee"
ARG HIVEMIND_REF=4bd43b77895019b20d18d81d0d0c1a5ab9a10847

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

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda install -y python~=3.10.12 pip && \
    python -m pip install --no-cache-dir "torch>=1.12" wheel grpcio-tools && \
    git clone https://github.com/learning-at-home/hivemind.git /tmp/hivemind && \
    git -C /tmp/hivemind checkout "${HIVEMIND_REF}" && \
    python -c "from pathlib import Path; pyproject = Path('/tmp/hivemind/pyproject.toml'); text = pyproject.read_text(); old = 'license = \"MIT\"'; new = 'license = {text = \"MIT\"}'; (old in text) or (_ for _ in ()).throw(SystemExit('Expected Hivemind pyproject license field not found')); pyproject.write_text(text.replace(old, new, 1))" && \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/hivemind && \
    conda clean --all -y && rm -rf ~/.cache/pip /tmp/hivemind

VOLUME /cache
ENV BLOOMBEE_CACHE=/cache

COPY . bloombee/
RUN python -c "from pathlib import Path; needle = 'hivemind @ git+https://github.com/learning-at-home/hivemind.git@${HIVEMIND_REF}'; paths = (Path('/home/bloombee/setup.py'), Path('/home/bloombee/setup.cfg')); [path.write_text('\\n'.join(line for line in path.read_text().splitlines() if needle not in line) + '\\n') for path in paths]" && \
    python -m pip install --no-cache-dir --no-build-isolation -e /home/bloombee

WORKDIR /home/bloombee/
CMD ["bash"]

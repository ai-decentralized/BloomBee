FROM yottalabsai/pytorch:2.9.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
LABEL maintainer="Yotta Labs"
LABEL repository="bloombee"
ARG HIVEMIND_REF=4bd43b77895019b20d18d81d0d0c1a5ab9a10847

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
ENV JUPYTER_PASSWORD=bloombee-dev
ENV BLOOMBEE_CACHE=/home/user/.cache/bloombee
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /home/user
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  wget \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

RUN id -u user >/dev/null 2>&1 || useradd -m -s /bin/bash user

RUN if ! command -v conda >/dev/null 2>&1; then \
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install_miniconda.sh && \
      bash /tmp/install_miniconda.sh -b -p "${CONDA_DIR}" && \
      rm -f /tmp/install_miniconda.sh; \
    fi && \
    if conda --help 2>/dev/null | grep -q '\btos\b'; then \
      conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
      conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    fi && \
    conda create -n bb python=3.10 -y

COPY . /home/user/BloomBee
COPY docker/start-yotta-dev.sh /usr/local/bin/start-yotta-dev.sh

RUN source "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    conda activate bb && \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir wheel grpcio-tools && \
    git clone https://github.com/learning-at-home/hivemind.git /tmp/hivemind && \
    git -C /tmp/hivemind checkout "${HIVEMIND_REF}" && \
    python -c "from pathlib import Path; pyproject = Path('/tmp/hivemind/pyproject.toml'); text = pyproject.read_text(); old = 'license = \"MIT\"'; new = 'license = {text = \"MIT\"}'; (old in text) or (_ for _ in ()).throw(SystemExit('Expected Hivemind pyproject license field not found')); pyproject.write_text(text.replace(old, new, 1))" && \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/hivemind && \
    python -c "from pathlib import Path; needle = 'hivemind @ git+https://github.com/learning-at-home/hivemind.git@${HIVEMIND_REF}'; paths = (Path('/home/user/BloomBee/setup.py'), Path('/home/user/BloomBee/setup.cfg')); [path.write_text('\\n'.join(line for line in path.read_text().splitlines() if needle not in line) + '\\n') for path in paths]" && \
    python -m pip install --no-cache-dir --no-build-isolation -e /home/user/BloomBee && \
    conda clean --all -y && \
    rm -rf ~/.cache/pip /tmp/hivemind && \
    mkdir -p /home/user/.cache/bloombee /home/user/.local /home/user/.cursor-server /home/user/.vscode-server /home/user/.vscode-remote /home/user/.npm-global && \
    printf '\nsource %s/etc/profile.d/conda.sh\nconda activate bb\n' "${CONDA_DIR}" >> /home/user/.bashrc && \
    printf '\nexport PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$PATH"\n' >> /home/user/.bashrc && \
    chmod +x /usr/local/bin/start-yotta-dev.sh && \
    chown -R user:user /home/user /opt/conda/envs/bb && \
    chmod 755 /home/user

EXPOSE 22 8888 31340

WORKDIR /home/user/BloomBee
CMD ["/usr/local/bin/start-yotta-dev.sh"]

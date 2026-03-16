FROM yottalabsai/pytorch:2.9.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
LABEL maintainer="Yotta Labs"
LABEL repository="bloombee"
ARG HIVEMIND_REF=4bd43b77895019b20d18d81d0d0c1a5ab9a10847

ENV DEBIAN_FRONTEND=noninteractive
ENV JUPYTER_PASSWORD=bloombee-dev
ENV BLOOMBEE_CACHE=/home/user/.cache/bloombee

WORKDIR /home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/*

RUN id -u user >/dev/null 2>&1 || useradd -m -s /bin/bash user

COPY . /home/user/BloomBee

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir wheel grpcio-tools && \
    git clone https://github.com/learning-at-home/hivemind.git /tmp/hivemind && \
    git -C /tmp/hivemind checkout "${HIVEMIND_REF}" && \
    python -c "from pathlib import Path; pyproject = Path('/tmp/hivemind/pyproject.toml'); text = pyproject.read_text(); old = 'license = \"MIT\"'; new = 'license = {text = \"MIT\"}'; (old in text) or (_ for _ in ()).throw(SystemExit('Expected Hivemind pyproject license field not found')); pyproject.write_text(text.replace(old, new, 1))" && \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps /tmp/hivemind && \
    python -c "from pathlib import Path; needle = 'hivemind @ git+https://github.com/learning-at-home/hivemind.git@${HIVEMIND_REF}'; paths = (Path('/home/user/BloomBee/setup.py'), Path('/home/user/BloomBee/setup.cfg')); [path.write_text('\\n'.join(line for line in path.read_text().splitlines() if needle not in line) + '\\n') for path in paths]" && \
    python -m pip install --no-cache-dir --no-build-isolation -e /home/user/BloomBee && \
    rm -rf ~/.cache/pip /tmp/hivemind && \
    mkdir -p /home/user/.cache/bloombee && \
    chown -R user:user /home/user/BloomBee /home/user/.cache

EXPOSE 22 8888 31340

WORKDIR /home/user/BloomBee

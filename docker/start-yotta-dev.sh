#!/bin/bash
set -euo pipefail

mkdir -p \
  /home/user \
  /home/user/.cache \
  /home/user/.cache/bloombee \
  /home/user/.cursor-server \
  /home/user/.local \
  /home/user/.vscode-server \
  /home/user/.vscode-remote \
  /home/user/.npm-global

chown -R user:user /home/user /opt/conda/envs/bb
chmod 755 /home/user

exec /start.sh

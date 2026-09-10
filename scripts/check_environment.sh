#!/usr/bin/env bash
set -euo pipefail

command -v docker
command -v nvidia-smi
docker version --format '{{.Server.Version}}'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .

if docker image inspect nvcr.io/nim/nvidia/canary-1b:latest >/dev/null 2>&1; then
  docker image inspect nvcr.io/nim/nvidia/canary-1b:latest --format '{{json .RepoDigests}} {{.Id}}'
else
  echo "NIM image is not present locally"
fi

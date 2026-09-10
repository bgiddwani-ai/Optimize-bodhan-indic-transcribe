#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINER_NAME=${CONTAINER_NAME:-indic-flex-latest}
NIM_IMAGE=${NIM_IMAGE:-nvcr.io/nim/nvidia/canary-1b:latest}
GPU_DEVICE=${GPU_DEVICE:-1}

: "${HF_MODEL_DIR:?Set HF_MODEL_DIR to the downloaded Indic-Transcribe Flex checkpoint}"
: "${BASE_CANARY_DIR:?Set BASE_CANARY_DIR to the unpacked compatible Canary NeMo archive}"
: "${DATA_DIR:?Set DATA_DIR to a writable artifact directory}"
: "${NEMO_SRC:?Set NEMO_SRC to the reconstructed NeMo source tree}"

for required_dir in "$HF_MODEL_DIR" "$BASE_CANARY_DIR" "$DATA_DIR" "$NEMO_SRC"; do
  test -d "$required_dir" || { echo "Missing directory: $required_dir" >&2; exit 1; }
done

docker run --name "$CONTAINER_NAME" \
  --gpus "device=$GPU_DEVICE" \
  --network host \
  --ipc host \
  --ulimit nofile=1048576:1048576 \
  -it \
  -v "$REPO_ROOT:/workspace" \
  -v "$REPO_ROOT/workflow:/home/CanaryNIM-Bharat-Final" \
  -v "$NEMO_SRC:/home/CanaryNIM-Bharat-Final/NeMo" \
  -v "$DATA_DIR:/data" \
  -v "$HF_MODEL_DIR:/source-model:ro" \
  -v "$BASE_CANARY_DIR:/base-canary:ro" \
  "$NIM_IMAGE" \
  /bin/bash

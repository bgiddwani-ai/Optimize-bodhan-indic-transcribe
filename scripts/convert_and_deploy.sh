#!/usr/bin/env bash
set -euo pipefail

test -d /source-model
test -d /base-canary
test -d /data
test -d /home/CanaryNIM-Bharat-Final/NeMo

python3 /workspace/tools/hf_to_nemo.py \
  --hf-dir /source-model \
  --base-dir /base-canary \
  --output /data/indic-transcribe-flex.nemo \
  --report /data/hf_to_nemo_report.json

python3 /workspace/tools/validate_nemo.py \
  --nemo /data/indic-transcribe-flex.nemo \
  --hf-dir /source-model \
  --report /data/nemo_validation_report.json

cd /home/CanaryNIM-Bharat-Final
bash 01_nemo2riva.sh
bash 02_conformer_onnx_to_trt.sh

python3 /workspace/tools/validate_trt_engine.py \
  --engine /data/indic-transcribe-flex-trtllm-engine/encoder/encoder.plan \
  --report /data/trt_encoder_validation_report.json

bash 03_riva2rmir.sh
bash 04_rmir2repo.sh
bash 05_replace_trt_encoder_and_canary_bls.sh

python3 /workspace/tools/validate_riva_vocab.py \
  --vocab /data/models/riva-trt-canary-1b-multi-asr-offline-am-streaming-offline/1/engine/decoder/vocab.json \
  --report /data/riva_vocab_validation_report.json

echo "Conversion and deployment completed. Start the service with /workspace/scripts/start_server.sh"

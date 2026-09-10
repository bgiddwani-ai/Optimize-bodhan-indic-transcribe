#!/usr/bin/env bash
set -euo pipefail

python3 /opt/NeMo/examples/asr/speech_to_text_eval.py \
    dataset_manifest=/data/dataset/hi64/hi_manifest_clean_final_inference_results.json \
    use_cer=True \
    only_score_manifest=True

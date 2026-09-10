#!/usr/bin/env bash
set -euo pipefail

python3 merge-manifest.py \
    /data/dataset/hi64/hi_manifest_cleaned_final.json \
    /data/dataset/hi64/64_indicvoices_random_indic_flex_nim_riva_trt.json \
    --output /data/dataset/hi64/hi_manifest_clean_final_inference_results.json

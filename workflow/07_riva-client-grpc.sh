#!/usr/bin/env bash
set -euo pipefail

riva_asr_client \
    --automatic_punctuation=true \
    --word_time_offsets=false \
    --print_transcripts=false \
    --num_iterations=1 \
    --language_code=hi \
    --audio_file=/data/dataset/hi64/hi_manifest_cleaned_final.json \
    --num_parallel_requests=64 \
    --output_filename=/data/dataset/hi64/64_indicvoices_random_indic_flex_nim_riva_trt.json

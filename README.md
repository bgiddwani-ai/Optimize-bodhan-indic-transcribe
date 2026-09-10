<<<<<<< HEAD
# Indic-Transcribe Flex: NeMo, TensorRT, and Riva optimization

This repository preserves the code and measured evidence used to convert [`bodhan-ai/indic-transcribe-flex`](https://huggingface.co/bodhan-ai/indic-transcribe-flex) from its Hugging Face Safetensors checkpoint into a NeMo `EncDecMultiTaskModel`, export it to Riva, build TensorRT and TensorRT-LLM engines, deploy an RMIR model repository, run Riva/Triton, and benchmark the service.

The validated deployment ran on Brev instance `minimum-fuchsia-mule`. The host has two NVIDIA H100 NVL GPUs; the captured `indic-flex-latest` container used GPU 1 for a single Riva replica.

## Repository layout

```text
.
├── workflow/                  # Canary-to-Riva/TensorRT workflow used on the instance
├── tools/                     # HF-to-NeMo conversion and validation utilities
├── scripts/                   # Reproducible container, build, server, and benchmark entry points
├── third_party/               # Pinned NeMo revision, patch, and added source files
├── results/                   # Reports, summaries, raw transcripts, and client logs
├── docs/                      # Status snapshot and technical slide deck
├── README.md
└── LEARNINGS.md
```

Generated model weights and engines are intentionally excluded. Rebuilding them requires the gated Flex checkpoint, a compatible Canary NeMo archive, and NVIDIA container access.

## Validated environment

| Component | Validated value |
|---|---|
| Host | Brev `minimum-fuchsia-mule` |
| GPU inventory | 2 × NVIDIA H100 NVL |
| Deployed replica | GPU 1 only |
| Container | `nvcr.io/nim/nvidia/canary-1b:latest` |
| Riva | 2.24.0 |
| NeMo | 2.8.0rc0 |
| TensorRT | 10.13.2.6 |
| TensorRT-LLM | 1.1.0 |
| PyTorch | 2.8.0 |
| NeMo source revision | `801051a88f11a97b8606212632db71b90606435d` plus `third_party/nemo-local.patch` and `nemo_overlay/` |
| nemo2riva revision | `68c578f0a5e617d97578d2041b9a37d4409d3879` |

The generated `/data` tree occupied approximately 22 GB, the Flex checkpoint 4.6 GB, and the unpacked base Canary archive 3.8 GB. Container layers require additional space; start with at least 100 GB free.

## Installation

### 1. Prerequisites

- Linux host with an NVIDIA GPU supported by the selected NIM image
- NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit
- Git, Git LFS, `rsync`, Python 3, and at least 100 GB of free disk
- Access to the gated Flex model and NVIDIA NGC container registry

Do not write Hugging Face or NGC tokens into this repository. Authenticate interactively or pass secrets through a secure shell prompt:

```bash
huggingface-cli login
read -s NGC_API_KEY
printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
unset NGC_API_KEY
```

### 2. Reconstruct the supplied NeMo source

The 587 MB third-party checkout is represented by a pinned upstream commit, a tracked patch, and four added files:

```bash
./scripts/reconstruct_nemo.sh "$PWD/vendor/NeMo"
```

Review [`third_party/README.md`](third_party/README.md) for provenance and manual reconstruction steps.

### 3. Prepare model inputs

Set up these host directories:

```text
<HF_MODEL_DIR>/
  config.json
  model.safetensors
  tokenizer_spl_tokens.model
  tokenizer_multilingual.model

<BASE_CANARY_DIR>/
  model_config.yaml
  model_weights.ckpt
  tokenizer files from a compatible Canary NeMo archive

<DATA_DIR>/
  writable output directory
```

The base Canary archive supplies the NeMo structure and preprocessing tensors. The converter replaces model weights, model dimensions, prompt format, and tokenizer assets with Flex values.

### 4. Pull and enter the NIM container

```bash
export HF_MODEL_DIR=/absolute/path/to/indic-transcribe-flex
export BASE_CANARY_DIR=/absolute/path/to/unpacked-canary-nemo
export DATA_DIR=/absolute/path/to/indic-flex-data
export NEMO_SRC="$PWD/vendor/NeMo"
export GPU_DEVICE=1

docker pull nvcr.io/nim/nvidia/canary-1b:latest
./scripts/run_container.sh
```

Set `GPU_DEVICE=0` on a one-GPU host. The launcher uses host networking and host IPC, sets the `nofile` limit to 1,048,576, and mounts the repository at `/workspace`.

## Conversion and deployment

Run inside the container started above:

```bash
/workspace/scripts/convert_and_deploy.sh
```

The entry point performs these stages:

1. Map all Hugging Face tensors into a NeMo `EncDecMultiTaskModel` archive.
2. Restore the `.nemo` file and compare source tensors exactly.
3. Export `.riva`, TensorRT-LLM checkpoint data, and the ONNX encoder.
4. Build the TensorRT encoder with a maximum batch size of 64 and feature length 3001.
5. Run `riva-build`, `riva-deploy`, and replace the generated encoder/BLS files.
6. Set Triton preferred batches to 32 and 64 with a 1 ms queue delay.
7. Validate the TensorRT encoder and deployed decoder vocabulary.

Expected artifacts under `/data`:

```text
indic-transcribe-flex.nemo
indic-transcribe-flex.riva
indic-transcribe-flex-trtllm-checkpoint/
indic-transcribe-flex-trtllm-engine/
indic-transcribe-flex-trt-bs64-decoupled.rmir
models/
hf_to_nemo_report.json
nemo_validation_report.json
trt_encoder_validation_report.json
```

## Running the server

Run inside the container and keep it in a persistent `screen` or `tmux` session:

```bash
/workspace/scripts/start_server.sh
```

Readiness checks:

```bash
curl -fsS http://localhost:8002/v2/health/ready
curl -fsS http://localhost:8002/v2/models/riva-trt-canary-1b-multi-asr-offline-am-streaming-offline/ready
```

Offline gRPC smoke test:

```bash
riva_asr_client \
  --automatic_punctuation=true \
  --word_time_offsets=false \
  --print_transcripts=true \
  --language_code=hi \
  --audio_file=/data/dataset/sample_hi.flac \
  --num_parallel_requests=1
```

## Running the concurrency benchmark

Install the small benchmark-only Python dependencies inside the container if they are missing:

```bash
python3 -m pip install -r /workspace/requirements-benchmark.txt
```

Prepare 1-second, 5-second, full-clip, and 15-second workloads from a 64-row, 16 kHz mono source manifest:

```bash
python3 /workspace/scripts/prepare_chunk_sweep.py \
  --source-manifest /data/dataset/hi64/hi_manifest_cleaned_final.json \
  --output-root /data/dataset/chunk_sweep
```

Run two trials at concurrency 64, 128, and 256:

```bash
python3 /workspace/scripts/run_chunk_sweep.py \
  --root /data/dataset/chunk_sweep \
  --durations 1s 5s full 15s \
  --concurrencies 64 128 256 \
  --trials 2
```

Validate exact chunk duration, response counts, non-empty transcripts, and deterministic repeated outputs:

```bash
python3 /workspace/scripts/validate_chunk_sweep.py \
  --root /data/dataset/chunk_sweep \
  --durations 1s 5s full 15s \
  --concurrencies 64 128 256 \
  --trials 2
```

RTFX is calculated as total audio seconds divided by client run time. Higher is better. It measures throughput, not response latency.

## Measured performance

Median RTFX across two one-pass trials:

| Audio workload | c64 | c128 | c256 |
|---|---:|---:|---:|
| 1 second | 118.41 | 147.08 | 173.29 |
| 5 seconds | 138.76 | 175.20 | 216.90 |
| Full clips, 6.92 seconds average | 275.15 | 346.65 | 410.05 |
| 15 seconds | 392.68 | 471.03 | 536.54 |

Median P95 client latency in milliseconds:

| Audio workload | c64 | c128 | c256 |
|---|---:|---:|---:|
| 1 second | 536.97 | 861.87 | 1,462.80 |
| 5 seconds | 2,297.95 | 3,653.80 | 5,870.65 |
| Full clips, 6.92 seconds average | 1,601.35 | 2,529.55 | 4,277.90 |
| 15 seconds | 2,262.45 | 4,040.70 | 6,607.40 |

The separate sustained concurrency-64 run used four iterations and processed 1,771.2 audio seconds in 3.6292 seconds: **488.05 RTFX**, 903.65 ms median latency, and 1,005.5 ms P95 latency. It reuses client and server state and therefore forms a different workload class from the one-pass sweep.

### Correctness evidence

- 1,923 source tensors mapped and restored with zero missing, unexpected, unequal, or shape-mismatched tensors.
- 64-output and sustained 256-output comparisons produced zero transcript mismatches against the prior Flex baseline.
- The 1-second sweep produced 896 of 896 non-empty responses. Repeated trials were deterministic.
- The final throughput sweep did not rerun WER/CER. Treat parity as a regression gate, not a substitute for a production accuracy evaluation.

See [`results/sweep_summary.csv`](results/sweep_summary.csv), [`results/sweep_summary.json`](results/sweep_summary.json), [`results/integrity_summary.json`](results/integrity_summary.json), and the per-run logs and transcripts under `results/`.

## Important operating notes

- The current single replica uses one H100. A second replica on the other GPU is a proposed improvement, not a measured result.
- Riva can divide a concurrency wave before it reaches the Python backend. Inspect Triton `batch_stats`, `inference_count`, and `execution_count` before attributing low throughput to the TensorRT engine.
- The final BLS coalesces requests within each execute list. It cannot merge separate Riva front-end waves.
- The TensorRT-LLM KV-cache free-memory fraction is `0.3`. The earlier `0.9` setting reduced memory headroom and caused instability.
- Stop stale benchmark clients and wait for GPU memory to release before restarting Triton.
- Chunked audio benchmarks can cross source-utterance boundaries. Use them for serving throughput, not WER.

Detailed observations and improvement priorities are in [`LEARNINGS.md`](LEARNINGS.md).

## Licensing

The Flex checkpoint uses the Indic Open Model License and inherits terms from its Canary base model. NeMo and nemo2riva retain their upstream licenses. Review all model, NGC container, and source-code licenses before redistributing artifacts or operating a hosted API.
=======
# Optimize-bodhan-indic-transcribe
Optimize canary with NIM
>>>>>>> 80f422273734f6dd4905b852ef3edd71f7463998

# Learnings

## 1. Architectural compatibility does not guarantee serving parity

Indic-Transcribe Flex follows the Canary-2 structure: a 32-layer FastConformer encoder, a 24-layer Transformer decoder, and a 7,152-token aggregate vocabulary. That compatibility enabled conversion, but RTFX still depended on request grouping, Riva front-end behavior, Triton scheduling, memory reservation, and clip duration.

Compare architectures only after matching the full inference boundary. A native engine benchmark, a Riva client benchmark, and a sustained multi-iteration benchmark answer different questions.

## 2. Exact checkpoint validation prevented silent conversion errors

The conversion mapped 1,923 Hugging Face tensors into NeMo names. Validation required:

- zero missing tensors
- zero unexpected tensors
- zero shape mismatches
- exact tensor equality after NeMo restore
- an exact tied language-model head
- tokenizer offsets and special token IDs verified independently

The restored tokenizer used `bos_id=4`, `eos_id=3`, and `pad_id=2`. EOS ID 3 was correct. Long or fixed-length decoder outputs should be diagnosed by inspecting EOS positions and sequence lengths before changing token configuration.

## 3. Build TensorRT artifacts in the serving runtime

The first encoder build exposed TensorRT runtime compatibility risk between build and serving containers. Rebuilding the encoder in `nvcr.io/nim/nvidia/canary-1b:latest` removed that boundary.

Pin and record the image digest, TensorRT version, TensorRT-LLM version, NeMo revision, engine profile, and GPU architecture for every reproducible build.

## 4. Triton counters locate the real batching boundary

The Riva client/front end sometimes delivered a concurrency-64 wave as groups such as 1 and 63. Increasing queue delay alone could not guarantee a single 64-request execution.

Use these Triton fields together:

- `inference_count` confirms how many requests reached the model
- `execution_count` shows how many backend executions served them
- `batch_stats` reveals the actual GPU batch sizes

The custom BLS coalesces requests within a Triton execute list. It does not merge separate lists emitted by the Riva front end.

## 5. Short clips are dominated by fixed serving overhead

At concurrency 256, median RTFX rose from 173.29 for 1-second chunks to 536.54 for 15-second chunks. The model architecture was unchanged. Longer audio amortized request setup, serialization, scheduling, and decoder overhead.

Report duration alongside concurrency. A single RTFX number without the audio-length distribution is not portable.

## 6. Memory headroom matters during repeated load

A TensorRT-LLM KV-cache free-memory fraction of 0.9 left insufficient headroom and contributed to unstable repeated runs. The final configuration used 0.3.

Stale clients and incomplete server shutdowns can hold GPU allocations. Before restarting Triton:

1. Stop benchmark clients.
2. Confirm the server processes exited.
3. Wait for GPU memory to release.
4. Start one clean server instance.

## 7. Correctness and speed require separate gates

The final conversion passed exact tensor checks, real NeMo inference, finite TensorRT encoder output, vocabulary validation, and transcript parity. Performance tuning proceeded only after those checks.

Transcript parity protects against deployment regressions, but it does not replace WER/CER evaluation. The final 1-second, 5-second, full, and 15-second throughput sweep did not recompute WER. Production acceptance still needs per-language accuracy, noise, code-mixing, and low-resource-language evaluation.

## 8. Benchmark methodology must remain explicit

The one-pass sweep used exactly N manifest rows and N parallel requests for each concurrency. Every duration/concurrency pair had two trials.

- 1 second: 64 evenly spaced items from 442 exact chunks
- 5 seconds: 64 evenly spaced items from 88 exact chunks
- full clips: the original 64 IndicVoices clips, averaging 6.92 seconds
- 15 seconds: 29 exact chunks cycled into a deterministic 64-item base

The 64-item base was repeated twice for concurrency 128 and four times for concurrency 256. This controls request count, but the 15-second set has lower unique-audio diversity. Concatenated chunks may cross utterance boundaries and should not be used for accuracy claims.

## 9. The sustained run is a separate workload class

The four-iteration concurrency-64 run achieved 488.05 RTFX by processing 1,771.2 audio seconds in 3.6292 seconds. It reused the same client and server across iterations. Compare this result with other steady-state runs, not directly with one-pass figures.

## 10. Current scale-out opportunity

The host contains two H100 NVL GPUs, while the captured container used GPU 1 only. A second independent Riva replica with load balancing is the next high-confidence capacity experiment.

Measure per-GPU and aggregate throughput, GPU memory, response latency, and transcript parity. Do not assume linear scaling before running the full concurrency matrix.

## Improvement priorities

1. Reduce the Riva/gRPC/Python BLS request-path overhead for 1-second workloads.
2. Benchmark two independent replicas across the two H100 GPUs.
3. Add duration buckets and tune preferred batch sizes and queue delay per workload.
4. Evaluate FP8 or INT8 and CUDA graphs only with matched accuracy and finite-output gates.
5. Rerun WER/CER by language and noise condition under the final serving configuration.

## Security and reproducibility

The gated model and NGC image require credentials. Authenticate interactively and never store raw tokens in scripts, shell history, logs, Git, or benchmark reports. The repository intentionally contains no credentials, model weights, TensorRT engines, RMIRs, or deployed model repository.

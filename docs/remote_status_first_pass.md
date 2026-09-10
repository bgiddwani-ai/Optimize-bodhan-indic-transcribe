# Indic Transcribe Flex optimization status

Completed on Brev instance `minimum-fuchsia-mule` in:

`/home/nvidia/indic-transcribe-optimization`

## Result

The Hugging Face checkpoint `bodhan-ai/indic-transcribe-flex` was converted to a restorable NeMo archive and passed through the attached Indic Canary optimization workflow to produce Riva, RMIR, TensorRT-LLM, and deployed model-repository artifacts.

The final serving test used `nvcr.io/nim/nvidia/canary-1b:1.5.0` on one H100 NVL. The encoder was rebuilt in that exact NIM image after a TensorRT runtime compatibility failure was detected with the first NeMo-container build.

## Primary artifacts

- NeMo: `/home/nvidia/indic-transcribe-optimization/data/indic-transcribe-flex.nemo`
- Riva: `/home/nvidia/indic-transcribe-optimization/data/indic-transcribe-flex.riva`
- RMIR: `/home/nvidia/indic-transcribe-optimization/data/indic-transcribe-flex-trt-bs64-decoupled.rmir`
- TRT-LLM checkpoint: `/home/nvidia/indic-transcribe-optimization/data/indic-transcribe-flex-trtllm-checkpoint`
- TRT-LLM engine: `/home/nvidia/indic-transcribe-optimization/data/indic-transcribe-flex-trtllm-engine`
- Riva model repository: `/home/nvidia/indic-transcribe-optimization/data/models`
- Checksums: `/home/nvidia/indic-transcribe-optimization/data/artifact_sha256.txt`

## NeMo conversion validation

- Model class: `EncDecMultiTaskModel`
- Parameters: 1,221,439,472
- Encoder layers: 32
- Decoder layers: 24
- Model width: 1024
- Vocabulary: 7,152 tokens
- State tensors checked against Hugging Face: 1,923/1,923 exactly equal
- Unequal tensors: 0
- Tied language-model head: exact
- NeMo SHA-256: `e26d6d111addbec8ace2a6eedd1a5c61c1430215c863f00ef349b34babe8c729`

Reports:

- `/home/nvidia/indic-transcribe-optimization/data/hf_to_nemo_report.json`
- `/home/nvidia/indic-transcribe-optimization/data/nemo_validation_report.json`
- `/home/nvidia/indic-transcribe-optimization/data/nemo_inference_validation_report.json`

The NeMo archive also completed a real encoder/decoder inference pass on a 5.729-second Hindi sample. Its normalized transcript matched the original Hugging Face result exactly. The archive exposes aggregate tokenizer groups named `spl_tokens` and `multilingual`, so stock NeMo's convenience `transcribe(target_lang="hi")` alias is not directly available; the validation used the model's real prompt IDs and generation path. The deployed custom Canary BLS maps all language codes directly.

## TensorRT and Riva validation

- Final encoder plan SHA-256: `76a1818c01337797cb15489814fafd3439c11240df8ac46e1919e686de595129`
- The source and deployed encoder plans have identical checksums.
- Encoder plan deserialized and executed at its minimum optimization profile.
- Output shape: `[1, 4, 1024]`
- Output was finite; mean absolute value: `0.12766054272651672`
- Decoder engine SHA-256: `ad345596a83ebd7e770a0445a6092ef000dd9a06ac39fa1880dd144d247f35aa`
- Riva vocabulary: 7,152 contiguous IDs
- All 27 configured Indic language prompt tokens present
- All required Canary2 prompt tokens present

Reports:

- `/home/nvidia/indic-transcribe-optimization/data/trt_encoder_validation_report_nim.json`
- `/home/nvidia/indic-transcribe-optimization/data/riva_vocab_validation_report.json`
- `/home/nvidia/indic-transcribe-optimization/data/inference_parity_report.json`

The single-sample Riva transcript matched both NeMo and Hugging Face exactly after outer-whitespace normalization.

## 64-request benchmark and evaluation

The cached IndicVoices Hindi source was filtered before inference. Of 67 rows read, 3 `unintelligible` rows were excluded and 64 were accepted, with zero preparation errors.

- Samples: 64/64 responses
- Total audio: 442.808 seconds
- Parallel requests: 64
- Run time: 1.8643 seconds
- Aggregate throughput: 237.51 RTFX
- Median latency: 1849.7 ms
- P90 latency: 1854.8 ms
- P95 latency: 1855.7 ms
- P99 latency: 1857.7 ms
- Average latency: 1483.6 ms
- WER: 10.72%
- CER: 5.04%

Integrity checks found 64 reference rows, 64 result rows, 64 merged rows, zero missing references, zero missing predictions, and zero duplicate audio paths.

Benchmark artifacts:

- `/home/nvidia/indic-transcribe-optimization/data/dataset/hi64/prepare_summary.json`
- `/home/nvidia/indic-transcribe-optimization/data/dataset/hi64/hi_manifest_cleaned_final.json`
- `/home/nvidia/indic-transcribe-optimization/data/dataset/hi64/64_indicvoices_random_indic_flex_nim_riva_trt.json`
- `/home/nvidia/indic-transcribe-optimization/data/dataset/hi64/hi_manifest_clean_final_inference_results.json`
- `/home/nvidia/indic-transcribe-optimization/data/dataset/hi64/integrity_summary.json`
- `/home/nvidia/indic-transcribe-optimization/data/dataset/hi64/eval.log`

## Reproduction notes

- Attached workflow: `/home/nvidia/indic-transcribe-optimization/indic-canary-optimization-main/CanaryNIM-Bharat-Final`
- Public custom exporter checkout: commit `68c578f0`
- NeMo conversion/runtime image: `nvcr.io/nvidia/nemo:25.02` (NeMo 2.2.0rc4)
- Serving/runtime image: `nvcr.io/nim/nvidia/canary-1b:1.5.0`
- Serving image digest: `sha256:2b23453997c74409916f962f0efe5949cdfaa0392a67608348f03f8bc54b592b`
- Authentication was performed interactively without writing credentials into project files or reports.

The working scripts and full session log remain under `/home/nvidia/indic-transcribe-optimization`.

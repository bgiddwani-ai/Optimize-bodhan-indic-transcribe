#!/usr/bin/env python3
"""Run one real-audio NeMo transcription and compare it to the prior HF result."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--hf-tokenizer-config", required=True)
    parser.add_argument("--hf-prediction")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    model = EncDecMultiTaskModel.restore_from(args.nemo, map_location="cpu")
    model = model.cuda().eval()
    restore_seconds = time.perf_counter() - started
    decoding_cfg = model.cfg.decoding
    decoding_cfg.beam.max_generation_delta = int(model.cfg.beam_search.max_generation_delta)
    model.change_decoding_strategy(decoding_cfg)

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16 kHz audio, got {sample_rate}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).cuda()
    audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device="cuda")
    tokenizer_config = json.loads(Path(args.hf_tokenizer_config).read_text())
    prompt_ids = tokenizer_config["prompt_ids_by_lang"][args.language]

    infer_started = time.perf_counter()
    with torch.inference_mode():
        _, _, encoder_states, encoder_mask = model.forward(
            input_signal=audio_tensor,
            input_signal_length=audio_length,
        )
        hypotheses = model.decoding.decode_predictions_tensor(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_mask,
            decoder_input_ids=torch.tensor(
                [prompt_ids], dtype=torch.long, device="cuda"
            ),
            return_hypotheses=True,
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - infer_started
    hypothesis = hypotheses
    while isinstance(hypothesis, (list, tuple)):
        hypothesis = hypothesis[0]
    token_ids = hypothesis.y_sequence
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    prediction_raw = model.tokenizer.ids_to_text(token_ids)
    prediction = prediction_raw.strip()

    prior = None
    if args.hf_prediction:
        prior = json.loads(Path(args.hf_prediction).read_text().splitlines()[0])
    prior_text = prior.get("pred_text") if prior else None
    report = {
        "audio": args.audio,
        "language": args.language,
        "prompt_ids": prompt_ids,
        "generated_token_ids": token_ids,
        "nemo_hypothesis_text": getattr(hypothesis, "text", None),
        "prediction": prediction,
        "prediction_raw": prediction_raw,
        "prediction_nonempty": bool(prediction.strip()),
        "prior_hf_prediction": prior_text,
        "exact_match_prior_hf": prediction == prior_text if prior_text is not None else None,
        "restore_seconds": restore_seconds,
        "inference_seconds": inference_seconds,
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["prediction_nonempty"]:
        raise SystemExit("Empty transcription")


if __name__ == "__main__":
    main()

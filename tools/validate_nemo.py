#!/usr/bin/env python3
"""Restore and audit the converted Flex NeMo archive against HF weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from nemo.collections.asr.models import EncDecMultiTaskModel


def hf_to_nemo_name(name: str) -> str:
    if name.startswith("model.encoder."):
        return name[len("model.") :]
    if name.startswith("model.decoder.embedding."):
        return "transf_decoder._embedding." + name[len("model.decoder.embedding.") :]
    if name.startswith("model.decoder.layers."):
        return "transf_decoder._decoder.layers." + name[len("model.decoder.layers.") :]
    if name.startswith("model.decoder.final_layer_norm."):
        return "transf_decoder._decoder.final_layer_norm." + name[
            len("model.decoder.final_layer_norm.") :
        ]
    if name.startswith("lm_head."):
        return "log_softmax.mlp.layer0." + name[len("lm_head.") :]
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    model = EncDecMultiTaskModel.restore_from(str(args.nemo), map_location="cpu")
    model.eval()
    state = model.state_dict()
    unequal = []
    checked = 0
    with safe_open(args.hf_dir / "model.safetensors", framework="pt", device="cpu") as sf:
        for hf_name in sf.keys():
            nemo_name = hf_to_nemo_name(hf_name)
            if nemo_name not in state:
                unequal.append({"hf": hf_name, "nemo": nemo_name, "reason": "missing"})
                continue
            source = sf.get_tensor(hf_name)
            restored = state[nemo_name]
            checked += 1
            if source.shape != restored.shape or not torch.equal(source, restored):
                unequal.append(
                    {
                        "hf": hf_name,
                        "nemo": nemo_name,
                        "source_shape": list(source.shape),
                        "restored_shape": list(restored.shape),
                        "reason": "shape_or_value_mismatch",
                    }
                )

    embedding = state["transf_decoder._embedding.token_embedding.weight"]
    lm_head = state["log_softmax.mlp.layer0.weight"]
    report = {
        "restored_class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensors": len(state),
        "source_tensors_checked": checked,
        "encoder_layers": int(model.cfg.encoder.n_layers),
        "decoder_layers": int(model.cfg.transf_decoder.config_dict.num_layers),
        "vocab_size": int(model.tokenizer.vocab_size),
        "prompt_format": str(model.prompt_format),
        "tokenizer_lang_groups": list(model.tokenizer.langs),
        "bos_id": int(model.tokenizer.bos_id),
        "eos_id": int(model.tokenizer.eos_id),
        "pad_id": int(model.tokenizer.pad_id),
        "tied_lm_head_exact": bool(torch.equal(embedding, lm_head)),
        "unequal_source_tensors": unequal,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if unequal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild a NeMo Canary archive from the Indic-Transcribe HF checkpoint.

The Bodhan HF port deliberately mirrors the NeMo module layout.  This tool
maps those tensors back to EncDecMultiTaskModel names, uses the public
Canary-1B NeMo archive as a structural/configuration template, replaces the
tokenizer with the two tokenizer models shipped by Indic-Transcribe, and
emits an uncompressed .nemo tar archive plus a machine-readable audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
import yaml
from safetensors import safe_open


ENCODER_LAYER_RE = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    raise ValueError(f"unmapped HF tensor: {name}")


def expected_template(base_state: dict[str, torch.Tensor], encoder_layers: int) -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}
    layer_templates: dict[str, tuple[int, ...]] = {}
    for name, tensor in base_state.items():
        match = ENCODER_LAYER_RE.match(name)
        if match:
            if match.group(1) == "0":
                layer_templates[match.group(2)] = tuple(tensor.shape)
            if int(match.group(1)) >= encoder_layers:
                continue
        expected[name] = tuple(tensor.shape)
    for layer in range(encoder_layers):
        for suffix, shape in layer_templates.items():
            expected[f"encoder.layers.{layer}.{suffix}"] = shape
    return expected


def configure_model(base_config: dict, hf_config: dict) -> dict:
    config = base_config
    config["encoder"]["n_layers"] = int(hf_config["encoder_layers"])
    config["encoder"]["d_model"] = int(hf_config["d_model"])
    config["encoder"]["n_heads"] = int(hf_config["encoder_attention_heads"])
    config["encoder"]["conv_kernel_size"] = int(hf_config["conv_kernel_size"])
    config["encoder"]["subsampling_factor"] = int(hf_config["subsampling_factor"])
    config["encoder"]["subsampling_conv_channels"] = int(
        hf_config["subsampling_conv_channels"]
    )

    decoder = config["transf_decoder"]["config_dict"]
    decoder["max_sequence_length"] = int(hf_config["max_target_positions"])
    decoder["hidden_size"] = int(hf_config["d_model"])
    decoder["inner_size"] = int(hf_config["decoder_ffn_dim"])
    decoder["num_layers"] = int(hf_config["decoder_layers"])
    decoder["num_attention_heads"] = int(hf_config["decoder_attention_heads"])

    config["head"]["hidden_size"] = int(hf_config["d_model"])
    config["head"]["num_classes"] = int(hf_config["vocab_size"])
    config["model_defaults"]["asr_enc_hidden"] = int(hf_config["d_model"])
    if "lm_enc_hidden" in config["model_defaults"]:
        config["model_defaults"]["lm_enc_hidden"] = int(hf_config["d_model"])
    config["beam_search"]["max_generation_delta"] = int(hf_config["max_generation_delta"])
    if "decoding" in config and "beam" in config["decoding"]:
        config["decoding"]["beam"]["max_generation_delta"] = int(
            hf_config["max_generation_delta"]
        )
    if "loss" in config:
        config["loss"]["pad_id"] = int(hf_config["pad_token_id"])
    config["prompt_format"] = "canary2"

    def tok(path: str) -> dict:
        return {
            "dir": None,
            "type": "bpe",
            "model_path": f"nemo:{path}",
            "vocab_path": None,
            "spe_tokenizer_vocab": None,
        }

    config["tokenizer"]["dir"] = None
    config["tokenizer"]["type"] = "agg"
    config["tokenizer"]["langs"] = {
        "spl_tokens": tok("tokenizer_spl_tokens.model"),
        "multilingual": tok("tokenizer_multilingual.model"),
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    hf_config = json.loads((args.hf_dir / "config.json").read_text())
    base_config = yaml.safe_load((args.base_dir / "model_config.yaml").read_text())
    base_state_obj = torch.load(
        args.base_dir / "model_weights.ckpt", map_location="cpu", weights_only=True
    )
    base_state = base_state_obj.get("state_dict", base_state_obj)

    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    hf_shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(args.hf_dir / "model.safetensors", framework="pt", device="cpu") as sf:
        for hf_name in sf.keys():
            nemo_name = hf_to_nemo_name(hf_name)
            tensor = sf.get_tensor(hf_name)
            if nemo_name in mapped:
                raise ValueError(f"duplicate mapped tensor: {nemo_name}")
            mapped[nemo_name] = tensor
            hf_shapes[nemo_name] = tuple(tensor.shape)

    embedding_name = "transf_decoder._embedding.token_embedding.weight"
    lm_head_name = "log_softmax.mlp.layer0.weight"
    if lm_head_name not in mapped:
        mapped[lm_head_name] = mapped[embedding_name].clone()
        hf_shapes[lm_head_name] = tuple(mapped[lm_head_name].shape)

    for name in ("preprocessor.featurizer.window", "preprocessor.featurizer.fb"):
        mapped[name] = base_state[name].clone()

    expected = expected_template(base_state, int(hf_config["encoder_layers"]))
    expected[embedding_name] = tuple(mapped[embedding_name].shape)
    expected[lm_head_name] = tuple(mapped[lm_head_name].shape)
    expected["log_softmax.mlp.layer0.bias"] = tuple(mapped["log_softmax.mlp.layer0.bias"].shape)
    position_name = "transf_decoder._embedding.position_embedding.pos_enc"
    expected[position_name] = tuple(mapped[position_name].shape)

    # The base Canary archive has fewer encoder blocks.  Expected shapes for
    # extra blocks were expanded from block 0 above; all decoder dimensions are
    # checked against the original 24-layer Canary template.
    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    mismatched = sorted(
        {
            name: {"expected": expected[name], "actual": tuple(mapped[name].shape)}
            for name in set(expected) & set(mapped)
            if expected[name] != tuple(mapped[name].shape)
        }.items()
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            json.dumps(
                {"missing": missing, "unexpected": unexpected, "mismatched": mismatched},
                indent=2,
            )
        )

    output_config = configure_model(base_config, hf_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="indic-flex-nemo-") as temp:
        stage = Path(temp)
        torch.save(mapped, stage / "model_weights.ckpt")
        (stage / "model_config.yaml").write_text(
            yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True)
        )
        for name in ("tokenizer_spl_tokens.model", "tokenizer_multilingual.model"):
            shutil.copy2(args.hf_dir / name, stage / name)
        with tarfile.open(args.output, "w") as archive:
            for path in sorted(stage.iterdir()):
                archive.add(path, arcname=path.name, recursive=False)

    report = {
        "format": "NeMo EncDecMultiTaskModel archive",
        "source_hf_dir": str(args.hf_dir),
        "base_template_dir": str(args.base_dir),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "tensor_count": len(mapped),
        "encoder_layers": int(hf_config["encoder_layers"]),
        "decoder_layers": int(hf_config["decoder_layers"]),
        "vocab_size": int(hf_config["vocab_size"]),
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "shape_mismatches": mismatched,
        "tied_lm_head_exact": bool(torch.equal(mapped[lm_head_name], mapped[embedding_name])),
        "tokenizer_layout": {"spl_tokens": [0, 1152], "multilingual": [1152, 7152]},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

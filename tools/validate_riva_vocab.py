#!/usr/bin/env python3
"""Validate the exported Riva vocabulary and Canary2 prompt coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LANGUAGES = [
    "as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi",
    "hne", "kn", "ks", "kok", "mai", "ml", "mni", "mr", "ne", "or",
    "pa", "sa", "sat", "sd", "ta", "te", "ur",
]
PROMPT_TOKENS = [
    "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
    "<|pnc|>", "<|noitn|>", "<|noromanized|>", "<|notimestamp|>",
    "<|nodiarize|>",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    vocab = json.loads(Path(args.vocab).read_text())
    by_lang = {
        lang: {token: int(idx) for idx, token in mapping.items()}
        for lang, mapping in vocab["tokens"].items()
    }
    special = by_lang["spl_tokens"]
    missing_prompt_tokens = [token for token in PROMPT_TOKENS if token not in special]
    missing_languages = [lang for lang in LANGUAGES if f"<|{lang}|>" not in special]
    prompt_ids = {
        lang: [
            special["<|startofcontext|>"],
            vocab["bos_id"],
            special["<|emo:undefined|>"],
            special[f"<|{lang}|>"],
            special[f"<|{lang}|>"],
            special["<|pnc|>"],
            special["<|noitn|>"],
            special["<|noromanized|>"],
            special["<|notimestamp|>"],
            special["<|nodiarize|>"],
        ]
        for lang in LANGUAGES
        if f"<|{lang}|>" in special
    }
    all_ids = sorted(idx for mapping in by_lang.values() for idx in mapping.values())
    report = {
        "vocab_size_declared": vocab["size"],
        "token_count": len(all_ids),
        "id_min": min(all_ids),
        "id_max": max(all_ids),
        "ids_contiguous": all_ids == list(range(vocab["size"])),
        "offsets": vocab["offsets"],
        "languages_requested": LANGUAGES,
        "missing_languages": missing_languages,
        "missing_prompt_tokens": missing_prompt_tokens,
        "prompt_ids": prompt_ids,
        "bos_id": vocab.get("bos_id"),
        "eos_id": vocab.get("eos_id"),
        "pad_id": vocab.get("pad_id"),
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing_languages or missing_prompt_tokens or not report["ids_contiguous"]:
        raise SystemExit("Riva vocabulary validation failed")


if __name__ == "__main__":
    main()

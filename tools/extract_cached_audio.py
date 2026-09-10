#!/usr/bin/env python3
"""Extract one self-contained audio example from a cached HF Arrow dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Audio, Dataset


def scalar_preview(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--target-manifest")
    args = parser.parse_args()

    target_text = None
    if args.target_manifest:
        first_line = Path(args.target_manifest).read_text().splitlines()[0]
        target_text = str(json.loads(first_line)["text"])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    chosen = None
    for arrow in sorted(Path(args.cache).rglob("*.arrow")):
        ds = Dataset.from_file(str(arrow))
        if not len(ds):
            continue
        audio_col = next((c for c in ds.column_names if "audio" in c.lower()), None)
        if audio_col is None:
            continue
        raw_ds = ds.cast_column(audio_col, Audio(decode=False))
        row = raw_ds[0]
        preview = {
            key: scalar_preview(value)
            for key, value in row.items()
            if key != audio_col
        }
        entry = {
            "arrow": str(arrow),
            "rows": len(ds),
            "audio_column": audio_col,
            "preview": preview,
        }
        candidates.append(entry)
        raw_values = {str(v) for v in preview.values()}
        values = {value.strip().lower() for value in raw_values}
        if chosen is None and (
            target_text in raw_values
            or args.language.lower() in values
            or "hindi" in values
        ):
            chosen = (entry, row[audio_col])

    if chosen is None and candidates:
        entry = candidates[0]
        ds = Dataset.from_file(entry["arrow"]).cast_column(
            entry["audio_column"], Audio(decode=False)
        )
        chosen = (entry, ds[0][entry["audio_column"]])
    if chosen is None:
        raise RuntimeError("No cached Arrow dataset with an audio column found")

    entry, audio = chosen
    source_name = Path(audio.get("path") or "sample.flac").name
    suffix = Path(source_name).suffix or ".flac"
    output_audio = out / f"sample_{args.language}{suffix}"
    if audio.get("bytes") is not None:
        output_audio.write_bytes(audio["bytes"])
    elif audio.get("path"):
        output_audio.write_bytes(Path(audio["path"]).read_bytes())
    else:
        raise RuntimeError(f"Unsupported audio payload: {audio.keys()}")

    report = {
        "selected": entry,
        "output_audio": str(output_audio),
        "output_bytes": output_audio.stat().st_size,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    (out / "cached_audio_extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

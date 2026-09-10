#!/usr/bin/env python3
"""Create a persistent, unintelligible-filtered manifest from cached Arrow audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset


def reference_text(row: dict) -> str:
    for key in ("text", "normalized", "unsanitized_normalized", "verbatim"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("no populated transcript field")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-prefix", required=True)
    parser.add_argument("--language", default="hi")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    extraction = json.loads(Path(args.extraction_report).read_text())
    selected = extraction["selected"]
    audio_col = selected["audio_column"]
    ds = Dataset.from_file(selected["arrow"]).cast_column(audio_col, Audio(decode=False))
    output = Path(args.output_dir)
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "hi_manifest_cleaned_final.json"
    records = []
    stats = {"seen": 0, "accepted": 0, "filtered_unintelligible": 0, "errors": 0}

    for source_index, row in enumerate(ds):
        if len(records) >= args.samples:
            break
        stats["seen"] += 1
        try:
            text = reference_text(row)
            if "unintelligible" in text.casefold():
                stats["filtered_unintelligible"] += 1
                continue
            audio = row[audio_col]
            suffix = Path(audio.get("path") or "sample.flac").suffix or ".flac"
            name = f"{source_index:06d}{suffix}"
            local_audio = audio_dir / name
            if audio.get("bytes") is not None:
                local_audio.write_bytes(audio["bytes"])
            elif audio.get("path"):
                local_audio.write_bytes(Path(audio["path"]).read_bytes())
            else:
                raise ValueError("audio payload has neither bytes nor path")
            duration = float(row.get("duration") or sf.info(local_audio).duration)
            records.append(
                {
                    "audio_filepath": f"{args.manifest_prefix}/audio/{name}",
                    "duration": duration,
                    "text": text,
                    "lang": args.language,
                    "dataset_config": "hindi",
                    "dataset_split": "valid",
                    "source_index": source_index,
                }
            )
        except Exception:
            stats["errors"] += 1

    stats["accepted"] = len(records)
    if len(records) != args.samples:
        raise RuntimeError(f"Prepared {len(records)} samples; expected {args.samples}")
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    summary = {
        **stats,
        "source_arrow": selected["arrow"],
        "manifest": str(manifest_path),
        "total_audio_seconds": sum(row["duration"] for row in records),
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

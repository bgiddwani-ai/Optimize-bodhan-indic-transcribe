#!/usr/bin/env python3
"""Create deterministic 1 s, 5 s, full-clip, and 15 s Riva manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-size", type=int, default=64)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[64, 128, 256])
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_manifest)
    if len(source_rows) != args.base_size:
        raise ValueError(f"Expected {args.base_size} source rows, found {len(source_rows)}")
    if any(value % args.base_size for value in args.concurrencies):
        raise ValueError("Every concurrency must be a multiple of base-size")

    waveforms = []
    for row in source_rows:
        samples, sample_rate = sf.read(row["audio_filepath"], dtype="float32", always_2d=False)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected 16 kHz audio: {row['audio_filepath']} has {sample_rate} Hz")
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        waveforms.append(samples)

    merged = np.concatenate(waveforms)
    summary = {
        "source_manifest": str(args.source_manifest),
        "source_rows": len(source_rows),
        "source_audio_sec": len(merged) / SAMPLE_RATE,
        "base_size": args.base_size,
        "concurrencies": args.concurrencies,
        "workloads": {},
        "manifests": {},
    }

    bases: dict[str, list[dict]] = {"full": source_rows}
    summary["workloads"]["full"] = {
        "pool_chunks": len(source_rows),
        "selected_items": args.base_size,
        "average_duration_sec": sum(float(row["duration"]) for row in source_rows) / len(source_rows),
        "selection": "original source rows",
    }

    for duration_sec in (1, 5, 15):
        label = f"{duration_sec}s"
        chunk_samples = duration_sec * SAMPLE_RATE
        pool_size = len(merged) // chunk_samples
        discarded_samples = len(merged) - pool_size * chunk_samples
        audio_dir = args.output_root / label / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        pool = []
        for index in range(pool_size):
            audio_path = audio_dir / f"chunk_{index:04d}.flac"
            segment = merged[index * chunk_samples : (index + 1) * chunk_samples]
            sf.write(audio_path, segment, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            info = sf.info(audio_path)
            if info.frames != chunk_samples or info.samplerate != SAMPLE_RATE or info.channels != 1:
                raise RuntimeError(f"Invalid generated chunk: {audio_path}")
            pool.append({"audio_filepath": str(audio_path), "duration": float(duration_sec), "text": ""})

        if pool_size >= args.base_size:
            selected_indices = np.linspace(0, pool_size - 1, args.base_size, dtype=int).tolist()
            base = [pool[index] for index in selected_indices]
            selection = "evenly spaced across exact-chunk pool"
        else:
            selected_indices = [index % pool_size for index in range(args.base_size)]
            base = [pool[index] for index in selected_indices]
            selection = "cycled exact-chunk pool to base size"
        bases[label] = base
        summary["workloads"][label] = {
            "exact_chunk_sec": duration_sec,
            "pool_chunks": pool_size,
            "selected_items": len(base),
            "discarded_audio_sec": discarded_samples / SAMPLE_RATE,
            "selection": selection,
            "selected_indices": selected_indices,
        }

    for label, base in bases.items():
        for concurrency in args.concurrencies:
            rows = (base * (concurrency // args.base_size))[:concurrency]
            path = args.output_root / label / f"manifest_c{concurrency}.json"
            write_jsonl(path, rows)
            summary["manifests"][f"{label}_c{concurrency}"] = {
                "path": str(path),
                "rows": len(rows),
                "unique_audio_paths": len({row["audio_filepath"] for row in rows}),
                "total_audio_sec": sum(float(row["duration"]) for row in rows),
            }

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "workloads": summary["workloads"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

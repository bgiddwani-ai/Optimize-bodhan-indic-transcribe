#!/usr/bin/env python3
"""Validate chunk durations and deterministic Riva benchmark outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--durations", nargs="+", default=["1s", "5s", "full", "15s"])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    checks: dict[str, object] = {}
    path_transcripts: defaultdict[str, set[str]] = defaultdict(set)
    bad_audio = []

    for duration in args.durations:
        if duration != "full":
            duration_sec = int(duration.removesuffix("s"))
            expected_frames = duration_sec * 16000
            audio_paths = sorted((args.root / duration / "audio").glob("chunk_*.flac"))
            for path in audio_paths:
                info = sf.info(path)
                if info.samplerate != 16000 or info.frames != expected_frames or info.channels != 1:
                    bad_audio.append(
                        {
                            "path": str(path),
                            "samplerate": info.samplerate,
                            "frames": info.frames,
                            "channels": info.channels,
                        }
                    )
            checks[f"{duration}_audio_files"] = len(audio_paths)

        for concurrency in args.concurrencies:
            trial_rows = []
            for trial in range(1, args.trials + 1):
                path = args.root / "results" / f"{duration}_c{concurrency}_trial{trial}.json"
                rows = load_jsonl(path)
                trial_rows.append(rows)
                nonempty = sum(bool(row.get("text", "").strip()) for row in rows)
                checks[f"{duration}_c{concurrency}_trial{trial}"] = {
                    "rows": len(rows),
                    "expected_rows": concurrency,
                    "nonempty": nonempty,
                }
                for row in rows:
                    path_transcripts[row["audio_filepath"]].add(row.get("text", ""))

            baseline = Counter((row["audio_filepath"], row.get("text", "")) for row in trial_rows[0])
            checks[f"{duration}_c{concurrency}_deterministic"] = all(
                Counter((row["audio_filepath"], row.get("text", "")) for row in rows) == baseline
                for rows in trial_rows[1:]
            )

    result_checks_passed = all(
        value is True
        or (isinstance(value, int) and value > 0)
        or (
            isinstance(value, dict)
            and value["rows"] == value["expected_rows"]
            and value["nonempty"] == value["expected_rows"]
        )
        for value in checks.values()
    )
    summary = {
        "bad_audio": bad_audio,
        "result_checks": checks,
        "audio_paths_with_multiple_transcripts": sum(len(values) > 1 for values in path_transcripts.values()),
        "all_checks_passed": (
            not bad_audio
            and result_checks_passed
            and all(len(values) == 1 for values in path_transcripts.values())
        ),
    }
    (args.root / "integrity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    if not summary["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

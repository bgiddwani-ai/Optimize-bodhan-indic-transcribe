#!/usr/bin/env python3
"""Benchmark a running Riva service across duration and concurrency manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "riva-trt-canary-1b-multi-asr-offline-am-streaming-offline"


def model_stats(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        model = json.load(response)["model_stats"][0]
    return {
        "inference_count": int(model["inference_count"]),
        "execution_count": int(model["execution_count"]),
        "batch_stats": {
            str(item["batch_size"]): int(item["compute_infer"]["count"])
            for item in model.get("batch_stats", [])
        },
    }


def parse_client_output(text: str) -> dict:
    latency = re.search(
        r"Latencies \(ms\):\s*Median\s+90th\s+95th\s+99th\s+Avg\s*\n\s*"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        text,
    )
    runtime = re.search(r"Run time:\s*([0-9.]+) sec", text)
    audio = re.search(r"Total audio processed:\s*([0-9.]+) sec", text)
    rtfx = re.search(r"Throughput:\s*([0-9.]+) RTFX", text)
    if not all((latency, runtime, audio, rtfx)):
        raise RuntimeError("Could not parse riva_asr_client output")
    median_ms, p90_ms, p95_ms, p99_ms, average_ms = map(float, latency.groups())
    return {
        "run_time_sec": float(runtime.group(1)),
        "total_audio_sec": float(audio.group(1)),
        "rtfx": float(rtfx.group(1)),
        "latency_ms": {
            "median": median_ms,
            "p90": p90_ms,
            "p95": p95_ms,
            "p99": p99_ms,
            "average": average_ms,
        },
    }


def batch_delta(before: dict, after: dict) -> dict:
    keys = set(before["batch_stats"]) | set(after["batch_stats"])
    return {
        key: after["batch_stats"].get(key, 0) - before["batch_stats"].get(key, 0)
        for key in sorted(keys, key=int)
        if after["batch_stats"].get(key, 0) - before["batch_stats"].get(key, 0)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--durations", nargs="+", default=["1s", "5s", "full", "15s"])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--language-code", default="hi")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--triton-http", default="http://localhost:8002")
    args = parser.parse_args()

    results_dir = args.root / "results"
    logs_dir = args.root / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stats_url = f"{args.triton_http}/v2/models/{args.model}/stats"

    records = []
    for duration in args.durations:
        for concurrency in args.concurrencies:
            manifest = args.root / duration / f"manifest_c{concurrency}.json"
            manifest_rows = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
            if manifest_rows != concurrency:
                raise ValueError(f"{manifest}: expected {concurrency} rows, found {manifest_rows}")

            for trial in range(1, args.trials + 1):
                output = results_dir / f"{duration}_c{concurrency}_trial{trial}.json"
                log_path = logs_dir / f"{duration}_c{concurrency}_trial{trial}.log"
                if output.exists():
                    output.unlink()

                before = model_stats(stats_url)
                command = [
                    "riva_asr_client",
                    "--automatic_punctuation=true",
                    "--word_time_offsets=false",
                    "--print_transcripts=false",
                    "--num_iterations=1",
                    f"--language_code={args.language_code}",
                    f"--audio_file={manifest}",
                    f"--num_parallel_requests={concurrency}",
                    f"--output_filename={output}",
                ]
                started = time.perf_counter()
                process = subprocess.run(command, text=True, capture_output=True, check=True)
                wall_time_sec = time.perf_counter() - started
                after = model_stats(stats_url)
                combined = process.stdout + "\nSTDERR\n" + process.stderr
                log_path.write_text(combined, encoding="utf-8")

                response_rows = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                nonempty = sum(bool(row.get("text", "").strip()) for row in response_rows)
                parsed = parse_client_output(combined)
                record = {
                    "duration": duration,
                    "concurrency": concurrency,
                    "trial": trial,
                    "manifest": str(manifest),
                    "output": str(output),
                    "log": str(log_path),
                    **parsed,
                    "wall_time_sec": wall_time_sec,
                    "response_rows": len(response_rows),
                    "nonempty_transcripts": nonempty,
                    "triton_inference_delta": after["inference_count"] - before["inference_count"],
                    "triton_execution_delta": after["execution_count"] - before["execution_count"],
                    "triton_batch_size_counts_delta": batch_delta(before, after),
                }
                if record["response_rows"] != concurrency:
                    raise RuntimeError(f"Missing responses: {record}")
                if record["nonempty_transcripts"] != concurrency:
                    raise RuntimeError(f"Empty transcripts: {record}")
                if record["triton_inference_delta"] != concurrency:
                    raise RuntimeError(f"Triton request count mismatch: {record}")
                records.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)

    aggregates = []
    for duration in args.durations:
        for concurrency in args.concurrencies:
            group = [
                row for row in records
                if row["duration"] == duration and row["concurrency"] == concurrency
            ]
            aggregates.append(
                {
                    "duration": duration,
                    "concurrency": concurrency,
                    "trials": len(group),
                    "audio_seconds": group[0]["total_audio_sec"],
                    "rtfx_median": statistics.median(row["rtfx"] for row in group),
                    "rtfx_min": min(row["rtfx"] for row in group),
                    "rtfx_max": max(row["rtfx"] for row in group),
                    "run_seconds_median": statistics.median(row["run_time_sec"] for row in group),
                    "latency_median_ms": statistics.median(row["latency_ms"]["median"] for row in group),
                    "p95_ms_median": statistics.median(row["latency_ms"]["p95"] for row in group),
                    "executions_median": statistics.median(row["triton_execution_delta"] for row in group),
                    "all_outputs_present": all(row["response_rows"] == concurrency for row in group),
                    "all_predictions_nonempty": all(row["nonempty_transcripts"] == concurrency for row in group),
                }
            )

    summary = {
        "method": "one client iteration, exactly N manifest rows, N parallel requests",
        "trials_per_configuration": args.trials,
        "records": records,
        "aggregates": aggregates,
    }
    (args.root / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = list(aggregates[0])
    with (args.root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregates)


if __name__ == "__main__":
    main()

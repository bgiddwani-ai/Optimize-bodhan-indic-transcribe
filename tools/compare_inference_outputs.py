#!/usr/bin/env python3
"""Compare HF, reconstructed NeMo, and deployed Riva transcripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--riva", required=True)
    parser.add_argument("--nemo", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    riva = json.loads(Path(args.riva).read_text())["text"]
    nemo = json.loads(Path(args.nemo).read_text())["prediction"]
    hf = json.loads(Path(args.hf).read_text().splitlines()[0])["pred_text"]
    normalized = {"riva": riva.strip(), "nemo": nemo.strip(), "hf": hf.strip()}
    report = {
        "normalized_transcripts": normalized,
        "riva_matches_hf": normalized["riva"] == normalized["hf"],
        "nemo_matches_hf": normalized["nemo"] == normalized["hf"],
        "all_exact_after_outer_whitespace": len(set(normalized.values())) == 1,
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_exact_after_outer_whitespace"]:
        raise SystemExit("Transcript parity failed")


if __name__ == "__main__":
    main()

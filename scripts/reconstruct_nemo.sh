#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DESTINATION=${1:-"$REPO_ROOT/vendor/NeMo"}
NEMO_COMMIT=$(tr -d '[:space:]' < "$REPO_ROOT/third_party/nemo-commit.txt")

test ! -e "$DESTINATION" || { echo "Destination already exists: $DESTINATION" >&2; exit 1; }
mkdir -p "$(dirname "$DESTINATION")"

git clone https://github.com/NVIDIA/NeMo.git "$DESTINATION"
git -C "$DESTINATION" checkout "$NEMO_COMMIT"
git -C "$DESTINATION" apply "$REPO_ROOT/third_party/nemo-local.patch"
rsync -a "$REPO_ROOT/third_party/nemo_overlay/" "$DESTINATION/"

echo "NeMo reconstructed at $DESTINATION"
git -C "$DESTINATION" status --short

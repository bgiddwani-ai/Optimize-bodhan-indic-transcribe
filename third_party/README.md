# NeMo source reconstruction

The validated container mounted a NeMo working tree based on commit:

```text
801051a88f11a97b8606212632db71b90606435d
```

The supplied `NeMo.zip` working tree differed from that commit. This repository preserves those tracked changes in `nemo-local.patch` and the added source files in `nemo_overlay/`.

Run:

```bash
./scripts/reconstruct_nemo.sh "$PWD/vendor/NeMo"
```

Manual equivalent:

```bash
git clone https://github.com/NVIDIA/NeMo.git vendor/NeMo
git -C vendor/NeMo checkout 801051a88f11a97b8606212632db71b90606435d
git -C vendor/NeMo apply ../../third_party/nemo-local.patch
rsync -a third_party/nemo_overlay/ vendor/NeMo/
```

`third_party/nemo-status.txt` records the original working-tree state. `.claude/settings.local.json` was deliberately excluded because it is local configuration rather than runtime source code.

NeMo remains subject to its upstream license. This directory does not replace or relicense the upstream project.

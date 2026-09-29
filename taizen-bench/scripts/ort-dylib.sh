#!/usr/bin/env bash
# Prints the ONNX Runtime shared library that ort (load-dynamic) should open.
# ORT_VERSION must match what ort =2.0.0-rc.11 targets (1.23.x); a mismatch fails
# at session creation with an API-version error.
set -euo pipefail
ORT_VERSION="${ORT_VERSION:-1.23.2}"
uv run --no-project --with "onnxruntime==${ORT_VERSION}" python - <<'PY'
import glob, os, onnxruntime
libs = glob.glob(os.path.join(os.path.dirname(onnxruntime.__file__), "capi", "libonnxruntime*"))
print(sorted(libs, key=len)[0])
PY

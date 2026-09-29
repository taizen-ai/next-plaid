# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "pylate-onnx-export",
#     "torch==2.9.0",
#     "pylate==1.6.0",
#     "transformers==5.3.0",
#     "sentence-transformers==5.3.0",
#     "onnx==1.23.0",
#     "onnxscript==0.7.2",
#     "onnxruntime==1.30.0",
#     "huggingface-hub==1.33.0",
# ]
#
# [tool.uv.sources]
# pylate-onnx-export = { path = "../../next-plaid-onnx/python" }
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Export mLateOn at the pinned revision to ONNX (fp32 + int8) for next-plaid-onnx.

The upstream exporter chains each Dense layer's `linear` and drops the residual
branch that mLateOn's Dense layers use, so its graph is not the PyTorch model.
This swaps in a wrapper that applies the Dense layers the way PyLate does and
reuses everything else (tokenizer, onnx_config.json, int8 quantization).

    uv run taizen-bench/scripts/export_mlateon.py OUT_DIR

The environment is pinned above and locked in export_mlateon.py.lock, so the
exported graph changes only when this script or its lock changes.
"""

import json
import sys
from pathlib import Path

import torch
from colbert_export import export as upstream
from huggingface_hub import snapshot_download

MODEL = "lightonai/mLateOn"
REVISION = "edd378f99593c0ac8a15518b97ad89786b02685e"


class ColBERTWithResidualDense(upstream.ColBERTForONNX):
    def __init__(self, pylate_model, uses_token_type_ids: bool = True):
        super().__init__(pylate_model, uses_token_type_ids)
        self.dense_layers = torch.nn.ModuleList(m for m in list(pylate_model)[1:] if hasattr(m, "linear"))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        for dense in self.dense_layers:
            projected = dense.activation_function(dense.linear(hidden))
            if dense.use_residual:
                projected = projected + (dense.residual(hidden) if dense.in_features != dense.out_features else hidden)
            hidden = projected
        return torch.nn.functional.normalize(hidden, p=2, dim=-1)


def main(out_dir: Path) -> None:
    source = snapshot_download(MODEL, revision=REVISION, ignore_patterns=["*.onnx"])
    upstream.ColBERTForONNX = ColBERTWithResidualDense
    upstream.export_model(source, output_dir=out_dir, quantize=True, force=True)
    config_path = out_dir / "onnx_config.json"
    config = json.loads(config_path.read_text())
    config["model_name"] = f"{MODEL}@{REVISION}"
    config_path.write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

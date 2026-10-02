# /// script
# requires-python = "==3.12.*"
# dependencies = ["huggingface-hub==1.33.0", "pyarrow==21.0.0"]
# ///
"""Writes the first N BeIR/scifact abstracts (public) as `{"id", "text"}` lines:
off-domain documents for `plaid_buffer_check` to append to the fiqa index.

    uv run taizen-bench/scripts/scifact_sample.py OUT.jsonl 150
"""

import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

out, count = sys.argv[1], int(sys.argv[2])
path = hf_hub_download("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset")
rows = pq.read_table(path).slice(0, count).to_pylist()
with open(out, "w") as sink:
    for row in rows:
        sink.write(json.dumps({"id": f"scifact-{row['_id']}", "text": f"{row['title']}\n{row['text']}"}) + "\n")

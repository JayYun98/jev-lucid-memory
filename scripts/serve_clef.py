"""Serve a locally downloaded, revision-pinned Clef release. No cloud inference.

The model directory includes Cloudflare's reviewed joint_schema_model.py.
This loads that code explicitly; never point --model-path at an untrusted directory.
"""

import argparse
import importlib.util
import sys
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-path", type=Path, required=True)
p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
p.add_argument("--port", type=int, default=8766)
p.add_argument("--max-tokens", type=int, default=4096)
args = p.parse_args()
model_path = args.model_path.resolve()
spec = importlib.util.spec_from_file_location("clef_release", model_path / "joint_schema_model.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
model, processor = module.load_release_model(
    model_path, device=args.device, dtype=torch.bfloat16, attn_implementation="sdpa"
)
lock = threading.Lock()
app = FastAPI(docs_url=None, redoc_url=None)


@app.get("/health")
def health():
    return {
        "model": "clef-flash",
        "revision": model_path.name,
        "device": args.device,
        "max_tokens": args.max_tokens,
    }


@app.post("/v1/systemone")
def decide(body: dict):
    # Text-only, bounded API. Reject excess inputs instead of silently losing evidence.
    if (
        set(body) - {"model", "state", "questions"}
        or not isinstance(body.get("questions"), dict)
        or not 1 <= len(body["questions"]) <= 205
    ):
        raise HTTPException(422, "invalid request")
    if body.get("model") != "clef-flash":
        raise HTTPException(422, "unknown model")
    if len(str(body)) > 160000:
        raise HTTPException(413, "request too large")
    try:
        encoded = module.encode_record(
            processor.tokenizer, body, max_length=1000000, processor=processor
        )
        if len(encoded.input_ids) > args.max_tokens:
            raise HTTPException(413, "token limit exceeded; no evidence was truncated")
        with lock:
            result = module.systemone(model, processor, body, max_length=args.max_tokens)
        result["usage"]["cost"] = 0.0  # API charges only; electricity/hardware are not estimated.
        return result
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422, "invalid question schema") from None


uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

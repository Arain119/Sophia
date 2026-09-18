"""HTTP chat backend for Sophia — stdlib only, one model resident.

    python -m ml.cli.chat_server --checkpoint sophia/sophia.pt [--port 8001]

POST /api/chat
  {"history": [{"role": "user"|"assistant", "content": str}, ...],
   "temperature": 0.7, "max_new_tokens": 320}
  → {"answer": str, "think": str, "hit_eos": bool, "elapsed_s": float}

GET /api/health → {"ok": true, "step": int, "device": str}

Generation is serialized: one rollout at a time on the resident engine.
Guards cap history length, per-message bytes and prompt budget so the
endpoint survives casual abuse on a small host.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from ml.training.rl.engine import RolloutEngine, chat_turn, load_policy

MAX_HISTORY = 20
MAX_MSG_CHARS = 4000
CTX = 4096

ENGINE = None
TOKENIZER = None
META = {}
LOCK = threading.Lock()


def _cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class Handler(BaseHTTPRequestHandler):
    server_version = "SophiaChat/1.0"

    def log_message(self, fmt: str, *args) -> None:  # noqa: D105 - quieter
        pass

    def _send(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        _cors(self)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send({"ok": True, **META})
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send({"error": "not found"}, 404)
            return
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 256 * 1024)
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send({"error": "bad json"}, 400)
            return

        raw_hist = req.get("history")
        if not isinstance(raw_hist, list) or not raw_hist:
            self._send({"error": "history required"}, 400)
            return
        history = []
        for m in raw_hist[-MAX_HISTORY:]:
            role = m.get("role")
            content = str(m.get("content", ""))[:MAX_MSG_CHARS]
            if role in ("user", "assistant") and content.strip():
                history.append({"role": role, "content": content})
        if not history or history[-1]["role"] != "user":
            self._send({"error": "history must end with a user turn"}, 400)
            return

        temperature = float(req.get("temperature", 0.7))
        temperature = min(max(temperature, 0.0), 2.0)
        max_new = min(max(int(req.get("max_new_tokens", 320)), 8), 1024)

        with LOCK:
            started = time.time()
            sample = chat_turn(
                ENGINE,
                TOKENIZER,
                history,
                temperature=temperature,
                max_new_tokens=max_new,
                ctx=CTX,
            )
            elapsed = time.time() - started

        self._send(
            {
                "answer": sample.answer,
                "think": sample.think,
                "hit_eos": bool(sample.hit_eos),
                "elapsed_s": round(elapsed, 2),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-spec", default="configs/model/sophia.json")
    parser.add_argument("--tokenizer-path", default="ml/modeling/text")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    global ENGINE, TOKENIZER, META
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    model, TOKENIZER, META = load_policy(
        checkpoint=args.checkpoint,
        model_spec=args.model_spec,
        tokenizer_path=args.tokenizer_path,
        batch_size=1,
        device=args.device,
        dtype=dtype,
    )
    ENGINE = RolloutEngine(
        model=model, tokenizer=TOKENIZER, device=args.device, max_batch=1
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"sophia chat server on {args.host}:{args.port} (step {META.get('step')})")
    server.serve_forever()


if __name__ == "__main__":
    main()

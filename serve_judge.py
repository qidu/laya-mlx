"""HTTP wrapper exposing Laya's typed decisions as a judge sidecar.

Serves the protocol in the proxy's
``docs/architecture/design_tool_judge_sidecar_protocol.md``:

    POST /judge   {"state": ..., "questions": {<qid>: <question>}} -> {<qid>: <answer>}
    GET  /health  -> {"status": "healthy", "model_version": ..., "mode": ...}

The question and answer objects are passed through unchanged: Laya's answer for a
``choice`` question already carries ``type``/``confidence``/``action``/``choice``/
``probabilities`` (doc §2.4) and for ``noul`` ``type``/``confidence``/``action``/
``noul`` (doc §2.5). The wrapper only unwraps the ``answers`` map from Laya's
envelope and drops ``model``/``usage``.

Run it (from this directory, or with the package installed):

    uv run serve_judge.py --model convaiinnovations/laya --port 8081

The model is loaded once at startup and then warmed with a throwaway inference
per question type, so the first real request does not pay MLX's lazy pipeline
setup; inference is serialized by a lock because a single MLX ``Agent`` is not
safe to drive from concurrent threads.
"""

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from laya_mlx import __version__
from laya_mlx.agent import DTYPES, Agent

MAX_BODY_BYTES = 8 * 1024 * 1024

log = logging.getLogger("serve_judge")

# Throwaway requests mirroring what the proxy actually sends (doc §4): one
# `choice` question for single-tool mode, two `noul` questions for batch mode.
# Both shapes are run so whichever mode the proxy is configured for meets a
# warm model on its first real request.
WARMUP_STATE = (
    "User: What is the weather in Paris today?\n\n"
    "Question: Should this tool be kept or discarded based on the user prompt and context?\n"
    "Tool:\n1. get_weather\n   - Get the current weather for a city."
)
WARMUP_CHOICE = {
    "decision": {
        "type": "choice",
        "instructions": "Should this tool be kept or discarded based on the user prompt and context?",
        "criteria": ["keep", "discard"],
    }
}
WARMUP_NOUL = {
    name: {"type": "noul", "instructions": f"Keep {name} tool?"}
    for name in ("Read", "get_weather")
}


def warmup(agent):
    """Run one throwaway inference per question type before the port is bound.

    MLX builds its Metal pipelines and allocates its buffers on first execution,
    so the first ``POST /judge`` after a restart costs noticeably more than
    steady state. Against the proxy's 50ms default ``timeout_ms`` that one-off
    can tip the first request into fail-open, which silently keeps every tool.
    Paying it here also proves the loaded model can actually answer, extending
    the load-before-bind contract above: startup fails loudly rather than
    serving 500s.

    Returns the total warm-up time in milliseconds.
    """
    total_ms = 0.0
    for label, questions in (("choice", WARMUP_CHOICE), ("noul", WARMUP_NOUL)):
        started = time.perf_counter()
        answers = agent.system_one(WARMUP_STATE, questions)["answers"]
        elapsed_ms = (time.perf_counter() - started) * 1000
        total_ms += elapsed_ms
        if set(answers) != set(questions):
            # Same check `do_POST` applies, so a model that cannot answer is
            # caught before the port opens rather than on the first client.
            raise RuntimeError(
                f"warm-up ({label}) answered {sorted(answers)}, expected {sorted(questions)}"
            )
        log.info("warm-up %s: %d question(s) in %.1fms", label, len(answers), elapsed_ms)
    return total_ms


def parse_request(payload):
    """Validate a ``POST /judge`` body and return ``(state, questions)``."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    state = payload.get("state")
    if not isinstance(state, str) or not state:
        raise ValueError("Request is missing a non-empty string 'state'")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Request is missing a non-empty 'questions' object")
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"Question {qid!r} must be a JSON object")
    return state, questions


class JudgeHandler(BaseHTTPRequestHandler):
    server_version = f"laya-judge/{__version__}"

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/health":
            self.send_error(404, "Not Found")
            return
        self._send_json(
            200,
            {
                "status": "healthy",
                "model_version": self.server.model_version,
                # The wrapper answers whatever question types it is sent, so it
                # reports both rather than claiming one exclusive mode.
                "mode": "choice+noul",
            },
        )

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/judge":
            self.send_error(404, "Not Found")
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        try:
            state, questions = parse_request(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        started = time.perf_counter()
        try:
            with self.server.lock:
                result = self.server.agent.system_one(state, questions)
        except ValueError as exc:
            # Laya rejects malformed question definitions with ValueError.
            log.warning("bad request: %s", exc)
            self._send_json(400, {"error": str(exc)})
            return
        except FloatingPointError as exc:
            log.error("inference produced non-finite outputs: %s", exc, exc_info=True)
            self._send_json(500, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - report any inference failure to the caller
            log.error("inference failed: %s", exc, exc_info=True)
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        elapsed_ms = (time.perf_counter() - started) * 1000

        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            log.error(
                "answer/question mismatch: asked %s, got %s",
                sorted(questions),
                sorted(answers) if isinstance(answers, dict) else type(answers).__name__,
            )
            self._send_json(500, {"error": "model returned answers for a different question set"})
            return

        log.info(
            "judged %d question(s) in %.1fms (%d input tokens)",
            len(answers),
            elapsed_ms,
            result.get("usage", {}).get("input_tokens", -1),
        )
        self._send_json(200, answers)

    def _read_json(self):
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Missing Content-Length")
        try:
            length = int(raw_length)
        except ValueError:
            raise ValueError(f"Invalid Content-Length: {raw_length!r}") from None
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError(f"Body length {length} out of range")
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Body is not valid UTF-8 JSON: {exc}") from None

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.debug("%s - %s", self.address_string(), fmt % args)


def build_parser():
    parser = argparse.ArgumentParser(prog="serve_judge.py", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--subfolder")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the startup inference; the first /judge then pays MLX's lazy setup",
    )
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load before binding so a broken checkpoint fails loudly at startup
    # instead of answering requests with 500s.
    log.info("loading %s (dtype=%s, device=%s)", args.model, args.dtype, args.device)
    agent = Agent(
        args.model,
        device=args.device,
        dtype=args.dtype,
        revision=args.revision,
        subfolder=args.subfolder,
        batch_size=args.batch_size,
    )

    # Still before the bind: warm-up failures mean every later request would
    # 500, and an un-warmed port would answer the first call too slowly to be
    # useful to the proxy's 50ms default timeout.
    if args.no_warmup:
        log.info("warm-up skipped (--no-warmup)")
    else:
        try:
            log.info("warming up on %s (one pass per question type)", args.device)
            log.info("warm-up done in %.1fms", warmup(agent))
        except Exception as exc:  # noqa: BLE001 - any failure here would 500 every request
            log.error("warm-up failed: %s", exc, exc_info=True)
            return 1

    server = ThreadingHTTPServer((args.host, args.port), JudgeHandler)
    server.agent = agent
    server.lock = threading.Lock()
    server.model_version = str(args.revision or agent.model_id)
    log.info("listening on http://%s:%d (model_version=%s)", args.host, args.port, server.model_version)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

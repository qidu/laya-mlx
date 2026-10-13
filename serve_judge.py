"""HTTP wrapper exposing Laya's typed decisions as a judge sidecar.

Serves the protocol in the proxy's
``docs/architecture/design_tool_judge_sidecar_protocol.md``:

    POST /judge     {"state": ..., "questions": {<qid>: <question>}} -> {<qid>: <answer>}
    POST /decision  {"model": ..., "state": ..., "questions": {...}}
                    -> {"model": ..., "answers": {...}, "usage": {...}}
    GET  /health    -> {"status": "healthy", "model_version": ..., "mode": ...}

For ``/judge`` the question and answer objects are passed through unchanged: Laya's
answer for a ``choice`` question already carries ``type``/``confidence``/``action``/
``choice``/``probabilities`` (doc §2.4) and for ``noul`` ``type``/``confidence``/
``action``/``noul`` (doc §2.5). The wrapper only unwraps the ``answers`` map from
Laya's envelope and drops ``model``/``usage``.

``/decision`` serves the Clef (System One) contract in the proxy's
``docs/api/decision/clef-schema-input.json`` and ``clef-schema-output.json`` instead,
for general judging rather than tool filtering. It validates the request against the
input schema and returns the ``{model, answers, usage}`` envelope the output schema
requires, dropping Laya's ``action`` field, which Clef has no concept of. ``model`` is
required but not restricted to Clef's selectors -- any name is accepted and echoed
back, so the endpoint can front Laya or a future judge. ``images`` is accepted and
ignored: Clef sends images ahead of the state, but Laya is a text-only encoder.

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
from laya_mlx.common import QTYPES

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_QUESTIONS = 64
MAX_IMAGES = 4

# Fields the Clef output schema requires per answer type. Laya additionally returns
# ``action``, which is not part of Clef and is dropped from ``/decision`` responses.
ANSWER_FIELDS = {
    "noul": ("type", "noul"),
    "choice": ("type", "choice", "probabilities", "confidence"),
    "score": ("type", "score", "legend", "probabilities", "confidence"),
}

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


def _validate_question(qid, question):
    """Validate one question against the Clef input schema.

    A valid Clef question is also a valid Laya question, so nothing is rewritten:
    both spell the option set ``criteria`` (a label -> description map for
    ``choice``, an ordered level list for ``score``, ``{"true": ..., "false": ...}``
    for ``noul``) and carry the question text in ``instructions``.
    """
    if not isinstance(question, dict):
        raise ValueError(f"Question {qid!r} must be a JSON object")
    qtype = question.get("type")
    if qtype not in QTYPES:
        raise ValueError(f"Question {qid!r}: 'type' must be one of {sorted(QTYPES)}, got {qtype!r}")
    instructions = question.get("instructions")
    if isinstance(instructions, str):
        if not instructions:
            raise ValueError(f"Question {qid!r}: 'instructions' must not be empty")
    elif not isinstance(instructions, (dict, list)):
        raise ValueError(
            f"Question {qid!r}: 'instructions' must be a non-empty string or an object/array"
        )
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError(f"Question {qid!r}: choice 'criteria' must be an object of option -> description")
        if not 2 <= len(criteria) <= 255:
            raise ValueError(
                f"Question {qid!r}: choice 'criteria' needs 2 to 255 options, got {len(criteria)}"
            )
        if not all(isinstance(label, str) and label for label in criteria):
            raise ValueError(f"Question {qid!r}: choice option ids must be non-empty strings")
    elif qtype == "score":
        if not isinstance(criteria, list):
            raise ValueError(f"Question {qid!r}: score 'criteria' must be an ordered array of levels")
        if not 2 <= len(criteria) <= 10:
            raise ValueError(
                f"Question {qid!r}: score 'criteria' needs 2 to 10 levels, got {len(criteria)}"
            )
    elif criteria is not None:
        if not isinstance(criteria, dict):
            raise ValueError(
                f"Question {qid!r}: noul 'criteria' must be an object describing 'true' and 'false'"
            )
        unknown = sorted(set(criteria) - {"true", "false"})
        if unknown:
            raise ValueError(f"Question {qid!r}: noul 'criteria' has unknown keys {unknown}")


def _validate_images(images):
    """Validate the Clef ``images`` extension and return how many were sent.

    Callers drop them: Laya is a text-only encoder, so an image cannot reach the
    decision. The count comes back so the caller can say so out loud rather than
    answer as if the images had no effect on the request.
    """
    if images is None:
        return 0
    if not isinstance(images, list):
        raise ValueError("'images' must be an array")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"'images' accepts at most {MAX_IMAGES} items, got {len(images)}")
    for i, image in enumerate(images):
        if isinstance(image, str):
            if not image.lower().startswith("data:"):
                raise ValueError(f"Image {i}: a string image must be a data URL")
        elif isinstance(image, dict):
            for field in ("content_type", "base64"):
                if not isinstance(image.get(field), str):
                    raise ValueError(f"Image {i}: '{field}' must be a string")
        else:
            raise ValueError(f"Image {i}: must be a data URL string or a base64 image object")
    return len(images)


def parse_decision_request(payload):
    """Validate a ``POST /decision`` body against the Clef input schema.

    Returns ``(model, state, questions)``, with ``questions`` ready to hand to
    ``Agent.system_one`` unchanged. ``model`` is only required to be a non-empty
    string -- it is echoed back rather than matched against Clef's selectors, so the
    endpoint can front Laya today and another judge later. ``state`` may be a string,
    object, or array; Laya serializes the structured forms before encoding.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("Request is missing a non-empty string 'model'")
    state = payload.get("state")
    if state is None:
        raise ValueError("Request is missing 'state'")
    if isinstance(state, str):
        if not state:
            raise ValueError("'state' must not be an empty string")
    elif not isinstance(state, (dict, list)):
        raise ValueError("'state' must be a string, object, or array")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Request is missing a non-empty 'questions' object")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"Too many questions: {len(questions)} (max {MAX_QUESTIONS})")
    for qid, question in questions.items():
        _validate_question(qid, question)
    images = _validate_images(payload.get("images"))
    if images:
        log.warning("/decision dropped %d image(s): the encoder is text-only", images)
    return model, state, questions


def clef_response(model, result):
    """Build the Clef output envelope from Laya's ``system_one`` result.

    Laya's per-answer shapes already match Clef's for ``choice`` and ``score``; the
    only difference is that Laya also returns ``action`` (and a ``confidence`` on
    ``noul``, which Clef does not define), so answers are narrowed to the fields the
    output schema names. A field the schema requires but Laya did not produce is an
    error rather than something to paper over with a default.
    """
    answers = {}
    for qid, answer in result["answers"].items():
        fields = ANSWER_FIELDS.get(answer.get("type"))
        if fields is None:
            raise ValueError(f"Question {qid!r}: unexpected answer type {answer.get('type')!r}")
        missing = [field for field in fields if field not in answer]
        if missing:
            raise ValueError(f"Question {qid!r}: {answer['type']} answer is missing {missing}")
        answers[qid] = {field: answer[field] for field in fields}
    usage = result["usage"]
    return {
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        },
    }


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
        path = self.path.split("?", 1)[0]
        if path not in ("/judge", "/decision"):
            self.send_error(404, "Not Found")
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        try:
            if path == "/judge":
                # /judge answers with the bare answers map, so there is no model to echo.
                model = None
                state, questions = parse_request(payload)
            else:
                model, state, questions = parse_decision_request(payload)
                log.info("received /decision request: model=%s, %d question(s)", model, len(questions))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._answer(model, state, questions)

    def _answer(self, model, state, questions):
        """Run inference for either endpoint and write the answer.

        ``model`` is ``None`` for ``/judge`` (bare answers map) and the requested
        selector for ``/decision``, which wraps the answers in Clef's envelope.
        """
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

        # Log which endpoint is being used (model is None for /judge, set for /decision)
        endpoint = "/decision" if model is not None else "/judge"
        log.info(
            "judged %d question(s) via %s in %.1fms (%d input tokens)",
            len(questions),
            endpoint,
            elapsed_ms,
            result.get("usage", {}).get("input_tokens", -1),
        )

        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            log.error(
                "answer/question mismatch: asked %s, got %s",
                sorted(questions),
                sorted(answers) if isinstance(answers, dict) else type(answers).__name__,
            )
            self._send_json(500, {"error": "model returned answers for a different question set"})
            return

        if model is None:
            self._send_json(200, answers)
            return
        try:
            body = clef_response(model, result)
        except (KeyError, ValueError) as exc:
            # Laya produced an answer Clef has no shape for: a server-side bug, so fail
            # loudly rather than return a partial envelope.
            log.error("cannot render the Clef response: %s", exc, exc_info=True)
            self._send_json(500, {"error": f"cannot render the Clef response: {exc}"})
            return
        self._send_json(200, body)

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

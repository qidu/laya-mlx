"""HTTP-level tests for the ``/judge`` and ``/decision`` sidecar endpoints.

``serve_judge.py`` lives at the repository root rather than inside ``laya_mlx``, so it
is loaded from its path; the server under test is the real ``JudgeHandler`` driven over
a real socket, backed by the tiny checkpoint fixture.
"""

import importlib.util
import json
import logging
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from laya_mlx import Agent

_SPEC = importlib.util.spec_from_file_location(
    "serve_judge", Path(__file__).resolve().parents[1] / "serve_judge.py"
)
serve_judge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(serve_judge)

NOUL = {"type": "noul", "instructions": "Is it true?"}


def post(url, path, payload):
    """POST ``payload`` as JSON and return ``(status, decoded body)``.

    An HTTP error is a result here, not an exception: the tests assert on the status
    and error message the handler chose.
    """
    request = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def decision_payload(**overrides):
    """A minimal valid ``/decision`` body, with fields replaced by ``overrides``."""
    payload = {"model": "clef", "state": "hello", "questions": {"yes": NOUL}}
    payload.update(overrides)
    return payload


@pytest.fixture
def judge_server(tiny_checkpoint):
    """The real handler on an ephemeral port, wired the way ``main`` wires it."""
    agent = Agent(tiny_checkpoint, dtype="float32")
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve_judge.JudgeHandler)
    server.agent = agent
    server.lock = threading.Lock()
    server.model_version = "tiny-checkpoint"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield types.SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", agent=agent)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def clef_questions():
    """Every question type in the Clef input schema.

    ``choice`` uses the object form of ``criteria``, which Clef requires and Laya also
    accepts; Laya's bare-list form is exercised separately against ``/judge``.
    """
    return {
        "topic": {
            "type": "choice",
            "instructions": "Which topic?",
            "criteria": {"a": "the first", "b": "the second", "c": None},
        },
        "level": {"type": "score", "instructions": "Rate it", "criteria": ["low", "mid", "high"]},
        "yes": {
            "type": "noul",
            "instructions": "Is it true?",
            "criteria": {"true": "it holds", "false": "it does not"},
        },
    }


def test_judge_still_returns_the_bare_answers_map(judge_server, questions):
    """``/judge`` passes Laya's answers through untouched, as the sidecar protocol says."""
    state = "hello"
    status, body = post(judge_server.url, "/judge", {"state": state, "questions": questions})
    assert status == 200
    assert body == judge_server.agent.system_one(state, questions)["answers"]
    assert set(body) == set(questions)
    assert body["topic"]["action"]["act_probability"] is not None


def test_decision_returns_the_clef_envelope(judge_server, clef_questions):
    status, body = post(
        judge_server.url,
        "/decision",
        {"model": "clef", "state": "hello", "questions": clef_questions},
    )
    assert status == 200
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "clef"
    assert set(body["answers"]) == set(clef_questions)
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0
    for qid, answer in body["answers"].items():
        fields = serve_judge.ANSWER_FIELDS[clef_questions[qid]["type"]]
        assert set(answer) == set(fields)
        assert answer["type"] == clef_questions[qid]["type"]


def test_decision_answer_values_follow_the_output_schema(judge_server, clef_questions):
    status, body = post(
        judge_server.url,
        "/decision",
        {"model": "laya", "state": "hello", "questions": clef_questions},
    )
    assert status == 200
    answers = body["answers"]

    choice = answers["topic"]
    assert choice["choice"] in clef_questions["topic"]["criteria"]
    assert set(choice["probabilities"]) == set(clef_questions["topic"]["criteria"])
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    score = answers["level"]
    assert 0 <= score["score"] <= 2
    assert score["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert sum(score["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    noul = answers["yes"]
    assert 0 <= noul["noul"] <= 1
    # Clef defines no confidence for noul, and no answer type carries Laya's action.
    assert set(noul) == {"type", "noul"}

    for qid in ("topic", "level"):
        assert 0 <= answers[qid]["confidence"] <= 1


@pytest.mark.parametrize("model", ["clef", "clef-flash", "laya", "jev"])
def test_decision_accepts_any_model_name_and_echoes_it(judge_server, model):
    status, body = post(judge_server.url, "/decision", decision_payload(model=model))
    assert status == 200
    assert body["model"] == model


def test_decision_serializes_structured_state(judge_server, clef_questions):
    """A dict or list state is JSON-encoded, so it decides the same as that JSON text."""
    record = {"messages": [{"role": "user", "text": "hello"}], "score": 3}
    object_status, object_body = post(
        judge_server.url,
        "/decision",
        {"model": "clef", "state": record, "questions": clef_questions},
    )
    json_status, json_body = post(
        judge_server.url,
        "/decision",
        {"model": "clef", "state": json.dumps(record), "questions": clef_questions},
    )
    assert (object_status, json_status) == (200, 200)
    assert object_body == json_body
    list_status, _ = post(
        judge_server.url,
        "/decision",
        {"model": "clef", "state": [{"a": 1}, {"b": 2}], "questions": clef_questions},
    )
    assert list_status == 200


def test_decision_validates_then_drops_images(judge_server, caplog):
    """Images cannot reach a text-only encoder, so the request is answered but reported."""
    payload = decision_payload(
        images=[
            "data:image/png;base64,iVBORw0KGgo=",
            {"content_type": "image/jpeg", "base64": "/9j/4AAQ"},
        ]
    )
    with caplog.at_level(logging.WARNING, logger="serve_judge"):
        status, body = post(judge_server.url, "/decision", payload)
    assert status == 200
    assert set(body) == {"model", "answers", "usage"}
    assert any("dropped 2 image(s)" in record.getMessage() for record in caplog.records)


def test_choice_criteria_form_differs_between_endpoints(judge_server):
    """Laya takes a bare label list; Clef's schema requires the label -> description map."""
    list_form = {"q": {"type": "choice", "instructions": "Pick one", "criteria": ["a", "b"]}}
    judge_status, judge_body = post(
        judge_server.url, "/judge", {"state": "hello", "questions": list_form}
    )
    assert judge_status == 200
    assert judge_body["q"]["choice"] in ("a", "b")

    decision_status, decision_body = post(
        judge_server.url, "/decision", decision_payload(questions=list_form)
    )
    assert decision_status == 400
    assert "must be an object of option -> description" in decision_body["error"]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([1, 2, 3], "must be a JSON object"),
        ({}, "non-empty string 'model'"),
        (decision_payload(model=7), "non-empty string 'model'"),
        (decision_payload(model=""), "non-empty string 'model'"),
        ({"model": "clef", "questions": {"yes": NOUL}}, "missing 'state'"),
        (decision_payload(state=None), "missing 'state'"),
        (decision_payload(state=""), "must not be an empty string"),
        (decision_payload(state=5), "'state' must be a string, object, or array"),
        (decision_payload(questions={}), "non-empty 'questions' object"),
        (decision_payload(questions=[]), "non-empty 'questions' object"),
        (decision_payload(questions=[{"type": "noul", "instructions": "x"}]), "'questions' object"),
        (
            decision_payload(questions={f"q{i}": NOUL for i in range(65)}),
            "Too many questions: 65 (max 64)",
        ),
        (decision_payload(questions={"q": "noul"}), "must be a JSON object"),
        (decision_payload(questions={"q": {"instructions": "x"}}), "'type' must be one of"),
        (
            decision_payload(questions={"q": {"type": "maybe", "instructions": "x"}}),
            "'type' must be one of ['choice', 'noul', 'score']",
        ),
        (decision_payload(questions={"q": {"type": "noul"}}), "'instructions' must be a non-empty"),
        (
            decision_payload(questions={"q": {"type": "noul", "instructions": ""}}),
            "'instructions' must not be empty",
        ),
        (decision_payload(questions={"q": {"type": "noul", "instructions": 4}}), "'instructions'"),
        (
            decision_payload(questions={"q": {"type": "choice", "instructions": "x"}}),
            "choice 'criteria' must be an object",
        ),
        (
            decision_payload(
                questions={"q": {"type": "choice", "instructions": "x", "criteria": {"a": "1"}}}
            ),
            "needs 2 to 255 options, got 1",
        ),
        (
            decision_payload(
                questions={
                    "q": {
                        "type": "choice",
                        "instructions": "x",
                        "criteria": {f"o{i}": None for i in range(256)},
                    }
                }
            ),
            "needs 2 to 255 options, got 256",
        ),
        (
            decision_payload(
                questions={"q": {"type": "choice", "instructions": "x", "criteria": {"": "empty"}}}
            ),
            "option ids must be non-empty strings",
        ),
        (
            decision_payload(questions={"q": {"type": "score", "instructions": "x"}}),
            "score 'criteria' must be an ordered array",
        ),
        (
            decision_payload(
                questions={"q": {"type": "score", "instructions": "x", "criteria": {"a": "b"}}}
            ),
            "score 'criteria' must be an ordered array",
        ),
        (
            decision_payload(
                questions={"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}
            ),
            "needs 2 to 10 levels, got 1",
        ),
        (
            decision_payload(
                questions={
                    "q": {"type": "score", "instructions": "x", "criteria": list("abcdefghijk")}
                }
            ),
            "needs 2 to 10 levels, got 11",
        ),
        (
            decision_payload(
                questions={
                    "q": {"type": "noul", "instructions": "x", "criteria": {"perhaps": "maybe"}}
                }
            ),
            "unknown keys ['perhaps']",
        ),
        (
            decision_payload(
                questions={"q": {"type": "noul", "instructions": "x", "criteria": ["a", "b"]}}
            ),
            "must be an object describing 'true' and 'false'",
        ),
        (decision_payload(images="nope"), "'images' must be an array"),
        (decision_payload(images=["data:image/png;base64,AA"] * 5), "at most 4 items, got 5"),
        (decision_payload(images=["https://example.com/a.png"]), "must be a data URL"),
        (decision_payload(images=[{"content_type": "image/png"}]), "'base64' must be a string"),
        (decision_payload(images=[7]), "must be a data URL string or a base64 image object"),
    ],
)
def test_decision_rejects_invalid_requests(judge_server, payload, match):
    status, body = post(judge_server.url, "/decision", payload)
    assert status == 400
    assert match in body["error"]


def test_decision_rejects_a_non_json_body(judge_server):
    request = urllib.request.Request(
        judge_server.url + "/decision", data=b"{not json", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 400
    assert "not valid UTF-8 JSON" in json.loads(error.value.read())["error"]


def test_health_reports_the_model_and_only_get_is_served(judge_server):
    with urllib.request.urlopen(judge_server.url + "/health") as response:
        assert response.status == 200
        assert json.loads(response.read()) == {
            "status": "healthy",
            "model_version": "tiny-checkpoint",
            "mode": "choice+noul",
        }
    for path in ("/judge", "/decision"):
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(judge_server.url + path)
        assert error.value.code == 404
    status, _ = post(judge_server.url, "/nope", {})
    assert status == 404


def test_clef_response_narrows_laya_answers_to_the_schema():
    result = {
        "answers": {
            "yes": {
                "type": "noul",
                "confidence": 0.9,
                "action": {"act_probability": 0.5},
                "noul": 0.75,
            }
        },
        "usage": {"input_tokens": 11, "output_tokens": 0},
    }
    assert serve_judge.clef_response("laya", result) == {
        "model": "laya",
        "answers": {"yes": {"type": "noul", "noul": 0.75}},
        "usage": {"input_tokens": 11, "output_tokens": 0},
    }


@pytest.mark.parametrize(
    ("answers", "match"),
    [
        ({"q": {"type": "choice", "choice": "a"}}, "is missing \\['probabilities', 'confidence'\\]"),
        ({"q": {"type": "surprised", "choice": "a"}}, "unexpected answer type 'surprised'"),
    ],
)
def test_clef_response_fails_loudly_on_an_answer_clef_cannot_express(answers, match):
    result = {"answers": answers, "usage": {"input_tokens": 3, "output_tokens": 0}}
    with pytest.raises(ValueError, match=match):
        serve_judge.clef_response("clef", result)

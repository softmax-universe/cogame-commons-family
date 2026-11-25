"""The LLM decision path, with a stubbed transport. No network, ever.

Covers clean JSON, JSON with trailing prose, prose before the brace, strict
validation without retries or scripted fallbacks, single-call timeout/throttle/
transport failure reporting, parallel batch dispatch across all six seats, and
the no-credentials path making zero network calls.
"""

from __future__ import annotations

import threading
import time

import pytest

from coworld.examples.commons_family.game.baselines import make_baseline
from coworld.examples.commons_family.game.engine import (
    CommonsConfig,
    module_for,
    new_game,
    observation,
    parse_decision,
)
from coworld.examples.commons_family.game.llm import (
    LlmDecider,
    LlmThrottled,
    LlmTimeout,
    LlmTransportError,
    build_transport,
    extract_json,
)


class StubTransport:
    """Answers with a canned script; counts every call it is asked to make."""

    def __init__(self, replies, barrier: threading.Barrier | None = None) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.bodies: list[dict] = []
        self.barrier = barrier
        self.lock = threading.Lock()

    def complete(self, body: dict, timeout: float, slot: int) -> str:
        if self.barrier is not None:
            # Times out (BrokenBarrierError) unless every seat is here at once.
            self.barrier.wait(timeout=5.0)
        with self.lock:
            self.calls += 1
            self.bodies.append(body)
            reply = self.replies[min(len(self.replies) - 1, self.calls - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply


class ExplodingTransport:
    """Any use at all is a failure: the no-credentials path must not call out."""

    def complete(self, body: dict, timeout: float, slot: int) -> str:  # pragma: no cover
        raise AssertionError("the disabled client made a network call")


def setup(module: str = "cleanup", **overrides):
    overrides.setdefault("rounds", 5)
    config = CommonsConfig(module=module, num_agents=6, **overrides)
    state = new_game(config)
    return config, state, module_for(config)


def decide_one(transport, module: str = "cleanup", **overrides):
    config, state, module_impl = setup(module, **overrides)
    decider = LlmDecider(config, module_impl, transport=transport, model="claude-haiku-4-5")
    obs = observation(state, config, 0, module_impl)
    answers = decider.decide({0: (obs, "take one apple a round")}, time.monotonic() + 10)
    return answers[0], decider, config, state, module_impl, obs


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def test_clean_json_is_taken_as_is():
    assert extract_json('{"harvest": 2, "clean": 1}') == {"harvest": 2, "clean": 1}


def test_outer_markdown_json_fence_is_extracted():
    assert extract_json('```json\n{"harvest": 1, "clean": 1}\n```') == {"harvest": 1, "clean": 1}


def test_full_json_reply_under_prefill_is_not_corrupted():
    transport = StubTransport(['{"harvest": 1, "clean": 1}'])
    (reply, cause), decider, *_ = decide_one(transport)
    assert cause == ""
    assert reply == {"harvest": 1, "clean": 1}
    assert transport.calls == 1


def test_trailing_prose_is_tolerated():
    assert extract_json('{"harvest": 1} — one apple, as agreed.') == {"harvest": 1}


def test_prose_before_the_brace_is_tolerated():
    assert extract_json('Let me think. I will take one.\n{"harvest": 1}') == {"harvest": 1}


def test_the_first_balanced_span_wins_and_nesting_is_respected():
    assert extract_json('{"a": {"b": 1}, "harvest": 2}{"harvest": 9}') == \
        {"a": {"b": 1}, "harvest": 2}


def test_a_brace_inside_a_string_does_not_close_the_object():
    assert extract_json('{"message": "use { and }", "harvest": 1}') == \
        {"message": "use { and }", "harvest": 1}


@pytest.mark.parametrize("raw", ["", "no json here", "{ not json", "[1,2,3]", "}{"])
def test_unusable_replies_extract_to_nothing(raw):
    assert extract_json(raw) is None


# ---------------------------------------------------------------------------
# the seat
# ---------------------------------------------------------------------------


def test_a_clean_reply_is_used_and_the_prefill_brace_is_restored():
    transport = StubTransport(['"harvest": 2, "clean": 1}'])
    (reply, cause), decider, *_ = decide_one(transport)
    assert cause == ""
    assert reply == {"harvest": 2, "clean": 1}
    assert transport.calls == 1
    assert decider.requests == 1
    # haiku-4.5 gets the assistant prefill, and its smaller max_tokens.
    assert transport.bodies[0]["messages"][-1] == {"role": "assistant", "content": "{"}
    assert transport.bodies[0]["max_tokens"] == 300


def test_out_of_range_and_illegal_fields_are_rejected_without_retry():
    transport = StubTransport(['"harvest": 99, "sanction": 0, "message": "hi"}', '"harvest": 1, "clean": 1}'])
    (reply, cause), decider, *_ = decide_one(transport, sanctions_enabled=True)
    assert reply is None
    assert cause == "invalid_action"
    assert transport.calls == 1
    assert decider.requests == 1
    assert "raw_reply" in decider.last_failures[0]

    # Optional fields default cleanly when omitted, while valid core fields pass.
    ok_transport = StubTransport(['"harvest": 1}'])
    (ok_reply, ok_cause), _, config, state, module, _ = decide_one(ok_transport, sanctions_enabled=True)
    assert ok_cause == ""
    decision = parse_decision(ok_reply, 0, config, state, module, "llm")
    assert decision.harvest == 1
    assert decision.clean == 0
    assert decision.sanction is None
    assert decision.message is None


def test_the_user_message_is_the_observation_with_type_and_round_seconds_dropped():
    transport = StubTransport(['"harvest": 1}'])
    _, _, _, _, _, obs = decide_one(transport)
    body = transport.bodies[0]
    user = body["messages"][0]["content"]
    assert '"type"' not in user
    assert '"round_seconds"' not in user
    assert obs["alias"] in user
    assert obs["alias"] in body["system"]
    assert "take one apple a round" in body["system"]     # standing orders, verbatim


def test_an_unusable_reply_is_rejected_on_single_call_with_raw_reply():
    transport = StubTransport(["not json at all", '{"harvest": 1}'])
    (reply, cause), decider, *_ = decide_one(transport)
    assert reply is None
    assert cause == "parse"
    assert transport.calls == 1, "one call per prompt seat, no hidden retries"
    assert decider.last_failures[0]["raw_reply"].endswith("not json at all")


def test_a_timeout_makes_one_call_and_is_reported_as_timeout():
    transport = StubTransport([LlmTimeout("slow"), '"harvest": 1}'])
    (reply, cause), *_ = decide_one(transport)
    assert (reply, cause) == (None, "timeout")
    assert transport.calls == 1


def test_a_transport_error_makes_one_call_and_is_reported_as_transport():
    transport = StubTransport([LlmTransportError("dns"), '"harvest": 1}'])
    (reply, cause), *_ = decide_one(transport)
    assert (reply, cause) == (None, "transport")
    assert transport.calls == 1


def test_an_unclassified_transport_exception_is_reported_without_retry():
    """`MessagesTransport` re-raises non-429/529 `HTTPError` (e.g. 503, 401)."""
    from urllib.error import HTTPError  # noqa: PLC0415

    error = HTTPError("https://api.anthropic.com/v1/messages", 503,
                      "Service Unavailable", {}, None)
    transport = StubTransport([error, '"harvest": 1}'])
    (reply, cause), decider, *_ = decide_one(transport)
    assert (reply, cause) == (None, "transport")
    assert transport.calls == 1


def test_an_unclassified_exception_in_one_seat_does_not_stop_the_batch():
    class EveryThirdCallExplodes:
        def __init__(self) -> None:
            self.calls = 0
            self.lock = threading.Lock()

        def complete(self, body: dict, timeout: float, slot: int) -> str:
            with self.lock:
                self.calls += 1
                boom = self.calls % 3 == 0
            if boom:
                raise KeyError("content")          # the key the parser expects
            return '"harvest": 1}'

    config, state, module = setup()
    decider = LlmDecider(config, module, transport=EveryThirdCallExplodes(),
                         model="claude-haiku-4-5")
    requests = {
        slot: (observation(state, config, slot, module), "")
        for slot in range(config.num_agents)
    }
    answers = decider.decide(requests, time.monotonic() + 10)
    assert sorted(answers) == list(range(6))
    assert all(cause in ("", "transport") for _, cause in answers.values())


def test_throttling_fails_immediately_on_single_call_without_sleep_ladder():
    transport = StubTransport([LlmThrottled("429"), '"harvest": 1}'])
    started = time.monotonic()
    (reply, cause), decider, *_ = decide_one(transport)
    assert (reply, cause) == (None, "throttled")
    assert transport.calls == 1
    assert decider.requests == 1
    assert time.monotonic() - started < 1.0


def test_expired_round_deadline_fails_before_calling_transport():
    transport = StubTransport(['"harvest": 1}'])
    config, state, module = setup()
    decider = LlmDecider(config, module, transport=transport, model="claude-haiku-4-5")
    obs = observation(state, config, 0, module)
    answers = decider.decide({0: (obs, "")}, time.monotonic() - 0.1)
    assert answers[0] == (None, "timeout")
    assert transport.calls == 0


def test_six_throttled_seats_respect_the_rolling_budget_with_one_call_each():
    transport = StubTransport([LlmThrottled("429")])
    config, state, module = setup(llm_max_requests_per_minute=4)
    decider = LlmDecider(config, module, transport=transport, model="claude-haiku-4-5")
    requests = {
        slot: (observation(state, config, slot, module), "")
        for slot in range(config.num_agents)
    }
    answers = decider.decide(requests, time.monotonic() + 1.5)
    assert sorted(answers) == list(range(6))
    assert all(cause in ("throttled", "rate_budget") for _, cause in answers.values())
    assert decider.requests == 4
    assert transport.calls == 4


def test_the_rate_budget_reports_rate_budget_rather_than_waiting():
    transport = StubTransport(['"harvest": 1}'])
    config, state, module = setup(llm_max_requests_per_minute=1)
    decider = LlmDecider(config, module, transport=transport, model="claude-haiku-4-5")
    obs = observation(state, config, 0, module)
    deadline = time.monotonic() + 10
    assert decider.decide({0: (obs, "")}, deadline)[0][1] == ""
    assert decider.decide({0: (obs, "")}, deadline)[0] == (None, "rate_budget")
    assert transport.calls == 1


# ---------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------


def test_all_six_seats_go_out_as_one_parallel_batch():
    """A barrier in the stub: it only clears if all six calls are in flight."""
    barrier = threading.Barrier(6)
    transport = StubTransport(['"harvest": 1}'], barrier=barrier)
    config, state, module = setup()
    decider = LlmDecider(config, module, transport=transport, model="claude-haiku-4-5")
    requests = {
        slot: (observation(state, config, slot, module), "")
        for slot in range(config.num_agents)
    }
    answers = decider.decide(requests, time.monotonic() + 10)
    assert sorted(answers) == list(range(6))
    assert all(cause == "" for _, cause in answers.values())
    assert transport.calls == 6
    assert decider.requests == 6


def test_each_seat_gets_its_own_system_prompt_built_once():
    transport = StubTransport(['"harvest": 1}'])
    config, state, module = setup()
    decider = LlmDecider(config, module, transport=transport, model="claude-haiku-4-5")
    requests = {
        slot: (observation(state, config, slot, module), f"orders for {slot}")
        for slot in range(2)
    }
    decider.decide(requests, time.monotonic() + 10)
    decider.decide(requests, time.monotonic() + 10)
    systems = {body["system"] for body in transport.bodies}
    assert len(systems) == 2                      # one per seat, reused
    assert any("orders for 0" in system for system in systems)
    assert any("orders for 1" in system for system in systems)


# ---------------------------------------------------------------------------
# no credentials
# ---------------------------------------------------------------------------


def test_without_credentials_the_client_is_disabled_and_makes_zero_calls(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_URI",
                 "COWORLD_LLM_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    transport, model = build_transport()
    assert transport is None
    assert model == ""

    config, state, module = setup()
    decider = LlmDecider(config, module)
    assert decider.enabled is False
    obs = observation(state, config, 0, module)
    answers = decider.decide({0: (obs, "")}, time.monotonic() + 10)
    assert answers == {0: (None, "disabled")}
    assert decider.requests == 0


def test_a_disabled_decider_never_touches_its_transport():
    config, state, module = setup()
    decider = LlmDecider(config, module, transport=None, model="")
    assert decider.enabled is False
    decider.transport = None
    obs = observation(state, config, 0, module)
    assert decider.decide({0: (obs, "")}, time.monotonic() + 10) == {0: (None, "disabled")}


def test_an_episode_with_no_credentials_fails_explicitly():
    from coworld.examples.commons_family import headless  # noqa: PLC0415

    config, _, module = setup(rounds=4)
    decider = LlmDecider(config, module, transport=None, model="")
    with pytest.raises(RuntimeError, match="disabled"):
        headless.run_episode(
            config,
            headless.build_policies(["steward"] * 6),
            parallel_seats=True,
            decider=decider,
            prompts={0: "take one", 1: "take two"},
        )


def test_the_credential_ladder_prefers_the_native_sidecar(monkeypatch):
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", "http://sidecar:8000")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    transport, model = build_transport()
    assert type(transport).__name__ == "MessagesTransport"
    assert model == "anthropic/claude-haiku-4.5"
    assert transport.base == "http://sidecar:8000"
    assert transport.api_key == "sidecar"


def test_the_credential_ladder_falls_through_to_the_direct_api(monkeypatch):
    monkeypatch.delenv("COWORLD_LLM_ENDPOINT", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    transport, model = build_transport()
    assert type(transport).__name__ == "MessagesTransport"
    assert "haiku" in model


def test_the_credential_ladder_reads_the_key_uri_last(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("sk-from-uri\n", encoding="utf-8")
    monkeypatch.delenv("COWORLD_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY_URI", key_file.as_uri())
    transport, _ = build_transport()
    assert type(transport).__name__ == "MessagesTransport"
    assert transport.api_key == "sk-from-uri"


def test_hosted_transport_posts_native_messages_with_seat_attribution(monkeypatch):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers["X-Coworld-Player-Slot"], body))
            response = json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://retired.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "local-key-must-not-be-used")
    try:
        transport, model = build_transport()
        for slot in (0, 1):
            assert transport.complete({"model": model, "max_tokens": 64, "system": "rules", "messages": [{"role": "user", "content": "view"}]}, 5, slot) == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert [slot for _, slot, _ in received] == ["0", "1"]
    for path, _, body in received:
        assert path == "/v1/messages"
        assert body["model"] == "anthropic/claude-sonnet-4.6"
        assert "anthropic_version" not in body

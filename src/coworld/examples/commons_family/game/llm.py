"""Game-owned inference batches use the native Coworld LLM sidecar.

Hosted requests prefer COWORLD_LLM_ENDPOINT, carry the requesting seat, and
use canonical OpenRouter models. Local play can use provider credentials.
The game owns parallel dispatch with one call per prompt seat and no hidden
retries or scripted fallbacks.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from coworld.examples.commons_family.game.engine import (
    NOTE_MAX_CHARS,
    PROMPT_MAX_RUNES,
    CommonsConfig,
    parse_decision,
    truncate_runes,
)
from coworld.examples.commons_family.game.modules.base import Module
from coworld.examples.commons_family.game.prompts import (
    CANONICAL_SELECTION,
    SYSTEM_TEMPLATES,
    PromptSelection,
)
from coworld.examples.commons_family.game.training_capture import TrainingCapture
from coworld.examples.commons_family.shared.checkpoint import checkpoint_model
from coworld.examples.commons_family.shared.log_shipper import get_logger

logger = get_logger("commons_family.llm")

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

SANCTION_RULES = "\nYou may also sanction one cog per round: you pay {cost}, they lose {burn}."


class LlmThrottled(Exception):
    """The provider asked us to slow down."""


class LlmTimeout(Exception):
    """The call did not answer inside its deadline."""


class LlmTransportError(Exception):
    """Anything else that is not the model's fault."""


class LlmRateBudget(Exception):
    """The rolling requests-per-minute budget is spent for this minute."""


_AUTO = object()


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class MessagesTransport:
    """Native Messages API over the hosted sidecar or a local provider."""

    def __init__(self, model: str, api_key: str, base: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base = base.rstrip("/")
        self.attempts: dict[int, list[dict]] = {}
        self.capture_enabled = bool(os.environ.get("COGAME_SAVE_TRAJECTORY_URI"))

    def complete(self, body: dict, timeout: float, slot: int) -> str:
        payload = {
            "model": self.model,
            "max_tokens": body["max_tokens"],
            "system": body["system"],
            "messages": body["messages"],
        }
        if "checkpoint_model" in body:
            payload["model"] = checkpoint_model(body["checkpoint_model"])
            payload["thinking"] = {"type": "disabled"}
            payload["temperature"] = 0
            payload["top_p"] = 1
        request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base}/v1/messages",
            data=request_bytes,
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": self.api_key,
                "user-agent": "cogame-commons-family/0.1",
                "X-Coworld-Player-Slot": str(slot),
            },
        )
        evidence: dict[str, Any] = {
            "attempt_id": f"{slot}-{len(self.attempts.get(slot, []))}",
            "policy": "registered-prompt",
            "origin": "model",
            "accepted": False,
            "model": self.model,
            "prompt": [
                {"role": "system", "content": payload["system"]},
                *payload["messages"],
            ],
            "request": payload,
            "decoder": {
                key: payload[key] for key in ("max_tokens", "thinking", "temperature", "top_p") if key in payload
            },
        }
        started = time.monotonic()

        def received(response: Any) -> bytes:
            chunks: list[bytes] = []
            evidence["response_complete"] = False
            evidence["http_status"] = response.status
            evidence["response_headers"] = dict(response.headers.items())
            call_id = response.headers.get("X-Softmax-Llm-Call-Id")
            if call_id:
                evidence["platform_call_id"] = call_id
            request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
            if request_id:
                evidence["provider_request_id"] = request_id
            for header, field in (
                ("X-Coworld-Checkpoint-Sha256", "model_identity"),
                ("X-Coworld-Tokenizer-Sha256", "tokenizer_identity"),
                ("X-Coworld-Chat-Template-Sha256", "chat_template_identity"),
            ):
                if value := response.headers.get(header):
                    evidence[field] = value
            try:
                reader = getattr(response, "read1", response.read)
                while chunk := reader(65536):
                    chunks.append(chunk)
                evidence["response_complete"] = True
            finally:
                data = b"".join(chunks)
                evidence["response_body_b64"] = base64.b64encode(data).decode("ascii")
                try:
                    evidence["raw_response"] = data.decode("utf-8")
                except UnicodeDecodeError:
                    pass  # Original bytes remain in response_body_b64.
            return data

        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(received(response))
                evidence["provider_response"] = result
                if result.get("model"):
                    evidence["model"] = result["model"]
                if result.get("id"):
                    evidence["generation_id"] = result["id"]
                if result.get("stop_reason"):
                    evidence["stop_reason"] = result["stop_reason"]
                for name in ("input_tokens", "output_tokens"):
                    if name in result.get("usage", {}):
                        evidence[name] = result["usage"][name]
                text = next(
                    (block["text"] for block in result["content"] if block["type"] == "text"),
                    "",
                )
                evidence["response"] = text
                if "checkpoint_model" in body and (
                    evidence.get("model_identity") != payload["model"].removeprefix("checkpoint/")
                    or not evidence.get("tokenizer_identity")
                    or not evidence.get("chat_template_identity")
                    or result.get("model") != payload["model"]
                ):
                    evidence["rejection_reason"] = "checkpoint serving identity mismatch"
                    raise LlmTransportError("checkpoint serving identity mismatch")
                return text
        except HTTPError as error:
            evidence["rejection_reason"] = f"HTTP {error.code}"
            received(error)
            if error.code in (429, 529):
                raise LlmThrottled(str(error.code)) from error
            raise
        except TimeoutError as error:
            evidence["rejection_reason"] = "timeout"
            raise LlmTimeout(str(error)) from error
        except URLError as error:
            evidence["rejection_reason"] = str(error)
            if isinstance(error.reason, TimeoutError):
                raise LlmTimeout(str(error)) from error
            raise LlmTransportError(str(error)) from error
        finally:
            evidence["latency_ms"] = (time.monotonic() - started) * 1000
            if self.capture_enabled:
                self.attempts.setdefault(slot, []).append(evidence)


def build_transport() -> tuple[object | None, str]:
    """The credential ladder. Returns `(transport, model)`; `(None, "")` disables."""
    endpoint = os.environ.get("COWORLD_LLM_ENDPOINT", "").strip()
    if endpoint:
        model = os.environ.get("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
        key = os.environ.get("COWORLD_LLM_API_KEY", "sidecar")
        return MessagesTransport(model, key, endpoint), model
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        uri = os.environ.get("ANTHROPIC_API_KEY_URI")
        if uri:
            try:
                from coworld.examples.commons_family.shared.artifact_io import (
                    read_data,
                )

                key = read_data(uri).decode("utf-8").strip()
            except Exception as error:  # noqa: BLE001 - a bad URI must not kill the episode
                logger.warning("could not read ANTHROPIC_API_KEY_URI: %s", error)
                key = ""
    if key:
        model = os.environ.get("COMMONS_FAMILY_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return MessagesTransport(
            model,
            key,
            os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        ), model
    return None, ""


# ---------------------------------------------------------------------------
# Reply extraction
# ---------------------------------------------------------------------------


def _strip_outer_markdown_fence(raw: str) -> str:
    """Strip a standard outer Markdown JSON/code fence if present."""
    stripped = raw.strip()
    if not stripped.startswith("```"):
        return raw
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return raw


def extract_json(raw: str) -> dict | None:
    """The first balanced `{...}` span, supporting standard outer Markdown JSON fences."""
    raw = _strip_outer_markdown_fence(raw)
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(raw[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


# ---------------------------------------------------------------------------
# The decider
# ---------------------------------------------------------------------------


class LlmDecider:
    """One batch of LLM calls per round, one call per prompt seat.

    All prompt seats go out together on a single thread pool: this is a
    simultaneous-decision game, and issuing the calls sequentially is how an
    LLM coworld blows its play budget.
    """

    def __init__(
        self,
        config: CommonsConfig,
        module: Module,
        transport: Any = _AUTO,
        model: str = "",
        *,
        prompt_selection: PromptSelection = CANONICAL_SELECTION,
    ) -> None:
        if transport is _AUTO:
            transport, model = build_transport()
        self.capture = TrainingCapture()
        self.attempts: dict[int, list[dict]] = {}
        self.last_failures: dict[int, dict[str, Any]] = {}
        self.config = config
        self.prompt_selection = prompt_selection
        self.module = module
        self.transport: Any = transport
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.requests = 0
        self._systems: dict[int, str] = {}
        self._budget: deque[float] = deque()
        self._lock = threading.Lock()
        self.checkpoint_models: dict[int, str] = {}
        if self.transport is None:
            logger.info("no LLM credentials configured: prompt seats cannot run without an LLM endpoint")

    @property
    def enabled(self) -> bool:
        return self.transport is not None

    # -- prompt ------------------------------------------------------------

    def system_prompt(self, obs: dict, standing_orders: str) -> str:
        sanction_rules = ""
        if obs["sanctions_enabled"]:
            sanction_rules = SANCTION_RULES.format(cost=obs["sanction_cost"], burn=obs["sanction_burn"])
        norm_line = f"\nPosted norm: {obs['norm_text']}" if obs["norm_text"] else ""
        prompt = self.prompt_selection.introduction(
            obs["alias"], obs["num_players"], obs["rounds"], obs["module"]
        ) + SYSTEM_TEMPLATES[self.prompt_selection.selected].format(
            alias=obs["alias"],
            num_players=obs["num_players"],
            rounds=obs["rounds"],
            module=obs["module"],
            module_rules=self.module.rules_text(self.config),
            effort_budget=obs["effort_budget"],
            sanction_rules=sanction_rules,
            norm_line=norm_line,
            schema_line=self.module.schema_line(self.config),
        )
        prompt += (
            f'\nThe optional "note" is private and must contain at most {NOTE_MAX_CHARS} Unicode characters.'
        )
        if obs["chat_enabled"]:
            prompt += (
                f' The optional "message" is public and must contain at most {obs["chat_max_chars"]} Unicode characters.'
            )
        else:
            prompt += ' Omit "message" because public chat is disabled.'
        prompt += " These limits count characters, not UTF-8 bytes."
        orders = truncate_runes(standing_orders, PROMPT_MAX_RUNES)
        if orders:
            prompt += f"\n\nSTANDING ORDERS\n{orders}"
        return prompt

    def user_message(self, obs: dict) -> str:
        payload = {key: value for key, value in obs.items() if key not in ("type", "round_seconds")}
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    # -- the batch ---------------------------------------------------------

    def decide(
        self,
        requests: dict[int, tuple[dict, str]],
        deadline: float,
        *,
        checkpoint_models: dict[int, str] | None = None,
    ) -> dict[int, tuple[dict | None, str]]:
        """One parallel batch with one call per prompt seat.

        Returns `{slot: (reply | None, cause)}`. `cause` is empty on success and
        one of `timeout, throttled, parse, invalid_action, rate_budget,
        transport, disabled` otherwise. Nothing here blocks past `deadline`.
        """
        self.checkpoint_models = {
            slot: checkpoint_model(model) for slot, model in (checkpoint_models or {}).items()
        }
        if self.checkpoint_models.keys() - requests.keys():
            raise ValueError("Checkpoint routes require a participating prompt seat")
        self.attempts = {}
        self.last_failures = {}
        self.capture.pending_attempts = self.attempts
        if isinstance(self.transport, MessagesTransport):
            self.transport.attempts = self.attempts
        if not requests:
            return {}
        if not self.enabled:
            self.last_failures = {
                slot: {"cause": "disabled", "reason": "no LLM credentials configured"}
                for slot in requests
            }
            return {slot: (None, "disabled") for slot in requests}
        slots = sorted(requests)
        with ThreadPoolExecutor(max_workers=max(1, len(slots))) as executor:
            answers = list(
                executor.map(
                    lambda slot: self._decide_seat(slot, *requests[slot], deadline),
                    slots,
                )
            )
        return dict(zip(slots, answers))

    def _record_failure(self, slot: int, cause: str, reason: str, raw_reply: str = "") -> None:
        entry: dict[str, Any] = {"cause": cause, "reason": reason}
        if raw_reply:
            entry["raw_reply"] = raw_reply
        with self._lock:
            self.last_failures[slot] = entry

    def _decide_seat(self, slot: int, obs: dict, standing_orders: str, deadline: float) -> tuple[dict | None, str]:
        if slot not in self._systems:
            self._systems[slot] = self.system_prompt(obs, standing_orders)
        system = self._systems[slot]
        user = self.user_message(obs)
        try:
            raw = self._complete(system, user, deadline, slot)
        except LlmRateBudget as error:
            self._record_failure(slot, "rate_budget", str(error))
            return None, "rate_budget"
        except LlmTimeout as error:
            self._record_failure(slot, "timeout", str(error))
            return None, "timeout"
        except LlmThrottled as error:
            self._record_failure(slot, "throttled", str(error))
            return None, "throttled"
        except LlmTransportError as error:
            logger.warning("slot %d transport error: %s", slot, error)
            self._record_failure(slot, "transport", str(error))
            return None, "transport"
        except Exception as error:
            logger.exception("slot %d unclassified LLM failure", slot)
            self._record_failure(slot, "transport", str(error))
            return None, "transport"
        if raw is None:
            self._record_failure(slot, "transport", "empty response")
            return None, "transport"
        parsed = extract_json(raw)
        if parsed is None:
            if self.attempts.get(slot):
                self.attempts[slot][-1]["rejection_reason"] = "no JSON object"
            logger.warning("slot %d unusable reply", slot)
            self._record_failure(slot, "parse", "no JSON object", raw)
            return None, "parse"
        try:
            parse_decision(parsed, slot, self.config, None, self.module, "llm")
        except ValueError as error:
            if self.attempts.get(slot):
                self.attempts[slot][-1]["rejection_reason"] = str(error)
            logger.warning("slot %d invalid decision: %s", slot, error)
            self._record_failure(slot, "invalid_action", str(error), raw)
            return None, "invalid_action"
        if self.attempts.get(slot):
            self.attempts[slot][-1]["_proposal"] = parsed
        return parsed, ""

    def _complete(self, system: str, user: str, deadline: float, slot: int) -> str | None:
        # Pre-4.6 models narrate their analysis and never reach the JSON unless
        # an assistant prefill forces the reply to BE the JSON object. 4.6+
        # models reject prefill and need max_tokens headroom instead.
        prefill = any(marker in self.model for marker in ("haiku-4-5", "sonnet-4-5", "haiku-4.5", "sonnet-4.5"))
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        if prefill:
            messages.append({"role": "assistant", "content": "{"})
        body = {
            "max_tokens": 300 if prefill else 4000,
            "system": system,
            "messages": messages,
        }
        if model := self.checkpoint_models.get(slot):
            # Route only inference. Keep the game's teacher scaffold, including
            # its assistant prefill and token ceiling, identical for the student.
            body["checkpoint_model"] = model
        if not prefill:
            body["output_config"] = {"effort": "low"}

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LlmTimeout("round deadline reached")
        timeout = min(self.config.decision_timeout_seconds, remaining)
        if not self._take_budget():
            raise LlmRateBudget("requests-per-minute budget spent")
        completion = self.transport.complete(body, timeout, slot)  # type: ignore[union-attr]
        if completion is None:
            return None
        if prefill:
            return completion if completion.lstrip().startswith(("{", "```")) else "{" + completion
        return completion

    def _take_budget(self) -> bool:
        """A rolling requests-per-minute budget shared across prompt seats."""
        now = time.monotonic()
        with self._lock:
            while self._budget and now - self._budget[0] > 60.0:
                self._budget.popleft()
            if len(self._budget) >= self.config.llm_max_requests_per_minute:
                return False
            self._budget.append(now)
            self.requests += 1
            return True

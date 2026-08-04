"""A canned completion function, so every node is testable without a network.

Injected into `Router(completion_fn=...)`. The response objects are deliberately
minimal duck-types rather than real `litellm.ModelResponse` instances — the router
detects that and skips LiteLLM pricing for them, which is exactly the code path a
gateway-hosted model takes in production.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FakeMessage:
    content: str


@dataclass(frozen=True, slots=True)
class FakeChoice:
    message: FakeMessage


@dataclass(frozen=True, slots=True)
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage


def _estimate_tokens(text: str) -> int:
    """Roughly four characters per token. Good enough for a fake."""
    return max(1, len(text) // 4)


@dataclass
class FakeCompletion:
    """Returns scripted replies and records what it was asked.

    `replies` may be a sequence consumed in order (the last one repeats once
    exhausted) or a callable taking the request kwargs and returning the reply text.
    """

    replies: Sequence[str] | Callable[[dict[str, Any]], str] = ("ok",)
    fail_times: int = 0
    failure: type[BaseException] = ConnectionError
    calls: list[dict[str, Any]] = field(default_factory=list)
    _index: int = 0
    _failures_emitted: int = 0

    async def __call__(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)

        if self._failures_emitted < self.fail_times:
            self._failures_emitted += 1
            raise self.failure(f"injected failure {self._failures_emitted}")

        if callable(self.replies):
            text = self.replies(kwargs)
        else:
            index = min(self._index, len(self.replies) - 1)
            text = self.replies[index]
            self._index += 1

        messages = kwargs.get("messages") or []
        prompt_text = "".join(str(m.get("content", "")) for m in messages)
        return FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content=text))],
            usage=FakeUsage(
                prompt_tokens=_estimate_tokens(prompt_text),
                completion_tokens=_estimate_tokens(text),
            ),
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

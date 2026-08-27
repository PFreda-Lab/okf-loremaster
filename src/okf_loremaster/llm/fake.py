"""A canned completion function, so every node is testable without a network.

Injected into `Router(completion_fn=...)`. The response objects are deliberately
minimal duck-types rather than real `litellm.ModelResponse` instances — the router
detects that and skips LiteLLM pricing for them, which is exactly the code path a
gateway-hosted model takes in production.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FakeMessage:
    content: str


@dataclass(frozen=True, slots=True)
class FakeChoice:
    message: FakeMessage
    # "length" is how a provider says it stopped mid-sentence at `max_tokens`. The
    # default is the ordinary case, so only a test about truncation has to mention it.
    finish_reason: str = "stop"


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
        # A reply longer than the budget it was given is what truncation looks like
        # from outside, so the fake reports it the way a provider would.
        budget = kwargs.get("max_tokens")
        ceiling = budget if isinstance(budget, int) else 0
        cut_off = ceiling > 0 and _estimate_tokens(text) > ceiling
        finish = "length" if cut_off else "stop"
        completion_tokens = _estimate_tokens(text)

        # ...and a provider reports it differently once a schema is in play, which is
        # the whole reason this is modeled rather than assumed. A `response_format` is
        # delivered as a forced tool call; a tool call truncated mid-arguments comes
        # back as a *clean stop*, with the budget fully spent and the partial JSON
        # discarded. Measured on the balanced model 2026-08-04 at three budgets: 64/64,
        # 256/256 and 1024/1024 tokens, `finish_reason='stop'`, `content='{}'` every
        # time. A fake that reported "length" here let a truncation bug ship green.
        if cut_off and kwargs.get("response_format") is not None:
            text, finish, completion_tokens = "{}", "stop", ceiling

        return FakeResponse(
            choices=[FakeChoice(message=FakeMessage(content=text), finish_reason=finish)],
            usage=FakeUsage(
                prompt_tokens=_estimate_tokens(prompt_text),
                completion_tokens=completion_tokens,
            ),
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class FakeStreamingCompletion:
    """Answers with a stream instead of a finished reply, so the watchdog is testable.

    Yields real LiteLLM chunk types rather than the duck-types above, because
    `Router._drain` hands them to `litellm.stream_chunk_builder`: anything else would
    exercise a reassembly that never runs in production.

    `stall_before` is the index of the piece to go silent in front of. The silence is
    unbounded — a stalled provider does not politely finish — so the watchdog is the only
    thing that can end the test.
    """

    pieces: Sequence[str] = ("ok",)
    stall_before: int | None = None
    usage: tuple[int, int] | None = (11, 7)
    finish_reason: str = "stop"
    closed: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append(kwargs)
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

        def chunk(content: str | None = None, finish: str | None = None) -> Any:
            return ModelResponseStream(
                model="gateway/deployment",
                choices=[
                    StreamingChoices(index=0, delta=Delta(content=content), finish_reason=finish)
                ],
            )

        try:
            for index, piece in enumerate(self.pieces):
                if self.stall_before == index:
                    await asyncio.sleep(3600)
                yield chunk(content=piece)
            yield chunk(finish=self.finish_reason)
            if self.usage is not None:
                last = chunk()
                last.usage = Usage(
                    prompt_tokens=self.usage[0],
                    completion_tokens=self.usage[1],
                    total_tokens=sum(self.usage),
                )
                yield last
        finally:
            self.closed = True

    @property
    def call_count(self) -> int:
        return len(self.calls)

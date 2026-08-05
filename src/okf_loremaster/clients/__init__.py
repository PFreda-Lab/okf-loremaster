"""HTTP clients for the four data sources.

E-utilities, BioC and PubTator are all `*.ncbi.nlm.nih.gov` and the rate limit is
enforced **per IP across all of them**, so they share one `HttpClient` and therefore one
token bucket. Giving each client its own limiter is the obvious design and it is wrong:
three clients at 8 rps each is 24 rps from NCBI's point of view.

iCite is a different host with its own budget, so it gets its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from okf_loremaster.clients._http import (
    ICITE_RPS,
    NCBI_CEILING_WITH_KEY,
    NCBI_CEILING_WITHOUT_KEY,
    NCBI_RPS_WITH_KEY,
    NCBI_RPS_WITHOUT_KEY,
    DiskCache,
    HttpClient,
    HttpStats,
    RateLimiter,
)
from okf_loremaster.clients.bioc import BioCClient, BioCDocument, BioCSection
from okf_loremaster.clients.eutils import (
    Author,
    ESearchResult,
    EUtilsClient,
    MeshTerm,
    PubMedRecord,
)
from okf_loremaster.clients.icite import CitationMetrics, ICiteClient
from okf_loremaster.clients.pubtator import (
    AnnotatedDocument,
    Annotation,
    PubTatorClient,
)
from okf_loremaster.config import ConfigError, Settings
from okf_loremaster.events import EventBus
from okf_loremaster.retention import http_cache_path

__all__ = [
    "AnnotatedDocument",
    "Annotation",
    "Author",
    "BioCClient",
    "BioCDocument",
    "BioCSection",
    "CitationMetrics",
    "Clients",
    "DiskCache",
    "ESearchResult",
    "EUtilsClient",
    "HttpClient",
    "HttpStats",
    "ICiteClient",
    "MeshTerm",
    "PubMedRecord",
    "PubTatorClient",
    "RateLimiter",
    "build_clients",
    "ncbi_rate",
]


def ncbi_rate(settings: Settings) -> float:
    """Requests per second to allow against NCBI, given whether a key is configured."""
    has_key = bool(settings.ncbi_api_key)
    rate = NCBI_RPS_WITH_KEY if has_key else NCBI_RPS_WITHOUT_KEY
    ceiling = NCBI_CEILING_WITH_KEY if has_key else NCBI_CEILING_WITHOUT_KEY
    return min(rate, ceiling)


@dataclass
class Clients:
    """Every data source, sharing the right limiters."""

    eutils: EUtilsClient
    bioc: BioCClient
    pubtator: PubTatorClient
    icite: ICiteClient
    ncbi_http: HttpClient
    icite_http: HttpClient
    # Both clients share it, and a run sweeps it. Held here rather than reached for
    # through one of them, since it belongs to neither.
    cache: DiskCache

    @property
    def stats(self) -> HttpStats:
        """Combined counters across both hosts."""
        a, b = self.ncbi_http.stats, self.icite_http.stats
        return HttpStats(
            requests=a.requests + b.requests,
            cache_hits=a.cache_hits + b.cache_hits,
            retries=a.retries + b.retries,
            bytes_downloaded=a.bytes_downloaded + b.bytes_downloaded,
        )

    async def aclose(self) -> None:
        await self.ncbi_http.aclose()
        await self.icite_http.aclose()


def build_clients(
    settings: Settings,
    *,
    bus: EventBus | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    require_email: bool = True,
) -> Clients:
    """Wire all four clients.

    `transport` is how a cassette gets injected: it sits below the cache and the
    limiter, so replayed tests run the same code path as a live call.
    """
    if require_email and not settings.ncbi_email:
        raise ConfigError(
            "NCBI requires a contact address on every request and throttles traffic "
            "that omits it. Set OKF_LOREMASTER_NCBI_EMAIL in your .env."
        )

    cache = DiskCache(
        http_cache_path(settings),
        ttl_days=settings.http_cache_ttl_days,
        enabled=settings.http_cache_enabled,
    )
    user_agent = f"{settings.ncbi_tool} (mailto:{settings.ncbi_email or 'unset'})"

    if settings.ca_bundle is not None and not settings.ca_bundle.is_file():
        raise ConfigError(
            f"OKF_LOREMASTER_CA_BUNDLE points at {settings.ca_bundle}, which is not a "
            "file. Leave it unset to use the default trust store."
        )

    def make(rate: float) -> HttpClient:
        return HttpClient(
            limiter=RateLimiter(rate),
            cache=cache,
            transport=transport,
            bus=bus,
            timeout=settings.http_timeout,
            max_retries=settings.http_max_retries,
            user_agent=user_agent,
            ca_bundle=settings.ca_bundle,
        )

    ncbi_http = make(ncbi_rate(settings))
    icite_http = make(ICITE_RPS)

    return Clients(
        eutils=EUtilsClient(ncbi_http, settings),
        bioc=BioCClient(ncbi_http),
        pubtator=PubTatorClient(ncbi_http),
        icite=ICiteClient(icite_http),
        ncbi_http=ncbi_http,
        icite_http=icite_http,
        cache=cache,
    )

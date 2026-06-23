"""
Rate Limiter Service

Implements per-provider token bucket rate limiting.
Each provider gets its own independent rate limiter instance.
"""

import time
import logging
import anyio
import database as db

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.

    Limits requests to a specified number per second (TPS).
    """

    def __init__(self, tps: float = 0):
        self.tps = tps
        self.interval = 1.0 / tps if tps > 0 else 0
        self.last_request_time = 0
        self.lock = anyio.Lock()

    def set_rate(self, tps: float):
        self.tps = tps
        self.interval = 1.0 / tps if tps > 0 else 0
        logger.info(f"Rate limiter set to {tps} TPS")

    async def wait(self):
        if self.tps <= 0:
            return

        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time

            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.3f}s (TPS: {self.tps})")
                await anyio.sleep(wait_time)
                self.last_request_time = time.time()
            else:
                self.last_request_time = now

    @property
    def is_enabled(self) -> bool:
        return self.tps > 0


class PerProviderRateLimiter:
    """
    Manages independent rate limiters per provider.

    Each provider ID gets its own RateLimiter with its own lock and timing,
    so limits are enforced independently.
    """

    def __init__(self, global_tps: float = 0):
        self._global = RateLimiter(global_tps)
        self._providers: dict[int, RateLimiter] = {}

    def set_global_rate(self, tps: float):
        self._global.set_rate(tps)

    def get_or_create_provider_limiter(self, provider_id: int, tps: float) -> RateLimiter:
        if provider_id not in self._providers:
            self._providers[provider_id] = RateLimiter(tps)
        elif tps != self._providers[provider_id].tps:
            self._providers[provider_id].set_rate(tps)
        return self._providers[provider_id]

    async def wait_for_provider(self, provider_id: int, provider_tps: float = None):
        """
        Wait using the provider-specific limiter if it has a rate set,
        otherwise fall back to the global limiter.
        """
        if provider_tps and provider_tps > 0:
            limiter = self.get_or_create_provider_limiter(provider_id, provider_tps)
            await limiter.wait()
        else:
            await self._global.wait()

    def remove_provider(self, provider_id: int):
        self._providers.pop(provider_id, None)

    @property
    def global_limiter(self) -> RateLimiter:
        return self._global


# Global instances
_rate_limiter: RateLimiter = None
_per_provider_limiter: PerProviderRateLimiter = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        tps = db.get_rate_limit_tps()
        _rate_limiter = RateLimiter(tps)
    return _rate_limiter


def get_per_provider_limiter() -> PerProviderRateLimiter:
    global _per_provider_limiter
    if _per_provider_limiter is None:
        tps = db.get_rate_limit_tps()
        _per_provider_limiter = PerProviderRateLimiter(tps)
    return _per_provider_limiter


def init_rate_limiter(tps: float = None) -> RateLimiter:
    global _rate_limiter, _per_provider_limiter
    if tps is None:
        tps = db.get_rate_limit_tps()
    _rate_limiter = RateLimiter(tps)
    _per_provider_limiter = PerProviderRateLimiter(tps)
    return _rate_limiter
"""
Rate Limiter Service

Implements token bucket rate limiting to protect free providers.
"""

import asyncio
import time
import logging
import database as db

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    
    Limits requests to a specified number per second (TPS).
    """
    
    def __init__(self, tps: float = 0):
        """
        Initialize rate limiter.
        
        Args:
            tps: Target requests per second (0 = disabled)
        """
        self.tps = tps
        self.interval = 1.0 / tps if tps > 0 else 0
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    def set_rate(self, tps: float):
        """
        Update the rate limit.
        
        Args:
            tps: New requests per second (0 = disabled)
        """
        self.tps = tps
        self.interval = 1.0 / tps if tps > 0 else 0
        logger.info(f"Rate limiter set to {tps} TPS")
    
    async def wait(self):
        """Wait if necessary to maintain the rate limit."""
        if self.tps <= 0:
            return
        
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
                self.last_request_time = time.time()
            else:
                self.last_request_time = now
    
    @property
    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.tps > 0


# Global instance
_rate_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        # Initialize with current database setting
        tps = db.get_rate_limit_tps()
        _rate_limiter = RateLimiter(tps)
    return _rate_limiter


def init_rate_limiter(tps: float = None) -> RateLimiter:
    """Initialize the global rate limiter."""
    global _rate_limiter
    if tps is None:
        tps = db.get_rate_limit_tps()
    _rate_limiter = RateLimiter(tps)
    return _rate_limiter
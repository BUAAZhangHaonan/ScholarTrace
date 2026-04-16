"""Token pool for DeepXiv API access.

Manages multiple DeepXiv tokens with:
- Auto-registration of new accounts
- Round-robin rotation
- Automatic rotation on rate limit (429)
- Async-safe via asyncio.Lock
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """Metadata for a single DeepXiv token."""
    token: str
    username: str = ""
    is_active: bool = True
    fail_count: int = 0


class TokenPool:
    """Pool of DeepXiv API tokens with rotation and auto-registration.

    Usage:
        pool = TokenPool(initial_tokens=["token1", "token2"])
        pool = TokenPool.from_env()  # reads DEEPXIV_TOKENS or DEEPXIV_TOKEN
        pool = TokenPool(auto_register=True, pool_size=3)

        token = await pool.get_token()    # get current best token
        await pool.rotate()               # move to next token (on 429)
    """

    def __init__(
        self,
        initial_tokens: list[str] | None = None,
        auto_register: bool = True,
        pool_size: int = 3,
    ):
        self._tokens: list[TokenInfo] = []
        self._index = 0
        self._lock = asyncio.Lock()
        self._auto_register = auto_register
        self._pool_size = pool_size

        if initial_tokens:
            for t in initial_tokens:
                if t.strip():
                    self._tokens.append(TokenInfo(token=t.strip()))

    @classmethod
    def from_env(cls) -> TokenPool:
        """Create pool from environment variables.

        Reads DEEPXIV_TOKENS (comma-separated) or DEEPXIV_TOKEN (single).
        """
        tokens_str = os.environ.get("DEEPXIV_TOKENS", "")
        if tokens_str:
            tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]
        else:
            single = os.environ.get("DEEPXIV_TOKEN", "")
            tokens = [single] if single.strip() else []

        return cls(initial_tokens=tokens)

    @property
    def size(self) -> int:
        """Number of tokens in pool."""
        return len(self._tokens)

    @property
    def active_count(self) -> int:
        """Number of active tokens."""
        return sum(1 for t in self._tokens if t.is_active)

    async def get_token(self) -> str:
        """Get the current best token.

        If pool is empty and auto_register is enabled, registers new tokens.
        If all tokens are inactive, re-activates the least-failed one.
        """
        async with self._lock:
            # Auto-register if pool is empty
            if not self._tokens:
                if self._auto_register:
                    await self._register_fill()
                else:
                    raise RuntimeError("DeepXiv token pool is empty and auto_register is disabled")

            # Find the next active token
            for _ in range(len(self._tokens)):
                info = self._tokens[self._index]
                if info.is_active:
                    return info.token
                self._index = (self._index + 1) % len(self._tokens)

            # All inactive — reactivate the least-failed one
            if self._tokens:
                best = min(self._tokens, key=lambda t: t.fail_count)
                best.is_active = True
                self._index = self._tokens.index(best)
                logger.info("Reactivated token %s (fail_count=%d)", best.username or "unknown", best.fail_count)
                return best.token

            raise RuntimeError("DeepXiv token pool exhausted")

    async def rotate(self) -> None:
        """Move to the next token (call on 429 or auth error).

        Marks the current token as failed and advances the pointer.
        """
        async with self._lock:
            if not self._tokens:
                return

            current = self._tokens[self._index]
            current.fail_count += 1
            logger.warning(
                "Token %s failed (count=%d), rotating",
                current.username or current.token[:8],
                current.fail_count,
            )

            self._index = (self._index + 1) % len(self._tokens)

            # If pool is getting thin and auto_register is on, add more
            active = self.active_count
            if active < self._pool_size and self._auto_register:
                await self._register_fill()

    async def _register_fill(self) -> None:
        """Register new tokens up to pool_size."""
        from .reader import DeepXivReader

        needed = self._pool_size - len(self._tokens)
        for _ in range(min(needed, 2)):  # Register at most 2 at a time
            token = await DeepXivReader.register()
            if token:
                self._tokens.append(TokenInfo(token=token))
                logger.info("Auto-registered new DeepXiv token (pool now has %d)", len(self._tokens))
            else:
                logger.warning("Auto-registration failed")
                break

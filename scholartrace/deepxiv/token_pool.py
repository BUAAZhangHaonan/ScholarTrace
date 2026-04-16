"""Token pool for DeepXiv API access."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from scholartrace.config import Settings, get_settings
from scholartrace.services import runtime_limits

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN = timedelta(seconds=60)


class TokenState(str, Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


@dataclass
class TokenInfo:
    token: str
    username: str = ""
    state: TokenState = TokenState.ACTIVE
    fail_count: int = 0
    cooldown_until: datetime | None = None


class TokenPool:
    """Pool of DeepXiv tokens with proactive routing and explicit token state."""

    def __init__(
        self,
        initial_tokens: list[str] | None = None,
        *,
        auto_register: bool = False,
        pool_size: int = 3,
        register_sdk_secret: str = "",
    ):
        self._tokens: list[TokenInfo] = []
        self._index = 0
        self._lock = asyncio.Lock()
        self._auto_register = auto_register
        self._pool_size = pool_size
        self._register_sdk_secret = register_sdk_secret

        for token in initial_tokens or []:
            if token.strip():
                self._tokens.append(TokenInfo(token=token.strip()))

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> TokenPool:
        runtime_settings = settings or get_settings()
        tokens = [
            token.strip()
            for token in runtime_settings.deepxiv_tokens.split(",")
            if token.strip()
        ]
        return cls(
            initial_tokens=tokens,
            auto_register=runtime_settings.deepxiv_auto_register,
            pool_size=runtime_settings.deepxiv_pool_size,
            register_sdk_secret=runtime_settings.deepxiv_register_sdk_secret,
        )

    @classmethod
    def from_env(cls) -> TokenPool:
        """Compatibility helper that still uses canonical ScholarTrace settings."""
        return cls.from_settings(get_settings())

    @property
    def size(self) -> int:
        return len(self._tokens)

    @property
    def active_count(self) -> int:
        return sum(1 for token in self._tokens if token.state == TokenState.ACTIVE)

    async def get_token(self) -> str:
        async with self._lock:
            await self._ensure_tokens_locked()
            self._refresh_states_locked()

            for offset in range(len(self._tokens)):
                index = (self._index + offset) % len(self._tokens)
                info = self._tokens[index]
                if info.state != TokenState.ACTIVE:
                    continue
                self._index = (index + 1) % len(self._tokens)
                return info.token

            if any(token.state == TokenState.COOLDOWN for token in self._tokens):
                raise RuntimeError("All DeepXiv tokens are cooling down or disabled")
            raise RuntimeError("No available DeepXiv tokens remain")

    async def mark_success(self, token: str) -> None:
        async with self._lock:
            info = self._find_token_locked(token)
            if info is None:
                return
            if info.state == TokenState.COOLDOWN and info.cooldown_until and info.cooldown_until <= datetime.utcnow():
                info.state = TokenState.ACTIVE
                info.cooldown_until = None

    async def mark_rate_limited(self, token: str, retry_after_seconds: int | None = None) -> None:
        async with self._lock:
            info = self._find_token_locked(token)
            if info is None:
                return
            info.fail_count += 1
            info.state = TokenState.COOLDOWN
            info.cooldown_until = datetime.utcnow() + timedelta(
                seconds=retry_after_seconds or _DEFAULT_COOLDOWN.total_seconds()
            )
            logger.warning(
                "DeepXiv token cooldown for %s until %s",
                info.username or token[:8],
                info.cooldown_until.isoformat(),
            )

    async def mark_auth_failed(self, token: str) -> None:
        async with self._lock:
            info = self._find_token_locked(token)
            if info is None:
                return
            info.fail_count += 1
            info.state = TokenState.DISABLED
            info.cooldown_until = None
            logger.warning("DeepXiv token disabled for %s", info.username or token[:8])

    def _find_token_locked(self, token: str) -> TokenInfo | None:
        for info in self._tokens:
            if info.token == token:
                return info
        return None

    def _refresh_states_locked(self) -> None:
        now = datetime.utcnow()
        for info in self._tokens:
            if info.state == TokenState.COOLDOWN and info.cooldown_until and info.cooldown_until <= now:
                info.state = TokenState.ACTIVE
                info.cooldown_until = None

    async def _ensure_tokens_locked(self) -> None:
        self._refresh_states_locked()
        if self._tokens and any(token.state == TokenState.ACTIVE for token in self._tokens):
            if self._auto_register and self.active_count < self._pool_size:
                await self._register_fill_locked()
                self._refresh_states_locked()
                if any(token.state == TokenState.ACTIVE for token in self._tokens):
                    return
            else:
                return

        if self._auto_register and self.active_count < self._pool_size:
            await self._register_fill_locked()
            self._refresh_states_locked()

        if self._tokens:
            return

        if self._auto_register and not self._register_sdk_secret:
            raise RuntimeError(
                "SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET is required when SCHOLARTRACE_DEEPXIV_AUTO_REGISTER=true",
            )
        raise RuntimeError(
            "DeepXiv is not configured. Set SCHOLARTRACE_DEEPXIV_TOKENS or enable SCHOLARTRACE_DEEPXIV_AUTO_REGISTER with SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET.",
        )

    async def _register_fill_locked(self) -> None:
        if not self._auto_register:
            return
        if not self._register_sdk_secret:
            raise RuntimeError(
                "SCHOLARTRACE_DEEPXIV_REGISTER_SDK_SECRET is required when SCHOLARTRACE_DEEPXIV_AUTO_REGISTER=true",
            )

        from scholartrace.deepxiv.reader import DeepXivReader

        needed = max(1, self._pool_size - self.active_count)
        for _ in range(needed):
            logger.info("Attempting explicit DeepXiv auto-registration")
            async with runtime_limits.budget_manager.enforce(
                runtime_limits.AUTO_REGISTER_POLICY,
                "deepxiv-auto-register",
            ):
                token = await DeepXivReader.register(self._register_sdk_secret)
            if not token:
                logger.warning("DeepXiv auto-registration failed")
                break
            self._tokens.append(TokenInfo(token=token))
            logger.info("Registered new DeepXiv token; pool size is now %d", len(self._tokens))

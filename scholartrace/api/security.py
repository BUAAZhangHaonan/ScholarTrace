from __future__ import annotations

import hmac
import logging
from collections.abc import Mapping

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from scholartrace.api.contracts import error_response

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_loopback_host(host: str) -> bool:
    return host in _LOOPBACK_HOSTS


def extract_access_token(headers: Headers | Mapping[str, str]) -> str:
    auth_header = headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return headers.get("x-scholartrace-token", "").strip()


def is_valid_access_token(expected_token: str, provided_token: str) -> bool:
    return bool(expected_token) and bool(provided_token) and hmac.compare_digest(
        expected_token,
        provided_token,
    )


def rest_auth_error_response(request: Request):
    logger.warning("REST auth failure for %s", request.url.path)
    return error_response(401, "unauthorized", "Authentication required")


class AccessTokenMiddleware:
    def __init__(self, app: ASGIApp, access_token: str):
        self._app = app
        self._access_token = access_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._access_token:
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        provided = extract_access_token(headers)
        if is_valid_access_token(self._access_token, provided):
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        logger.warning("MCP auth failure for %s", path)
        response = error_response(401, "unauthorized", "Authentication required")
        await response(scope, receive, send)

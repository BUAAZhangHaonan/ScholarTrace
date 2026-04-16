from __future__ import annotations

import json

from fastapi import HTTPException
from fastapi.responses import JSONResponse


_STATUS_TO_CODE = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    429: "rate_limited",
}


def error_payload(code: str, message: str, *, retryable: bool = False) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    }


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, message, retryable=retryable),
    )


def safe_http_exception_response(exc: HTTPException) -> JSONResponse:
    code = _STATUS_TO_CODE.get(exc.status_code, "request_failed")
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return error_response(
        exc.status_code,
        code,
        detail,
        retryable=exc.status_code >= 500 or exc.status_code == 429,
    )


def tool_error_json(
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> str:
    return json.dumps(
        error_payload(code, message, retryable=retryable),
        ensure_ascii=False,
    )

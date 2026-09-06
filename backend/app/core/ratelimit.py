"""Простой in-memory rate limit по IP.

Без Redis намеренно: сервис однопроцессный и не хранит состояние между
перезапусками — тащить сюда внешнее хранилище ради счётчика незачем.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request

from app.core.config import settings
from app.core.errors import AppError

_hits: dict[str, deque] = defaultdict(deque)
WINDOW = 60.0


def _client(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate(request: Request) -> None:
    key = _client(request)
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= settings.rate_limit_per_minute:
        raise AppError(
            "Слишком много запросов.", status_code=429,
            hint=f"Не больше {settings.rate_limit_per_minute} запросов в минуту. "
                 "Подождите немного.")
    q.append(now)

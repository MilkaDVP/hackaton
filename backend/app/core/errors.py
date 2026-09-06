"""Ошибки наружу — всегда в одном формате и всегда по-русски.

Трейсбек в ответ не попадает никогда: он уходит в лог, наружу идёт
понятная человеку формулировка и, где возможно, подсказка что делать.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

log = logging.getLogger("risk.errors")


class AppError(Exception):
    """Ожидаемая ошибка, которую не стыдно показать пользователю."""

    def __init__(self, message: str, *, status_code: int = 400,
                 hint: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.hint = hint
        self.details = details or {}


def _payload(message: str, hint: str | None = None, details: dict | None = None):
    body = {"error": {"message": message}}
    if hint:
        body["error"]["hint"] = hint
    if details:
        body["error"]["details"] = details
    return body


def install(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code,
                            content=_payload(exc.message, exc.hint, exc.details))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        fields = []
        for e in exc.errors():
            loc = ".".join(str(p) for p in e.get("loc", ()) if p != "body")
            fields.append({"field": loc, "problem": e.get("msg", "")})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_payload("Данные анкеты заполнены неверно.",
                             "Проверьте выделенные поля.", {"fields": fields}))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Трейсбек — только в лог. Наружу ничего внутреннего.
        log.exception("необработанная ошибка на %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload("Внутренняя ошибка сервиса.",
                             "Попробуйте ещё раз. Если повторяется — проверьте формат файла."))

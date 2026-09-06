from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core import errors
from app.core.config import settings
from app.services.predictor import registry


def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _setup_logging()
    registry.load()          # модели грузятся один раз при старте
    yield


app = FastAPI(
    title="Риск незачёта",
    description=("Список студентов в зоне риска для куратора. "
                 "Загруженные файлы обрабатываются в памяти и не сохраняются."),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

errors.install(app)
app.include_router(router)

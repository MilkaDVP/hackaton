# ---------- сборка зависимостей ----------
FROM python:3.11-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /w
COPY backend/requirements.txt .
# Чистим только то, что заведомо не нужно в рантайме.
# ВАЖНО: каталоги `tests` у numpy и scipy НЕ трогаем — numpy.testing импортирует
# numpy._core.tests._natype, и без него scipy.sparse не поднимается вообще.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt \
 && find /opt/venv/lib/python3.11/site-packages/pandas -type d -name tests -prune -exec rm -rf {} + \
 && find /opt/venv/lib/python3.11/site-packages/sklearn -type d -name tests -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name include -prune -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyx" -delete \
 && find /opt/venv -type f -name "*.c" -delete \
 && rm -rf /opt/venv/lib/python3.11/site-packages/pip \
           /opt/venv/lib/python3.11/site-packages/setuptools \
 # Дым-тест прямо в сборке: если чистка что-то сломала, образ не соберётся,
 # а не упадёт потом в рантайме crash-loop'ом.
 && /opt/venv/bin/python -c "import numpy, scipy.sparse, pandas, sklearn, joblib; from sklearn.calibration import CalibratedClassifierCV; print('импорты ок')"

# ---------- рантайм ----------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ARTIFACTS_DIR=/app/riskml/artifacts \
    DEMO_DATA=/app/data/student-por.csv

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
# riskml нужен рантайму: joblib пиклит engineer() по ссылке на модуль,
# без импортируемого riskml.pipeline артефакт не загрузится
COPY riskml/ /app/riskml/
COPY backend/app/ /app/app/
COPY data/student-por.csv /app/data/student-por.csv

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

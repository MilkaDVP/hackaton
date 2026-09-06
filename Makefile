.DEFAULT_GOAL := help
COMPOSE ?= docker compose
PY ?= python

.PHONY: help up dev down logs build rebuild train test lint format clean

help: ## Показать эту справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

up: ## Собрать и поднять приложение (http://localhost:3000)
	$(COMPOSE) up --build -d
	@echo "фронтенд  http://localhost:3000"
	@echo "API       http://localhost:8000/docs"

dev: ## Режим разработки с hot reload (http://localhost:5173)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up --build

down: ## Остановить и убрать контейнеры
	$(COMPOSE) down

logs: ## Логи всех сервисов
	$(COMPOSE) logs -f --tail=100

build: ## Только собрать образы
	$(COMPOSE) build

rebuild: ## Пересобрать с нуля, без кэша
	$(COMPOSE) build --no-cache

train: ## Переобучить модель и обновить riskml/artifacts
	$(PY) -m riskml.train --data data/student-por.csv --reference research/results.json

test: ## Тесты бэкенда (включая паритет с ноутбуком)
	cd backend && $(PY) -m pytest tests -q

lint: ## Проверить стиль python
	$(PY) -m ruff check riskml backend/app backend/tests

format: ## Отформатировать python
	$(PY) -m ruff format riskml backend/app backend/tests
	$(PY) -m ruff check --fix riskml backend/app backend/tests

clean: ## Убрать контейнеры и мусор сборки
	$(COMPOSE) down -v --remove-orphans
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache backend/.pytest_cache frontend/dist

# Makefile для сервера. Лежит в /opt/risk/deploy, запускать из /opt/risk:
#   make -f deploy/Makefile deploy
.DEFAULT_GOAL := help

C := docker compose --env-file deploy/.env -f deploy/docker-compose.prod.yml

.PHONY: help deploy up down restart logs ps health prune

help: ## Показать команды
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

deploy: ## Обновить сайт (сборка + перезапуск + откат при неудаче)
	bash deploy/deploy.sh

up: ## Поднять стек
	$(C) up -d

down: ## Остановить стек
	$(C) down

restart: ## Перезапустить
	$(C) restart

logs: ## Логи (LOG=caddy для одного сервиса)
	$(C) logs -f --tail=100 $(LOG)

ps: ## Состояние контейнеров
	$(C) ps

health: ## Проверить бэкенд изнутри сети
	$(C) exec -T backend python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/health').read().decode()[:200])"

prune: ## Убрать неиспользуемые образы (диск всего 10 ГБ)
	docker system prune -af --filter "until=168h"
	df -h / | tail -1

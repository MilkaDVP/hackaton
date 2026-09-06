#!/usr/bin/env bash
# Поднять/обновить боевой стек. Запускается на сервере из /opt/risk.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f deploy/.env ] || { echo "нет deploy/.env — скопируйте из .env.example"; exit 1; }
set -a; . deploy/.env; set +a

echo "=== сборка (может занять несколько минут на одном ядре) ==="
docker compose --env-file deploy/.env -f deploy/docker-compose.prod.yml build

echo "=== запуск ==="
docker compose --env-file deploy/.env -f deploy/docker-compose.prod.yml up -d

echo "=== ожидание готовности ==="
for i in $(seq 1 40); do
  st=$(docker inspect --format '{{.State.Health.Status}}' risk-backend-prod 2>/dev/null || echo none)
  [ "$st" = healthy ] && break
  sleep 5
done
docker compose --env-file deploy/.env -f deploy/docker-compose.prod.yml ps

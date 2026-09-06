#!/usr/bin/env bash
# Обновление боевого стека. Запускается НА СЕРВЕРЕ из /opt/risk:
#   bash deploy/deploy.sh
#
# Логика простая и без сюрпризов: собрать новые образы, поднять, дождаться
# healthy. Если новый бэкенд не поднялся — откатиться на предыдущий образ.
set -uo pipefail
cd "$(dirname "$0")/.."

C="docker compose --env-file deploy/.env -f deploy/docker-compose.prod.yml"
# .env НЕ сорсим: в bcrypt-хэше есть $2a$14$, шелл на нём ломается.

log() { echo -e "\n=== $* ==="; }

[ -f deploy/.env ] || { echo "нет deploy/.env"; exit 1; }

log "метка текущих образов (для отката)"
docker tag risk-backend:prod  risk-backend:rollback  2>/dev/null || true
docker tag risk-frontend:prod risk-frontend:rollback 2>/dev/null || true
echo "ок"

log "сборка"
if ! $C build; then
  echo "сборка упала — ничего не меняли, старая версия работает"
  exit 1
fi

log "перезапуск"
$C up -d

log "проверка здоровья"
healthy=0
for _ in $(seq 1 40); do
  n=$($C ps --format '{{.Status}}' | grep -c healthy || true)
  if [ "$n" -ge 2 ]; then healthy=1; break; fi
  sleep 5
done

if [ "$healthy" -ne 1 ]; then
  log "ОТКАТ: сервисы не поднялись"
  docker tag risk-backend:rollback  risk-backend:prod  2>/dev/null || true
  docker tag risk-frontend:rollback risk-frontend:prod 2>/dev/null || true
  $C up -d
  $C ps
  exit 1
fi

log "уборка"
# Диск всего 10 ГБ, а каждая пересборка оставляет старые слои и кэш билдера
# (~1.7 ГБ за раз). Без этого через два деплоя место кончается.
docker image prune -af  --filter "label!=keep" >/dev/null 2>&1 || true
docker builder prune -af >/dev/null 2>&1 || true
df -h / | tail -1

log "готово"
$C ps --format '{{.Service}}  {{.Status}}'
curl -sf -o /dev/null -w "локальная проверка health: %{http_code}\n" \
  http://127.0.0.1/api/health || echo "health недоступен снаружи caddy (это норма, он под авторизацией)"

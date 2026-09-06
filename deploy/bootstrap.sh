#!/usr/bin/env bash
# Подготовка чистого сервера. Идемпотентен: повторный запуск ничего не ломает.
#   bash bootstrap.sh
set -euo pipefail

log() { echo -e "\n=== $* ==="; }

# ---------------------------------------------------------------- swap
# 1 ядро и 4 ГБ. Сборка фронтенда (vite + esbuild) на одном ядре легко упирается
# в память, поэтому swap делаем ДО первой сборки, а не после первого OOM.
log "swap"
if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl -w vm.swappiness=10 >/dev/null
  grep -q 'vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
  echo "swap 2 ГБ создан"
else
  echo "swap уже есть: $(swapon --show --noheadings | head -1)"
fi

# ---------------------------------------------------------------- пакеты
log "базовые пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg ufw ncdu >/dev/null
echo "ок"

# ---------------------------------------------------------------- docker
log "docker"
if ! command -v docker >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  # Ubuntu 26.04 может ещё не иметь своего пула в репозитории Docker —
  # берём кодовое имя ближайшего LTS, если своего нет.
  CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  if ! curl -fsI "https://download.docker.com/linux/ubuntu/dists/$CODENAME/Release" >/dev/null 2>&1; then
    echo "для $CODENAME пула нет, откатываемся на noble"
    CODENAME=noble
  fi
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $CODENAME stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin >/dev/null
fi
docker --version && docker compose version

# ---------------------------------------------------------------- логи docker
# 10 ГБ диска: логи контейнеров забьют его быстрее всего остального.
log "ротация docker-логов"
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
JSON
systemctl restart docker
echo "ок"

# ---------------------------------------------------------------- firewall
log "ufw"
ufw allow 22/tcp  >/dev/null
ufw allow 80/tcp  >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null
ufw status numbered | head -8

# ---------------------------------------------------------------- уборка
log "еженедельная уборка docker"
cat > /etc/cron.weekly/docker-prune <<'CRON'
#!/bin/sh
docker system prune -af --filter "until=168h" >/dev/null 2>&1
CRON
chmod +x /etc/cron.weekly/docker-prune
echo "ок"

log "готово"
free -m | head -2
df -h / | tail -1

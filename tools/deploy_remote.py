# -*- coding: utf-8 -*-
"""Разворачивает проект на сервере за ОДНО SSH-подключение.

Отдельные подключения на каждую команду провайдер начинает резать как брутфорс,
поэтому здесь одна сессия: загрузка архива + все шаги + проверки.

    SRV_KEY=~/.ssh/shazram_ed25519 python tools/deploy_remote.py путь_к_архиву
"""
from __future__ import annotations

import os
import sys
import time

import paramiko

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HOST = os.environ.get("SRV_HOST", "2.56.120.183")
USER = os.environ.get("SRV_USER", "root")
KEY = os.path.expanduser(os.environ.get("SRV_KEY", "~/.ssh/shazram_ed25519"))
REMOTE_DIR = "/opt/risk"


def connect(attempts: int = 8) -> paramiko.SSHClient:
    """Подключаемся с бэкоффом: сервер может временно резать соединения."""
    last = None
    for i in range(attempts):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST, username=USER, key_filename=KEY, timeout=30,
                      banner_timeout=30, auth_timeout=30,
                      look_for_keys=False, allow_agent=False)
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            wait = min(15 * (i + 1), 60)
            print(f"  подключение не удалось ({type(e).__name__}), жду {wait} с")
            time.sleep(wait)
    raise SystemExit(f"не удалось подключиться: {last}")


def sh(c: paramiko.SSHClient, cmd: str, label: str = "", check: bool = True) -> str:
    if label:
        print(f"\n=== {label} ===")
    _, out, err = c.exec_command(cmd, timeout=None)
    buf = []
    for line in iter(out.readline, ""):
        line = line.rstrip()
        buf.append(line)
        print(line)
    code = out.channel.recv_exit_status()
    e = err.read().decode("utf-8", "replace").strip()
    if e:
        print(e[:2000], file=sys.stderr)
    if check and code != 0:
        raise SystemExit(f"шаг «{label or cmd[:40]}» упал с кодом {code}")
    return "\n".join(buf)


def upload(c: paramiko.SSHClient, local: str, remote: str) -> None:
    size = os.path.getsize(local)
    print(f"\n=== загрузка {size/1e6:.1f} МБ ===")
    sftp = c.open_sftp()
    done = [0]

    def cb(sent, total):
        pct = sent * 100 // max(total, 1)
        if pct // 20 > done[0] // 20:
            print(f"  {pct}%")
        done[0] = pct

    sftp.put(local, remote, callback=cb)
    sftp.close()
    print("  загружено")


def main() -> int:
    archive = sys.argv[1]
    if not os.path.exists(archive):
        raise SystemExit(f"нет архива: {archive}")

    c = connect()
    try:
        sh(c, f"mkdir -p {REMOTE_DIR}", "подготовка каталога")
        upload(c, archive, "/root/risk.tar.gz")
        sh(c, f"tar -xzf /root/risk.tar.gz -C {REMOTE_DIR} && rm -f /root/risk.tar.gz && "
              f"du -sh {REMOTE_DIR} && ls {REMOTE_DIR}", "распаковка")
    finally:
        c.close()
    print("\nпроект на сервере")
    return 0


if __name__ == "__main__":
    sys.exit(main())

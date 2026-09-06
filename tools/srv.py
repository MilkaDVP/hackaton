# -*- coding: utf-8 -*-
"""Мини-обёртка над SSH для деплоя: локально нет sshpass, поэтому ходим paramiko.

    python tools/srv.py run  "uname -a"
    python tools/srv.py put  local.tar /root/local.tar
    python tools/srv.py cmd  file_with_commands.sh

Пароль/ключ берутся из окружения, в репозиторий не попадают:
    SRV_HOST, SRV_USER, SRV_PASS  либо  SRV_KEY (путь к приватному ключу)
"""
from __future__ import annotations

import os
import sys

import paramiko

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

HOST = os.environ.get("SRV_HOST", "2.56.120.183")
USER = os.environ.get("SRV_USER", "root")
PASS = os.environ.get("SRV_PASS")
KEY = os.environ.get("SRV_KEY")


def connect(attempts: int = 8) -> paramiko.SSHClient:
    """С бэкоффом: провайдер режет частые подключения как брутфорс,
    и обрыв на этапе SSH-баннера — это именно он, а не проблема с ключом."""
    import time
    key = os.path.expanduser(KEY) if KEY else None
    last = None
    for i in range(attempts):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            kw = dict(timeout=30, banner_timeout=30, auth_timeout=30,
                      look_for_keys=False, allow_agent=False)
            if key and os.path.exists(key):
                c.connect(HOST, username=USER, key_filename=key, **kw)
            else:
                c.connect(HOST, username=USER, password=PASS, **kw)
            return c
        except Exception as e:  # noqa: BLE001
            last = e
            wait = min(15 * (i + 1), 60)
            print(f"  подключение не удалось ({type(e).__name__}), жду {wait} с",
                  file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"не удалось подключиться: {last}")


def run(c: paramiko.SSHClient, cmd: str, quiet: bool = False) -> int:
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=False, timeout=None)
    for line in iter(stdout.readline, ""):
        if not quiet:
            print(line.rstrip())
    err = stderr.read().decode("utf-8", "replace").strip()
    code = stdout.channel.recv_exit_status()
    if err and not quiet:
        print(err, file=sys.stderr)
    return code


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    action = sys.argv[1]
    c = connect()
    try:
        if action == "run":
            return run(c, sys.argv[2])
        if action == "cmd":
            script = open(sys.argv[2], encoding="utf-8").read()
            sftp = c.open_sftp()
            with sftp.file("/tmp/_step.sh", "w") as f:
                f.write(script)
            sftp.close()
            return run(c, "bash /tmp/_step.sh")
        if action == "put":
            sftp = c.open_sftp()
            sftp.put(sys.argv[2], sys.argv[3])
            sftp.close()
            print(f"загружено -> {sys.argv[3]}")
            return 0
        print(f"неизвестное действие: {action}")
        return 2
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())

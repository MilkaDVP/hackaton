# -*- coding: utf-8 -*-
"""Запуск долгой команды на сервере так, чтобы она пережила обрыв SSH.

Провайдер режет частые подключения, а сборка на одном ядре идёт минутами —
держать канал открытым всё это время ненадёжно. Поэтому: одно подключение,
скрипт улетает через stdin (без SFTP), запускается через setsid+nohup и пишет
в лог на сервере. Дальше состояние опрашивается отдельными редкими заходами.

    python tools/remote_run.py start script.sh   # запустить в фоне
    python tools/remote_run.py tail  [n]         # хвост лога
    python tools/remote_run.py wait              # ждать завершения
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

LOG = "/opt/risk/remote.log"
DONE = "/opt/risk/remote.done"
SCRIPT = "/opt/risk/remote_task.sh"


def connect(attempts: int = 10) -> paramiko.SSHClient:
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
            wait = min(20 * (i + 1), 90)
            print(f"  подключение не удалось ({type(e).__name__}), жду {wait} с",
                  file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"не удалось подключиться: {last}")


def exec_out(c: paramiko.SSHClient, cmd: str) -> tuple[int, str]:
    _, out, err = c.exec_command(cmd, timeout=120)
    text = out.read().decode("utf-8", "replace")
    code = out.channel.recv_exit_status()
    e = err.read().decode("utf-8", "replace").strip()
    return code, text + (("\n" + e) if e else "")


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "tail"
    c = connect()
    try:
        if action == "start":
            body = open(sys.argv[2], encoding="utf-8").read()
            # Скрипт уезжает одной командой в base64: и SFTP, и запись через
            # stdin рвутся на этом сервере, а тут всё помещается в argv.
            import base64
            b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
            code, t = exec_out(c, f"echo '{b64}' | base64 -d > {SCRIPT} && wc -l {SCRIPT}")
            if code != 0:
                print(t)
                return 1
            print(t.strip())
            code, t = exec_out(
                c,
                f"rm -f {LOG} {DONE}; "
                f"setsid nohup bash -c 'bash {SCRIPT} > {LOG} 2>&1; "
                f"echo $? > {DONE}' < /dev/null > /dev/null 2>&1 & echo запущено")
            print(t.strip())
            return 0

        if action == "tail":
            n = sys.argv[2] if len(sys.argv) > 2 else "40"
            _, t = exec_out(c, f"tail -n {n} {LOG} 2>/dev/null; "
                               f"echo '---'; cat {DONE} 2>/dev/null || echo 'ещё работает'")
            print(t)
            return 0

        if action == "wait":
            for _ in range(120):
                code, t = exec_out(c, f"cat {DONE} 2>/dev/null || echo RUNNING")
                if "RUNNING" not in t:
                    rc = t.strip().splitlines()[0]
                    print(f"завершено, код {rc}")
                    _, tail = exec_out(c, f"tail -n 30 {LOG}")
                    print(tail)
                    return 0 if rc == "0" else 1
                time.sleep(20)
            print("не дождались")
            return 1

        print(__doc__)
        return 2
    finally:
        c.close()


if __name__ == "__main__":
    sys.exit(main())

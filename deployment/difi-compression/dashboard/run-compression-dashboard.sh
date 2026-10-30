#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p logs
if [ "${1:-}" = '--stop' ]; then
  tmux kill-session -t cusat-dashboard 2>/dev/null || true
  python3 -c 'import importlib.util; s=importlib.util.spec_from_file_location("demo","compression-demo.py"); d=importlib.util.module_from_spec(s); s.loader.exec_module(d); [d.transport.connect(n) for n in d.HOSTS]; d.stop()'
  echo 'Dashboard and compression modems stopped.'
  exit 0
fi
if tmux has-session -t cusat-dashboard 2>/dev/null; then
  if curl --silent --fail --max-time 2 http://localhost:8765/api/state | python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if all(d.get("hosts",{}).get(n,{}).get("connected") for n in ("11","12")) else 1)' 2>/dev/null; then
    echo 'Dashboard is running: http://localhost:8765'
    exit 0
  fi
  echo 'Existing session has no dashboard server. Inspect: tmux attach -t cusat-dashboard' >&2
  exit 1
fi
# Establish SSH in the foreground so a password prompt is visible when keys/masters expire.
python3 -c 'import launcher; [launcher.connect(n) for n in ("11","12")]'
tmux new-session -d -s cusat-dashboard -n dashboard "python3 -u dashboard-server.py --start 2>&1 | tee logs/dashboard-server.log"
echo 'Starting radios and traffic in tmux cusat-dashboard…'
for attempt in $(seq 1 150); do
  if curl --silent --fail --max-time 2 http://localhost:8765/api/state | python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if all(d.get("hosts",{}).get(n,{}).get("connected") for n in ("11","12")) else 1)' 2>/dev/null; then
    echo 'Ready: http://localhost:8765'
    echo 'Console: tmux attach -t cusat-dashboard'
    exit 0
  fi
  if ! tmux has-session -t cusat-dashboard 2>/dev/null; then
    tail -30 logs/dashboard-server.log >&2
    exit 1
  fi
  sleep 1
done
echo 'Startup timed out. Inspect logs/dashboard-server.log or tmux attach -t cusat-dashboard.' >&2
exit 1

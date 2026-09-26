#!/bin/sh
# A local MongoDB for offline development and the tests, on :27100. The data
# lives in .local/mongo inside this checkout (git ignores it), so nothing
# touches a system install. --fork is unsupported on macOS, hence nohup.
#   scripts/mongo_local.sh        start (no-op if already up)
#   scripts/mongo_local.sh stop   stop it
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="${CYCLOPSDIARY_MONGO_DIR:-$ROOT/.local/mongo}"
PORT="${CYCLOPSDIARY_MONGO_PORT:-27100}"
if [ "$1" = "stop" ]; then
  pkill -f "mongod --dbpath $DB/db" && echo "mongod on :$PORT stopped" || echo "not running"; exit 0
fi
mkdir -p "$DB/db" "$DB/log"
if nc -z localhost "$PORT" 2>/dev/null; then echo "already up on :$PORT"; exit 0; fi
nohup mongod --dbpath "$DB/db" --logpath "$DB/log/mongod.log" --port "$PORT" --bind_ip 127.0.0.1 \
      >/dev/null 2>&1 &
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1; nc -z localhost "$PORT" 2>/dev/null && { echo "mongod up on :$PORT ($DB)"; exit 0; }
done
echo "mongod failed to start; see $DB/log/mongod.log"; exit 1

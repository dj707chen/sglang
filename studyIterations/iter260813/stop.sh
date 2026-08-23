#!/usr/bin/env bash
# Stop the local SGLang server and every process it spawned.
#
#   ./stop.sh          # SIGTERM the tree, escalate to SIGKILL after 10s
#   ./stop.sh --force  # SIGKILL immediately
#
# Also sweeps for orphaned "sglang::scheduler" / "sglang::detokenizer"
# processes: they normally die with the parent (the scheduler installs a
# PR_SET_PDEATHSIG-equivalent watchdog), but a SIGKILL'd parent can leave them.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# Depth-first walk so children are collected before their parent.
collect_tree() {
    local pid=$1 child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        collect_tree "$child"
    done
    echo "$pid"
}

TARGETS=()
if [[ -f "$PID_FILE" ]]; then
    ROOT=$(cat "$PID_FILE")
    if kill -0 "$ROOT" 2>/dev/null; then
        mapfile -t TARGETS < <(collect_tree "$ROOT")
    else
        echo "pid file points at a dead process ($ROOT); cleaning up"
    fi
fi

# Orphans: match the scheduler/detokenizer proc titles and the launch command.
while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ " ${TARGETS[*]-} " == *" $pid "* ]] && continue
    TARGETS+=("$pid")
done < <(pgrep -f 'sglang::|sglang\.launch_server' 2>/dev/null)

if (( ${#TARGETS[@]} == 0 )); then
    echo "nothing to stop"
    rm -f "$PID_FILE"
    exit 0
fi

echo "stopping ${#TARGETS[@]} process(es):"
for pid in "${TARGETS[@]}"; do
    printf '  %-7s %s\n' "$pid" "$(ps -p "$pid" -o command= 2>/dev/null | cut -c1-100)"
done

SIG=TERM
(( FORCE )) && SIG=KILL
for pid in "${TARGETS[@]}"; do kill "-$SIG" "$pid" 2>/dev/null; done

if (( ! FORCE )); then
    for _ in $(seq 1 20); do
        alive=0
        for pid in "${TARGETS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        (( alive )) || break
        sleep 0.5
    done
    for pid in "${TARGETS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  pid $pid ignored SIGTERM -> SIGKILL"
            kill -KILL "$pid" 2>/dev/null
        fi
    done
fi

rm -f "$PID_FILE"

# The ZMQ ipc:// sockets are NamedTemporaryFiles under $TMPDIR, created by
# PortArgs.init_new(); they are not cleaned up here because their names are
# random and nothing else depends on them.

echo "stopped"

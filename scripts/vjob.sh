#!/usr/bin/env bash
# Run a long job as a transient systemd service, so it survives ssh/tmux and is supervised by systemd.
#   scripts/vjob.sh start NAME LOGFILE -- command args...   start unit vpipe-NAME, stdout+stderr appended to LOGFILE
#   scripts/vjob.sh status NAME                              active/inactive + result + exit code
#   scripts/vjob.sh stop NAME                                stop the unit (and the containers it started)
set -euo pipefail
cmd=$1 name=$2
unit="vpipe-${name}"
case "$cmd" in
  start)
    log=$3; shift 3; [ "${1:-}" = "--" ] && shift
    mkdir -p "$(dirname "$log")"
    # SELinux denies systemd exec of scripts on /data (default_t); have bash read them instead
    case "$1" in *.sh) set -- /bin/bash "$@" ;; esac
    systemd-run --unit "$unit" --collect --same-dir \
      --setenv=PATH="/root/.local/bin:/usr/local/bin:/usr/bin:/bin" \
      --property=StandardOutput="append:$log" --property=StandardError="append:$log" \
      --property=KillMode=control-group "$@"
    echo "started $unit -> $log"
    ;;
  status)
    systemctl show "$unit" -p ActiveState -p Result -p ExecMainStatus -p ActiveEnterTimestamp 2>/dev/null || echo "no unit $unit"
    ;;
  stop)
    systemctl stop "$unit"
    ;;
  *) echo "usage: $0 start|status|stop NAME ..." >&2; exit 2 ;;
esac

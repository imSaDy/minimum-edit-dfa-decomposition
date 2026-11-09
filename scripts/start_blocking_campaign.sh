#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source remote/environment.sh
reference="$project_dir/remote/results/paper-final-3-10"
output="$project_dir/remote/results/paper-blocking-20261014"
session="dfa-paper-blocking-20261014"
case "${1:-start}" in
  prepare)
    python scripts/blocking_campaign.py prepare --reference "$reference" --output "$output"
    ;;
  start)
    if systemctl --user is-active --quiet "$session.service" || tmux has-session -t "$session" 2>/dev/null; then
      echo "Blocking campaign already running: $session"
      exit 0
    fi
    bash scripts/start_blocking_campaign.sh prepare
    printf -v launch '%q ' bash "$project_dir/scripts/start_blocking_campaign.sh" foreground
    tmux new-session -d -s "$session" "$launch"
    echo "Started $session; progress: $output/progress.json"
    ;;
  foreground)
    trap 'code=$?; printf "%s\n" "$code" > "$output/exit-code.txt"' EXIT
    read -r maximum high < <(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["aggregate_memory_max_gib"],r["aggregate_memory_high_gib"])' "$output/runtime-resources.json")
    systemd-run --user --unit="$session" --collect --wait --service-type=exec \
      --setenv="PATH=$PATH" --property="MemoryMax=${maximum}G" --property="MemoryHigh=${high}G" \
      --property=MemorySwapMax=0 --property=OOMPolicy=continue --property=KillMode=control-group \
      --property=TimeoutStopSec=30s --property="WorkingDirectory=$project_dir" \
      /bin/bash "$project_dir/scripts/start_blocking_campaign.sh" managed >> "$output/run.log" 2>&1
    ;;
  managed)
    python -u remote/dispatch_paper.py --output "$output"
    python scripts/blocking_campaign.py check --reference "$reference" --output "$output"
    ;;
  monitor)
    exec bash remote/monitor_paper.sh --output "$output" --interval 30 "${@:2}"
    ;;
  status)
    cat "$output/progress.json"
    ;;
  check)
    python scripts/blocking_campaign.py check --reference "$reference" --output "$output"
    ;;
  *) echo "Usage: bash scripts/start_blocking_campaign.sh [start|prepare|monitor|status|check]"; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source remote/environment.sh
output="$project_dir/remote/results/paper-final-3-10"
protocol="$project_dir/remote/experiments/paper-20261005/verification-variants.json"
session="dfa-paper-final-3-10"
mode=${1:-start}
case "$mode" in
  prepare)
    python -u remote/run_paper.py --protocol "$protocol" --output "$output" --prepare
    ;;
  foreground)
    mkdir -p "$output"
    trap 'code=$?; printf "%s\n" "$code" > "$output/exit-code.txt"' EXIT
    if [[ -f "$output/runtime-resources.json" ]]; then
      read -r maximum high < <(python -c 'import json,sys; r=json.load(open(sys.argv[1])); a=r["aggregate_memory_max_gib"]; b=r["aggregate_memory_high_gib"]; assert type(a) is int and type(b) is int and 0 < b < a; print(a,b)' "$output/runtime-resources.json")
      systemd-run --user --unit="$session" --collect --wait --service-type=exec \
        --setenv="PATH=$PATH" \
        --property="MemoryMax=${maximum}G" --property="MemoryHigh=${high}G" \
        --property=MemorySwapMax=0 --property=OOMPolicy=continue \
        --property=KillMode=control-group --property=TimeoutStopSec=30s \
        --property="WorkingDirectory=$project_dir" \
        /bin/bash "$project_dir/remote/start_paper.sh" managed >> "$output/run.log" 2>&1
    else
      python -u "$project_dir/remote/dispatch_paper.py" --output "$output" >> "$output/run.log" 2>&1
    fi
    ;;
  managed)
    exec python -u "$project_dir/remote/dispatch_paper.py" --output "$output" >> "$output/run.log" 2>&1
    ;;
  start)
    if systemctl --user is-active --quiet "$session.service"; then
      echo "Campaign is already running in service $session.service"
      exit 0
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Campaign is already running in tmux session $session"
      exit 0
    fi
    python -u remote/run_paper.py --protocol "$protocol" --output "$output" --prepare
    printf -v launch '%q ' bash "$project_dir/remote/start_paper.sh" foreground
    tmux new-session -d -s "$session" "$launch"
    echo "Started/resumed session: $session"
    echo "Progress: $output/progress.json"
    echo "Log: $output/run.log"
    echo "Attach: tmux attach -t $session"
    ;;
  *) echo "Usage: bash remote/start_paper.sh [start|prepare]"; exit 2 ;;
esac

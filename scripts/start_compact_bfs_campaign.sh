#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
source remote/environment.sh
reference="$project_dir/remote/results/paper-final-3-10"
output="$project_dir/remote/results/paper-compact-bfs-20261012"
report="$project_dir/remote/reports/paper-four-bfs-20261012"
session="dfa-paper-compact-bfs-20261012"
case "${1:-start}" in
  prepare)
    python scripts/compact_bfs_campaign.py prepare --reference "$reference" --output "$output"
    ;;
  start)
    if systemctl --user is-active --quiet "$session.service" || tmux has-session -t "$session" 2>/dev/null; then
      echo "Campaign already running: $session"
      exit 0
    fi
    bash scripts/start_compact_bfs_campaign.sh prepare
    printf -v launch '%q ' bash "$project_dir/scripts/start_compact_bfs_campaign.sh" foreground
    tmux new-session -d -s "$session" "$launch"
    echo "Started $session; progress: $output/progress.json"
    ;;
  foreground)
    trap 'code=$?; printf "%s\n" "$code" > "$output/exit-code.txt"' EXIT
    if [[ -f "$output/runtime-resources.json" ]]; then
      read -r maximum high < <(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["aggregate_memory_max_gib"],r["aggregate_memory_high_gib"])' "$output/runtime-resources.json")
      systemd-run --user --unit="$session" --collect --wait --service-type=exec \
        --setenv="PATH=$PATH" --property="MemoryMax=${maximum}G" --property="MemoryHigh=${high}G" \
        --property=MemorySwapMax=0 --property=OOMPolicy=continue --property=KillMode=control-group \
        --property=TimeoutStopSec=30s --property="WorkingDirectory=$project_dir" \
        /bin/bash "$project_dir/scripts/start_compact_bfs_campaign.sh" managed >> "$output/run.log" 2>&1
    else
      bash scripts/start_compact_bfs_campaign.sh managed >> "$output/run.log" 2>&1
    fi
    ;;
  managed)
    python -u remote/dispatch_paper.py --output "$output"
    python scripts/compact_bfs_campaign.py compare --reference "$reference" --supplement "$output" --output "$report"
    ;;
  status)
    cat "$output/progress.json"
    ;;
  compare)
    python scripts/compact_bfs_campaign.py compare --reference "$reference" --supplement "$output" --output "$report"
    ;;
  *) echo "Usage: bash scripts/start_compact_bfs_campaign.sh [start|prepare|status|compare]"; exit 2 ;;
esac

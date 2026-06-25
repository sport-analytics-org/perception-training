#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="/court-vision/deploy/.env"
SESSION="court-vision-perception-training"

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

PERCEPTION_HOST="${PERCEPTION_HOST:-127.0.0.1}"
PERCEPTION_PORT="${PERCEPTION_PORT:-6001}"
COURT_SEGMENTATION_CHECKPOINT="$CHECKPOINTS_ROOT/basket-court-segmentation/vit-large-basket-seg-keypoints.pt"
COURT_DETECTION_CHECKPOINT="$CHECKPOINTS_ROOT/court-detection/rfdetr-large-allclasses-640/best.pt"

if [ "${1:-}" = "--run" ]; then
  export COURT_SEGMENTATION_CHECKPOINT COURT_DETECTION_CHECKPOINT
  uv run --project "$ROOT" uvicorn perception_training.api:app \
    --host "$PERCEPTION_HOST" \
    --port "$PERCEPTION_PORT"
fi

test "${1:-}" = ""
command -v curl >/dev/null
command -v tmux >/dev/null
command -v uv >/dev/null
test -f "$ENV_FILE"
test -f "$COURT_SEGMENTATION_CHECKPOINT"
test -f "$COURT_DETECTION_CHECKPOINT"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -c "$ROOT" "./deploy.sh --run"

for _ in $(seq 1 "${HEALTH_TIMEOUT_SECONDS:-120}"); do
  if curl -fsS --max-time 2 "http://$PERCEPTION_HOST:$PERCEPTION_PORT/health" >/dev/null 2>&1; then
    echo "started $SESSION"
    exit 0
  fi
  sleep 1
done

tmux capture-pane -pt "$SESSION" -S -200 >&2 || true
exit 1

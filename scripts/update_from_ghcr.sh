#!/usr/bin/env bash
# Pull a GHCR image and recreate the local compose service, then verify health.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REGISTRY_IMAGE_DEFAULT="ghcr.io/highkay/googleaisearch2api"
TAG_OR_IMAGE="${1:-${GOOGLEAISEARCH2API_IMAGE:-latest}}"
HOST_PORT="${APP_PORT:-9010}"
BASE_URL="${SMOKE_BASE_URL:-http://127.0.0.1:${HOST_PORT}}"
RUN_SMOKE="${RUN_SMOKE:-1}"

if [[ "$TAG_OR_IMAGE" == *"/"* ]]; then
  IMAGE="$TAG_OR_IMAGE"
else
  IMAGE="${REGISTRY_IMAGE_DEFAULT}:${TAG_OR_IMAGE}"
fi

export GOOGLEAISEARCH2API_IMAGE="$IMAGE"
export GOOGLEAISEARCH2API_PULL_POLICY="${GOOGLEAISEARCH2API_PULL_POLICY:-always}"

PIN_ENV="${PIN_ENV:-1}"

echo "==> Using image: $GOOGLEAISEARCH2API_IMAGE"
echo "==> Pull policy: $GOOGLEAISEARCH2API_PULL_POLICY"

if ! docker compose pull; then
  cat <<'EOF' >&2
docker compose pull failed.
If the GHCR package is private, login first:

  echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
EOF
  exit 1
fi

docker compose up -d --force-recreate --remove-orphans

echo "==> Waiting for healthz"
deadline=$((SECONDS + 120))
while true; do
  if curl -fsS "${BASE_URL}/healthz" >/tmp/googleaisearch2api-healthz.json 2>/dev/null; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "healthz did not become ready within 120s" >&2
    docker compose ps >&2 || true
    docker compose logs --tail 80 >&2 || true
    exit 1
  fi
  sleep 2
done

python3 - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path("/tmp/googleaisearch2api-healthz.json").read_text())
print(json.dumps(payload, ensure_ascii=False, indent=2))
if not payload.get("ok"):
    raise SystemExit("healthz ok=false")
if payload.get("accepting_requests") is False:
    raise SystemExit("healthz accepting_requests=false")
print("healthz OK")
PY

RUNNING_IMAGE="$(docker compose ps --format '{{.Image}}' 2>/dev/null | head -n1 || true)"
echo "==> Running image: ${RUNNING_IMAGE:-unknown}"

if [[ "$PIN_ENV" == "1" && -f .env ]]; then
  python3 - <<PY
from pathlib import Path
image = ${IMAGE@Q}
path = Path(".env")
lines = []
seen_image = False
seen_pull = False
for line in path.read_text().splitlines():
    if line.startswith("GOOGLEAISEARCH2API_IMAGE="):
        lines.append(f"GOOGLEAISEARCH2API_IMAGE={image}")
        seen_image = True
    elif line.startswith("GOOGLEAISEARCH2API_PULL_POLICY="):
        # Pin to never after a successful pull so later recreate won't depend on GHCR.
        lines.append("GOOGLEAISEARCH2API_PULL_POLICY=never")
        seen_pull = True
    else:
        lines.append(line)
if not seen_image:
    lines.append(f"GOOGLEAISEARCH2API_IMAGE={image}")
if not seen_pull:
    lines.append("GOOGLEAISEARCH2API_PULL_POLICY=never")
path.write_text("\\n".join(lines) + "\\n")
print(f"pinned .env -> {image} (PULL_POLICY=never)")
PY
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  if [[ -n "${API_TOKEN:-}" ]]; then
    echo "==> Running smoke_api.py"
    if command -v uv >/dev/null 2>&1; then
      uv run python scripts/smoke_api.py --base-url "$BASE_URL" || {
        echo "smoke failed (service is up; inspect logs)" >&2
        exit 1
      }
    else
      echo "uv not found; skip smoke_api.py" >&2
    fi
  else
    echo "==> API_TOKEN not set in shell; skip smoke_api.py (compose still uses .env)"
  fi
fi

echo "==> Update complete: $GOOGLEAISEARCH2API_IMAGE"

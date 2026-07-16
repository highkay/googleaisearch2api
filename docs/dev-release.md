# 开发 → 发布 → 本地更新流程

本文是本仓库**标准发布路径**。每次功能/修复上线都按此执行，避免本地 ad-hoc 镜像与 GHCR 分叉。

## 目标闭环

```text
改代码
  → uv run pytest / ruff
  → git commit + push origin main
  → GitHub Actions 构建并推送 GHCR
  → 本地 pull sha-* 镜像并 recreate
  → healthz / smoke 验证
```

镜像名：

```text
ghcr.io/highkay/googleaisearch2api
```

| 触发 | 标签 |
|------|------|
| push `main` | `latest`, `main`, `sha-<7位>` |
| tag `v*.*.*` | `v*.*.*`, `sha-<7位>` |
| `workflow_dispatch` | 分支名 + `sha-<7位>`（默认分支另加 `latest`） |

Workflow：`.github/workflows/docker-publish.yml`

---

## 1. 本地开发与自检

```bash
cd /path/to/googleaisearch2api
uv sync --extra dev
uv run ruff check .
uv run ruff format .
uv run pytest
```

改 Google AI 选择器/提取逻辑前先取证：

```bash
uv run python scripts/probe_google_ai.py --prompt "What changed in OpenAI Responses API?"
```

开发态 Compose（挂源码，非生产默认）：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

生产默认 Compose **不挂源码**，只跑镜像里的代码。

---

## 2. 审查后提交并推送

```bash
git status
git diff
uv run pytest -q
uv run ruff check .

git add <files>   # 不要提交 .env / .env.backup* / .deploy-backups/
git commit -m "简述改了什么以及为什么"
git push origin main
```

`push main` 即触发 GHCR 构建。记录短 SHA：

```bash
git rev-parse --short=7 HEAD
# 例: 25e1512
```

查看 Action：

```bash
# 有 gh CLI
gh run list --workflow=docker-publish.yml --limit 5
gh run watch

# 无 gh：浏览器打开
# https://github.com/highkay/googleaisearch2api/actions
```

也可用 API 轮询：

```bash
curl -sS "https://api.github.com/repos/highkay/googleaisearch2api/actions/runs?per_page=3" \
  | python3 -c 'import sys,json
for r in json.load(sys.stdin)["workflow_runs"][:3]:
  print(r["status"], r["conclusion"], r["head_sha"][:7], r["html_url"])'
```

**务必等 `status=completed` 且 `conclusion=success` 再拉镜像。**

---

## 3. 本地更新 GHCR 镜像（推荐脚本）

```bash
# 生产更稳：钉死本次 commit 的 sha
./scripts/update_from_ghcr.sh sha-<7位>

# 或跟踪 latest（每次 main 都会动）
./scripts/update_from_ghcr.sh latest
```

脚本会：

1. 设置 `GOOGLEAISEARCH2API_IMAGE` / `PULL_POLICY=always`
2. `docker compose pull`
3. `docker compose up -d --force-recreate`
4. 轮询 `/healthz` 直到 `ok=true`
5. 若 shell 有 `API_TOKEN`，可选跑 `scripts/smoke_api.py`

### 手工等价步骤

```bash
export GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:sha-<7位>
export GOOGLEAISEARCH2API_PULL_POLICY=always
docker compose pull
docker compose up -d --force-recreate
curl -fsS http://127.0.0.1:${APP_PORT:-9010}/healthz | python3 -m json.tool
```

建议同步写入 `.env`（数据卷不受影响）：

```env
GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:sha-<7位>
GOOGLEAISEARCH2API_PULL_POLICY=never
```

`PULL_POLICY=never` 适合镜像已 pull 到本地后，避免 GHCR 偶发网络抖动导致 recreate 失败；下次升级前再改为 `always` 或直接用脚本。

若包为 private：

```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

数据卷 `googleaisearch2api-data` 保留 SQLite（配置、日志、sticky 状态）；**换镜像不会清空会话池**。

---

## 4. 发布后验收清单

```bash
# 镜像是否为目标 sha
docker inspect googleaisearch2api-googleaisearch2api-1 --format '{{.Config.Image}}'

# 健康
curl -fsS http://127.0.0.1:9010/healthz | python3 -m json.tool
```

关注字段：

| 字段 | 期望 |
|------|------|
| `ok` / `accepting_requests` | true |
| `sticky_hot_pool_sessions` / `sticky_active_sessions` | 有 Google 时 ≥1 更好；0 则 auto 应走 Duck |
| `browser_gate.busy` | 空闲时 false |
| `proxy_auto_recovery.last_success` | canary 补到 active 后应为 true |
| `workers_with_errors` | 最好 0 |

真实请求：

```bash
curl -sS http://127.0.0.1:9010/v1/chat/completions \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"google-search","messages":[{"role":"user","content":"What is 19 plus 23? Reply with only the number."}]}'
```

---

## 5. 回滚

```bash
# 回上一已知好的 sha
./scripts/update_from_ghcr.sh sha-<old>

# 或改 .env 后
# GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:sha-<old>
# GOOGLEAISEARCH2API_PULL_POLICY=always
docker compose pull && docker compose up -d --force-recreate
```

---

## 6. 运行时约束（不要绕过）

- Google 入口是 `udm=50&aep=11`，不是 `google.com/ai`
- 每个 browser worker 独占 runner/browser/context
- Compose 保持 `init: true`，回收 Chrome 僵尸进程
- **Hot 池**只含 `status=active`；cooldown 到期不会自动再进线上选择
- recovery 与 Google worker 互斥 browser gate；`auto` 在热池为空或 recovery 运行时直走 Duck
- `proxy_auto_recovery.last_success=true` 表示 canary 达到 active 目标，不是“进程随便 exit 0”
- 300+ sticky IP 是**库存**；线上只需要少量 Hot active，质量靠 canary 动态筛

---

## 7. 一页速查（复制即用）

```bash
# --- 开发自检 ---
uv run ruff check . && uv run pytest -q

# --- 提交发布 ---
git add -A   # 先检查，排除 .env*
git commit -m "..."
git push origin main
SHA=$(git rev-parse --short=7 HEAD)
echo "waiting for GHCR sha-$SHA ..."

# --- 等 Action success 后本地更新 ---
./scripts/update_from_ghcr.sh "sha-$SHA"

# --- 验收 ---
curl -fsS http://127.0.0.1:9010/healthz | python3 -m json.tool
docker inspect googleaisearch2api-googleaisearch2api-1 --format '{{.Config.Image}}'
```

---

## 8. 并行开发建议

- 根因用真实 `/healthz`、SQLite、容器日志取证后再改
- 选择器/提取逻辑与并发模型拆开改，各自跑对应测试
- 大规模 `probe_proxy_sessions` 与在线流量错峰；recovery 已与 Google worker 互斥，仍避免无意义狂扫 300 IP

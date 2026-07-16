# 开发 → 发布 → 本地更新流程

本文把本仓库已经在用的发布路径固化成可重复步骤。目标：

```text
本地改代码 → pytest → push main → GitHub Actions 构建 GHCR 镜像 → 本地 pull/up → healthz/smoke
```

## 1. 本地开发

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format .
uv run pytest
```

如果改了 Google AI 选择器或页面提取逻辑，先重新取证：

```bash
uv run python scripts/probe_google_ai.py --prompt "What changed in OpenAI Responses API?"
```

开发态容器（挂源码，非默认生产路径）：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## 2. 提交与触发镜像构建

```bash
git status
git add <files>
git commit -m "..."
git push origin main
```

`main` 分支 push 会触发 `.github/workflows/docker-publish.yml`：

| 触发 | 镜像标签 |
|------|----------|
| push `main` | `latest`, `main`, `sha-<7位>` |
| tag `v*.*.*` | `v*.*.*`, `sha-<7位>` |
| `workflow_dispatch` | 当前分支名 + `sha-<7位>`（默认分支还会带 `latest`） |

镜像名：

```text
ghcr.io/highkay/googleaisearch2api
```

查看构建：

```bash
# 需要 gh 已登录
gh run list --workflow=docker-publish.yml --limit 5
gh run watch
```

没有 `gh` 时，直接打开：

```text
https://github.com/highkay/googleaisearch2api/actions
```

## 3. 本地从 GHCR 更新

推荐用仓库脚本（会 pull、recreate、healthz、可选 smoke）：

```bash
./scripts/update_from_ghcr.sh
```

指定不可变 sha（生产更稳）：

```bash
./scripts/update_from_ghcr.sh sha-abc1234
# 或
GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:sha-abc1234 \
  ./scripts/update_from_ghcr.sh
```

等价手工步骤：

```bash
# 若 GHCR 包为 private，先登录
# echo "$GITHUB_TOKEN" | docker login ghcr.io -u USERNAME --password-stdin

export GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:latest
export GOOGLEAISEARCH2API_PULL_POLICY=always
docker compose pull
docker compose up -d
curl -fsS http://127.0.0.1:9010/healthz
uv run python scripts/smoke_api.py --base-url http://127.0.0.1:9010
```

相关 `.env` 变量（见 `.env.example`）：

```env
GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:latest
GOOGLEAISEARCH2API_PULL_POLICY=always
```

数据卷 `googleaisearch2api-data` 会保留 SQLite（配置、请求日志、sticky session 状态），更新镜像不会清空会话池。

## 4. 回滚

```bash
# 回到上一个已知好的 sha
GOOGLEAISEARCH2API_IMAGE=ghcr.io/highkay/googleaisearch2api:sha-<old> \
  GOOGLEAISEARCH2API_PULL_POLICY=always \
  ./scripts/update_from_ghcr.sh
```

或在 `.env` 里改 `GOOGLEAISEARCH2API_IMAGE` 后：

```bash
docker compose pull && docker compose up -d
```

## 5. 发布后稳定性检查清单

1. `GET /healthz` 返回 `ok=true`，`accepting_requests=true`
2. 关注：
   - `sticky_active_sessions` / `sticky_selectable_sessions`
   - `workers_with_errors`
   - `proxy_auto_recovery.last_success`（为 `false` 时说明 canary 没补到 active）
3. 看容器日志是否反复出现：
   - `Google sticky proxy session pool is empty`
   - 同一批 `google_canary_blocked` 会话被 recovery 反复探测
4. 真实请求：

```bash
curl -sS http://127.0.0.1:9010/v1/chat/completions \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"google-search","messages":[{"role":"user","content":"What is 19 plus 23? Reply with only the number."}]}'
```

## 6. 已知运行时约束（不要绕过）

- 入口是 `udm=50&aep=11`，不是 `google.com/ai`
- 每个 browser worker 独占 runner/browser/context
- Docker 必须 `init: true`，避免 Chrome 僵尸进程堆 PID
- sticky session 自动恢复默认只做少量 Google canary；`last_success=true` 仅表示 probe 进程成功拿到 active 目标，不是“随便 exit 0”
- `SEARCH_ENGINE=auto` 会在 Google sticky 失败后降级 Duck.ai，但应先用 sticky 重试预算换会话，而不是一碰 block 就立刻放弃其它 ready session

## 7. 并行开发建议

- 稳定性根因用真实 `/healthz`、SQLite `proxy_sessions`/`request_logs`、容器日志取证，再改代码
- 选择器/提取逻辑与并发模型拆开改，并分别跑对应测试
- probe 脚本（`scripts/probe_*.py`）可在容器外并行跑，但不要和在线 worker 抢同一代理会话预算


## Sticky Hot Pool 行为（2026-07 起）

- **Hot 池**仅包含 `status=active` 的 sticky session（canary 成功或真实 Google 成功后晋升）。
- cooldown 到期**不会**自动重新进入线上选择；必须由 recovery canary 再次晋升。
- `auto` 在 Hot 池为空，或 recovery 占用 browser gate 时，**直接走 Duck**，不占 Google worker。
- recovery 与 Google worker 互斥占用 browser gate；Duck 不受此限制。
- Google block 的 IPv4/IPv6 会合并进 session 的 `ip_vector`，供 known-block 跳过匹配。

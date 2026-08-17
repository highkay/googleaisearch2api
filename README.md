# Google AI Search2API

`googleaisearch2api` 把 Gemini web 的 StreamGenerate 协议包装成一个 OpenAI 兼容的搜索 API（进程内纯 HTTP 引擎，无需浏览器），并提供本地 Web Console 查看配置、请求统计和实时探针结果。Duck.ai 浏览器引擎作为自动降级兜底。

## 运行时事实

- 主引擎是进程内 Gemini web HTTP 引擎（`gemini_web.py`，102 槽 StreamGenerate payload），请求时自动注入强制搜索指令，答案自带 grounding 引用。
- 代理选择是两层融合池：WARP 出口（Tier-1 稳定层，`GEMINI_WARP_PROXIES` 逗号分隔的 `socks5h://` URL，必须远端 DNS）→ 2260 sticky 会话（Tier-2 容量层）→ base 单次兜底。每层都先快速探测（真实 StreamGenerate POST）再选用，失败自动冷却轮换。
- Gemini web 的构建版本号（BL）会被缓存复用，避免每次请求都抓取；失败时回退到已知可用的默认 BL。
- Duck.ai 兜底仍走 `patchright + chrome` 浏览器（Playwright 官方镜像 Chromium，映射到 Patchright `chrome` channel）；原 Google AI Mode 浏览器抓取代码保留休眠，不再参与任何请求路由。
- 单进程并发通过常驻 browser worker 池实现；每个 worker 独占自己的 browser/context。

## 功能

- `GET /v1/models`
- `GET /query`
- `POST /query`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- Bearer Token 认证
- 可配置模型名、代理、语言、超时、worker 数和队列长度
- 可在 Gemini web、Duck.ai、gemini-upstream 和自动降级模式之间切换
- SQLite 持久化配置与请求日志
- 本地 Web Console: `/console`

## 推荐启动方式

默认推荐直接用 Docker Compose：

```bash
cp .env.example .env
```

先在 `.env` 里填一个你自己的 `API_TOKEN`，再启动：

```bash
docker compose pull
docker compose up -d
```

默认宿主端口是 `9010`。启动后访问：

- API: `http://127.0.0.1:9010`
- Console: `http://127.0.0.1:9010/console`

默认 Compose 使用 GitHub Actions 发布到 GHCR 的镜像：`ghcr.io/highkay/googleaisearch2api:latest`。它不再挂载源码目录，也不会在容器启动时重新执行 `uv sync`。这样可以直接复用镜像里已经构建好的运行环境，避免宿主机仓库里的 `.python-version=3.13` 触发容器冷启动下载 Python，导致服务长时间不可用。

**开发发布闭环**（必读）：[docs/dev-release.md](docs/dev-release.md)

**WARP 出口稳定性**（运维必读）：[docs/warp-stability.md](docs/warp-stability.md) —— 外部 warp-plus 舰队容器死亡根因、症状→诊断表、真实隧道健康检查配方与身份持久化。加固 overlay 见 `docker-compose.warpplus-hardened.yml`。

```bash
uv run pytest -q
git push origin main
# 等 Actions success 后（本机走 sparkcr 镜源）：
#   .env: GOOGLEAISEARCH2API_IMAGE=ghcr.sparkcr.cn/highkay/googleaisearch2api:sha-$(git rev-parse --short=7 HEAD)
#   docker compose pull && docker compose up -d --force-recreate
```

## 启动后验证

```bash
curl http://127.0.0.1:9010/healthz
```

```bash
curl http://127.0.0.1:9010/v1/models \
  -H "Authorization: Bearer your-strong-token"
```

```bash
uv run python scripts/smoke_api.py --base-url http://127.0.0.1:9010
```

## 开发态 Compose

如果你确实需要把本地源码目录挂进容器，再显式叠加开发态覆盖文件：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

这个覆盖文件会：

- 挂载当前仓库到 `/app`
- 保留容器内 `.venv`
- 在容器启动时执行 `uv sync --frozen --no-dev`
- 强制 `uv` 使用容器现成的 `/usr/bin/python3.12`

这样可以避免因为仓库根目录的 `.python-version=3.13` 导致容器在每次启动时重新下载 Python。开发态适合修改代码后重建/重启容器验证，但它不是默认推荐的运行方式。

## 本地直接运行

```bash
uv sync --extra dev
uv run patchright install chrome
cp .env.example .env
uv run googleaisearch2api
```

默认地址：

- API: `http://127.0.0.1:8000`
- Console: `http://127.0.0.1:8000/console`

## 关键配置

- `API_TOKEN`: OpenAI 兼容接口的 Bearer Token
- Console 使用同一个 `API_TOKEN` 登录；敏感字段默认不会回显到页面里
- `DEFAULT_MODEL`: 对外暴露的模型名
- `SEARCH_ENGINE`: 搜索引擎选择，支持 `gemini | duck | gemini-upstream | auto`，默认 `gemini`（Gemini web HTTP 引擎，走 Gemini web StreamGenerate）；`auto` 会优先跑 Gemini web，遇到上游不可用类错误再降级到 Duck.ai。
- `GEMINI_WEB_MODEL`: Gemini web HTTP 引擎使用的模型名；默认 `gemini-3.7-flash`。
- `GEMINI_WEB_COOKIE`: 可选，Gemini web 会话 Cookie（`__Secure-1PSID` 等），用于提升 Gemini web HTTP 引擎的可用性。
- `GEMINI_WEB_SAPISID`: 可选，Gemini web 会话的 `SAPISID` 值，用于 Gemini web HTTP 引擎请求签名。
- `GEMINI_FAST_PROBE_TIMEOUT_S`: 冷池时对候选 sticky 会话做 gemini.google.com 快速探测的单次超时（秒）；默认 `8.0`。
- `GEMINI_MAX_PROBE_SESSIONS`: 冷池时每次请求最多快速探测的候选会话数（用于轮换 IP）；默认 `3`。
- `GEMINI_WARP_PROXIES`: WARP 出口池（Tier-1 稳定层），逗号分隔的 `socks5h://` URL 列表（必须用 `socks5h` 远端 DNS，`socks5` 会握手失败）；留空 = 禁用。需要容器接入 `warp-plus_default` 外部 Docker 网络（docker-compose 已配好）。示例：`GEMINI_WARP_PROXIES=socks5h://warpplus-us:1080,socks5h://warpplus-ca:1080,...`。选中策略：先轮换探测 WARP 出口，全部不可用才落到 2260 sticky 会话，最后才是 base 单次兜底。
- `BROWSER_HEADLESS`: 是否无头运行
- `BROWSER_USER_AGENT`: 可选，覆盖浏览器级 UA；留空时服务会给 headless Chrome 使用普通 Chrome UA
- `BROWSER_WORKERS`: 常驻浏览器 worker 数
- `REQUEST_QUEUE_SIZE`: 内存等待队列容量；满了以后返回 `429`
- `REQUEST_LOG_MAX_ROWS`: SQLite 里最多保留多少条最近请求日志；默认 2000
- `GOOGLE_AI_BLOCKED_RETRY_COUNT`: Google 返回机器人/abuse block 页面时，回收当前 browser session 后重试的次数；默认 0。只有在代理会轮换到新出口 IP 时才建议调高；同一出口网络被 Google block 时，立即重试通常会提高失败率。
- `DUCK_AI_WORKERS`: Duck.ai 独立浏览器 worker 数；默认 4。实测 Duck.ai 对并发 burst 更敏感，如需调高请先小步验证。
- `DUCK_AI_QUEUE_SIZE`: Duck.ai 队列长度；默认 8。
- `DUCK_AI_COOLDOWN_SECONDS`: Duck.ai 返回限流时的本地熔断冷却时间；默认 120。
- `AI_MODE_HTTP_ENABLED`: 是否启用 Google AI Mode 混合 HTTP 快路径（浏览器铸 folif token + curl_cffi 查询）；默认 `false`，未验证协议，需先用 `scripts/probe_ai_mode_tokens.py` 测 token TTL 后再开启。
- `PROXY_AUTO_RECOVERY_ENABLED`: 启用 sticky session 自动恢复；恢复任务默认只做少量 Google canary，不跑 egress/IPLark，以避免恢复过程本身放大浏览器资源占用。
- `PROXY_AUTO_RECOVERY_MAX_PROBES`: 单次自动恢复最多执行多少个昂贵探针；默认 3。
- `PROXY_AUTO_RECOVERY_TARGET_ACTIVE`: 触发恢复时希望补到的 Google selectable session 数；这是恢复目标，不是保证值，实际数量仍取决于当前代理出口是否被 Google block。
- `GOOGLEAISEARCH2API_CPUS` / `GOOGLEAISEARCH2API_MEMORY_LIMIT` / `GOOGLEAISEARCH2API_PIDS_LIMIT`: Docker 资源护栏；默认 `2.0` CPU、`3g` 内存、`512` PID。
- `BROWSER_PROXY_SERVER`: 代理地址，例如 `http://127.0.0.1:7890` 或 `socks5h://user:pass@host:port`；HTTP 代理会把认证字段传给浏览器，`socks5`/`socks5h` 会先走本地 HTTP CONNECT 桥接层，再由桥接层向 SOCKS5 上游完成认证。

请求日志会自动脱敏常见密钥、Bearer token、`user:pass@host` 形式的内联凭据，并且不会把最终 Google URL 里的 `q=` 查询词原样持久化到 SQLite。

Docker Compose 默认使用 `init: true` 启动服务，让 `docker-init` 回收 Chrome/Crashpad 子进程；不要移除这个设置，否则浏览器异常退出后可能堆积僵尸进程并耗尽 PID。

如果容器里需要走宿主机代理：

```env
BROWSER_PROXY_SERVER=http://host.docker.internal:7890
```

如果要使用带认证的 SOCKS5 代理，推荐使用远端 DNS 解析的 `socks5h` 形式：

```env
BROWSER_PROXY_SERVER=socks5h://user:pass@192.168.1.18:2260
```

## API 示例

列出模型：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer your-strong-token"
```

轻量查询接口：

```bash
curl http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-strong-token" \
  -d '{
    "model": "google-search",
    "query": "What is the difference between Responses API and Chat Completions API? summarize in 3 points",
    "stream": false
  }'
```

快速 GET 查询：

```bash
curl "http://127.0.0.1:8000/query?q=What%20changed%20in%20OpenAI%20Responses%20API%3F" \
  -H "Authorization: Bearer your-strong-token"
```

Chat Completions：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-strong-token" \
  -d '{
    "model": "google-search",
    "messages": [
      {"role": "user", "content": "What is the difference between Responses API and Chat Completions API? summarize in 3 points"}
    ]
  }'
```

Responses API：

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-strong-token" \
  -d '{
    "model": "google-search",
    "input": "Summarize the latest differences between Responses API and Chat Completions API in 3 points."
  }'
```

## Tool Query 设计

`/query` 是给 tool wrapper 使用的轻量查询接口，不伪装成 OpenAI SDK 协议，但功能上覆盖现有 OpenAI 兼容接口能做的事：

- `POST /query` 使用 JSON body，适合工具调用和较长查询。
- `GET /query?q=...` 使用 query string，适合临时探测；长文本或敏感文本仍建议走 POST。
- 入参核心字段是 `query`，可选 `instructions` 和 `context` 用来表达 OpenAI chat/responses 里的 system prompt 与上下文。
- `context` 支持普通字符串，也支持 `{role, content}` 数组；角色支持 `system`、`developer`、`user`、`assistant`。
- 返回结构是工具友好的 `answer`、`citations`、`usage`、`google_ai`，不需要客户端解析 `choices` 或 `output`。
- `stream=true` 时返回 SSE：`query.created`、`answer.delta`、`query.completed`；和现有流式接口一样，这是拿到完整 Google AI 结果后的事件回放，不是 Google 原生流式协议透传。
- 认证、模型解析、请求日志、并发队列、浏览器 worker 和错误码都复用现有 `/v1/chat/completions` 与 `/v1/responses` 的基础设施。

## 已知边界

- Gemini web 引擎走的是逆向的 StreamGenerate 协议（102 槽 payload），不是 Google 官方公开 API；协议可能变化，槽位布局以 `gemini_web.py` 与上游 gemini-web2api 为参考，改动前需先重新取证。
- 强制搜索指令会让 Gemini 对事实类问题也走搜索 grounding（返回引用）；纯算术问题会自动跳过该指令。
- WARP 出口必须用 `socks5h://`（远端 DNS），`socks5://` 会 SOCKS 握手失败。
- 当前 streaming 是在拿到完整答案后按 OpenAI SSE 形状回放，不是 Google 原生流式协议透传。
- 代理能否连通取决于你自己的代理服务（WARP 出口需容器接入 `warp-plus_default` 网络，2260 sticky 为 HTTP 代理）。
- 目前只支持文本输入。`tools`、图片/文件输入、`tool`/`function` 一类消息角色会返回 `422`，而不是静默降级。
- 原 Google AI Mode 浏览器抓取（`browser.py`、`pool.py`、`proxy_recovery.py`、`scripts/probe_google_ai.py`）保留但休眠，不再参与请求路由。

## 友情链接

- [Linux.do](https://linux.do)

## 常用命令

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run python scripts/probe_google_ai.py --prompt "What changed in OpenAI Responses API?"
# 发布：.env 改 IMAGE sha → docker compose pull && docker compose up -d --force-recreate（见 docs/dev-release.md）
```

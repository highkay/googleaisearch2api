# Google AI Search2API

`googleaisearch2api` 把 Google AI 搜索页面上的真实浏览器交互包装成一个 OpenAI 兼容 API，并提供一个本地 Web Console 用来查看配置、请求统计和实时探针结果。

## 运行时事实

- 当前已验证稳定入口是 `https://www.google.com/search?udm=50&aep=11...`，不是 `https://google.com/ai`。
- 运行时浏览器策略已经收敛为单一 `patchright + chrome`。
- Docker 使用 Playwright 官方镜像内置 Chromium，并把它映射到 Patchright 的 `chrome` channel 兼容路径。
- 单进程并发通过常驻 browser worker 池实现；每个 worker 独占自己的 browser/context。
- 纯 `httpx` 直接请求同一 AI 搜索 URL 只能拿到 `enablejs` 壳页，不能把 Google AI 当成稳定公开 HTTP API。

## 功能

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- Bearer Token 认证
- 可配置模型名、代理、语言、超时、worker 数和队列长度
- SQLite 持久化配置与请求日志
- 本地 Web Console: `/console`

## 推荐启动方式

默认推荐直接用 Docker Compose：

```bash
cp .env.example .env
docker compose up --build -d
```

默认宿主端口是 `9010`。启动后访问：

- API: `http://127.0.0.1:9010`
- Console: `http://127.0.0.1:9010/console`

默认 Compose 不再挂载源码目录，也不会在容器启动时重新执行 `uv sync`。这样可以直接复用镜像里已经构建好的运行环境，避免宿主机仓库里的 `.python-version=3.13` 触发容器冷启动下载 Python，导致服务长时间不可用。

## 启动后验证

```bash
curl http://127.0.0.1:9010/healthz
```

```bash
curl http://127.0.0.1:9010/v1/models \
  -H "Authorization: Bearer change-me-google-search"
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
- `DEFAULT_MODEL`: 对外暴露的模型名
- `BROWSER_HEADLESS`: 是否无头运行
- `BROWSER_USER_AGENT`: 可选，覆盖浏览器级 UA；留空时服务会给 headless Chrome 使用普通 Chrome UA
- `BROWSER_WORKERS`: 常驻浏览器 worker 数
- `REQUEST_QUEUE_SIZE`: 内存等待队列容量；满了以后返回 `429`
- `BROWSER_PROXY_SERVER`: 代理地址，例如 `http://127.0.0.1:7890`

如果容器里需要走宿主机代理：

```env
BROWSER_PROXY_SERVER=http://host.docker.internal:7890
```

## API 示例

列出模型：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer change-me-google-search"
```

Chat Completions：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-google-search" \
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
  -H "Authorization: Bearer change-me-google-search" \
  -d '{
    "model": "google-search",
    "input": "Summarize the latest differences between Responses API and Chat Completions API in 3 points."
  }'
```

## 已知边界

- 这是浏览器自动化方案，不是 Google 官方公开 API。
- Google 页面结构可能变化；如果选择器或页面行为失效，先运行 `scripts/probe_google_ai.py` 重新取证，再改提取逻辑。
- 当前 streaming 是在拿到完整答案后按 OpenAI SSE 形状回放，不是 Google 原生流式协议透传。
- 代理支持已经接入浏览器启动参数，但代理能否连通取决于你自己的代理服务。

## 常用命令

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run python scripts/probe_google_ai.py --prompt "What changed in OpenAI Responses API?"
```

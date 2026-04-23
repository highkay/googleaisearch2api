# Google AI Search2API

`googleaisearch2api` 把 Google AI 搜索页面中的真实查询交互包装成一个 OpenAI API 兼容服务，并提供本地 Web Console 做配置、统计和探针验证。

## 当前结论

这是基于 2026-04-23 的真实探针结论设计的：

- 运行时浏览器策略已经收敛为单一 Chromium 内核路径。
- 本地优先使用 `patchright + chrome`；Docker 优先使用 Playwright 官方镜像内置 Chromium，并映射到 `chrome` channel 兼容路径。
- `https://google.com/ai` 这个入口在当前 `patchright` 路径下会超时，不适合作为运行时入口。
- 纯 `httpx` 直接请求同一 AI 搜索 URL，只会拿到 `enablejs` 壳页，没有答案正文，所以当前不能把 Google AI Mode 当成稳定的公开 HTTP API。
- 单进程并发通过常驻浏览器 worker 池实现：每个 worker 独占一个浏览器上下文，请求通过内存队列分发，队列满时返回 `429`。

因此本项目的最终结构是：

- 外部：标准 HTTP / OpenAI 兼容 API
- 内部：`patchright` 驱动的浏览器执行器

## 功能

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`
- Bearer Token 认证
- 可配置模型名，默认 `google-search`
- 可配置代理、语言、超时
- 可配置浏览器 worker 数和队列长度
- SQLite 持久化配置与请求日志
- 本地 Web Console
  - 统计面板
  - 配置编辑
  - 最近请求
  - 实时探针按钮

## 运行

1. 安装依赖

```bash
uv sync --extra dev
```

2. 安装 Patchright 管理的 Chrome stable

```bash
uv run patchright install chrome
```

3. 复制环境文件

```bash
cp .env.example .env
```

4. 启动服务

```bash
uv run googleaisearch2api
```

默认地址：

- API: `http://127.0.0.1:8000`
- Console: `http://127.0.0.1:8000/console`

## 配置说明

关键配置：

- `API_TOKEN`: OpenAI 兼容接口的 Bearer Token
- `DEFAULT_MODEL`: 对外暴露的模型名
- `BROWSER_HEADLESS`: 是否无头运行
- `BROWSER_USER_AGENT`: 可选，覆盖浏览器级 Chrome User-Agent；留空时，服务会为 headless Chrome 使用普通 Chrome UA，避免 `HeadlessChrome` 指纹
- `BROWSER_WORKERS`: 常驻浏览器 worker 数；每个 worker 一次只处理一个请求
- `REQUEST_QUEUE_SIZE`: 内存等待队列容量；队列满时 API 返回 `429`
- `BROWSER_PROXY_SERVER`: 代理地址，例如 `http://127.0.0.1:7890`

## API 示例

### 列出模型

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer change-me-google-search"
```

### Chat Completions

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

### Responses API

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
- Google 页面结构可能变化；若选择器失效，需要重新跑 `scripts/probe_google_ai.py` 验证并更新提取逻辑。
- 当前 streaming 是在拿到完整答案后按 OpenAI SSE 形状回放，不是 Google 原生流式网络协议透传。
- 代理支持已经接入浏览器启动参数，但是否能连通取决于你配置的代理本身。
- 在 Windows 系统代理场景下，Chrome 就算没有显式传 `proxy` 参数，也可能继承系统代理。
- 当前验证结果显示：`msedge + headless` 更容易被 Google 立即风控，而 `chrome + headless` 是当前更稳的默认策略，因此仓库已不再保留多浏览器分支。

## Docker

仓库已提供基于 Playwright 官方镜像的 Docker 构建路径。镜像使用
`mcr.microsoft.com/playwright:v1.58.2-noble` 内置 Chromium，不再在构建时从
`dl.google.com` 下载 Google Chrome。

构建时会把内置 Chromium 映射到 Patchright `channel="chrome"` 期望的路径，因此应用代码仍保持单一浏览器策略。

构建：

```bash
docker build -t googleaisearch2api .
```

运行：

```bash
docker run --rm -p 9010:8000 \
  -e APP_HOST=0.0.0.0 \
  -e API_TOKEN=change-me-google-search \
  -e BROWSER_WORKERS=2 \
  -e REQUEST_QUEUE_SIZE=8 \
  googleaisearch2api
```

开发态 Compose 启动：

```bash
docker compose up --build
```

Compose 会自动构建镜像、挂载当前代码目录到 `/app`、保留容器内 `.venv`，并在容器启动时执行
`uv sync --frozen --no-dev`，确保挂载进去的最新代码会重新同步到运行环境。SQLite 数据挂载到命名卷
`googleaisearch2api-data`。默认宿主端口是 `9010`，可以通过 `.env` 里的 `APP_PORT` 覆盖。

如果容器需要走宿主机代理：

```env
BROWSER_PROXY_SERVER=http://host.docker.internal:7890
```

## 常用命令

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run python scripts/probe_google_ai.py --prompt "What changed in OpenAI Responses API?"
```

## 友链

- [linux.do](https://linux.do/)

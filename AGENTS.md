# Repo Notes

## Runtime Truth

- 主搜索引擎是进程内 Gemini web HTTP 引擎（`gemini_web.py`，102 槽 StreamGenerate payload，`gemini_web_models.py` 提供模型映射）。
- 请求提示词会注入强制搜索指令（`prompting.adapt_prompt_for_gemini_web`），算术表达式自动跳过；Gemini 因此返回 grounding 引用，避免"短答案无引用"被质量检查拒绝。
- 代理选择是两层融合池（`app._select_gemini_session`）：
  1. **Tier-1 WARP 出口**（`gemini_proxy_pool.GeminiWarpPool`，`GEMINI_WARP_PROXIES` 逗号分隔的 `socks5h://` URL；必须远端 DNS，`socks5://` 会 SOCKS 握手失败；容器需接入 `warp-plus_default` 外部网络）——内存轮换 + 失败冷却 + 线程锁。
  2. **Tier-2 2260 sticky 会话**（`proxy_sessions` 冷池探测轮换）——容量层。
  3. 最后 base 单次兜底（不再反复重试同一 IP）。
- 每层选用前都先做快速探测（`fast_proxy_probe.probe_gemini_http_fast`：真实 StreamGenerate POST "ping"，非首页 GET——2260 抽风 IP 首页能通但 POST 挂起）。
- Gemini web 的 BL 版本号缓存在模块级（`gemini_web._resolve_bl`），失败回退 `DEFAULT_BL_FALLBACK`。
- 原 Google AI Mode 浏览器抓取（`browser.py`、`pool.py`、`proxy_recovery.py`、`probe_google_ai.py`）保留休眠，`SEARCH_ENGINE` 已无 `google` 选项（`gemini | duck | gemini-upstream | auto`）；Duck.ai 兜底仍走 `patchright + chrome`。
- 2026-04-23 浏览器时代验证结果（现仅适用于 Duck.ai 引擎）：`patchright + chrome` 可打开 AI 搜索页；`https://google.com/ai` 超时；纯 `httpx` 只返回 `enablejs` 壳页。

## Working Rules

- 改 Gemini web 提取逻辑（payload 槽位/解析）前，先用真实请求重新取证；槽位布局以 `gemini_web.py` 与上游 gemini-web2api（只读参考）为准，同步更新相关测试。
- 改并发模型时保留“每个 worker 独占 runner/browser/context”的线程边界（Duck/browser 路径）。
- 优先保持实现线性、少状态、少隐式缓存（例外：`gemini_web._cached_bl` 与 `GeminiWarpPool` 内存态，都有明确失效/冷却策略）。
- 对外兼容层可以演进，但不要把未验证的 Google 内部 HTTP 端点当成稳定协议写死。
- sticky **Hot 池**只含 `status=active`；cooldown 到期不会自动再进线上选择（冷池候选走 `list_gemini_candidates` 显式探测）。
- `auto` 链为 `gemini（进程内，融合池）→ duck`；recovery 与 browser gate 只影响 Duck 兜底路径。

## Release

标准闭环见 `docs/dev-release.md`：

```text
pytest → commit/push main → 等 Actions 推 GHCR → .env 改 GOOGLEAISEARCH2API_IMAGE=<sha>（sparkcr 镜源）→ docker compose pull && up -d --force-recreate
```

不要默认提交 `.env`、`.env.backup*`、`.deploy-backups/`。

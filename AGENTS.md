# Repo Notes

## Runtime Truth

- 在本仓库里，Google AI 的已验证稳定入口不是 `https://google.com/ai`，而是 `https://www.google.com/search?udm=50&aep=11...`。
- 2026-04-23 本地验证结果：
  - 运行时浏览器策略已经收敛为单一 `patchright + chrome`。
  - Docker 路径使用 Playwright 官方镜像内置 Chromium，并映射到 Patchright `chrome` channel 兼容路径。
  - 单实例并发应通过常驻 browser worker 池实现；不要在多个请求线程中共享同一个 `GoogleAiRunner`。
  - `patchright + chrome` 可以打开 AI 搜索页并真实提交 textarea 查询。
  - `https://google.com/ai` 在 `patchright` 下会超时。
  - 纯 `httpx` 请求相同 AI 搜索 URL 只返回 `enablejs` 壳页，不能直接拿到答案正文。

## Working Rules

- 修改浏览器选择器前，先运行 `scripts/probe_google_ai.py` 重新取证。
- 如果要改 Google AI 提取逻辑，必须同步更新相关测试。
- 如果要改并发模型，必须保留“每个 worker 独占 runner/browser/context”的线程边界。
- 优先保持实现线性、少状态、少隐式缓存。
- 对外兼容层可以演进，但不要把未验证的 Google 内部 HTTP 端点当成稳定协议写死。

# Proxy Session Allowlist 落地方案

状态：2026-05-26 调查完成，进入实现拆分前方案。
目标：把动态粘性代理从“直接使用基础代理用户名随机出口”改成“按当前代理用户名派生 `<base>.userN` 会话、出口发现、Google canary 晋级、运行时反馈淘汰”的闭环，降低 Google unusual traffic 阻断率。

## 1. 已验证事实

### 1.1 当前真实实现

- Google AI 请求使用 `patchright + chrome`，不走 `httpx` 或 `curl_cffi`。
- `src/googleaisearch2api/app.py` 的 `_run_google_ai()` 从 `ConfigStore.get_config()` 取配置，调用 `BrowserPool.execute(config, prompt)`。
- `src/googleaisearch2api/pool.py` 会对 `ServiceConfig` 做 deep copy，然后交给 worker 内部的 `GoogleAiRunner.run_prompt()`。
- `src/googleaisearch2api/browser.py` 的 `_build_session_signature()` 包含 `browser_proxy_username`。因此只要把本次请求的用户名从 `<base>` 改成 `<base>.userN`，worker 会自动重建独立 browser context。
- `RequestLogRow` 当前只记录 `proxy_enabled`，没有记录会话名、出口 IP、风险分或 Google block 中的 IP。

### 1.2 线上异常判断

线上错误：

```text
Google blocked the session while opening query page: our systems have detected unusual traffic from your computer network. please try your request again later. why did this happen? ip address: 66.187.6.127 ≠ 2a09:bac5:624d:2da5::48c:59 time: 2026-05-26t04:15:43z
```

判断：

- 这不是回答提取逻辑回退到 “You said” 的问题。
- 这不是 `httpx`/`curl_cffi` 请求 Google 导致的问题；Google 请求仍是 Patchright 浏览器。
- `≠` 更像同一次 Google 会话相关链路中出现两个不同出口地址，可能来自裸 `US` 动态出口、IPv4/IPv6 双栈出口、或代理服务内部链路漂移。
- `.userN` 能稳定固定一段时间出口，但稳定出口仍可能被 Google 拒绝，所以必须加入 Google canary 作为最终晋级门禁。

### 1.3 本地代理探针

以 `US` 为样例，已验证 `.userN` 比直接使用基础用户名稳定：

| 代理用户名 | 出口 | ASN/组织 | 结果 |
| --- | --- | --- | --- |
| `US.user1` | `216.73.156.47` | AS23470 ReliableSite.Net LLC | 两轮稳定 |
| `US.user2` | `23.254.194.215` | AS36352 HostPapa | 两轮稳定 |
| `US.user3` | `66.187.6.127` | AS399804 Hostodo | 两轮稳定，且在线上 Google block 日志中出现 |
| `US` | 多个 IPv4/IPv6 | 多个出口 | 5 轮出现 12 个唯一 IP，不适合作为 allowlist 基础 |

Google canary 结果：

- `US.user1` 被 Google unusual traffic 阻断，Google 报 `216.73.156.47`。
- `US.user2` 被 Google unusual traffic 阻断，Google 报 `23.254.194.215`。

结论：IP 稳定和风险分高不等于 Google 可用，晋级必须依赖真实 Google canary。

### 1.4 IPLark 可用性

已验证 `https://iplark.com/{ip}` 可以无代理查询任意 IP：

- `https://iplark.com/216.73.156.47` 页面显示 `IP评分 76/100`、`使用场景 数据中心`、`代理 否`。
- 页面会请求：
  - `/ipscore?ip=216.73.156.47&token=...`，返回 `{"ip":"216.73.156.47","quality_score":76}`。
  - `/ipintelligence?ip=216.73.156.47&token=...`，返回 ASN、国家、使用类型、代理、威胁、标签等 JSON。

已验证同款 Patchright headless 也可用：

- 使用项目默认 Chrome channel、headless、默认 UA 访问 `https://iplark.com/216.73.156.47`。
- 主文档状态是 `412`，但页面挑战完成后仍能加载正文和网络 JSON。
- `/ipscore` 返回 200，`quality_score=76`。
- `/ipintelligence` 返回 200。

已验证不应依赖单次 `curl_cffi`：

- `curl_cffi` impersonate Chrome 访问 IPLark 页面只能拿到挑战/跳转 HTML，不能稳定拿到 `quality_score`。
- `/ipscore` 需要页面生成的 token；单次 HTTP 请求不执行页面 JS，不适合作为稳定方案。

## 2. 方案原则

1. Console 增加 `Resin 粘性会话` 开关；不勾选时完全保持一般代理路径，勾选后才启用 `.userN` 会话清单。
2. 不再把基础代理用户名直接投入 Google 请求；启用 Resin 粘性会话后只使用按当前 base prefix 派生出的 `<base>.userN`。
3. 出口发现、风险元数据、Google canary 都是非热路径；线上用户请求只做一次本地 DB 会话选择。
4. 第三方评分和风险标签只作为观测元数据，不作为准入门槛；Google canary 和真实请求成功才是晋级 active 的硬门槛。
5. Google 一旦返回 unusual traffic，立即 cooldown 当前 session 和其出口 IP 向量，不等待累计失败。
6. 重复出口不重复投入；同一个出口向量只保留一个表现最好的 session。
7. 不记录代理密码，不把 prompt 发给 IP 风险服务。

## 3. 总体架构

新增 6 个组件：

| 组件 | 职责 | 热路径 |
| --- | --- | --- |
| `ProxySessionStore` | 保存 session、IP 观测、评分、事件、统计 | 是，读 active 会话 |
| `EgressProbe` | 通过候选 `.userN` 访问 IP echo 服务，得到出口向量 | 否 |
| `IplarkProbe` | 无代理 headless 浏览器打开 `iplark.com/{ip}`，监听 JSON 响应 | 否 |
| `GoogleCanaryProbe` | 使用同一候选 `.userN` 跑真实 Google AI nonce prompt | 否 |
| `ProxySessionSelector` | 从 active 会话中选一个，改写本次 `ServiceConfig.browser_proxy_username` | 是 |
| `ProxySessionFeedback` | 请求成功/阻断/超时后更新统计、cooldown 或 retire | 是 |

状态机：

```text
new
  -> egress_checked
  -> risk_checked
  -> canary_passed -> active
  -> cooldown
  -> retired
```

任意阶段可进入 `retired`：

- 无法稳定得到出口向量。
- Google canary unusual traffic。
- 同一 session 连续多轮出口漂移。

## 4. 数据模型

### 4.1 `proxy_sessions`

```sql
CREATE TABLE proxy_sessions (
  id INTEGER PRIMARY KEY,
  proxy_base_username TEXT NOT NULL,      -- US / JP / openai / default
  session_name TEXT NOT NULL,             -- user1
  proxy_username TEXT NOT NULL UNIQUE,    -- <base>.user1
  status TEXT NOT NULL,                   -- new/egress_checked/risk_checked/active/cooldown/retired
  epoch INTEGER NOT NULL DEFAULT 1,

  primary_ip TEXT,
  ip_vector_json TEXT NOT NULL DEFAULT '[]',
  ip_vector_hash TEXT,
  asn INTEGER,
  organization TEXT,

  iplark_min_quality_score INTEGER,
  iplark_usage_type TEXT,
  iplark_category TEXT,
  iplark_public_proxy INTEGER,
  iplark_threat TEXT,
  iplark_tag TEXT,

  google_canary_status TEXT NOT NULL DEFAULT 'unknown',
  google_canary_error TEXT,
  google_canary_checked_at DATETIME,

  request_success_count INTEGER NOT NULL DEFAULT 0,
  request_block_count INTEGER NOT NULL DEFAULT 0,
  request_error_count INTEGER NOT NULL DEFAULT 0,
  canary_success_count INTEGER NOT NULL DEFAULT 0,
  canary_block_count INTEGER NOT NULL DEFAULT 0,
  duplicate_of_session_id INTEGER,

  last_checked_at DATETIME,
  last_selected_at DATETIME,
  last_success_at DATETIME,
  last_blocked_at DATETIME,
  cooldown_until DATETIME,
  retired_at DATETIME,
  retire_reason TEXT,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL
);
```

索引：

```sql
CREATE INDEX ix_proxy_sessions_status ON proxy_sessions(status);
CREATE INDEX ix_proxy_sessions_ip_vector_hash ON proxy_sessions(ip_vector_hash);
CREATE INDEX ix_proxy_sessions_cooldown_until ON proxy_sessions(cooldown_until);
```

### 4.2 `proxy_ip_observations`

```sql
CREATE TABLE proxy_ip_observations (
  id INTEGER PRIMARY KEY,
  proxy_session_id INTEGER NOT NULL,
  epoch INTEGER NOT NULL,
  source TEXT NOT NULL,                   -- ipify4/ipify64/ipinfo/google_block/iplark
  ip TEXT NOT NULL,
  raw_json TEXT,
  observed_at DATETIME NOT NULL
);
```

用途：

- 回放一个 session 的出口历史。
- 判断同一 `.userN` 是否发生漂移。
- 把 Google block 中的 `A ≠ B` 两个 IP 也加入观测。

### 4.3 `proxy_session_events`

```sql
CREATE TABLE proxy_session_events (
  id INTEGER PRIMARY KEY,
  proxy_session_id INTEGER,
  event_type TEXT NOT NULL,               -- discovered/risk_checked/canary_ok/canary_block/request_ok/request_block/cooldown/retired
  message TEXT NOT NULL,
  raw_json TEXT,
  created_at DATETIME NOT NULL
);
```

### 4.4 扩展 `request_logs`

用当前项目的 `_ensure_column()` 模式做增量列：

```sql
ALTER TABLE request_logs ADD COLUMN proxy_session_id INTEGER;
ALTER TABLE request_logs ADD COLUMN proxy_base_username TEXT;
ALTER TABLE request_logs ADD COLUMN proxy_username TEXT;
ALTER TABLE request_logs ADD COLUMN proxy_primary_ip TEXT;
ALTER TABLE request_logs ADD COLUMN proxy_ip_vector_hash TEXT;
ALTER TABLE request_logs ADD COLUMN proxy_iplark_score INTEGER;
ALTER TABLE request_logs ADD COLUMN resin_sticky_session_enabled INTEGER;
ALTER TABLE request_logs ADD COLUMN google_block_ips_json TEXT;
ALTER TABLE request_logs ADD COLUMN google_block_mismatch INTEGER;
```

这样 console 最近请求表可以直接看出“哪一个 `.userN`、哪个出口、哪个分数、是否 Google block”。

### 4.5 扩展 `service_config`

Console 需要新增一个用户可见配置：

```sql
ALTER TABLE service_config ADD COLUMN resin_sticky_session_enabled INTEGER DEFAULT 0;
```

同步扩展：

- `ServiceConfig.resin_sticky_session_enabled: bool = False`
- `ServiceConfigUpdate.resin_sticky_session_enabled: bool = False`
- `AppSettings.resin_sticky_session_enabled`，环境变量 `RESIN_STICKY_SESSION_ENABLED=false`
- `ConfigStore._row_to_config()` 和 `ConfigStore.update_config()`
- `src/googleaisearch2api/templates/console.html` 设置表单

语义：

- 未勾选：`browser_proxy_server`、`browser_proxy_username`、`browser_proxy_password` 按当前方式直接传给 Patchright，等价于一般代理。
- 已勾选：当前代理用户名被视为 Resin base prefix，可以是 `US`、`JP`、`openai` 或任意服务商认可的用户名；运行时由 session selector 改写成 `<base>.user1`、`<base>.user2`。
- 已勾选但没有 active session：默认返回 503，不回退基础用户名，除非显式打开 `PROXY_ALLOW_FALLBACK_TO_BASE=true`。

## 5. 出口发现逻辑

### 5.1 候选生成

配置：

```text
RESIN_STICKY_SESSION_ENABLED=false
RESIN_STICKY_SUFFIX_TEMPLATE=.user{n}
PROXY_SESSION_START=1
PROXY_SESSION_END=200
PROXY_SESSION_BATCH_SIZE=5
PROXY_MIN_ACTIVE_SESSIONS=3
```

候选用户名：

```text
<base>.user1
<base>.user2
...
<base>.user200
```

`<base>` 来自当前 `ServiceConfig` 的代理用户名：

- 优先使用 `browser_proxy_username`。
- 如果代理 URL 内嵌账号且 `browser_proxy_username` 为空，可以解析 URL 内嵌 username 作为 base；实现时仍应拆分 server/auth，不把改写后的用户名重新拼进 URL 字符串。
- 如果开启 Resin 粘性会话但无法得到 base username，配置校验应失败并返回清晰错误，例如 `Resin sticky session requires a proxy username prefix`。

候选只保存 username 和 base username，不保存 password。真实代理 server/password 仍来自当前 `ServiceConfig`。

派生规则：

```text
base=US      -> US.user1, US.user2, ...
base=JP      -> JP.user1, JP.user2, ...
base=openai  -> openai.user1, openai.user2, ...
```

Console 字段映射：

- `代理服务器`：保持当前字段，例如 `http://192.168.1.18:2260`。
- `代理用户名`：未勾选 Resin 粘性会话时按一般用户名直接使用；勾选后作为 base prefix 使用，不预设国家或厂商，当前填 `US` 就派生 `US.userN`，填 `JP` 就派生 `JP.userN`，填 `openai` 就派生 `openai.userN`。
- `代理密码`：保持当前字段，不进入 `proxy_sessions` 表。
- `Resin 粘性会话`：新增 checkbox，控制是否启用 `.userN` 发现、评分、canary 和选择。

### 5.2 出口向量

每个候选 session 运行两轮，使用同一代理用户名访问：

- `https://api.ipify.org?format=json`
- `https://api64.ipify.org?format=json`
- `https://ipinfo.io/json`

输出：

```json
{
  "primary_ip": "216.73.156.47",
  "ip_vector": ["216.73.156.47"],
  "asn": 23470,
  "organization": "ReliableSite.Net LLC"
}
```

规则：

- 两轮所有服务都得到同一个 IP：稳定。
- 得到一个 IPv4 和一个 IPv6，但两轮完全一致：稳定双栈向量。
- 同一服务两轮变化，或向量数量超过 2：不稳定，进入 cooldown 或 retired。
- `ip_vector_hash = sha256(sorted(ip_vector).join(","))`。

为什么用向量而不是单 IP：

- Google block 可能出现 `IPv4 ≠ IPv6`。
- 如果只记录 primary IP，会漏掉同一 session 的另一个出口地址。
- 评分和重复判断都必须覆盖整个向量，最终取最差分。

### 5.3 重复出口处理

如果新候选的 `ip_vector_hash` 已存在：

1. 已存在 active：新候选标记 `retired`，`duplicate_of_session_id=existing.id`。
2. 已存在 cooldown：新候选也进入 cooldown，直到同一 `cooldown_until`。
3. 已存在 retired：如果 retired 原因是 Google block，新候选 retired；如果原因是旧 session 漂移，可重新 canary。

如果只是向量有交集，例如 `["1.1.1.1"]` 和 `["1.1.1.1","2606:..."]`：

- 先视为重复风险。
- 只有当 active 会话不足，并且新向量 Google canary 通过，才允许作为独立 session。

## 6. IPLark 风险评分

### 6.1 查询方式

不要直接请求 `/ipscore`，先用无代理 Patchright headless 打开：

```text
https://iplark.com/{ip}
```

监听网络响应：

- URL 包含 `/ipscore?ip={ip}`：解析 `quality_score`。
- URL 包含 `/ipintelligence?ip={ip}`：解析 `intelligence`、`history`。

主文档返回 `412` 不视为失败；只要两个 JSON 响应成功即可。

### 6.2 缓存

按 IP 缓存：

```text
IPLARK_CACHE_TTL_HOURS=168
IPLARK_BAD_CACHE_TTL_HOURS=24
```

同一个 IP 被多个 `.userN` 发现时，不重复查 IPLark。

### 6.3 风险元数据规则

规则：

- `quality_score`、`publicProxy`、`threat`、`category/usageType` 只写入 DB 和 console，用于后续分析。
- 第三方评分缺失、低分、查询失败、风险标签异常，都不阻止候选进入 Google canary。
- 能提升有效 IP 命中率的硬规则只保留出口稳定性、重复出口去重、Google canary 和真实请求反馈。
- 多 IP 向量只用于稳定性和重复判断，不用第三方分数做准入排序。

注意：本地已验证 `216.73.156.47` 的 IPLark 分数是 76，但 Google canary 仍被 block；线上也出现高分 IP 被 Google block 的情况。所以第三方分数只能作为观测，不能作为 active 判定。

## 7. Google Canary 晋级

### 7.1 canary prompt

每次 canary 使用唯一 nonce，避免把 echo 当成功：

```text
Return exactly this token and nothing else: gai-canary-20260526-{uuid}
```

成功条件：

- 没有触发 `GoogleAiBlockedError`。
- 返回正文包含 nonce 或能明确回答。
- `final_url` 不是 `/sorry/`。
- `body_excerpt` 不含 unusual traffic/captcha/not a robot。

失败条件：

- `GoogleAiBlockedError`：立即 `canary_block_count += 1`，session cooldown。
- timeout/runtime error：不立即 retired，先 cooldown 短周期后重试。

### 7.2 block IP 解析

从错误文案中解析：

```text
ip address: 66.187.6.127 ≠ 2a09:bac5:624d:2da5::48c:59 time: ...
```

得到：

```json
{
  "ips": ["66.187.6.127", "2a09:bac5:624d:2da5::48c:59"],
  "mismatch": true,
  "time": "2026-05-26T04:15:43Z"
}
```

处理：

- 写入 `proxy_ip_observations(source='google_block')`。
- 如果 block IP 不在当前 `ip_vector`，把 session 标记为 `egress_drift_block`。
- 当前 session cooldown。
- 同一 `ip_vector_hash` 的其他 session 同步 cooldown。

### 7.3 晋级

只有满足以下条件才进入 `active`：

- 出口向量稳定。
- Google canary 通过。
- 不是重复出口，或重复出口是被明确替换的旧 session。

active 记录：

```text
status=active
google_canary_status=passed
canary_success_count += 1
last_success_at=now
```

## 8. 运行时接入

### 8.1 选择时机

在 `app._run_google_ai()` 中：

1. 从 store 读取基础 `ServiceConfig`。
2. 如果 `resin_sticky_session_enabled=False`，直接使用基础 config，保持一般代理行为。
3. 如果 `resin_sticky_session_enabled=True`，调用 `ProxySessionSelector.select(config)` 选择 active session。
4. 用 `config.model_copy(deep=True, update={"browser_proxy_username": selected.proxy_username})` 得到本次请求配置。
5. `start_request()` 使用“已改写后的 config”记录日志，并记录 `resin_sticky_session_enabled`。
6. 调用 `BrowserPool.execute(selected_config, prompt)`。
7. 只有 Resin 粘性会话开启时，根据成功/异常调用 `ProxySessionFeedback`。

这样不需要改 Google runner 的核心浏览器逻辑。

### 8.2 无 active 会话时的策略

配置：

```text
PROXY_ALLOW_FALLBACK_TO_BASE=false
```

默认策略：

- 没有 active session：返回 503，提示 proxy session pool has no active sessions。
- 不自动退回基础用户名，因为直接使用基础用户名已验证可能产生多出口漂移。

临时救急可把 `PROXY_ALLOW_FALLBACK_TO_BASE=true`，此时使用当前 console 中保存的 proxy username。

### 8.3 选择算法

小流量 v1：

```text
score =
  iplark_min_quality_score
  + success_rate * 20
  - recent_block_penalty
  - datacenter_penalty
```

过滤：

- `status != active` 排除。
- `cooldown_until > now` 排除。
- `last_selected_at` 太近且还有其他 active 时降权。

选择：

- `worker_count == 1` 时，优先沿用上一个 active session，直到 `max_requests_per_session` 或失败。
- `worker_count > 1` 时，先用加权轮询；如果后续发现 context 频繁重建，再把选择下沉到 `BrowserPool._worker_main()` 做 worker affinity。

初始配置：

```text
PROXY_SESSION_MAX_REQUESTS=20
PROXY_SESSION_MIN_ROTATE_SECONDS=600
PROXY_SESSION_COOLDOWN_MINUTES=60
PROXY_SESSION_BLOCK_COOLDOWN_HOURS=12
```

### 8.4 成功反馈

一次 Google 请求成功：

- `request_success_count += 1`
- `last_success_at = now`
- `last_selected_at = now`
- 如果当前 session 是 `risk_checked` 但人工 probe 成功，也可晋级 active。

### 8.5 阻断反馈

捕获 `GoogleAiBlockedError`：

- 解析 block IP。
- `request_block_count += 1`
- `last_blocked_at = now`
- `status = cooldown`
- `cooldown_until = now + PROXY_SESSION_BLOCK_COOLDOWN_HOURS`
- 关闭或 reset pool，避免当前 worker 继续持有被 block 的 context。
- 下一个请求选择其他 active session。

捕获 timeout/runtime error：

- `request_error_count += 1`
- 不立即 retire。
- 连续 `N` 次同类错误后 cooldown。

## 9. 实现拆分

### 9.1 第一阶段：纯逻辑和数据

新增：

- `src/googleaisearch2api/proxy_sessions.py`
- `src/googleaisearch2api/iplark.py`
- `src/googleaisearch2api/egress.py`

测试：

- 解析 Google block IP，包括 `A ≠ B`、单 IP、无 IP。
- IP 向量 normalize/hash。
- 重复出口判断。
- IPLark `/ipscore` 和 `/ipintelligence` JSON 解析。
- selector 过滤 cooldown/retired/duplicate。

### 9.2 第二阶段：DB 和 store

新增 SQLAlchemy rows：

- `ProxySessionRow`
- `ProxyIpObservationRow`
- `ProxySessionEventRow`

扩展 `create_tables()`：

- `Base.metadata.create_all(engine)`
- `_ensure_request_log_proxy_columns(engine)`

新增 store 方法：

- `upsert_proxy_session()`
- `record_ip_observation()`
- `record_proxy_event()`
- `list_active_proxy_sessions()`
- `mark_proxy_session_cooldown()`
- `finish_request_success_with_proxy()`
- `finish_request_error_with_proxy()`

### 9.3 第三阶段：探针脚本

新增脚本：

```powershell
uv run python scripts/probe_proxy_sessions.py --base-username openai --start 1 --end 20 --egress --iplark
uv run python scripts/probe_proxy_sessions.py --base-username JP --start 1 --end 20 --google-canary
```

脚本行为：

1. 逐个生成 `.userN`。
2. egress 探测出口向量。
3. 查风险元数据；查询失败也继续。
4. 候选跑 Google canary。
5. 写入 SQLite。
6. 打印 active/cooldown/retired 汇总。

### 9.4 第四阶段：热路径接入

改 `Services`：

```python
@dataclass
class Services:
    settings: AppSettings
    store: ConfigStore
    pool: BrowserPool
    proxy_selector: ProxySessionSelector | None
```

改 `_run_google_ai()`：

- 先判断 `config.resin_sticky_session_enabled`；未启用时完全跳过 session selector。
- 启用时入队前选择 session。
- `start_request()` 记录 `proxy_session_id` 等字段。
- 启用时 success/error 都反馈给 session store。

### 9.5 第五阶段：console 可观测性

Console 增加：

- Settings 区域增加 `Resin 粘性会话` checkbox。
- checkbox 未勾选：下方 session 清单只读展示或隐藏，当前代理字段按一般代理解释。
- checkbox 已勾选：`代理用户名` 显示为 base prefix，旁边提示将派生 `<base>.userN`，不要在这里填已经派生后的 `openai.user1` 或 `JP.user1`。
- active/cooldown/retired 数量。
- active session 列表：username、IP、score、ASN、成功率、最近 block。
- 最近请求表增加 session、IP、score。
- 操作按钮：
  - scan next batch
  - run canary for candidates
  - cooldown selected
  - retire selected

## 10. 验证门禁

### 10.1 本地自动化

必须通过：

```powershell
uv run ruff check --fix
uv run pytest
```

新增测试至少覆盖：

- 未勾选 `resin_sticky_session_enabled` 时，`US`、`JP`、`openai` 或任意用户名都按一般代理直接使用，不触发 selector。
- 勾选 `resin_sticky_session_enabled` 后，base username 不会被直接选择，只会选择 active 的 `<base>.userN`。
- 候选生成测试必须覆盖至少 `US`、`JP`、`openai` 三个 base prefix，证明没有写死国家或厂商。
- 无 active 且 fallback disabled 返回 503。
- `GoogleAiBlockedError` 后 session cooldown，下一次选择不同 session。
- `RequestLogRow` 写入 session/IP/score/block IP。
- 重复出口只保留一个 active。

### 10.2 本地真实探针

必须跑：

```powershell
uv run python scripts/probe_proxy_sessions.py --base-username openai --start 1 --end 5
```

验收：

- `.userN` 能得到稳定出口向量。
- 风险元数据能被解析；若第三方查询失败，候选仍继续 canary。
- `216.73.156.47` 这类已知 IP 能得到 `quality_score=76` 附近的分数。

Google canary 只小批量跑：

```powershell
uv run python scripts/probe_proxy_sessions.py --base-username openai --start 1 --end 5 --google-canary
```

验收：

- 被 block 的 session 不进入 active。
- 错误里的 Google IP 被解析进 DB。
- 通过 canary 的 session 才进入 active。

### 10.3 容器验证

在 GHCR 镜像内验证：

```powershell
docker run --rm ghcr.io/... uv run python scripts/probe_proxy_sessions.py --base-username openai --start 1 --end 3 --egress --iplark
```

重点验证：

- Patchright headless 在容器里也能触发 IPLark `/ipscore` 和 `/ipintelligence`。
- 主文档 `412` 不被当成失败。
- Chrome channel 路径与当前 Google runner 一致。

### 10.4 线上发布验证

发布到 `admin@fnos` 后：

1. `RESIN_STICKY_SESSION_ENABLED=false` 启动，确认 DB migrations 成功。
2. 在容器内跑小批量 scan，得到至少 `PROXY_MIN_ACTIVE_SESSIONS` 个 active。
3. 在 console 勾选 `Resin 粘性会话`，或设置 `RESIN_STICKY_SESSION_ENABLED=true` 后重启。
4. 用 console probe 和 `/query` 各跑一次。
5. 持续观察 30 分钟：
   - 502 unusual traffic 是否下降。
   - request log 是否记录 session/IP/score。
   - block 后是否自动切换 session。

## 11. 回滚

最快回滚：

```text
RESIN_STICKY_SESSION_ENABLED=false
```

然后重启服务或调用 pool reset。

数据库表和 request_logs 增量列可以保留，不影响旧路径。

如果 manager enabled 但 active 清单为空：

- 默认返回 503，不回退基础用户名。
- 临时需要服务可用时，手动打开 `PROXY_ALLOW_FALLBACK_TO_BASE=true`，但这会回到已知不稳定路径。

## 12. 外部参考

- Google unusual traffic 说明：https://support.google.com/websearch/answer/86640
- IPinfo Privacy API：https://ipinfo.io/developers/privacy-standard-api
- IPQualityScore Proxy Detection API：https://www.ipqualityscore.com/documentation/proxy-detection-api/response-parameters
- AbuseIPDB API：https://docs.abuseipdb.com/

这些第三方服务只作为可选补充分数源。v1 以 IPLark 页面评分和 Google canary 为主。

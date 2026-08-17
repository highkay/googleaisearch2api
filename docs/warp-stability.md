# warp-plus 出口舰队稳定性手册

适用对象：外部 `warp-plus` 代理舰队（不在本仓库 compose 内，`docker-compose.warpplus-hardened.yml` 仅为加固参考）。这些容器为 googleaisearch2api 提供 Tier-1 WARP 出口（`GEMINI_WARP_PROXIES` 里的 `socks5h://warpplus-*:1080`）。本文档说明：为什么它们会死、怎么看诊断、怎么让它稳定。

现存舰队映射（端口为宿主机绑定，容器内 SOCKS 一律监听 1080）：

| 容器 | 服务 | 端口 |
|---|---|---|
| warpplus-de-warpplus | de | 1081 |
| warpplus-gb-warpplus | gb | 1082 |
| warpplus-ca-warpplus | ca | 1083 |
| warpplus-scan-dk-warpplus | scan-dk | 1091 |
| warpplus-scan-fi-warpplus | scan-fi | 1094 |
| warpplus-scan-no-warpplus | scan-no | 1102 |
| warpplus-scan-rs-warpplus | scan-rs | 1106 |

## 1. 为什么容器会死

### 1.1 启动期：fail-fast 的 os.Exit(1) 链

warp-plus 入口是「任何一步不通过就直接退出(1)」的启动链，重启后重来。任何一环失败都会造成崩溃循环：

- **identity 注册失败**：启动时要 Register/UpdateAccount；Cloudflare 拒绝（设备数超限、网络被拒、license 失效）→ 立即退出。
- **15 秒握手 gate**：与 WARP 端点握手有超时门槛（约 15s），超时未建立 → 退出。
- **5 秒连通性测试**：握手后做 ~5s 连通性验证，不通过 → 退出。
- **`--scan` 的 1 分钟 UDP 扫描风险**：`scan-*` 容器带 `--scan`，启动时跑约 1 分钟 UDP 端点扫描。从中国大陆直连（或路由被 QoS/封锁）时 UDP 大面积超时/丢包，扫描阶段最容易触发上面的握手/连通性超时而死亡。这是 scan 系列相对非 scan 系列易挂的一个独立因素。

### 1.2 启动后死亡

- **OOM（exit 137）**：每个连接维护 2×64KB 收发缓冲，多引擎 × 多连接叠加后内存快速膨胀；`mem_limit` 过低或宿主机内存紧张时被内核 OOM-killer 杀掉（`docker inspect` 里 `OOMKilled: true`）。这是最常见的「跑着跑着没了」。
- **psiphon 静默死亡 → 僵尸容器**：隧道引擎内部崩了但容器进程未退出，SOCKS 端口仍 accept，握手也能过，但隧道转发已失效。此时从主 API 发请求会表现为 **504**（代理看起来通，实际出不去）。这是最隐蔽的一类：`docker ps` 显示 Up、`docker inspect` 健康状态却可能 already-consuming 或 dockerfile 无 healthcheck 时根本看不出。**必须靠「真实隧道探测」型 healthcheck 兜底**。
- **共享 WARP+ license 设备数上限（~5）**：一个 WARP+ 订阅 license 最多同时约 5 台设备。7+ 台容器共享同一 license 时，启动阶段 Register/UpdateAccount 会因「设备已满」被拒 → fail-fast 退出。新增容器前先确认 license 设备余量。

### 1.3 死亡链在日志里的样子

典型的 fail-fast 崩溃循环（`docker logs --tail` 逐环对照）：

```text
# identity 注册/更新被拒 → 立即退出(1)
[ERROR] register failed / UpdateAccount error   →  exit 1
# 15s 握手 gate 超时
[ERROR] handshake timeout after 15s             →  exit 1
# 5s 连通性测试失败
[ERROR] connectivity check failed                 →  exit 1
# 僵尸容器：日志在正常路由日志后戛然而止，进程活着、SOCKS 还 accept
[INFO]  ... (最后一行一切正常，之后没有任何输出，容器却也 不 退出)
```

单看日志能区分两类：**崩溃型**（exit 1 + ERROR 尾部）和**僵尸型**（日志戛然而止 + 进程仍在 + `docker ps` 显示 Up）。

### 1.4 僵尸容器为什么表现为 504

warp-plus 内部是多引擎（含 psiphon、wireguard 等）混合隧道。psiphon 引擎静默崩溃后，SOCKS listener 与 WireGuard 出口可能仍在，甚至有半个引擎过节流/自愈逻辑：

1. 外部（主 API 的 Gemini warp 池 / curl）连上 1080，SOCKS 握手成功 —— 端口「活着」。
2. 数据发给远端隧道引擎，但引擎已经不再转发。
3. 请求超时 → 主 API 测到这个出口全部超时/挂起，被判定不可用。

这就是观测到的 **504** 来源：代理「看起来通、实际出不去」。只有用「真实隧道探测」（§3）才能把它和健康出口区分开。

## 2. 诊断表（症状 → docker inspect 线索 → 动作）

| 症状 | docker inspect 线索 | 动作 |
|---|---|---|
| 容器反复重启 | `Status.Restarting`、`ExitCode: 1`、日志有 `Register/UpdateAccount` 错误 | 换/续期 license；确认身份卷未损坏；确认 `--scan` 容器在受控网络 |
| 启动即死，无 OOM | 日志在握手/连通性测试阶段超时退出，`ExitCode: 1` | 换端点（重跑 scan 或指定 `--endpoints`）；检查网络丢包（尤其中国大陆直连） |
| 运行中内存持续上涨后退出 | `State.OOMKilled: true`、`ExitCode: 137`、`RestartCount` 增长 | 看 `docker inspect --format '{{.State.OOMKilled}}' <ctr>`；调 `mem_limit` 到 512m 并按需扩容；限制并行连接 |
| `docker ps` 显示 Up，但主 API 疯狂 504 | healthcheck 非 0（PORT accept 但 tunnel 死）；`docker logs` 尾部引擎崩溃不再输出 → **psiphon 僵尸** | 用 §3 真实隧道探测确认；unhealthy 阈值触发自动重启（restart: unless-stopped）；必要时手动 `docker restart` |
| 主 API 探测 WARP 全部失败、fallback 到 2260/base | 多个容器同时 unhealthy 或都是僵尸 | 先查 license 设备数上限是否被撞；再看共享宿主机是否 OOM 了一串容器 |

### 2.1 取证命令（一条看懂现状）

```bash
# 退出码 / 是否被 OOM 杀 / 重启次数
docker inspect --format '{{.State.ExitCode}} OOM={{.State.OOMKilled}} restarts={{.RestartCount}}' <ctr>

# 容器自带 healthcheck 时的健康状态
docker inspect --format '{{json .State.Health}}' <ctr>

# 最近的崩溃尾日志
docker logs --tail 50 --timestamps <ctr>

# 实时内存 / CPU（对照 §6 的 512m 护栏）
docker stats --no-stream <ctr>

# 健康状态随时间
watch -n 30 'docker ps --format "table {{.Names}}\t{{.Status}}"' 
```

任何猜测之前先把上面四行跑全：exit 1 vs 137 vs OOMKilled，配合日志尾部关键字（`register`/`handshake`/`connectivity`/`timeout`）基本能直接命中 §2 表格里的一行。

## 3. 健康检查配方（真实隧道探测）

不要只看端口在 listen —— 僵尸容器的端口照样 accept。要穿透 SOCKS 打到 Cloudflare 的连通性接口才算活：

```yaml
healthcheck:
  # socks5h:// 强制远端 DNS；目标接口返回 warp 状态（warp=on 表示隧道真的通）
  test: ["CMD", "curl", "-fsS",
         "-x", "socks5h://127.0.0.1:1080",
         "https://connectivity.cloudflareclient.com/cdn-cgi/trace"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 30s
```

更严格的可检查返回值里含 `warp=on`（需要镜像里有 `grep`）：

```yaml
test: ["CMD-SHELL",
       "curl -fsS -x socks5h://127.0.0.1:1080 https://connectivity.cloudflareclient.com/cdn-cgi/trace | grep -q 'warp=on'"]
```

说明：

- 容器内 `127.0.0.1:1080` 就是它自己的 SOCKS（`--bind 0.0.0.0:1080`）。
- healthcheck 返回非 0 时不会杀掉容器 —— **Docker 的 restart 策略只看进程是否退出，不响应 unhealthy**。`restart: unless-stopped` 只能自动拉起「崩溃退出的容器」；僵尸容器进程还活着，需要靠巡检脚本/外部调度在检测到 unhealthy 后显式 `docker restart`（参见 §5）。健康检查的价值是把「看不见的僵尸」变成 `docker ps` 里显眼的 `unhealthy`，给自愈工具一个明确的判定依据。
- 从 API 宿主侧做等效的一次性人工探测（等价于 §5 脚本干的活）：

```bash
curl -fsS -x socks5h://<warp-host>:1081 https://connectivity.cloudflareclient.com/cdn-cgi/trace
# 输出应包含 warp=on；超时/失败 = 该出口隧道不可用
```

- 镜像里必须带 `curl`（以及严格版需要 `grep`）；若镜像没有，改用镜像自带的 fetch 工具等价表达。

### 3.1 healthcheck 参数怎么调

- `interval: 30s`：对 7+ 容器 × 每次耗时 <1s 的探测成本可忽略；不要低于 10s，避免流量噪声。
- `timeout: 10s`：真实隧道探测是一次完整 TLS + HTTP 往返，5s 有抖动误报，10s 稳妥。
- `retries: 3` + `start_period: 30s`：启动期握手+连通性检查本身要几秒，`start_period` 期间的状态不计入失败，防止首次启动被误判。
- 连续 3 次失败后状态变 `unhealthy` —— 这就是自愈工具的触发信号。

## 4. 身份持久化（--cache-dir + 命名卷）

warp-plus 把 `x25519` 身份（private key，对应一个固定虚拟 IP）缓存在 cache 目录。一旦丢失，下次启动就要重新 Register/UpdateAccount → 既增加启动失败概率，又烧 license 设备名额。**身份必须落在命名卷里，而不是容器可写层**：

```yaml
command: ["/usr/bin/warp-plus", "--bind", "0.0.0.0:1080", "--cache-dir", "/warp-cache"]
volumes:
  - warp-ident-de:/warp-cache
```

`docker compose down`/`up` 甚至 `docker rm` 后身份还在卷里，新容器复用同一 identity，启动更稳、不重复占 license 设备。

注意：别把身份卷和「随机临时缓存」混用；`--scan` 的扫描结果如果也想保留，另放独立路径，别拿身份卷存扫描临时数据。

### 4.1 验证身份真的持久化了

```bash
# 同一台容器的两次重启之间，cache 目录应有固定文件（identity/key 等），内容不变
docker exec <ctr> ls -la /warp-cache

# 重建容器（卷还在）后对比 identity 是否复用：
docker compose down && docker compose up -d <svc>
docker exec <ctr> sha256sum /warp-cache/*   # 前后一致 = 卷挂载生效
```

如果每次重建 cache 目录都重新生成（新文件、启动日志再次出现 Register/UpdateAccount）→ 检查 `--cache-dir /warp-cache` 与卷挂载点是否真的对上（常见错误：卷挂到了别的路径，进程实际写的是容器可写层）。

## 5. 巡检脚本 scripts/warp_health.py

舰队侧的健康巡检脚本，对每个容器做宿主机级真实隧道探测（等价于 §3 的 curl，但在宿主机上以 `-x socks5h://127.0.0.1:<端口>` 打 Cloudflare 连通性接口），并汇总：

- 每个容器的 端口→curl 结果（区分「端口通但隧道死」的僵尸）；
- 需要 `restart` / unhealthy 时长超阈值的容器列表。

用法（在舰队宿主机上执行）：

```bash
python scripts/warp_health.py              # 扫默认端口清单
python scripts/warp_health.py --ports 1081,1082,1083
python scripts/warp_health.py --json       # 机器可读输出，供监控对接
```

> `scripts/warp_health.py` 属于外部舰队侧的工具，运行在 warp-plus 宿主上，不在本 API 仓库内。本仓库只提供 §3 的探针配方作为它接入同一判定标准。

自愈闭环建议（healthcheck 只负责「标出来」，拉起由下面任一承担）：

- **最快**：巡检脚本扫描到 `unhealthy` 即 `docker restart <ctr>`；僵尸容器 30s 内被拉回。
- **更自动**：宿主跑 `willfarrell/autoheal` 之类的 sidecar，对 `unhealthy*` 容器自动重启（它监听 docker events，基于 healthcheck 状态）。
- **兜底**：崩溃型（exit 1）本来就会被 `restart: unless-stopped` 拉起，僵尸型才需要上面两层。

## 6. 内存与 license 护栏（加固要点）

- **mem_limit: 512m**：单容器护栏（覆盖 2×64KB/连接的缓冲），防一个容器 OOM 拖垮共享宿主、连锁放倒一批出口。仍持续 OOM 就先限并行连接数，而不是无限加内存。
- **restart: unless-stopped**：崩溃/僵尸自动拉起，但保留手动 `stop` 的能力（别用 `always`，避免想停停不下来）。
- **license 设备上限 ~5**：7+ 台共享同一 WARP+ license 一定有一批 Register 失败。要么按设备数切分 license，要么砍掉冗余容器；新增出口前先查余量。

## 7. 加固 overlay 参考

`docker-compose.warpplus-hardened.yml`（本仓库根目录）把以上 §3/§4/§6 固化成一份「illustrative - real fleet's compose lives outside this repo」的对照清单，映射现网 7 个容器（端口 1081–1106、名称 warpplus-*）。真实舰队 compose 在仓库之外，用它做 diff 基准逐项核对即可：

```bash
docker compose -f docker-compose.warpplus-hardened.yml config --quiet   # 校验语法
```

## 8. 日常巡检清单

每次动 / 定期对一遍（每题可自检 is 5 分钟内完成）：

1. `docker ps`：有没有 `Restarting` / `(unhealthy)` —— `unhealthy` 是僵尸的头号信号。
2. 连续重启的容器：`docker inspect` 看 ExitCode（1=启动链，137=OOM）+ 日志尾部关键字。
3. `docker stats`：确认每台 RSS 没贴着/顶破 512m 上限持续涨。
4. 探活：宿主机 curl 打每个出口一次（§3），确认没有「端口能连但隧道死」。
5. license：新增/替换容器前确认 WARP+ license 设备余量 ≥ 设备数，别让第七台把前面六台一起拉进 Register 失败。
6. 身份：确认身份卷存在、重启后 identity 未重建（§4.1）。

一条命令盯全部健康状态：

```bash
watch -n 60 'docker ps --format "{{.Names}}\t{{.Status}}"'
```
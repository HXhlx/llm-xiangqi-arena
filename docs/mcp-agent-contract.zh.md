# MCP Agent 对局契约

面向外部 agent（通过 `mcp_server.py`）与平台的责任边界。思考与记忆由 agent 自管；平台强制的是**操作通道**与**工具审计**。

## 原则

本项目定位是**大模型执子对决**，不是程序/规则 bot 对决。

选手只通过本 MCP 工具集操作对局；思考与记忆由选手自管；长考与工具超时不罚；禁止借助象棋引擎、搜索树，以及用脚本/启发式代打；平台缺陷与服务中断不计入选手胜负责任。

规则投递（软约束，非沙箱硬拦）：

1. 宿主系统提示可挂 `prompts/player/mcp_external.md`（`XIANGQI_LANG=zh` 时为 `mcp_external.zh.md`）
2. 工具 `get_player_brief` 随时拉取
3. `claim_seat` / `create_duel` / `challenge_preset` 成功响应自动带 `player_brief` + `rules` + `contract_version`
4. `wait_my_turn` 在 `my_turn` 时带短 `rules_reminder`

## 席位令牌

- **统一入座剧本**（子智能体 / 多会话 / 远程 MCP 相同）：
  1. 裁判 `create_duel`（双 external 时默认 `auto_claim=false`）→ 只共享 `game_id`
  2. 各方 `claim_seat(side)` → 只保存本席 `seat_token`（响应含 `player_brief` / `rules`）
  3. 本地烟雾：`scripts/mcp_llm_duel_smoke.py` 只当裁判写提示，不预发双 token
- 单 external（如 `challenge_preset`）默认仍自动 claim 本席，方便单挑场内 LLM。
- 显式 `auto_claim=true` 可强制开局即发牌（双席会带回双方 token，**独立宿主勿用**）。
- 除 `list_presets` / `list_games` / `get_player_brief` / 开局类工具外，对局工具必填 `seat_token`。
- 无 token / 错 token / 代提交对方席 → `error_class=auth`（HTTP 401）。
- 裸 REST 若带合法 token 但无 `X-Xiangqi-Mcp-Tool`，审计记为 `rest_direct`（赛后分析用，**不自动判负**）。
- `sides.*.claimed`（state）与 `/api/games` 的 `external_seats[].joinable` 供发现空席；已 claimed 再 `claim_seat` 会**轮换** token（旧牌作废），但必须携带该席当前 `seat_token`，或裁判密钥（`XIANGQI_REFEREE_SECRET`，请求头 `X-Xiangqi-Referee-Secret`）。MCP 工具 `claim_seat` 会把可选参数 `referee_secret` 转发为该请求头，仅供裁判进程使用。环境变量与请求头都会先去掉首尾空白再比较；未设置或空密钥永远不匹配。不带凭证的再次 claim 会被拒绝。轮换之后旧 token 失效。不要把密钥或该参数交给选手 agent。选手简报只要求保存本席 `seat_token`。
- 快照不保存 `seat_tokens`。`POST /restore` 之后双方外部席都是未入座，任何一方都可以先 claim。快照也不保存 API key。恢复后的 `ai` 席会再次读取 `LITELLM_*`（不会改用预设自己的 endpoint）。

## 远程接入

```bash
# 对局 API（示例）
XIANGQI_API_BASE=http://127.0.0.1:8000 .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000

# MCP streamable HTTP（远程客户端连 /mcp；须能访问上面的 API）
XIANGQI_API_BASE=http://127.0.0.1:8000 .venv/bin/python mcp_server.py --http 8765 --host 127.0.0.1
# 仅受信任的局域网：--host <LAN-IP>
# 健康检查：GET http://127.0.0.1:8765/health （含 contract_version / remote_hint）
```

**不要在不受信任的网络上绑定 `0.0.0.0`。** 控制端点（创建、开始、暂停、resume、seek、重置、restore，以及空席的首次 claim）默认无认证。先暂停、再 seek、再 resume 可以把外部对局回退到更早的 ply，并丢掉已排队的外部着法。`POST /finish` 只接受本地对局（任一方为 `llm`、`ai` 或 `external` 都不接受），已占用席位的再次 claim 需要当前 token 或裁判密钥，但其余控制面只要端口可达即可调用。除非每个客户端都是受信任的裁判，否则请绑定 `127.0.0.1`。受信任的局域网上请绑定 `<LAN-IP>`，不要绑定全部网卡。

推荐流程（与子智能体相同）：

```
裁判: python scripts/mcp_llm_duel_smoke.py --api-base ... --work-dir /tmp/duel
     → 得到 game_id + red-prompt.txt / black-prompt.txt
红宿主: 按 red-prompt → claim_seat(red) → 逐步下
黑宿主: 按 black-prompt → claim_seat(black) → 逐步下
```

不要把对方的 `seat_token` 写进共享配置。

## 工具审计

- 主落盘：`logs/mcp/{game_id}.jsonl`（对标 LangGraph `turns` 的工具轨；**不含** reasoning）。
- 同时 `broadcast(tool_call|tool_result)` 进入 `game.events`，终局 `game.log` Event Log 可读。
- 不写 `logs/turns/{gid}.jsonl`（避免与 LLM 思维链格式混用）。

## 预演（ObserveSession）

- `preview` / `preview_reset` / `get_threats`：只改席位 work 盘，不改 live。
- `get_legal_moves`（带 token）：work 盘 annotated 列表。
- 正式落子只有 `submit_move` → `/move`；**没有**第二落子口。
- **走子阶段闸门（复盘/预盘禁用）**：`submit_move` 只在 `status=playing` 且轮到你时可用。MCP 层预检 + 服务端 `/move` 双重拦截，非 playing 一律 `error_class=state` 拒绝：
  - `waiting`（预盘/开局前）：未开局不走子，等 `wait_my_turn` 通知；
  - `paused`（复盘/暂停）：不走子，`resume` 回 `playing` 后再等轮次；
  - `finished`（终局复盘）：对局已结束，改用 `get_result`；
  - `interrupted`（中断）：不走子，等恢复或向裁判反馈。
  `get_board` / `get_legal_moves` / `preview` / `get_threats` 不受限（只读 / work 盘）。

## 超时与长考

- External 席豁免赛程墙钟 / 停滞杀局。
- `wait_my_turn` 超时是**正常结果**：`ok=true` + `waiting_done_reason=timeout`（无 `error_class`，不算错误），响应带 `suggested_retry_sec`；**不改变胜负**，可再调，默认单次等 30 分钟。
- 长时间不落子、会话卡住再续上视为正常；裁判 Truncate 只能人工 pause/resign。
- **沉默≠退出**：对方长考（半小时甚至更久）属正常；wait/state 响应的 `activity`（`current_side_thinking_sec` / `side_last_tool_sec_ago`）是客观活跃度信号；agent 任何时候不得因对方沉默而主动退出或认输，终局以 `status=finished` 为准。

## 选手必须怎么下（大模型对决）

每一手由**模型本人**根据当前局面决定着法，再调用工具提交。合法回合形态：

```
wait_my_turn →（可选 get_board / get_legal_moves / preview* / get_threats）
→ 在回复中写出选择理由与 ICCS
→ submit_move(..., note=一句人话理由)
```

| 允许 | 禁止 |
|------|------|
| 每手单独调 MCP（或单次 Shell 只提交**你已写明**的那一手） | 一条 Shell/Python `while` 循环自动下完多手 |
| 用自然语言记住开局意图（中炮、顺炮等） | 写死 `opening_seq` / `VAL` 子力表 / `move_score` 贪心代打 |
| `preview` 推演候选 | Pikafish / 其它引擎、MCTS、αβ 搜索代选着 |
| `note` 写拟人理由（贪心/嘴硬/松劲均可） | 把选着逻辑外包给不可见脚本 |

编排方抽查：`logs/mcp/{gid}.jsonl` 应能对应到 agent 逐步工具调用；若 transcript 里出现「整局一个启发式循环」视为**不合规实测**，不能当作正式擂台局。

## 防作弊

| 手段 | 说明 |
|------|------|
| 席位 token | 无 token 无法 move / resign / wait / 预演 |
| 不提供引擎 MCP 工具 | 不暴露 Pikafish bestmove / eval |
| 引擎 / MCTS 外挂 | **规则禁止**（开放 agent 无法硬防；非本迭代沙箱范围） |
| 本地启发式 / 脚本代打 | **规则禁止**（与「大模型对决」定位冲突；开放 agent 无法硬防，编排抽查） |
| 赛后着法与引擎 top-N 吻合率 | 契约允许抽查报告；**默认不自动判负** |

平台能强制的是「改棋必经 MCP」；不能保证开放 agent 不用外挂引擎或启发式脚本。

## error_class 归责

| class | 含义 | 归责 |
|-------|------|------|
| `rules` | 非法着、非己方行棋、重复 pending（非法着附 `reason`：蹩马腿/无炮架/送将等） | 选手 |
| `auth` | token 缺失/错误/席位不符 | 选手或编排 |
| `infra` | 5xx、服务宕机、MCP/连接失败 | 非选手；不因基建记负 |
| `state` | paused / interrupted / finished | 非自动判负；按返回值重试或停手 |

`wait_my_turn` 超时不是错误：`ok=true` + `waiting_done_reason=timeout`，无 `error_class`。

MCP 错误响应统一：`ok` + `error_class` + `error`。

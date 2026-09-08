# MCP 外部选手简报（大模型象棋对决）

本仓库是**大模型执子对决**擂台，不是启发式或引擎 bot 对局。完整契约见
`docs/mcp-agent-contract.zh.md`。

同一套规则会由 `get_player_brief`、`claim_seat` 以及开局类工具注入。
你不必依赖宿主粘贴本文件。

## 你是谁

你是 Xiangqi Arena 的外部选手。入座、等待、落子都通过 MCP 工具
（或 `import mcp_server` 调用同一组函数）。着法用 ICCS（例如 `h2e2`）。

## 统一入座剧本（子智能体 / 多会话 / 远程）

步骤相同，与宿主无关：

1. 裁判（人类、`scripts/mcp_llm_duel_smoke.py`、或某一个 agent）调用
   `create_duel` 开双外部席。**默认不发牌**（`auto_claim=false`）。
   只共享 `game_id` 和你的 `side`。
2. 你：`get_player_brief`（可选）→ `claim_seat(game_id, side, name=...)`
   → **只保存本席 `seat_token`**，然后阅读注入的 `rules`。
3. 或 `list_games` → `joinable_seats` 发现空席后补入座。

不要把对方的 token 写进共享 `meta.json`。子智能体不得向父会话索要双方 token。

## 每一手

1. `wait_my_turn(game_id, side, seat_token)` 等到轮到你（超时可续等，不罚）。
2. 用 `get_board` / `get_legal_moves` 读局面（可选 `preview` / `get_threats`）。
3. **由你（模型本人）决定**下一手：写出理由 + ICCS。
4. 只为这一手调用 `submit_move(..., note="一句人话理由")`。
5. 重复直到终局，或达到任务规定的手数上限。

## 长考与等待

- 对方长考（半小时甚至更久）属正常。**沉默≠对方已离开。**
  `wait_my_turn` 超时只表示「这次轮询未等到」，不是错误也不是认输。再调即可，不罚。
- 工具结果里的 `activity` 是客观信号：
  `current_side_thinking_sec`、`side_last_tool_sec_ago`。
- 只有服务端会结束对局（`status=finished`）。**不要因为对方安静而认输或退出。**

## 走子阶段闸门（复盘 / 预盘禁用）

`submit_move` **只在** `status=playing` 且轮到你时合法。
其它阶段返回 `error_class=state`，不会被接受：

- `waiting`（开局前）：先入座，再等 `wait_my_turn`。
- `paused`（复盘 / 暂停）：不要走子；继续等待直到 `resume`。
- `finished`：改用 `get_result`。
- `interrupted`：不要走子；等恢复或向裁判反馈。

这些阶段不要连打 `submit_move`——闸门是故意的，不是抖动。
`get_board` / `get_legal_moves` / `preview` 仍可用来看棋。

## 禁止

- 一条很长的 Shell / Python `while`/`for` 自动下完多手。
- 写死 `opening_seq`、`VAL` 子力表、或 `move_score` 贪心代打。
- Pikafish、其它引擎、MCTS、αβ 或任何搜索树代选着。
- 把「整局策略」编译进不可见脚本然后挂机。
- 索取或使用对方的 `seat_token`。

若必须从 shell 调 `mcp_server`：每个 shell **最多完成一手**
（等待 → 读盘 → 提交你已经写明的那一手）。选着必须出现在助手回复里，
不能藏在评分函数中。

## 推荐的 `note`

说人话，可以带点脾气（贪、嘴硬、松劲都可以）。不要写子力分，不要写引擎数字。

# AGENTS.md

## Project

This project extends TradingAgents into a BTCUSDT Binance Futures AI Signal and DEMO Trading System.

The system is for research, signal generation, backtesting, and simulated trading.

REAL Binance order execution must NOT be implemented.

---

# 1. Development Philosophy

Follow the existing TradingAgents architecture whenever possible.

Before modifying code:

1. Inspect the existing implementation.
2. Understand how the existing component works.
3. Reuse existing functionality when appropriate.
4. Only create new modules when necessary.
5. Avoid unnecessary changes to TradingAgents core.

Prefer adapters and separate modules over invasive modifications.

Keep responsibilities separated:

```text
Binance Data
     ↓
Indicators
     ↓
TradingAgents / LLM
     ↓
Signal Engine
     ↓
Demo Execution
     ↓
TP/SL Monitor
```

Do not mix these responsibilities unnecessarily.

---

# 2. Implementation Workflow

Implement the project one phase at a time according to:

```text
.opencode/plans/btcusdt-signal-engine.md
```

Never implement multiple future phases unless explicitly requested.

For every phase:

### Before coding

1. Read the relevant section of the plan.
2. Inspect existing code.
3. Identify files to modify.
4. Identify files to create.
5. Explain the implementation approach.

### During coding

- Make the smallest reasonable changes.
- Reuse existing code.
- Do not rewrite unrelated modules.
- Keep functions small and testable.
- Add tests for new functionality.

### After coding

1. Run relevant tests.
2. Run the full test suite when practical.
3. Run lint/type checks if available.
4. Report changed files.
5. Report test results.
6. Report any deviation from the plan.

Then STOP.

Do not automatically continue to the next phase.

---

# 3. Critical Trading Rule — One Active Signal

There must be at most ONE OPEN signal/position for BTCUSDT.

This rule has highest priority.

If an OPEN position exists:

```text
DO NOT CALL AI
DO NOT CREATE ANOTHER SIGNAL
DO NOT OPEN ANOTHER DEMO POSITION
```

The system must continue monitoring the existing position.

Only after the existing position is closed by TP or SL may the AI analyze a new signal.

---

# 4. Signal State Machine

Use:

```text
IDLE
  ↓
ANALYZING
  ↓
OPEN
  ↓
CLOSED
  ↓
IDLE
```

If AI returns WAIT:

```text
ANALYZING
     ↓
    WAIT
     ↓
    IDLE
```

WAIT must NOT create a position.

---

# 5. AI Analysis Rules

The primary AI analysis timeframe is:

```text
1H
```

AI analysis may only use CLOSED candles.

Never provide the AI with future information.

Never use an unfinished 1H candle as a completed candle.

The AI must not:

- invent prices
- invent indicators
- use future candles
- use future news
- place Binance orders
- modify an already-open signal

---

# 6. Indicator Rules

Indicators must be calculated deterministically by Python.

The LLM must NOT be responsible for calculating technical indicators.

Primary indicators:

```text
EMA20
EMA50
EMA200
RSI14
MACD
ATR14
Volume SMA
Volume Ratio
```

Optional:

```text
ADX
```

The calculated indicator values must be explicitly provided to the AI.

---

# 7. Signal Rules

Allowed signals:

```text
LONG
SHORT
WAIT
```

For LONG:

```text
SL < Entry < TP
```

For SHORT:

```text
TP < Entry < SL
```

Every LONG/SHORT signal must pass validation before being opened.

Invalid signals must be rejected.

---

# 8. Immutable Entry / TP / SL

After a signal becomes OPEN:

```text
entry_price
stop_loss
take_profit
```

are immutable.

The AI must never modify them.

No trailing stop is required for the first version.

Do not introduce dynamic TP/SL unless explicitly requested.

---

# 9. TP/SL Monitoring

TP/SL monitoring is independent from AI analysis.

AI analysis timeframe:

```text
1H
```

TP/SL monitoring:

```text
1m
```

or realtime market price.

Do NOT wait for the next 1H candle to detect TP/SL.

LONG:

```text
price >= TP → TAKE PROFIT
price <= SL → STOP LOSS
```

SHORT:

```text
price <= TP → TAKE PROFIT
price >= SL → STOP LOSS
```

---

# 10. Backtest Data Integrity

Backtests must never contain look-ahead bias.

A signal at time T may only use information available at time T.

Never use:

- future candles
- future indicators
- future closing prices
- future trade outcomes

If both TP and SL appear to be hit inside the same candle, do not guess the execution order.

Use lower timeframe data, preferably 1m data, to determine which happened first.

---

# 11. Demo Trading

The system supports:

```text
SIGNAL_ONLY
DEMO
LIVE
```

Current implementation:

```text
SIGNAL_ONLY
DEMO
```

LIVE is future architecture only.

Do NOT implement real Binance order execution.

---

# 12. DEMO Must Never Trade Real Money

DEMO mode must never:

- place Binance orders
- cancel Binance orders
- modify Binance orders
- use Binance trading API credentials
- require a real trading API key

DEMO only simulates execution using market data.

---

# 13. Demo Account Configuration

Demo Account must support:

```text
initial_balance
margin_per_trade
leverage
risk_percent
fee_rate
```

These values must NOT be hard-coded inside trading logic.

Example defaults:

```text
Initial Balance = 1000 USDT
Margin = 50 USDT
Leverage = 10x
Risk = 1%
Fee = 0.04%
```

Users must be able to change these values.

---

# 14. Demo Position Calculation

When using fixed margin:

```text
position_size = margin × leverage
```

Quantity:

```text
quantity = position_size / entry_price
```

Use Decimal or another appropriate precision-safe method for financial calculations.

Avoid unnecessary floating-point rounding errors.

---

# 15. Demo PnL

LONG:

```text
gross_pnl =
(exit_price - entry_price) × quantity
```

SHORT:

```text
gross_pnl =
(entry_price - exit_price) × quantity
```

Fees must be calculated separately.

```text
net_pnl = gross_pnl - fees
```

Balance must be updated using net PnL.

---

# 16. Risk Management

Support risk percentage.

Example:

```text
Balance = 1000 USDT
Risk = 1%
```

Target maximum risk:

```text
10 USDT
```

Risk calculation must consider:

```text
Entry Price
Stop Loss
Quantity
Margin
Leverage
```

Do not silently override user configuration.

If margin-based and risk-based sizing conflict, the implementation must make the selected sizing rule explicit.

---

# 17. Database Persistence

SQLite is the persistent source of truth for:

- signals
- demo accounts
- demo positions
- demo trades

Do not keep critical trading state only in memory.

The system must survive application restart.

---

# 18. Restart Recovery

If the application restarts while a Demo position is OPEN:

```text
Load Account
Load Position
Load Signal
Resume TP/SL Monitoring
DO NOT CALL AI
```

The position must not disappear.

If no position is OPEN:

```text
Wait for the next eligible 1H candle close
```

---

# 19. AI Lock

When a Demo position is OPEN:

```text
AI_LOCKED = true
```

No AI signal generation is allowed.

When TP or SL closes the position:

```text
AI_LOCKED = false
```

The next AI analysis may happen only at the next eligible 1H candle close.

---

# 20. LLM Provider Architecture

Use a provider abstraction.

Supported providers:

```text
DeepSeek
Ollama
```

Supported models include:

```text
DeepSeek V4 Flash
Qwen3 8B
Qwen3 4B
```

API keys must come from environment variables.

Never hard-code credentials.

Do not expose API keys in logs.

---

# 21. Binance API Rules

Public market data may be used without authentication where supported.

Trading/order endpoints must NOT be implemented in the current project.

Never add real order execution accidentally while implementing market data.

Keep market-data code separate from future execution code.

---

# 22. Security

Never commit:

```text
.env
API keys
API secrets
Telegram bot tokens
private credentials
```

Use `.env.example` for configuration documentation.

Never print secrets to logs.

---

# 23. Database Safety

Application startup must not delete trading history.

Do not automatically reset:

- Demo balance
- positions
- trades
- signals

Do not recreate the database destructively.

Database migrations must preserve existing data.

---

# 24. Testing Requirements

Every trading-critical calculation must have tests.

Minimum tests:

### Signal

- LONG validation
- SHORT validation
- WAIT behavior
- duplicate OPEN prevention
- immutable TP/SL

### TP/SL

- LONG TP
- LONG SL
- SHORT TP
- SHORT SL

### PnL

- LONG PnL
- SHORT PnL
- fee calculation
- net PnL

### Demo

- initial balance
- margin
- leverage
- quantity
- balance update
- equity
- drawdown

### State

- OPEN blocks AI
- TP unlocks AI
- SL unlocks AI
- restart recovery

### Backtest

- no look-ahead
- same-candle TP/SL handling

---

# 25. Code Organization

Prefer this architecture:

```text
binance/
    client.py
    market_data.py
    indicators.py

signal_engine/
    engine.py
    validator.py
    state.py
    monitor.py

demo/
    account.py
    position.py
    executor.py
    statistics.py

database/
    database.py
    models.py

notification/
    telegram.py

backtest/
    engine.py
    metrics.py
```

However, if TradingAgents already contains an appropriate implementation, reuse it instead of creating duplicate functionality.

---

# 26. Logging

Use structured and useful logging.

Log:

- data fetch errors
- AI analysis start/end
- signal creation
- signal rejection
- position open
- TP
- SL
- position close
- Demo balance update
- restart recovery

Never log:

- API keys
- API secrets
- Telegram tokens
- sensitive credentials

---

# 27. Error Handling

External API failures must not corrupt trading state.

If Binance market data fails:

```text
Do not create a signal.
```

If AI response is invalid:

```text
Reject signal.
```

If database write fails:

```text
Do not report the trade as successfully opened/closed.
```

Critical state transitions must be atomic where practical.

---

# 28. No Automatic Phase Expansion

If implementing Phase 3, do NOT also implement:

```text
Phase 4
Phase 5
Phase 6
...
```

unless explicitly instructed.

Do not add unrelated features.

Do not redesign the entire project without approval.

---

# 29. Definition of Done

A phase is complete only when:

- implementation matches the plan
- relevant tests pass
- no critical regression is introduced
- changed files are reported
- test results are reported
- deviations are reported

Then STOP.

---

# 30. Final Safety Rule

This project is for research, backtesting, signal generation, and simulated trading.

Do not claim that any strategy is profitable.

Do not guarantee win rate.

Do not guarantee returns.

Do not enable real-money trading automatically.

REAL Binance order execution requires a separate explicit future implementation and review.

# BTCUSDT AI Signal + Demo Trading Engine

## 1. Mục tiêu

Xây dựng hệ thống AI phân tích BTCUSDT Binance Futures trên timeframe 1H.

Hệ thống tạo:

- LONG
- SHORT
- WAIT

Sau khi có LONG/SHORT, hệ thống tạo một signal OPEN và mô phỏng giao dịch bằng DEMO ACCOUNT.

Mục tiêu hiện tại là:

- Signal analysis
- Demo trading
- Backtesting
- Model comparison

Không thực hiện giao dịch Binance thật.

---

# 2. Nguyên tắc quan trọng

## Single Active Signal

Tại mọi thời điểm chỉ được có tối đa một OPEN signal cho BTCUSDT.

Nếu đang có OPEN position:

```text
DO NOT CALL AI
```

AI không được tạo signal mới cho đến khi position hiện tại đóng.

---

## Signal Lifecycle

```text
IDLE
  ↓
ANALYZING
  ↓
LONG / SHORT
  ↓
OPEN
  ↓
TP / SL
  ↓
CLOSED
  ↓
IDLE
```

Nếu AI trả:

```text
WAIT
```

thì:

```text
ANALYZING
   ↓
WAIT
   ↓
IDLE
```

Không tạo position.

---

# 3. Timeframe

## AI Analysis

Timeframe chính:

```text
1H
```

AI chỉ được phân tích khi candle 1H đã đóng.

Không sử dụng candle 1H chưa đóng.

## TP/SL Monitoring

TP/SL phải được monitor độc lập với AI.

Có thể sử dụng:

```text
1m
```

hoặc realtime market price.

Không được chờ candle 1H mới kiểm tra TP/SL.

---

# 4. Binance Market Data

Sử dụng Binance USDT-M Futures public market data.

Symbol mặc định:

```text
BTCUSDT
```

Hỗ trợ:

```text
1m
5m
15m
1h
4h
```

OHLCV:

```text
timestamp
open
high
low
close
volume
```

Không implement Binance order execution.

Không cần API key cho public market data nếu API endpoint cho phép.

Phải có:

- timeout
- retry
- error handling
- data validation
- logging

---

# 5. Indicator Engine

Indicator phải được tính bằng Python.

Không yêu cầu LLM tự tính indicator.

Indicators:

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

Có thể hỗ trợ:

```text
ADX
```

sau này.

Indicator output phải có cấu trúc rõ ràng để truyền cho TradingAgents.

Phải có unit tests.

---

# 6. TradingAgents Integration

TradingAgents là AI reasoning engine.

Không rewrite TradingAgents core nếu không cần thiết.

Tái sử dụng các agent hiện có khi phù hợp:

```text
Technical Analyst
Bull Researcher
Bear Researcher
Trader
Risk Management
```

TradingAgents nhận:

```text
BTCUSDT
1H market data
recent OHLCV
technical indicators
market context
```

AI phải trả structured signal.

---

# 7. AI Signal Format

AI output phải có cấu trúc tương đương:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "signal": "LONG",
  "confidence": 82,
  "entry": 112500,
  "stop_loss": 111500,
  "take_profit": 114500,
  "risk_reward": 2.0,
  "reason": [
    "EMA20 above EMA50",
    "RSI bullish",
    "MACD positive",
    "Volume confirmation"
  ]
}
```

Signal phải là:

```text
LONG
SHORT
WAIT
```

Nếu WAIT:

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "signal": "WAIT"
}
```

AI không được:

- tự gọi Binance order API
- dùng dữ liệu tương lai
- bịa giá
- thay đổi TP sau khi position OPEN
- thay đổi SL sau khi position OPEN

---

# 8. Signal Validation

Trước khi tạo OPEN signal phải validate:

- signal hợp lệ
- symbol đúng
- timeframe đúng
- entry hợp lệ
- SL hợp lệ
- TP hợp lệ
- risk/reward hợp lệ
- confidence hợp lệ

LONG:

```text
SL < Entry < TP
```

SHORT:

```text
TP < Entry < SL
```

Nếu dữ liệu không hợp lệ:

```text
REJECT SIGNAL
```

Không tạo position.

---

# 9. Signal Database

Sử dụng SQLite.

Bảng:

```text
signals
```

Các field chính:

```text
id
symbol
timeframe
signal
confidence
entry_price
stop_loss
take_profit
risk_reward
reason
created_at
closed_at
status
result
exit_price
pnl_percent
pnl_usdt
```

Status:

```text
OPEN
CLOSED
```

WAIT không tạo OPEN record.

Database phải đảm bảo không có duplicate OPEN signal.

---

# 10. Demo Account

Hệ thống có 3 mode kiến trúc:

```text
SIGNAL_ONLY
DEMO
LIVE
```

Chỉ implement:

```text
SIGNAL_ONLY
DEMO
```

LIVE chỉ chuẩn bị architecture.

Không implement real Binance order execution.

---

# 11. Demo Configuration

Demo Account phải cho phép cấu hình:

```text
initial_balance
margin_per_trade
leverage
risk_percent
fee_rate
```

Ví dụ mặc định:

```text
Initial Balance = 1000 USDT
Margin = 50 USDT
Leverage = 10x
Risk = 1%
Fee = 0.04%
```

Không hard-code các giá trị này trong trading logic.

---

# 12. Demo Position Size

Nếu sử dụng fixed margin:

```text
Position Size = Margin × Leverage
```

Ví dụ:

```text
Margin = 50 USDT
Leverage = 10x

Position Size = 500 USDT
```

Quantity:

```text
Quantity = Position Size / Entry Price
```

Phải tính toán bằng Decimal hoặc phương pháp phù hợp để tránh sai số floating point.

---

# 13. Risk Management

Hỗ trợ:

```text
risk_percent
```

Ví dụ:

```text
Balance = 1000 USDT
Risk = 1%
```

Target risk:

```text
10 USDT
```

Risk calculation phải xem xét khoảng cách Entry → SL.

Không được để leverage làm cho risk calculation sai lệch.

Margin và risk phải được xử lý rõ ràng, không được âm thầm chọn một giá trị khác với cấu hình.

---

# 14. Demo Execution

Demo phải mô phỏng:

```text
LONG
SHORT
Entry
TP
SL
Quantity
Position Size
Margin
Leverage
Fee
PnL
Balance
Equity
Drawdown
```

DEMO không được gửi request tạo order Binance.

Demo chỉ mô phỏng execution.

Signal Engine và Demo Execution phải tách biệt.

---

# 15. LONG PnL

Với LONG:

```text
Gross PnL =
(Exit Price - Entry Price) × Quantity
```

SHORT:

```text
Gross PnL =
(Entry Price - Exit Price) × Quantity
```

Fee phải được tính riêng.

```text
Net PnL = Gross PnL - Fees
```

Balance phải cập nhật bằng Net PnL.

---

# 16. TP/SL

LONG:

```text
TP khi Price >= Take Profit
SL khi Price <= Stop Loss
```

SHORT:

```text
TP khi Price <= Take Profit
SL khi Price >= Stop Loss
```

Sau khi position OPEN:

```text
Entry
TP
SL
```

là immutable.

Không cho AI thay đổi.

---

# 17. Same Candle TP/SL

Trong backtest nếu cùng một candle có:

```text
High >= TP
Low <= SL
```

không được tự đoán TP trước hay SL trước.

Sử dụng timeframe thấp hơn, ưu tiên 1m data, để xác định thứ tự.

Không được dùng dữ liệu tương lai.

---

# 18. Demo Database

Tạo bảng:

```text
demo_accounts
```

Fields:

```text
id
name
initial_balance
balance
equity
margin_per_trade
leverage
risk_percent
fee_rate
created_at
updated_at
```

Tạo:

```text
demo_positions
```

Fields:

```text
id
account_id
signal_id
symbol
side
entry_price
quantity
position_size
margin
leverage
stop_loss
take_profit
unrealized_pnl
status
opened_at
closed_at
```

Tạo:

```text
demo_trades
```

Fields:

```text
id
account_id
position_id
signal_id
side
entry_price
exit_price
quantity
margin
position_size
leverage
gross_pnl
fee
net_pnl
pnl_percent
result
opened_at
closed_at
```

---

# 19. Restart Recovery

Application restart không được mất trạng thái.

Nếu database có OPEN Demo Position:

```text
Load position
Load signal
Resume monitoring
DO NOT CALL AI
```

Nếu không có OPEN position:

```text
Wait for next eligible 1H candle close
```

Phải có test restart recovery.

---

# 20. AI Lock

Implement rõ ràng AI lock.

Nếu:

```text
OPEN position exists
```

thì:

```text
AI_LOCKED = true
```

Không được gọi AI.

Khi TP hoặc SL:

```text
Position CLOSED
AI_LOCKED = false
```

AI chỉ có thể chạy ở candle 1H tiếp theo.

---

# 21. LLM Providers

Hỗ trợ:

```text
DeepSeek V4 Flash
Qwen3 8B
Qwen3 4B
```

Tạo provider abstraction.

Ví dụ:

```text
LLM Provider
├── DeepSeek
└── Ollama
    ├── Qwen3 8B
    └── Qwen3 4B
```

API keys phải lấy từ environment variables.

Không hard-code API key.

---

# 22. Scheduler

AI scheduler:

```text
1H candle closes
        ↓
Check active position
        ↓
OPEN?
 ┌──────┴──────┐
 YES           NO
  ↓             ↓
STOP         Analyze
```

TP/SL monitor chạy độc lập.

---

# 23. Telegram

Gửi notification khi signal OPEN:

```text
BTCUSDT LONG

Entry:
SL:
TP:
Confidence:
Risk/Reward:
```

Khi đóng:

```text
BTCUSDT LONG CLOSED

Result:
Entry:
Exit:
PnL:
Fee:
Net PnL:
```

Telegram token phải nằm trong environment variables.

---

# 24. Backtest

Backtest phải đảm bảo:

```text
NO LOOK-AHEAD BIAS
```

Signal tại thời điểm T chỉ được sử dụng dữ liệu có trước hoặc tại thời điểm T.

Không sử dụng future candles.

Tính:

```text
Total Trades
Wins
Losses
Win Rate
Gross PnL
Net PnL
Average PnL
Profit Factor
Maximum Drawdown
Average Win
Average Loss
Long Performance
Short Performance
```

---

# 25. Model Benchmark

Benchmark:

```text
DeepSeek V4 Flash
Qwen3 8B
Qwen3 4B
```

Các model phải dùng cùng:

```text
Dataset
Candles
Indicators
Prompt
Entry logic
TP/SL logic
Margin
Leverage
Fee
```

So sánh:

```text
Total Signals
LONG
SHORT
WAIT

Total Trades
Win Rate
Net PnL
Profit Factor
Maximum Drawdown
Average Trade
```

Không đánh giá model chỉ dựa trên Win Rate.

---

# 26. Dashboard / CLI

Hiển thị Demo Account:

```text
Balance
Equity
Margin
Leverage
Unrealized PnL
Realized PnL
Total Trades
Win Rate
Profit Factor
Max Drawdown
```

Current Position:

```text
Symbol
Side
Entry
Current Price
TP
SL
Margin
Leverage
PnL
```

Current Signal:

```text
LONG / SHORT / WAIT
Confidence
Entry
SL
TP
Risk/Reward
Reason
```

Trade History:

```text
Entry
Exit
Side
Result
PnL
Fee
```

Cho phép:

```text
Create Demo Account
Reset Demo Account
Change Balance
Change Margin
Change Leverage
Change Risk
View Statistics
```

Không cho phép reset account nếu đang có OPEN position, trừ khi user xác nhận rõ ràng.

---

# 27. Project Structure

Ưu tiên structure:

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

Không tạo duplicate functionality nếu repository đã có implementation phù hợp.

---

# 28. Development Rules

Implement từng phase một.

Không implement nhiều phase cùng lúc.

Trước khi sửa code:

1. Inspect existing implementation.
2. Xác định file cần sửa.
3. Xác định file cần tạo.
4. Kiểm tra khả năng reuse code hiện tại.

Sau khi code:

1. Run tests.
2. Run lint/type checks nếu có.
3. Report changed files.
4. Report test results.
5. Report any deviation from plan.

Sau mỗi phase phải STOP.

Không tự động chuyển sang phase tiếp theo.

---

# 29. Implementation Phases

## PHASE 0

Repository Audit.

Không sửa code.

Kiểm tra:

- architecture
- LLM providers
- data providers
- indicators
- TradingAgentsGraph
- configuration
- CLI
- tests

---

## PHASE 1

Binance Futures Market Data.

Implement:

- BTCUSDT
- OHLCV
- multiple timeframes
- closed candle handling
- retry
- timeout
- validation
- tests

Không order execution.

---

## PHASE 2

Indicator Engine.

Implement:

- EMA20
- EMA50
- EMA200
- RSI14
- MACD
- ATR14
- Volume SMA
- Volume Ratio

Tests.

---

## PHASE 3

SQLite + Signal Schema.

Implement:

- signals
- active signal
- create signal
- close signal
- duplicate protection
- persistence
- tests

---

## PHASE 4

LLM Provider.

Implement:

- DeepSeek
- Ollama
- Qwen3 8B
- Qwen3 4B
- environment configuration
- provider abstraction
- tests

---

## PHASE 5

TradingAgents Integration.

Connect:

```text
Market Data
+
Indicators
↓
TradingAgents
↓
Structured Signal
```

No future data.

No order execution.

---

## PHASE 6

Signal Engine.

Implement:

```text
IDLE
ANALYZING
OPEN
CLOSED
```

Critical rule:

```text
OPEN position
→ DO NOT CALL AI
```

Implement validation and immutable TP/SL.

---

## PHASE 7

TP/SL Monitor.

Implement:

- LONG TP
- LONG SL
- SHORT TP
- SHORT SL
- closing signal
- PnL
- persistence
- tests

---

## PHASE 8

Demo Account.

Implement:

- balance
- margin
- leverage
- risk
- fee
- position size
- LONG
- SHORT
- TP
- SL
- PnL
- balance update
- equity
- drawdown
- trade history

DEMO must never call Binance order API.

---

## PHASE 9

Restart Recovery.

Implement persistent recovery of OPEN Demo positions.

No AI call after restart if position is OPEN.

Tests.

---

## PHASE 10

1H Scheduler.

AI runs only at closed 1H candle.

TP/SL monitor remains independent.

---

## PHASE 11

Telegram Notifications.

Implement:

- OPEN notification
- CLOSE notification
- configuration through .env

---

## PHASE 12

Backtest.

Implement:

- historical data
- no look-ahead
- 1m TP/SL ordering
- fees
- margin
- leverage
- statistics

---

## PHASE 13

Model Benchmark.

Compare:

- DeepSeek V4 Flash
- Qwen3 8B
- Qwen3 4B

Same dataset and same trading configuration.

---

## PHASE 14

Dashboard / CLI.

Implement Demo Account management and monitoring.

---

## PHASE 15

Final Validation.

Verify:

1. No duplicate OPEN signal.
2. OPEN position blocks AI.
3. TP closes position.
4. SL closes position.
5. LONG PnL correct.
6. SHORT PnL correct.
7. Fees correct.
8. Margin correct.
9. Leverage correct.
10. Balance update correct.
11. Restart recovery correct.
12. WAIT creates no position.
13. No look-ahead.
14. Demo never sends Binance orders.
15. Only closed 1H candles trigger AI.

Run all tests.

Report final architecture and test results.

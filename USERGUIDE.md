# NEXTRA AI — Web App User Guide

This guide explains how to **use** the web dashboard: what each tab shows,
which buttons to press, and what happens next. No coding needed.

- Open the app at `http://SERVER:8000` (replace `SERVER` with your machine's
  address; on the same computer use `http://127.0.0.1:8000`).
- Everything here uses **demo (virtual) money** — no button in this app can
  touch a real Binance account.
- Top-right corner: 🌙 toggles dark/light theme, **EN/VI** switches language
  instantly. ⚙️ opens Settings.

## 1. Dashboard

The home tab. Read it top to bottom.

**Metric cards** — six numbers describing the demo account:

| Card      | Meaning                                                        |
|-----------|----------------------------------------------------------------|
| Balance   | Cash after closed trades (e.g. `$1,000.45`).                   |
| Equity    | Balance + unrealized PnL of open positions.                    |
| Net PnL   | Total profit/loss of all closed trades, after fees.            |
| Win Rate  | Share of closed trades that were profitable.                   |
| Peak      | Highest equity ever reached.                                   |
| Drawdown  | Largest % drop from peak — the pain meter.                     |

**Portfolio bar** — one chip per enabled symbol:

- Green `seeking` — no position, the AI may analyze the next 1H candle.
- Yellow chip — a position is being held (monitored, AI locked).
- `FULL` — max concurrent positions reached: analysis pauses, but TP/SL
  monitoring never stops.

**Position card** — one card per open position:

```
BTCUSDT 62                        [SHORT]
                                     [50x]
+$5.79 (+11.5%)
Entry $80,800.00               [ Chốt ]
TP $80,100.00 · +$5.79
SL $81,500.00 · −$1.43
0.0062 BTC · 50.00/500.00 · minimax-m3 · 12/09 14:00
```

- Top: symbol + AI confidence, direction chip, leverage chip.
- Big number: live unrealized PnL (green = profit, red = loss).
- TP/SL rows show the exit price and what each exit is currently worth.
- Bottom line: quantity, margin/position-size, AI model, analysis time.

**Close button ("Chốt")** — closes the position now at market price:

1. Press **Chốt**.
2. A confirm dialog shows the estimated PnL (e.g. `+$5.79 (+11.5%)`).
3. Confirm → the position closes, the trade + fee + balance update, and a
   Telegram message is posted. If the monitor closed it a second earlier,
   you get a "already closed" message instead — nothing breaks.

**Price chart** — reference only (the AI never sees it):

- Symbol dropdown and timeframe buttons (1H / 4H / 1D).
- ＋ − zoom (keeps your pan position), ◀ ▶ pan, **Reset** restores the view.
- Dashed lines mark the active signal's Entry / TP / SL.
- The bar under the chart shows the long/short crowd ratio.

**🤖 AI button** — opens a chat panel to ask about the current signal
(read-only: it explains, it cannot trade or change anything).

## 2. Signals

Every signal the AI ever produced. Columns:

- **Symbol / Direction / Entry / SL / TP** — the trade plan.
- **Status** — `PENDING_ENTRY` (waiting for price to touch entry),
  `OPEN` (running), `TP_HIT` / `SL_HIT` (closed), `CANCELLED`.
- **Result** — `WIN` / `LOSS` for closed signals; a small `Manual` tag marks
  positions you closed by hand (vs. natural TP/SL).
- Filters (status, direction, date range), paging, and delete buttons.
  OPEN signals can never be deleted; deleting history never touches money.

## 3. Daemon Log

One row per 1H candle the daemon processed: timestamp, symbol, decision
(`SKIP`, `WAIT`, `OPEN`, `CLOSE`, ...), and the reason. Click a row for
details. Use it to answer "why didn't the AI trade at 14:00?" — the reason
(confidence too low, event blackout, position already open, ...) is written
there. Old rows can be selected and cleared to save space.

## 4. Events

A monthly calendar of macro news (FOMC, CPI, NFP, ...). Days with events are
highlighted; click a day for details. When price action enters a blackout
window (e.g. ±3h around FOMC), a ⏸ banner appears at the top of the app and
no new positions are opened until it passes. Existing positions keep being
monitored.

## 5. Trades

One row per **closed** trade:

```
ID · Symbol · Side · Entry · Exit · Qty · Lev · Gross · Fee · Net PnL · Result · Opened · Closed
```

- **Lev** (e.g. `50x`) is the leverage used for that trade.
- **Gross** is PnL before fees, **Fee** is the round-trip cost,
  **Net PnL** is what actually changed your balance (green/red).
- **Opened/Closed** are Vietnam-time (+07) timestamps.

## 6. Statistics

The same numbers as the dashboard cards, plus per-symbol breakdowns and the
**quota panel**: how many AI calls were made, input/output tokens, success
rate, and estimated cost. A `~` before a token number means it is estimated,
not metered. Use it to check whether the AI bill matches your provider plan.

## 7. Settings

Four groups. Changes apply to the **next** candle — never mid-trade.

- **Strategy** — 7 knobs: min confidence, min risk/reward, fee rate, ATR
  bounds for SL/TP, funding cap, trend-bias. Higher confidence = fewer but
  pickier signals.
- **Demo** — balance, margin, leverage, risk %, fee. **Locked while any
  position is OPEN** (the server refuses with HTTP 409). Reset restores the
  saved initial balance and clears positions/trades, but keeps signal history.
- **AI provider** — Ollama or an OpenAI-compatible API (DeepSeek, OpenRouter,
  custom). Changing provider while a signal is active **stages** the change
  and applies it when the system goes idle. Test buttons verify the
  connection before saving.
- **Telegram** — bot token + two chat IDs: signals channel (opens/closes)
  and reports channel (daily P&L at 07:00 +07). One **Test** button per
  channel.
- **Symbols** — enable/disable symbols. A symbol holding an OPEN position
  cannot be disabled until the position closes.

## Golden rules

1. **One position per symbol.** While OPEN: no new AI analysis, no second
   position, no settings edits that affect it.
2. **Entry / TP / SL never change** once a position is OPEN. The chart lines
   are fixed until close.
3. **WAIT creates nothing.** It just returns the system to idle.
4. **TP/SL monitoring never sleeps** — not on FULL portfolio, not during
   event blackouts, not after a restart.
5. **Nothing here spends real money.** If a number looks wrong, it is a
   display or demo-math issue — check the Daemon Log and Trades tabs first.

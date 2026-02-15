# ALPHA-PRIME v2.0 - Daily Startup Checklist  
**From computer on → live trading in 5–10 minutes**

Last updated: February 11, 2026  
Optimized for: Indian market hours (9:15 AM – 3:30 PM IST)

***

## Pre-Market Checklist (Before 9:00 AM IST) – 1 minute

### Environmental Check (1 minute)

- [ ] Laptop plugged into power
- [ ] Internet stable – run:

  ```bash
  ping 8.8.8.8
  ```

  Confirm you see multiple `Reply from 8.8.8.8` lines (no `Request timed out`). [keepyourhomeip](https://keepyourhomeip.com/blogs/news/what-is-ping-and-how-to-use-it-to-troubleshoot-your-internet-connectivity)

- [ ] System clock correct (IST) – run:

  ```bash
  date
  ```

- [ ] No pending OS updates that might auto‑restart
- [ ] Trading workspace clear, no distractions

***

## Phase 1: Start Core Services – 2 minutes

### Step 1.1 – Start PostgreSQL (Database) – 10 seconds

**macOS:**

```bash
brew services start postgresql@14
brew services list | grep postgresql
# Confirm: status shows "started"
```

**Linux (Ubuntu/Debian):**

```bash
sudo systemctl start postgresql
sudo systemctl status postgresql
# Confirm: "active (running)" in output
```

**Windows:**

- [ ] Press `Win + R`, type `services.msc`, press Enter.
- [ ] Find **postgresql-x64-14** (or similar).
- [ ] Right‑click → **Start**.
- [ ] Status column should show **Running**.

- [ ] PostgreSQL status: **RUNNING** ✓

***

### Step 1.2 – Start Redis (Cache) – 10 seconds

**macOS:**

```bash
brew services start redis
redis-cli ping
# Confirm: PONG
```

**Linux (Ubuntu/Debian):**

```bash
sudo systemctl start redis
redis-cli ping
# Confirm: PONG
```

**Windows:**

```bash
cd C:\Redis
start redis-server.exe redis.windows.conf
# New window opens – keep it running

redis-cli ping
# Confirm: PONG
```

- [ ] Redis status: **PONG** ✓

***

### Step 1.3 – Navigate to Project – 5 seconds

**macOS / Linux:**

```bash
cd ~/Documents/alpha-prime-v2
pwd
# Confirm: path ends with /alpha-prime-v2
```

**Windows (CMD/PowerShell):**

```bash
cd C:\Users\YourName\Documents\alpha-prime-v2
cd
# Confirm: path shows \alpha-prime-v2
```

- [ ] Correct project directory confirmed ✓

***

### Step 1.4 – Activate Virtual Environment – 5 seconds

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

- [ ] Success indicator: command prompt starts with **(venv)** ✓

***

## Phase 2: System Health Check – 1–2 minutes

### Step 2.1 – Run Health Check Script – 30 seconds

```bash
python scripts/health_check.py --quick
```

Expected key lines:

```text
✓ Database: Connected (...)
✓ Redis: Connected (PONG)
✓ Broker API: Authenticated
✓ Market Data: Live feed active
...
Status: READY TO TRADE
```

- [ ] Database: **Connected** ✓  
- [ ] Redis: **Connected (PONG)** ✓  
- [ ] Broker API: **Authenticated** ✓  
- [ ] Market data: **Live feed active** ✓  
- [ ] No `❌ Critical` lines ✓

**If any ❌ Critical:**

```bash
tail -n 50 logs/alpha_prime.log
```

- [ ] Fix cause (DB/Redis/broker/market data), rerun health check ✓

***

### Step 2.2 – Verify Broker Connection – 30 seconds

```bash
python scripts/broker_test.py --verify
```

Expected key lines:

```text
✓ Zerodha login successful
✓ Account: AB1234
✓ Available margin: ₹x,xx,xxx
✓ Active orders: 0
✓ Open positions: 0
✓ Order placement: TEST OK
```

- [ ] Login successful ✓  
- [ ] Margin available ✓  
- [ ] Active orders as expected (usually 0 pre‑market) ✓  
- [ ] Open positions as expected (usually 0 intraday start) ✓

***

### Step 2.3 – Check Market Status – 10 seconds

```bash
python -c "from core.market_hours import is_market_open; print('Market Open' if is_market_open() else 'Market Closed')"
```

- [ ] If before 9:15 AM → expect **Market Closed** ✓  
- [ ] After 9:15 AM → expect **Market Open** ✓

***

## Phase 3: Risk Configuration Check – 1 minute

### Step 3.1 – Verify Risk Limits – 30 seconds

```bash
python scripts/show_config.py --risk
```

Verify on screen:

- [ ] Daily Max Loss: matches your daily risk (e.g. 3–5% of capital) ✓  
- [ ] Max Drawdown: within prop firm / broker rules ✓  
- [ ] Per‑Trade Risk: around 0.25–0.5% of capital ✓  
- [ ] Max Open Positions: within your comfort + rule set ✓  
- [ ] Kill Switch Drawdown: enabled and sensible ✓  

**If values wrong:**

```bash
python scripts/configure_risk.py --interactive
# or edit:
nano config/risk_config.yaml   # macOS/Linux
notepad config\risk_config.yaml  # Windows
```

- [ ] Risk config corrected and saved ✓

***

### Step 3.2 – Verify Strategy Configuration – 20 seconds

```bash
python scripts/show_config.py --strategies
```

Check:

- [ ] Only intended strategies show `[ENABLED]` ✓  
- [ ] No test / experimental strategies enabled ✓  
- [ ] Symbol lists match what you actually want to trade ✓  
- [ ] Position limits per strategy look correct ✓

***

## Phase 4: Start Trading Engine – 1 minute

### Step 4.1 – Start in Monitor Mode (Before 9:15 AM) – 30 seconds

```bash
python scripts/run_engine.py --mode monitor
```

Expected log snippets:

```text
INFO: ALPHA-PRIME v2.0 Engine Starting
INFO: Mode: MONITOR (dry-run)
INFO: Strategies loaded: X
INFO: Risk manager initialized
INFO: Market data feed: CONNECTED
INFO: Waiting for market open...
INFO: Engine ready (monitor mode)
```

- [ ] Engine in **MONITOR** mode visible ✓  
- [ ] `Strategies loaded: ...` ✓  
- [ ] `Market data feed: CONNECTED` ✓  
- [ ] No ERROR / CRITICAL lines on startup ✓

> Keep this terminal **open**. This is **Engine Terminal**.

***

### Step 4.2 – Switch to Live at 9:15 AM – 30 seconds

Open **another** terminal for command control:

- [ ] Navigate + activate `venv` again:

  ```bash
  cd ~/Documents/alpha-prime-v2          # macOS/Linux
  # or
  cd C:\Users\YourName\Documents\alpha-prime-v2   # Windows

  source venv/bin/activate               # macOS/Linux
  venv\Scripts\activate                  # Windows
  ```

- [ ] Switch to live:

  ```bash
  python scripts/run_engine.py --mode live --confirm
  ```

Prompt example:

```text
⚠️  WARNING: Switching to LIVE mode
This will place REAL trades with REAL money.

Type 'CONFIRM' to proceed: CONFIRM
```

Expected logs:

```text
INFO: Switching to LIVE mode
INFO: Risk limits verified
INFO: ✓ LIVE TRADING ACTIVE
INFO: Monitoring for signals...
```

- [ ] `LIVE TRADING ACTIVE` message seen in logs ✓  
- [ ] No immediate errors ✓

> Keep this command terminal open if engine uses it; otherwise, you can reuse it as **Command Terminal**.

***

## Phase 5: Dashboard & Monitoring – 1 minute

### Step 5.1 – Start Dashboard – 30 seconds

Open a **third** terminal:

```bash
cd ~/Documents/alpha-prime-v2          # macOS/Linux
# or
cd C:\Users\YourName\Documents\alpha-prime-v2  # Windows

source venv/bin/activate               # macOS/Linux
venv\Scripts\activate                  # Windows

python -m dashboard.app_v2
```

Expected output:

```text
INFO: Starting ALPHA-PRIME v2.0 Dashboard
INFO: WebSocket server: ws://localhost:8001
INFO: HTTP server: http://localhost:8000
INFO: Press CTRL+C to quit
```

- [ ] Dashboard server running, no ERROR / CRITICAL ✓

> This is **Dashboard Terminal** – keep it open.

***

### Step 5.2 – Open Dashboard in Browser – 30 seconds

- [ ] Open browser → go to `http://localhost:8000`
- [ ] Confirm:

  - [ ] Portfolio value visible ✓  
  - [ ] Today’s P&L: ₹0.00 (at start) ✓  
  - [ ] Status indicator: **TRADING ACTIVE** / similar green label ✓  
  - [ ] Strategies listed with **Running/Enabled** ✓  
  - [ ] Open positions: 0 (or expected) ✓  
  - [ ] No red error banner ✓

***

### Step 5.3 – Enable Real-Time Alerts (Optional) – 30 seconds

From **Command Terminal** (with `venv` active):

```bash
# Telegram alerts (critical only)
python scripts/enable_alerts.py --telegram --critical-only

# Desktop/terminal alerts (if supported)
python scripts/enable_alerts.py --desktop
```

- [ ] Alert channels enabled ✓

***

## Phase 6: Final Pre‑Trading Verification – 1 minute

### “Am I Ready?” Checklist

**Infrastructure**

- [ ] **Engine Terminal** open – logs scrolling, no errors ✓  
- [ ] **Dashboard Terminal** open – dashboard running ✓  
- [ ] **Command Terminal** free for commands ✓  
- [ ] Dashboard page visible in browser ✓  
- [ ] Internet still stable (optional re‑check):

  ```bash
  ping 8.8.8.8
  ```

**Risk & Compliance**

- [ ] Daily loss limit correct for today ✓  
- [ ] Per‑trade risk acceptable ✓  
- [ ] Only tested strategies enabled ✓  
- [ ] No manual override / manual hedge scripts running ✓  

**Market Conditions**

- [ ] Market open and liquid ✓  
- [ ] No major scheduled news in next 30 minutes (check calendar) ✓  
- [ ] Spreads normal on broker platform ✓  
- [ ] India VIX not at extreme level (per your rules) ✓  

**Personal**

- [ ] Well‑rested and focused ✓  
- [ ] No urgent tasks in next 2 hours ✓  
- [ ] Phone on silent or DND ✓  
- [ ] Emergency stop procedure clear in mind ✓  

- [ ] **READY TO TRADE** ✓

***

## End‑of‑Day Shutdown (3:30 PM onwards) – 5 minutes

### Step 7.1 – Stop New Trades (≈3:25 PM) – 30 seconds

From **Command Terminal**:

```bash
python scripts/run_engine.py --mode stop-new-trades
```

Expected:

```text
INFO: New trade entries disabled
INFO: Existing positions will be managed to exit according to rules
```

- [ ] New trades stopped ✓  
- [ ] Engine still running, managing open positions ✓

***

### Step 7.2 – Close All Positions (≈3:28 PM, Intraday Only) – 1 minute

If you want **no overnight positions**:

```bash
python scripts/close_all_positions.py --market-order --confirm
```

Prompt example:

```text
⚠️  This will close ALL open positions at MARKET price

Type 'CLOSE' to proceed: CLOSE
```

- [ ] All positions closed ✓  
- [ ] No pending orders ✓  
- [ ] Verified on broker platform ✓

***

### Step 7.3 – Generate Daily Report – 1 minute

```bash
python scripts/generate_report.py --today --detailed
```

Expected:

```text
📊 Daily Trading Report - YYYY-MM-DD
...
Report saved: reports/daily_YYYY-MM-DD.pdf
```

- [ ] Report generated ✓  
- [ ] P&L matches broker statement ✓  
- [ ] No anomalies in trade list ✓

***

### Step 7.4 – Backup Data – 30 seconds

```bash
python scripts/backup.py --daily --compress
```

- [ ] Backup completed (backup file created in backups folder) ✓

***

### Step 7.5 – Stop All Services – 1–2 minutes

**1. Stop Dashboard**

- [ ] In **Dashboard Terminal**: press `CTRL + C`  
  Confirm no more dashboard logs.

**2. Stop Engine**

- [ ] In **Engine Terminal**: press `CTRL + C`  
  Confirm engine stops and returns to shell prompt.

**3. Deactivate Virtual Environment**

From any terminal with `(venv)`:

```bash
deactivate
```

- [ ] Prompt no longer shows `(venv)` ✓

**4. Optional – Stop Redis**

- macOS:

  ```bash
  brew services stop redis
  ``` [danielabaron](https://danielabaron.me/blog/homebrew-postgresql-service-not-starting-resolved/)

- Linux:

  ```bash
  sudo systemctl stop redis
  ```

- Windows:

  - [ ] Close the `redis-server.exe` window.

**5. Optional – Stop PostgreSQL**

- macOS:

  ```bash
  brew services stop postgresql@14
  ``` [stackoverflow](https://stackoverflow.com/questions/7975556/how-can-i-start-postgresql-server-on-mac-os-x)

- Linux:

  ```bash
  sudo systemctl stop postgresql
  ```

- Windows:

  - [ ] In `services.msc`, right‑click PostgreSQL service → **Stop**.

- [ ] Dashboard stopped ✓  
- [ ] Engine stopped ✓  
- [ ] Environment deactivated ✓  
- [ ] Services stopped if desired ✓

***

### Step 7.6 – Daily Journal (Optional but Recommended) – 2 minutes

```bash
nano logs/daily_journal.txt         # macOS / Linux
notepad logs\daily_journal.txt      # Windows
```

Log briefly:

- [ ] Any manual interventions ✓  
- [ ] Unusual market conditions ✓  
- [ ] Strategy performance notes ✓  
- [ ] Issues & ideas ✓  

***

## Emergency Procedures

### 🚨 Emergency Stop – Immediate Action

**Trigger:** something looks very wrong – runaway losses, repeated errors, strange trades.

1. **Stop Engine Immediately**

   - [ ] In **Engine Terminal**: press `CTRL + C`.

2. **Check Pending Orders**

   ```bash
   python scripts/check_pending_orders.py
   ```

   - [ ] Confirm no unknown/unwanted orders ✓

3. **Flatten All Positions (if needed)**

   ```bash
   python scripts/emergency_flatten.py --all --market
   ```

   - [ ] Confirm all positions closed on broker platform ✓

4. **Check Damage**

   ```bash
   python scripts/show_pnl.py --today
   ```

5. **Review What Happened**

   ```bash
   tail -n 200 logs/alpha_prime.log
   ```

   - [ ] Understand cause before restarting ✓

***

### 🔴 Kill Switch Activated (Automatic)

If built‑in risk kill switch triggers, logs may show:

```text
CRITICAL: KILL SWITCH ACTIVATED
CRITICAL: Daily loss limit reached: -₹x,xxx
INFO: All strategies paused
INFO: Closing open positions...
INFO: Engine entering SAFE mode
```

When you see this:

- [ ] Do **NOT** restart engine immediately ✓  
- [ ] Confirm all positions closed (broker platform + `show_pnl.py`) ✓  
- [ ] Review logs to understand cause ✓  
- [ ] Decide if trading should stop for the day ✓  
- [ ] If restarting another day, adjust risk or disable problem strategy ✓

***

### 🟡 Broker Connection Lost

From **Command Terminal**:

```bash
python scripts/reconnect_broker.py
```

If still down:

- [ ] Log into broker web platform, manually manage open positions ✓  
- [ ] Do **not** restart engine until connection stable ✓  

Optional quick status check (example Zerodha):

```bash
# Just an example; you can also open status page in browser
curl -I https://kite.zerodha.com/
```

***

## Quick Reference Card (Print This)

```text
┌─────────────────────────────────────────┐
│ ALPHA-PRIME Daily Quick Start           │
├─────────────────────────────────────────┤
│ 1. Start services:                      │
│    # macOS                              │
│    brew services start postgresql@14    │
│    brew services start redis            │
│                                         │
│    # Linux                              │
│    sudo systemctl start postgresql      │
│    sudo systemctl start redis           │
│                                         │
│ 2. Activate env:                        │
│    cd ~/Documents/alpha-prime-v2        │
│    source venv/bin/activate             │
│                                         │
│ 3. Health check:                        │
│    python scripts/health_check.py --quick│
│                                         │
│ 4. Start engine (monitor):              │
│    python scripts/run_engine.py         │
│      --mode monitor                     │
│                                         │
│ 5. Start dashboard:                     │
│    python -m dashboard.app_v2           │
│    Open: http://localhost:8000          │
│                                         │
│ 6. At 9:15 AM, go live:                 │
│    python scripts/run_engine.py         │
│      --mode live --confirm              │
│                                         │
│ 7. End of day (3:30 PM):                │
│    - stop new trades                    │
│    - close positions (if intraday)      │
│    - generate report & backup           │
│    - CTRL+C engine + dashboard          │
│                                         │
│ EMERGENCY STOP:                         │
│  - CTRL+C engine                        │
│  - emergency_flatten.py --all --market  │
└─────────────────────────────────────────┘
```

***

## Time Budget Summary

| Phase                           | Duration (approx) | Cumulative |
|---------------------------------|-------------------|------------|
| Pre‑market checks               | 1 min             | 1 min      |
| Start services (DB + Redis)     | 2 min             | 3 min      |
| Health checks                   | 2 min             | 5 min      |
| Risk & strategy verification    | 1 min             | 6 min      |
| Start engine (monitor → live)   | 1 min             | 7 min      |
| Dashboard & alerts              | 1–2 min           | 8–9 min    |
| Final verification              | 1 min             | 9–10 min   |

With practice: **5–7 minutes total**.

***

## Personal Notes (Fill Once, Then Reuse)

**My typical daily routine:**

- [ ] Start checklist at: `_______`
- [ ] Coffee while services boot: ☕
- [ ] Quick NIFTY / BANKNIFTY pre‑market scan ✓
- [ ] Review yesterday’s report in `reports/` ✓
- [ ] Adjustments for today (news, events): `________________________`

**Reminders:**

- [ ] Friday: run **weekly backup** (`python scripts/backup.py --weekly`) ✓  
- [ ] Month‑end: run **performance review** ✓  
- [ ] Check API key expiry dates: `________________________` ✓
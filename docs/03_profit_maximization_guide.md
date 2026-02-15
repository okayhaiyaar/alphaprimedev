# ALPHA-PRIME v2.0 – Profit Maximization & Funded Account Master Guide  
**Turn disciplined execution into consistent profits**

Last updated: February 11, 2026

***

## Philosophy: The Path to Consistent Profits

### Core Truth

**You don't get rich from one big trade. You get rich from not blowing up.**

The ALPHA-PRIME system is built to execute a statistical **edge** with machine-like discipline. Your edge only matters if you:

1. Configure risk correctly  
2. Only run **proven** strategies  
3. Let the system work without emotional interference  
4. Scale up **slowly** and systematically  

This guide is your **operating manual** to convert ALPHA-PRIME’s edge into **consistent income**, while protecting your funded accounts from catastrophic loss.

***

## Part 1: Prop Firm Challenge Strategy

### 1.1 Know Your Prop Firm Rules (Fill This First)

**Typical prop firm rules (example, you MUST confirm your own):**

| Rule Type              | Typical Limit     | Your Firm Limit |
|------------------------|-------------------|-----------------|
| Daily Loss Limit       | 5% of account     | ____%           |
| Max Total Drawdown     | 10% of account    | ____%           |
| Max Trailing Drawdown  | 5% from peak      | ____%           |
| Min Trading Days       | 5–10 days         | ____ days       |
| Max Trading Days       | 30–60 days        | ____ days       |
| Profit Target          | 8–10%             | ____%           |

**Before doing ANYTHING:**

- [ ] Fill your firm’s exact rules in the table above  
- [ ] Write the **actual rupee amounts** for daily loss and max drawdown  
- [ ] Keep those numbers visible next to your screen  

***

### 1.2 The “Bulletproof Pass” Risk Settings

For challenges, **you never push the firm limits**. You operate at **40% of what they allow**.

**Recommended `config/risk_config.yaml` for prop challenges:**

```yaml
risk_limits:
  # USE ONLY ~40% OF FIRM'S LIMITS

  # If firm daily loss limit = 5%, set:
  daily_max_loss: 2.0        # 2% of account

  # If firm max total DD = 10%, set:
  total_max_drawdown: 4.0    # 4% of account

  # Per-trade risk (CRITICAL)
  per_trade_risk: 0.25       # 0.25% of account per trade (NEVER exceed 0.5%)

  # Position limits
  max_open_positions: 3
  max_correlated_positions: 1

  # Kill switch (hard stop BEFORE limits)
  kill_switch_daily_loss: 1.8    # auto-stop day before 2% reached
  kill_switch_total_dd: 3.5      # auto-stop account before 4% reached
```

**Why this works:**

- 40% of firm limits leaves buffer for **slippage, spreads, gaps, errors**  
- Hitting your internal limit **never** equals blowing the account  
- You can have a **bad day** and still stay in the game  

***

### 1.3 Strategy Selection for Challenges

Rule: **Run ONLY 1–2 uncorrelated, stable strategies** during evaluation.

| Strategy Type      | Win Rate (typical) | Risk:Reward | Max DD | Use in Challenge? |
|--------------------|--------------------|------------|--------|-------------------|
| Mean Reversion     | 60–70%             | 1:1.5      | 3–5%   | ✅ Primary        |
| Breakout Scalper   | 55–65%             | 1:1.2      | 5–8%   | ✅ Secondary      |
| Trend Following    | 40–50%             | 1:2+       | 10%+   | ❌ Too volatile   |
| Swing / Position   | 45–55%             | 1:3+       | 15%+   | ❌ Too slow       |

**Configuration example (challenge mode):**

```bash
# Enable ONLY proven, stable strategies
python scripts/configure_strategies.py \
  --enable MeanReversion_NIFTY50 \
  --enable Scalper_BANKNIFTY \
  --disable-all-others
```

**Verify:**

```bash
python scripts/show_config.py --strategies
# Expect: Exactly 1–2 strategies ENABLED
```

***

## Part 2: Risk Management (The Foundation)

### 2.1 Non-Negotiable Rules

#### Rule #1 – Fixed Fractional Position Sizing

Formula:

```text
Position Size = (Account Size × Risk %) / Stop Loss (points × value/point)
```

Example (NIFTY futures):

- Account: ₹1,00,000  
- Risk per trade: 0.25% → ₹250  
- Stop loss: 50 points, value = ₹5/point  

```text
Risk amount = 100,000 × 0.0025 = ₹250
Loss per lot at SL = 50 × 5 = ₹250
Position = 1 lot
```

**Rules:**

- Always size via formula, **never** via “1 lot looks fine”.  
- Never risk more than **0.25%** during challenge; absolute max **0.5%** even later.

***

#### Rule #2 – Daily Loss Circuit Breaker

**3-Strike Rule:**

- After **2 consecutive losing trades** → reduce risk to **50%** for next 3 trades  
- After **3 consecutive losses** → **stop trading for the day**  
- After **50% of daily loss limit** hit → **no new trades**, only manage open trades  

**Enable in ALPHA-PRIME:**

```bash
python scripts/configure_risk.py --enable-loss-circuit-breaker \
  --consecutive-losses 3 \
  --half-daily-limit-action pause
```

***

#### Rule #3 – Correlation Limits

You treat NIFTY, BANKNIFTY, and a basket of NIFTY-heavy stocks as **one risk cluster**.

- Never hold more than **1 correlated index exposure** at a time  
- Multiple stocks in same sector = correlated → count as 1  
- Long equity + short India VIX = correlated

**Enforcement:**

```bash
python scripts/configure_risk.py \
  --max-correlated-positions 1 \
  --correlation-threshold 0.7
```

***

#### Rule #4 – Time-Based Risk Reduction

Avoid structurally dangerous times:

| Time Window      | Multiplier | Rule                      |
|------------------|-----------:|---------------------------|
| 9:15–9:30 AM     |    0.5×    | Half size (open volatility) |
| 12:00–1:00 PM    |    0×      | No new trades (illiquid) |
| 3:15–3:30 PM     |    0.5×    | Half size (close volatility) |
| Major news/RBI   |    0×      | No trades                 |

**Configure:**

```bash
python scripts/configure_risk.py --enable-time-filters \
  --avoid-open-close \
  --block-news-events
```

***

### 2.2 Position Sizing Frameworks

You will **not** actually trade full Kelly, but you should understand it.

**Kelly formula:**

```text
Kelly % = W - [(1 - W) / R]
```

- W = win rate  
- R = avg win / avg loss  

Example: W = 0.60, R = 1.5 → Kelly ≈ 33%.  
You never go above **25–50% of Kelly**. During challenges, you ignore Kelly and stick to **fixed 0.25%**.

**Your default:**

- [ ] **Fixed fractional 0.25% risk per trade** (prop challenge & early funded)  

***

### 2.3 Drawdown Management

**Stages:**

| Drawdown vs Equity Peak | Action                |
|-------------------------|----------------------|
| 0–1%                    | Normal               |
| 1–2%                    | Review trades        |
| 2–3%                    | Size to 75%         |
| 3–4%                    | Size to 50%         |
| 4–5%                    | Stop new trades      |
| >5%                     | Full stop + review   |

**Configure drawdown stages:**

```bash
python scripts/configure_risk.py --drawdown-stages \
  --stage1-threshold 2 --stage1-action reduce_75 \
  --stage2-threshold 3 --stage2-action reduce_50 \
  --stage3-threshold 4 --stage3-action pause \
  --stage4-threshold 5 --stage4-action stop
```

***

## Part 3: Backtesting for Profitability

### 3.1 What to Backtest (Beyond Just Returns)

Run a **full backtest** before any real money:

```bash
python scripts/backtest.py --strategy MeanReversion_NIFTY50 \
  --start 2023-01-01 \
  --end 2026-02-11 \
  --metrics all \
  --report detailed
```

**Critical metrics and thresholds:**

| Metric                | Goal               | Hard Stop (no trade) |
|-----------------------|--------------------|----------------------|
| Annualized Return     | > 20%              | < 8%                 |
| Max Drawdown          | < 8%               | > 10%                |
| Worst Single Day      | < 2%               | > 5%                 |
| Win Rate              | > 50%              | < 40%                |
| Profit Factor         | > 1.5              | < 1.2                |
| Consecutive Losses    | < 5                | > 8                  |
| Sharpe Ratio          | > 1.5              | < 1.0                |
| Max Daily Loss        | < 2%               | > 5%                 |

If any metric breaks the **Hard Stop** column → **do not trade that strategy live**.

***

### 3.2 Walk-Forward Optimization (No Curve Fitting)

Use **walk‑forward** instead of optimizing on all data:

```bash
python scripts/backtest.py --walk-forward \
  --train-period 365 \
  --test-period 90 \
  --step 30 \
  --strategy MeanReversion_NIFTY50
```

You want to see:

- Performance **consistent** across test segments  
- No single test segment with >5% loss  
- Out-of-sample metrics within **80–120%** of in-sample metrics  

***

### 3.3 Regime Testing

Test across **different market regimes**:

```bash
# Bullish period
python scripts/backtest.py --strategy MeanReversion_NIFTY50 \
  --start 2024-01-01 --end 2024-06-30 --regime bull

# Bearish period
python scripts/backtest.py --strategy MeanReversion_NIFTY50 \
  --start 2023-01-01 --end 2023-06-30 --regime bear

# Sideways period
python scripts/backtest.py --strategy MeanReversion_NIFTY50 \
  --start 2022-07-01 --end 2022-12-31 --regime sideways
```

**Rule:** strategy must be at least **breakeven** in all regimes, not just one.

***

### 3.4 Monte Carlo Stress Testing

```bash
python scripts/backtest.py --monte-carlo \
  --simulations 10000 \
  --strategy MeanReversion_NIFTY50
```

You want:

- 95th percentile **max drawdown** < firm limit  
- 5th percentile **return** ≥ 0%  
- Probability of hitting **profit target** (e.g. 8%) is reasonable (>60% is great)  

If Monte Carlo shows high probability of violation → tweak or reject.

***

## Part 4: Strategy Optimization

### 4.1 What to Optimize For

**Bad objective:**

```python
# ❌ Avoid
objective = maximize(total_return)
```

**Good objective:**

```python
# ✅ Better
objective = maximize(
    risk_adjusted_return
)  # e.g. return / (max_drawdown ** 2)
# subject to:
#   max_drawdown < 8
#   worst_day < 2
#   consecutive_losses < 5
```

Run:

```bash
python scripts/optimize_strategy.py --strategy MeanReversion_NIFTY50 \
  --objective risk_adjusted \
  --max-drawdown-constraint 8 \
  --max-daily-loss-constraint 2 \
  --test-robust
```

***

### 4.2 Catching Overfitting

**In-sample vs out-of-sample** comparison:

| Metric    | In-Sample | Out-of-Sample | Verdict     |
|-----------|-----------|---------------|-------------|
| Return    | 45%       | 42%           | ✅ Good      |
| Return    | 55%       | 12%           | 🚨 Overfit  |
| Sharpe    | 2.1       | 1.9           | ✅ Good      |
| Sharpe    | 3.5       | 0.8           | 🚨 Overfit  |
| Win rate  | 68%       | 64%           | ✅ Good      |
| Win rate  | 78%       | 51%           | 🚨 Overfit  |

**Rule:** out-of-sample metrics should be within **80–120%** of in-sample.

***

### 4.3 Volatility Filters

Avoid trading when volatility is too low or too insane:

```bash
python scripts/configure_strategy.py --strategy MeanReversion_NIFTY50 \
  --enable-volatility-filter \
  --min-volatility 0.5 \
  --max-volatility 2.5 \
  --volatility-lookback 20
```

Concept:

- Measures 20-day rolling volatility / VIX-like measure  
- Pauses strategy if vol < 0.5× average (dead) or > 2.5× average (insane)  

***

## Part 5: Daily Execution Rules

### 5.1 Pre-Market Routine (8:45–9:15 AM)

- [ ] Check global markets (US, Asia) – general trend  
- [ ] Check India VIX:
  - VIX > 25 → **half size**  
  - VIX > 35 → **no trades**  
- [ ] Check calendar (RBI, CPI, GDP, budget days)  
- [ ] Quick NIFTY / BANKNIFTY chart scan – any obvious regime shift?  
- [ ] Review yesterday’s ALPHA‑PRIME report (drawdown, biggest loss)  
- [ ] Confirm margin and leverage are fine  
- [ ] Check broker status page, no ongoing issues  

If **multiple red flags** → reduce size or skip day.

***

### 5.2 Intraday Monitoring

**Frequency:**

- 9:15–9:45 → every 10 minutes  
- 9:45–3:00 → every 30 minutes  
- 3:00–3:30 → every 10 minutes  

**Checks:**

- [ ] Any position > 1% loss? Is stop correct?  
- [ ] Daily P&L near 50% of daily limit → prepare to stop new trades  
- [ ] Any broker rejections / margin warnings?  
- [ ] Any red alert on dashboard?  

**Intervene only when:**

- System or broker error  
- Risk limit misbehavior  
- Connection or data feed broken  
- Extreme news (RBI, circuit breakers, war, etc.)  

**Do NOT intervene when:**

- Trade is losing but stop intact  
- Strategy skipped a trade you “felt” was good  
- Daily P&L slightly negative within limits  

***

### 5.3 End-of-Day Review (After 3:30 PM)

Run:

```bash
python scripts/generate_report.py --today
```

Review:

- Daily P&L (as % of equity)  
- Largest win / largest loss  
- Strategy contribution breakdown  
- Any approach/near miss of risk limits  
- Execution issues (slippage, rejections)  

If anything seems out of normal range → note in journal and investigate the next morning.

***

## Part 6: Funded Account Management (Post-Challenge)

### 6.1 Transition from Challenge to Funded

**Do not** instantly double risk when you pass.

| Weeks Funded | Per-Trade Risk | Max Positions | Daily Max Loss | Notes             |
|--------------|----------------|---------------|----------------|-------------------|
| 1–2          | 0.25%          | 3             | 2%             | Same as challenge |
| 3–4          | 0.30%          | 4             | 2.5%           | Slight increase   |
| 5–8          | 0.35%          | 4             | 3%             | If all good       |
| 9+           | 0.40%          | 5             | 3.5%           | Stable regime     |

Never go beyond **0.5% risk per trade**, even fully funded.

***

### 6.2 Scaling Rules

Scaling formula (slow and conservative):

```text
New Risk % = Base Risk % × (1 + Cumulative Return / 20)
```

Example:

- Start: 0.25%, account ₹1,00,000  
- After +20% → ₹1,20,000  
- New risk = 0.25% × 1.20 ≈ 0.30%  

**Hard rule:** never increase risk more than **0.05% per month**.

***

### 6.3 Withdrawal Policy

**Monthly rule of thumb:**

- Withdraw **50–70% of net monthly profit**  
- Leave **30–50%** to compound slowly  

Example:

- Start: ₹1,00,000  
- End of month: ₹1,08,000 (profit ₹8,000)  
- Withdraw: ₹5,000  
- New trading balance: ₹1,03,000  

This protects profits from future drawdown and gives real cash flow.

***

## Part 7: Common Ways to Blow Up (Avoid These)

### 7.1 Fatal Mistakes

1. **Revenge Trading**

- Losing day → doubling size next day to “make it back”  
- Fix: always follow fixed fractional risk, unaffected by yesterday.

2. **Override Syndrome**

- System says “no trade” → you manually override and trade anyway  
- Fix: hard rule – **no manual orders** on challenge/funded accounts.

3. **Martingale / Averaging Down**

- Adding position to a loser instead of taking the stop  
- Fix: never add size to a losing trade; only pyramid **winners** if system allows.

4. **Over-Optimization**

- Strategy with 90% win rate in backtest but collapses live  
- Fix: require walk-forward, regime tests, Monte Carlo before deployment.

5. **Ignoring Correlation**

- Stacking NIFTY, BANKNIFTY, and heavy NIFTY stocks all same direction  
- Fix: enforce `max_correlated_positions = 1` and obey it.

***

### 7.2 Psychological Traps

- **Profit Euphoria:** multiple green days → temptation to increase risk quickly  
- **Drawdown Panic:** normal DD → urge to scrap strategies and “start fresh”  
- **“Just This Once” Override:** biggest killer; you violate rules one time, then again  

**Anti-dote:** write your red lines (below) and **never** cross them.

***

## Part 8: Performance Tracking & Analytics

### 8.1 Daily Metrics

```bash
python scripts/show_performance.py --today
```

Track:

- [ ] Daily P&L (%)  
- [ ] Win rate vs backtest expectation  
- [ ] Avg win vs avg loss  
- [ ] Max intraday drawdown  
- [ ] Slippage vs benchmarks  
- [ ] Number of trades vs signals  

***

### 8.2 Weekly Review (e.g. Sunday)

```bash
python scripts/generate_report.py --week --detailed
```

Answer:

- Weekly return compared to target  
- Which strategies performed best / worst and why  
- Any consistent underperformers  
- Any risk rule violations  
- Did you intervene manually, how often, outcome?  

Actions:

- Disable strategy if underperforming for >3 weeks vs backtest  
- Reduce size if risk limits frequently approached  
- Create solutions for repeated execution issues  

***

### 8.3 Monthly Performance

```bash
python scripts/generate_report.py --month --pdf --detailed
```

Compare month vs targets:

- Total return  
- Max drawdown  
- Worst day  
- Win rate  
- Profit factor  
- Sharpe ratio  
- Trade count  

Use this to decide:

- Scale up / maintain / scale down  
- Add another strategy or hold  
- Tighten risk if drawdown creeping up  

***

## Part 9: Market Conditions & Adaptation

### 9.1 Detecting the Current Regime

```bash
python scripts/detect_regime.py --current
```

Example output:

```text
Current Market Regime: Choppy/Range
Confidence: 78%

Recommendation:
- Enable: MeanReversion strategies
- Disable: Trend following
- Risk adjustment: Normal (1.0×)
```

Your adjustments:

- Trend regime → favor trend / breakout, slightly higher risk  
- Choppy regime → favor mean reversion, normal risk  
- High vol regime → reduce size 50%, fewer trades  

***

### 9.2 Seasonal Patterns (India Focus, Just Awareness)

Not trading advice, but patterns to note:

- March (FY end) → volatility, partial size reduction  
- Budget / RBI days → system pause or strict filters  
- Festive season → often strong trending phases  

Use as **context**, never as an excuse to override rules.

***

## Part 10: My Personal Playbook (Fill This)

**Fill this once, then update monthly.**

### 10.1 My Prop Firm Rules

- Firm name: __________________________  
- Account size: ₹_____________________  
- Daily loss limit: ______% = ₹________  
- Max total drawdown: ______% = ₹______  
- Profit target: ______% = ₹___________  
- Min trading days: ______  
- Evaluation period: ______ days  

***

### 10.2 My Risk Configuration

```text
# My actual live settings (write real numbers)

per_trade_risk:        _____ %
daily_max_loss:        _____ %
max_drawdown:          _____ %
max_positions:         _____
max_correlated:        _____
kill_switch_daily:     _____ %
kill_switch_total:     _____ %
```

***

### 10.3 My Enabled Strategies

**Challenge phase:**

- Primary: __________________________  
- Secondary: ________________________  

All others: **DISABLED**

**After 3+ profitable funded months:**

- Additional strategy candidates: ______________________  

***

### 10.4 My No-Trade Times

- [ ] 9:15–9:25 AM (open)  
- [ ] ____________________  
- [ ] ____________________  
- [ ] RBI policy days  
- [ ] ____________________  

***

### 10.5 My Daily Routine

**Pre-market (8:45–9:15):**

- [ ] Run `02_daily_startup_checklist`  
- [ ] Global & VIX check  
- [ ] Review yesterday’s report  

**Intraday:**

- [ ] Dashboard check every ______ minutes  

**End-of-day (after 3:30):**

- [ ] Close positions (if intraday)  
- [ ] Generate report & backup  
- [ ] Journal key notes  

***

### 10.6 My Red Lines (Never Cross)

- Never trade if India VIX > ______  
- Never exceed ______ open positions  
- Never risk more than ______% per trade  
- Never trade during ____________________  
- Never override system stop losses  
- Never add to losing positions  
- Never trade when angry / tired / tilted  

***

### 10.7 My Monthly Targets

**This month (__/__/2026):**

- Target return: _____%  
- Max acceptable drawdown: _____%  
- Minimum win rate: _____%  
- Target trade count: _____  
- Key focus (process): ______________________________  

***

### 10.8 My Mistake Log

Keep a simple table in a journal file:

```text
Date       | What I Did Wrong                | Loss/Impact | Lesson | Prevention
-----------+----------------------------------+------------+--------+----------
YYYY-MM-DD | e.g. manually closed winner ... | -₹____      | ...    | ...
```

***

## Part 11: Advanced Topics

### 11.1 Adding a New Strategy Safely

1. **Backtest deeply (min 3 years):**

```bash
python scripts/backtest.py --strategy NewStrategy \
  --start 2023-01-01 \
  --metrics all \
  --walk-forward \
  --monte-carlo
```

2. **Paper trade for at least 30 days:**

```bash
python scripts/run_engine.py --mode paper \
  --strategy NewStrategy \
  --duration 30
```

3. **If live:** start at **10% of normal size**:

```bash
python scripts/configure_strategy.py --strategy NewStrategy \
  --enable \
  --size-multiplier 0.1
```

4. Gradually scale to 100% over ~2 months, only if results match expectations.

***

### 11.2 Multi‑Strategy Portfolio

Goal: combine **low‑correlated** strategies.

Example allocation:

```text
Strategy A (MeanReversion): 40% of risk budget
Strategy B (Breakout):      30% of risk budget
Strategy C (Momentum):      30% of risk budget
```

Check correlations:

```bash
python scripts/analyze_strategies.py --correlation-matrix --strategies all
```

**Rule:** keep pairwise correlations < **0.5** where possible.

***

## Summary: ALPHA-PRIME Golden Rules (Print This)

```text
╔═══════════════════════════════════════════════════╗
║             ALPHA-PRIME GOLDEN RULES              ║
╠═══════════════════════════════════════════════════╣
║ 1. Never risk more than 0.25% per trade          ║
║ 2. Never exceed your daily loss limit            ║
║ 3. Never override the system's stops             ║
║ 4. Never add to losing positions                 ║
║ 5. Never trade on emotion or tilt                ║
║ 6. Never change parameters mid-challenge         ║
║ 7. Never ignore correlation limits               ║
║ 8. Never trade high VIX without reducing size    ║
║ 9. Never revenge trade after a loss              ║
║ 10. Never forget: SURVIVAL > PROFITS             ║
║                                                   ║
║ WHEN IN DOUBT: DO NOTHING                        ║
║ Preservation of capital is JOB #1                ║
╚═══════════════════════════════════════════════════╝
```

***

## Final Word

ALPHA-PRIME v2.0 can become a **printing press** for you **only if**:

- Your **risk config** is conservative and aligned with prop rules  
- Your **strategies** are genuinely robust, not curve‑fit  
- Your **execution** is fully delegated to the system  
- Your **discipline** is stronger than your emotions  

Most traders fail not because their systems are bad, but because they **cannot follow their own rules**.

You now have:

- A challenge‑ready risk template  
- A process to validate strategies  
- Daily and weekly execution rules  
- A personal playbook section to hard‑code your decisions  

The edge is there.  
Your job now is **boring, relentless execution**.
# ALPHA-PRIME v2.0 - Quick Start Installation  
**Get from zero → running system in 30 minutes**

Last updated: February 11, 2026

***

## What You Need First

### Hardware / Software Requirements

- [ ] Computer with **Windows 10/11**, **macOS 10.15+**, or **Ubuntu 20.04+**
- [ ] At least **8 GB RAM** (more is better)
- [ ] At least **10 GB free disk space**
- [ ] **Stable internet connection**

### Accounts / Credentials You Need Ready

- [ ] **Broker account** (Zerodha / Upstox) OR **prop firm funded account**
- [ ] **Broker API credentials** (API Key, Secret, User ID, Password / PIN)
- [ ] **Email account** for alerts (Gmail works best)
- [ ] **Telegram account** (optional, for mobile alerts)

> Tip: If you don’t have API keys yet, you can still install everything and plug them in later in the `.env` file.

***

## Section 1: Install Prerequisites (≈15 minutes)

In this section you install the tools ALPHA-PRIME needs: Python (language), PostgreSQL (database), Redis (cache), Git (to download code).

***

### Step 1.1 – Install Python 3.11

**Why:** ALPHA-PRIME is written in Python. Python is the “language” the system speaks. [docs.python](https://docs.python.org/3.11/using/windows.html)

***

#### Windows

- [ ] Open your browser and go to:  
      `https://www.python.org/downloads/` [docs.python](https://docs.python.org/3.11/using/windows.html)
- [ ] Click the **big yellow button** that says **“Download Python 3.11.x”**  
      (If it shows 3.12, click “View all Python releases” and choose **3.11.x**.)
- [ ] Run the downloaded file `python-3.11.x-amd64.exe`
- [ ] On the first screen:
  - [ ] **CHECK** the box **“Add Python 3.11 to PATH”** (this is critical: it lets you run `python` from anywhere). [docs.python](https://docs.python.org/3/using/windows.html)
  - [ ] Click **“Install Now”**.
- [ ] When it finishes, close the installer.

**Verify Python:**

- [ ] Open **Command Prompt**:
  - Press `Win` key, type **cmd**, press Enter.
- [ ] Type (copy‑paste exactly):

  ```bash
  python --version
  ```

  You should see something like:

  ```text
  Python 3.11.x
  ```

**If you see:**

- `python is not recognized`  
  → Close Command Prompt, open it again, try `python --version` one more time.  
  → Still fails? Run the installer again and make sure **“Add Python to PATH”** is checked. [stackoverflow](https://stackoverflow.com/questions/6318156/adding-python-to-path-on-windows)

***

#### macOS

- [ ] Open **Terminal**:
  - Press `Cmd + Space`, type **Terminal**, press Enter.
- [ ] First install **Homebrew** (a tool that installs other tools), if you don’t have it:

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

  When it finishes, run:

  ```bash
  brew --version
  ```

  If you see a version number, Homebrew is ready. [dataquest](https://www.dataquest.io/blog/install-postgresql-14-7-for-macos/)

- [ ] Install Python 3.11:

  ```bash
  brew install python@3.11
  ```

- [ ] Verify:

  ```bash
  python3 --version
  ```

  You should see `Python 3.11.x`.

**If you see “command not found: brew”:**

- [ ] Run the Homebrew install command above again and follow on‑screen instructions. [dataquest](https://www.dataquest.io/blog/install-postgresql-14-7-for-macos/)

***

#### Linux (Ubuntu / Debian)

- [ ] Open **Terminal**.
- [ ] Update packages:

  ```bash
  sudo apt update
  ```

- [ ] Install Python 3.11 and tools:

  ```bash
  sudo apt install python3.11 python3.11-venv python3-pip
  ```

- [ ] Verify:

  ```bash
  python3 --version
  ```

  You should see `Python 3.11.x`.

***

### Step 1.2 – Install PostgreSQL (Database)

**Why:** ALPHA-PRIME stores orders, positions, trades, and history in a PostgreSQL database (think of it as a powerful spreadsheet that the system can query fast). [postgresql](https://www.postgresql.org/download/)

***

#### Windows

- [ ] Go to: `https://www.postgresql.org/download/windows/` [postgresql](https://www.postgresql.org/download/windows/)
- [ ] Click **“Download the installer”** and choose **PostgreSQL 14** (or 15 if 14 is not available; ALPHA-PRIME assumes 14+).
- [ ] Run the installer:
  - [ ] Keep all defaults (especially port **5432**).
  - [ ] **Set a password** for the **postgres** user and **write it down**. You will need this in `.env`.
- [ ] When finished, allow it to install **pgAdmin** as well (the GUI admin tool).

**Quick check:**

- [ ] Open **pgAdmin** from Start Menu.
- [ ] It should connect to `PostgreSQL 14` automatically or ask for the password you set.

***

#### macOS

- [ ] In **Terminal**, install PostgreSQL via Homebrew:

  ```bash
  brew install postgresql@14
  ```

- [ ] Start PostgreSQL:

  ```bash
  brew services start postgresql@14
  ```

- [ ] Verify:

  ```bash
  psql --version
  ```

  You should see something like `psql (PostgreSQL) 14.x`. [dataquest](https://www.dataquest.io/blog/install-postgresql-14-7-for-macos/)

**If `psql: command not found`:**

- [ ] Add PostgreSQL to PATH (only needed on some setups):

  ```bash
  echo 'export PATH="/usr/local/opt/postgresql@14/bin:$PATH"' >> ~/.zshrc
  source ~/.zshrc
  ```

- [ ] Try `psql --version` again. [dataquest](https://www.dataquest.io/blog/install-postgresql-14-7-for-macos/)

***

#### Linux (Ubuntu / Debian)

- [ ] In **Terminal**:

  ```bash
  sudo apt install postgresql postgresql-contrib
  ```

- [ ] Start PostgreSQL and enable it at boot:

  ```bash
  sudo systemctl start postgresql
  sudo systemctl enable postgresql
  ```

- [ ] Verify:

  ```bash
  psql --version
  ```

  You should see a PostgreSQL version.

***

### Step 1.3 – Install Redis (Cache)

**Why:** Redis is a super‑fast in‑memory store used for live prices, session info, and internal queues. [liquidweb](https://www.liquidweb.com/blog/install-redis-macos-windows/)

***

#### Windows

Redis on Windows is easiest using WSL (Linux on Windows) or the community build.

**Simple (dev) option – use Redis for Windows port:**

- [ ] Open browser: `https://github.com/microsoftarchive/redis/releases` [linkedin](https://www.linkedin.com/pulse/redis-in-depth-comprehensive-guide-setup-mac-windows-kumar-shanu)
- [ ] Download the latest **`Redis-x64`** ZIP.
- [ ] Extract to `C:\Redis`.
- [ ] Open **Command Prompt as Administrator**.
- [ ] Run:

  ```bash
  cd C:\Redis
  redis-server.exe redis.windows.conf
  ```

- [ ] Leave this window **open** (Redis server is running here).

**Quick check (optional):**

- [ ] Open a second Command Prompt:

  ```bash
  cd C:\Redis
  redis-cli ping
  ```

  You should see:

  ```text
  PONG
  ```

***

#### macOS

- [ ] In **Terminal** (Homebrew must be installed):

  ```bash
  brew install redis
  ```

- [ ] Start Redis:

  ```bash
  brew services start redis
  ```

- [ ] Verify:

  ```bash
  redis-cli ping
  ```

  You should see `PONG`. [liquidweb](https://www.liquidweb.com/blog/install-redis-macos-windows/)

***

#### Linux (Ubuntu / Debian)

- [ ] In **Terminal**:

  ```bash
  sudo apt install redis-server
  sudo systemctl start redis
  sudo systemctl enable redis
  ```

- [ ] Verify:

  ```bash
  redis-cli ping
  ```

  Expected: `PONG`. [liquidweb](https://www.liquidweb.com/blog/install-redis-macos-windows/)

***

### Step 1.4 – Install Git

**Why:** Git downloads the project code from GitHub (or your repo). It’s like “copying the project” in a safe and updateable way.

***

#### Windows

- [ ] Go to: `https://git-scm.com/download/win`
- [ ] Download the installer and run it.
- [ ] Keep **default options** (just click Next until Finish). [feaforall](https://feaforall.com/how-to-install-python-3-on-windows-and-set-the-path/)
- [ ] Verify in Command Prompt:

  ```bash
  git --version
  ```

***

#### macOS

- [ ] In **Terminal**:

  ```bash
  brew install git
  ```

- [ ] Verify:

  ```bash
  git --version
  ```

***

#### Linux (Ubuntu / Debian)

- [ ] In **Terminal**:

  ```bash
  sudo apt install git
  git --version
  ```

If you see a version number, Git is installed.

***

## Section 2: Download ALPHA-PRIME Project (≈5 minutes)

You now download the actual ALPHA-PRIME v2.0 code.

***

### Step 2.1 – Choose a Folder for the Project

Pick a simple path you can remember.

**Windows example:**

- [ ] Open **Command Prompt** and run:

  ```bash
  cd C:\Users\YourName\Documents
  ```

  (Replace `YourName` with your actual username.)

**macOS / Linux example:**

- [ ] In **Terminal**:

  ```bash
  cd ~/Documents
  ```

***

### Step 2.2 – Clone from GitHub (recommended)

Replace the URL below with your actual repo if different.

- [ ] Run:

  ```bash
  git clone https://github.com/yourusername/alpha-prime-v2.git
  cd alpha-prime-v2
  ```

You should now be **inside** the `alpha-prime-v2` folder.

***

### Step 2.3 – If You Have a ZIP Instead

If you were given a ZIP file (e.g. `alpha-prime-v2.zip`):

- [ ] Extract it to a folder like:
  - Windows: `C:\alpha-prime-v2`
  - macOS/Linux: `~/alpha-prime-v2`
- [ ] Open a terminal/Command Prompt and navigate there:

  ```bash
  cd path/to/alpha-prime-v2
  ```

  (Replace `path/to` with the real folder path.)

***

## Section 3: Automated Setup (≈10 minutes)

The setup script does the heavy lifting for you:

- Creates a **virtual environment** (isolated Python sandbox)
- Installs dependencies
- Initializes the database
- Creates configuration templates

***

### Step 3.1 – Make Setup Script Executable (macOS / Linux only)

- [ ] From inside the project folder:

  ```bash
  chmod +x scripts/setup.sh
  ```

(Windows users **skip** this; you will run the script via Git Bash or WSL if needed.)

***

### Step 3.2 – Run the Setup Script

**All OS (from project root):**

- [ ] Run:

  ```bash
  bash scripts/setup.sh --dev
  ```

What you should see:

- Lines like:
  - `Creating virtual environment...`
  - `Installing dependencies...`
  - `Setting up database...`
  - `✓ Setup complete!`

This can take **a few minutes** depending on your internet speed.

***

### If You See Errors During Setup

**Error: `python: command not found` or `python3: command not found`**

- [ ] Go back to **Step 1.1** and confirm Python is installed and `python` / `python3` works.

**Error: `psql: could not connect to server` or mentions PostgreSQL**

- [ ] Ensure PostgreSQL is running:
  - Windows: Start **pgAdmin**, if needed restart the PostgreSQL service from Services.
  - macOS:

    ```bash
    brew services start postgresql@14
    ```

  - Linux:

    ```bash
    sudo systemctl start postgresql
    ```

- [ ] Re-run:

  ```bash
  bash scripts/setup.sh --dev
  ```

**Error: `could not connect to Redis`**

- [ ] Ensure Redis is running (see Step 1.3), then re-run setup.

**Error: `Permission denied` (macOS / Linux)**

- [ ] Try:

  ```bash
  sudo bash scripts/setup.sh --dev
  ```

  (You will be asked for your system password.)

***

### Step 3.3 – Verify Setup Created Key Files

From the project folder:

- [ ] Check that `venv` folder exists.
- [ ] Check that `.env` file exists (or `.env.example` was copied).

If `venv` or `.env` is missing, the setup script may have failed. Fix the error and run it again.

***

## Section 4: Configure Your Credentials (≈5 minutes)

Your `.env` file tells ALPHA-PRIME how to talk to your broker, database, email, etc.

***

### Step 4.1 – Open the `.env` File

Location: **project root folder**, file named `.env`.

**Windows:**

- [ ] In File Explorer, open the `alpha-prime-v2` folder.
- [ ] If you don’t see `.env`, make sure “Hidden items” are visible.
- [ ] Right‑click `.env` → **Open with** → choose **Notepad**.

**macOS:**

- [ ] In **Terminal** (from project root):

  ```bash
  open -e .env
  ```

  or use any editor (VS Code, etc.).

**Linux:**

- [ ] In **Terminal**:

  ```bash
  nano .env
  ```

  (or open with your preferred editor.)

***

### Step 4.2 – Fill in Broker and API Keys

Inside `.env`, find the lines similar to:

```bash
# Broker API (Zerodha example)
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_secret_here
ZERODHA_USER_ID=your_user_id_here
ZERODHA_PASSWORD=your_password_here
ZERODHA_PIN=your_pin_here
```

- [ ] Replace `your_api_key_here` etc. with your **real** values.

**Where to get Zerodha API keys (example):**

- [ ] Log into **Kite** in your browser.
- [ ] Go to `https://kite.trade/` → **Developers** / **Create App**.
- [ ] Create an app, then copy:
  - **API Key**
  - **API Secret**
- [ ] Fill them into your `.env`.

*For Upstox or other brokers, fill the corresponding variables in the same file.*

***

### Step 4.3 – Set Up Email Alerts (optional but recommended)

In `.env`, find:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_TO=your_email@gmail.com
```

- [ ] Replace `your_email@gmail.com` with your Gmail address (or other provider).
- [ ] For Gmail:
  - Create an **App Password** in Google Account → Security → App passwords.
  - Use that as `SMTP_PASSWORD`.
- [ ] Save the file.

***

### Step 4.4 – Set Up Telegram Alerts (optional)

Find:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

- [ ] Create a bot via `@BotFather` in Telegram (send `/newbot`).
- [ ] Copy the bot token into `TELEGRAM_BOT_TOKEN`.
- [ ] For Chat ID, send a message to your bot, then use a simple script (provided in docs) or an online tool to get your chat ID.
- [ ] Fill `TELEGRAM_CHAT_ID`.

***

### Step 4.5 – Database Credentials

The setup script usually configures PostgreSQL access accordingly, but verify lines like:

```bash
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/alpha_prime_v2
```

- [ ] Make sure `your_password` matches the **postgres** password you set in Step 1.2.
- [ ] Save the `.env` file after editing.

***

### Step 4.6 – Security Reminder

- [ ] Never share `.env` with anyone.
- [ ] `.env` is usually in `.gitignore` so it’s not committed to Git. Don’t remove it from there.
- [ ] Treat `.env` like your **password vault**.

***

## Section 5: Test That Everything Works (≈5–10 minutes)

Here you just check that core pieces (Python, DB, Redis, broker) are wired correctly.

***

### Step 5.1 – Activate the Virtual Environment

Do this **every time** you work with ALPHA-PRIME in a new terminal.

**Windows (Command Prompt or PowerShell):**

- [ ] From the project folder:

  ```bash
  venv\Scripts\activate
  ```

You should see your prompt change to start with `(venv)`.

**macOS / Linux:**

- [ ] From the project folder:

  ```bash
  source venv/bin/activate
  ```

You should also see `(venv)` at the start of your prompt.

***

### Step 5.2 – Run Smoke Tests

These are quick health checks for configuration, DB, Redis, and broker.

- [ ] From the project root with `(venv)` active:

  ```bash
  pytest tests/smoke/ -v
  ```

**Expected output (example):**

```text
tests/smoke/test_config.py  ✓
tests/smoke/test_database.py ✓
tests/smoke/test_redis.py   ✓
tests/smoke/test_broker.py  ✓

====== 4 passed in 2.5s ======
```

**If a test fails:**

- `test_config.py` fails  
  → Check `.env` for missing / incorrect values.

- `test_database.py` fails  
  → PostgreSQL may not be running, or `DATABASE_URL` is wrong.
  - Start PostgreSQL again (see Section 1.2).

- `test_redis.py` fails  
  → Redis may not be running.
  - Start Redis again (Section 1.3).

- `test_broker.py` fails  
  → Broker API keys might be wrong, or broker API is down.
  - Double‑check keys in `.env`.
  - Try logging into broker’s web platform manually.

***

### Step 5.3 – Start the Dashboard

This starts the ALPHA-PRIME web interface on your machine.

- [ ] Ensure `(venv)` is active.
- [ ] Run:

  ```bash
  python -m dashboard.app_v2
  ```

**Expected output:**

```text
INFO: Starting ALPHA-PRIME v2.0 Dashboard
INFO: Running on http://127.0.0.1:8000
INFO: Press CTRL+C to quit
```

***

### Step 5.4 – Open the Dashboard in Your Browser

- [ ] Open your browser (Chrome/Edge/Safari/Firefox).
- [ ] Go to: `http://localhost:8000`
- [ ] You should see the ALPHA-PRIME dashboard with:
  - Portfolio overview
  - Equity curve (likely empty on first run)
  - Strategy controls
  - Live positions (empty if nothing running yet)

**Visual description (example):**

```text
╔════════════════════════════════════════╗
║  ALPHA-PRIME v2.0                      ║
║  Dashboard                             ║
╠════════════════════════════════════════╣
║  Portfolio Value: $100,000.00          ║
║  Today's P&L: $0.00                    ║
║  Open Positions: 0                     ║
║                                        ║
║  [Start Trading] [Stop] [Settings]     ║
╚════════════════════════════════════════╝
```

If you see this (or similar), your install is **successfully running**.

To stop the dashboard:

- [ ] Go to the terminal window where it’s running.
- [ ] Press `CTRL + C`.

***

## Section 6: You’re Done!

What you just did:

- [x] Installed Python
- [x] Installed PostgreSQL (database)
- [x] Installed Redis (cache)
- [x] Installed Git
- [x] Downloaded ALPHA-PRIME v2.0
- [x] Ran the automated setup
- [x] Configured your `.env` with broker and alert credentials
- [x] Ran smoke tests
- [x] Started the dashboard and saw it in browser

This was a **one‑time setup**. You should **not** need to repeat these steps unless you reinstall your system.

***

## Quick Daily Reference (after first install)

**Start everything (day‑to‑day use):**

```bash
# 1. Open terminal / command prompt
cd path/to/alpha-prime-v2

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Start dashboard
python -m dashboard.app_v2
```

**Stop the system:**

- In the terminal where ALPHA-PRIME is running:
  - Press `CTRL + C`.

***

## Troubleshooting – Common Problems & Fixes

### “Command not found: python”

- [ ] Try:

  ```bash
  python3 --version
  ```

- If that works, use `python3` instead of `python`:

  ```bash
  python3 -m dashboard.app_v2
  ```

- If neither `python` nor `python3` works, revisit **Step 1.1**.

***

### “Permission denied” on macOS / Linux

This usually means the script isn’t marked as executable.

- [ ] Run:

  ```bash
  chmod +x scripts/setup.sh
  chmod +x scripts/run_engine.py
  ```

- [ ] If still denied, prefix with `sudo` (only when needed):

  ```bash
  sudo bash scripts/setup.sh --dev
  ```

***

### “Port 8000 already in use”

Another program is already using port 8000.

**macOS / Linux:**

- [ ] Find what’s using it:

  ```bash
  lsof -i :8000
  ```

- [ ] Kill that process (carefully) using the PID shown, or change ALPHA‑PRIME’s port in the dashboard config.

**Windows:**

- [ ] In Command Prompt:

  ```bash
  netstat -ano | findstr :8000
  ```

- [ ] Find the PID, then kill it via Task Manager or:

  ```bash
  taskkill /PID <PID> /F
  ```

***

### Dashboard Loads but Shows Errors

- [ ] Check logs:

  ```bash
  # From project root
  tail -f logs/alpha_prime.log    # macOS / Linux
  ```

  On Windows (no `tail`):

  - Open `logs/alpha_prime.log` in Notepad and scroll to bottom.
- [ ] Look for lines with `ERROR`.
- [ ] Common hints:
  - Database connection errors → PostgreSQL not running or `DATABASE_URL` wrong.
  - Redis errors → Redis not running.
  - Broker errors → API credentials wrong or broker API down.

***

### Can’t Connect to Broker API

- [ ] Verify values in `.env` (no extra spaces, correct case).
- [ ] Try logging into broker web platform (Kite, etc.) to confirm credentials.
- [ ] Check broker API status (e.g. Zerodha status page).
- [ ] Ensure your API app is activated and has permissions.

***

### PostgreSQL Service Fails to Start

**macOS (Homebrew):**

- [ ] Check logs:

  ```bash
  tail -f /usr/local/var/log/postgres.log
  ```

- [ ] Restart service:

  ```bash
  brew services restart postgresql@14
  ``` [dataquest](https://www.dataquest.io/blog/install-postgresql-14-7-for-macos/)

**Linux:**

- [ ] Check status:

  ```bash
  sudo systemctl status postgresql
  ```

- [ ] Restart:

  ```bash
  sudo systemctl restart postgresql
  ```

***

### Redis Not Responding

- [ ] Run:

  ```bash
  redis-cli ping
  ```

  If not `PONG`:

  - macOS:

    ```bash
    brew services restart redis
    ```

  - Linux:

    ```bash
    sudo systemctl restart redis-server
    ```

  - Windows: close and re‑run `redis-server.exe` in `C:\Redis`.

***

## Emergency Reset – Start Fresh If Things Are Messed Up

If configuration is broken and you just want a clean slate (keeping a backup):

> Warning: This deletes ALL portfolio positions, trades, and orders in the ALPHA‑PRIME database, but it creates a backup first.

1. **Stop everything**

   - [ ] In any ALPHA-PRIME terminal windows, press `CTRL + C`.

2. **Deactivate virtual environment**

   - [ ] If your prompt shows `(venv)`, run:

     ```bash
     deactivate
     ```

3. **Reset portfolio db (with backup)**

   - [ ] From project root:

     ```bash
     python scripts/reset_portfolio.py --mode full --backup --confirm
     ```

   - This will:
     - Create a backup `.sql` file in `backups/`.
     - Wipe positions/trades/orders and cache for a fresh start.

4. **Re-run setup**

   - [ ] Run:

     ```bash
     bash scripts/setup.sh --dev
     ```

5. **Re-check `.env`**

   - [ ] Open `.env`, ensure broker/API settings are correct.

6. **Run smoke tests again**

   - [ ] Activate `venv` and run:

     ```bash
     pytest tests/smoke/ -v
     ```

If smoke tests pass and dashboard starts, you are back to a working clean state.

***

## Support & Next Steps

If you get stuck:

- [ ] Check `logs/alpha_prime.log` for detailed error messages.
- [ ] Copy the error message and search it in **Perplexity** or the broker’s help center.
- [ ] Check broker API docs (Zerodha / Upstox / your broker) for any extra steps (like whitelisting IPs, enabling APIs).

**After installation, read next:**

- [ ] `docs/02_daily_startup_checklist.md` – how to start and stop the system **every day**.
- [ ] `docs/03_profit_maximization_guide.md` – how to configure strategies and risk settings for your own trading.

Once you have this guide done once, you **never** need to repeat it on this machine unless you:

- Wipe or reinstall your OS, or
- Intentionally remove ALPHA-PRIME and want a clean reinstall.
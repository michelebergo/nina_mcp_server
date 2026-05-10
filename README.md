# NINA Advanced API MCP Server

A **Model Context Protocol (MCP) server** that lets AI agents (Claude Desktop, the NINA *AI Assistant* plugin, or any MCP-capable client) control [N.I.N.A.](https://nighttime-imaging.eu/) over its [Advanced API](https://github.com/christian-photo/ninaAPI).

Talk to your observatory: *"connect the camera, cool to -10°C, slew to M31, take a 30-second exposure"* — and the agent executes it.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NINA 3.x](https://img.shields.io/badge/NINA-3.x-green.svg)](https://nighttime-imaging.eu/)
[![NINA Advanced API 2.2.13+](https://img.shields.io/badge/Advanced%20API-2.2.13+-orange.svg)](https://github.com/christian-photo/ninaAPI)
[![Tools: 176](https://img.shields.io/badge/MCP%20tools-176-brightgreen.svg)](#-tool-coverage)

</div>

> ⚠️ This software controls real telescope hardware. Always test in a safe state before running unattended. Use at your own responsibility.

---

## Table of contents

- [What this is (and isn't)](#what-this-is-and-isnt)
- [Prerequisites](#prerequisites)
- [Installing the prerequisites (for first-timers)](#-installing-the-prerequisites-for-first-timers)
- [Quick start (5 minutes)](#-quick-start-5-minutes)
  - [Step 1 — Install NINA Advanced API plugin](#step-1--install-nina-advanced-api-plugin-inside-nina)
  - [Step 2 — Get this MCP server](#step-2--get-this-mcp-server)
  - [Step 3 — Verify the server runs](#step-3--verify-the-server-runs)
  - [Step 4 — Connect a client](#step-4--connect-an-mcp-client)
  - [Step 5 — Talk to it](#step-5--talk-to-it)
- [Configuration reference](#-configuration-reference)
- [Usage examples](#-usage-examples)
- [Tool coverage](#-tool-coverage)
- [FAQ](#-faq)
- [Troubleshooting](#-troubleshooting)
- [Repository contents](#-repository-contents)
- [Contributing / License / Credits](#-contributing)

---

## What this is (and isn't)

| | |
|---|---|
| **NINA** | the desktop application that runs your imaging session. You install and run it. |
| **NINA Advanced API plugin** | a plugin *inside* NINA, by [christian-photo](https://github.com/christian-photo/ninaAPI). When NINA is running with this plugin enabled, NINA exposes a REST API on `http://localhost:1888/v2/api`. |
| **This repo (`nina_advanced_mcp.py`)** | a small Python process that speaks **MCP** on stdio and translates each tool call into a REST call to the Advanced API. AI clients talk to **this**, this talks to NINA. |

```
┌─────────────────┐   stdio (MCP)   ┌──────────────────────┐   HTTP REST   ┌──────────────┐
│  Claude Desktop │ ──────────────▶ │  nina_advanced_mcp   │ ────────────▶ │  NINA + API  │
│  / AI Assistant │                 │      .py (this)      │ :1888         │   plugin     │
└─────────────────┘                 └──────────────────────┘               └──────┬───────┘
                                                                                  │
                                                                          ┌───────▼────────┐
                                                                          │  Mount, camera,│
                                                                          │  focuser, etc. │
                                                                          └────────────────┘
```

**You need all three running** for natural-language equipment control to work.

---

## Prerequisites

- **NINA 3.x** ([download](https://nighttime-imaging.eu/download/))
- **NINA Advanced API plugin 2.2.13 or later** — installed from inside NINA (see Step 1)
- **Python 3.10+** ([download](https://www.python.org/downloads/))
- **One of:**
  - [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (recommended — single-line install, runs the server with no manual venv)
  - or plain `pip` + `venv` (works everywhere)
- **An MCP client**, either:
  - [Claude Desktop](https://claude.ai/download) (free, easiest to test with), or
  - the [NINA AI Assistant plugin](https://github.com/michelebergo/nina.plugin.aiassistant) (integrated chat panel inside NINA)

---

## 📦 Installing the prerequisites (for first-timers)

If you already have Python, git, and an MCP client, **skip to [Quick start](#-quick-start-5-minutes)**. If anything below sounds unfamiliar, expand the section and follow the steps.

<details>
<summary><b>Python 3.10+</b> — required to run the server script</summary>

**Windows:**
1. Go to https://www.python.org/downloads/windows/ and click *Download Python 3.x*.
2. Run the installer. **⚠️ On the first screen, check ☑ "Add python.exe to PATH"** before clicking *Install Now*. Forgetting this is the #1 cause of "python not found" later.
3. Open a **new** PowerShell window (not an existing one — PATH only refreshes for new shells) and verify:
   ```powershell
   python --version
   ```
   You should see `Python 3.12.x` (or whatever version you installed).

**macOS:**
```bash
brew install python@3.12       # if you have Homebrew
# or download the .pkg installer from python.org
python3 --version
```

**Linux:** use your distro's package manager (`apt install python3 python3-pip`, `dnf install python3`, etc.).

</details>

<details>
<summary><b>git</b> — to clone this repository (or use the ZIP download workaround)</summary>

**With git (recommended):**
- Windows: https://git-scm.com/download/win → run the installer, accept defaults.
- macOS: `xcode-select --install` (installs the Apple command-line tools) or `brew install git`.
- Linux: `sudo apt install git` (or your distro's equivalent).

Verify:
```bash
git --version
```

**Without git (ZIP workaround):**
1. Go to https://github.com/michelebergo/nina_mcp_server
2. Click the green **Code** button → **Download ZIP**
3. Extract somewhere stable (e.g. `C:\Users\<YOU>\Documents\nina_mcp_server\`)
4. In the Quick start, *skip the `git clone` line* and `cd` into the extracted folder instead.

</details>

<details>
<summary><b>uv</b> — modern Python launcher (recommended over manual venv)</summary>

`uv` is a single binary that installs Python packages and runs scripts in isolated environments — much faster than `pip` + `venv`, and it skips the "which Python is active?" confusion.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
If you see a *running scripts is disabled on this system* error, use the command above as-is — the `-ExecutionPolicy ByPass` flag bypasses the policy for this one command only.

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative — via pip** (works on any OS, useful in locked-down environments):
```bash
pip install uv
```

**After installing, open a new terminal** so the PATH refreshes, then verify:
```bash
uv --version
```
You should see something like `uv 0.5.x`.

> Don't want to install uv? Use the **venv + pip** path instead in [Step 2 — Option B](#step-2--get-this-mcp-server).

</details>

<details>
<summary><b>Claude Desktop</b> — the easiest MCP client to test with</summary>

1. Download from https://claude.ai/download for your OS.
2. Install and sign in (free account works; you'll get rate-limited but it's fine for testing).
3. **Important — fully quitting Claude Desktop**:
   - **Windows**: closing the window doesn't quit it. Right-click the Claude icon in the system tray (bottom-right, you may need to click the `^` arrow) → **Quit**.
   - **macOS**: closing the window doesn't quit it. From the menu bar: *Claude → Quit Claude* (or `Cmd+Q`).
   - You'll need a full quit-and-relaunch every time you edit `claude_desktop_config.json`.

</details>

<details>
<summary><b>A JSON-aware editor</b> — for editing <code>claude_desktop_config.json</code> without breaking it</summary>

The MCP config is JSON. A missing comma or a smart quote (`"` from Word/Notepad) will silently break it and Claude Desktop will start with no servers — usually with no error message.

**Free editors that highlight JSON errors:**
- [VS Code](https://code.visualstudio.com/) (recommended — also helpful for the rest of your astrophotography life)
- [Notepad++](https://notepad-plus-plus.org/)
- Built-in *TextEdit* on macOS works **only** in plain-text mode (*Format → Make Plain Text*) — otherwise it inserts smart quotes that break JSON.
- Avoid Windows Notepad for anything more than 5 lines; it doesn't warn about JSON errors.

**Quick validate:** paste the file content into https://jsonlint.com/ — it'll point at the first error.

</details>

---

## 🚀 Quick start (5 minutes)

### Step 1 — Install NINA Advanced API plugin (inside NINA)

1. Open NINA → **Options → Plugins**
2. Search for **"Advanced API"** (by *christian-photo*)
3. Click **Install** → restart NINA
4. After restart, open **Options → Plugins → Advanced API** and verify:
   - Plugin is **Enabled**
   - **Port** = `1888` (default — leave it unless you have a conflict)
   - **Use Access Token** = OFF for first-time setup (you can enable it later)
5. Make sure NINA stays running for everything below — the API is gone when NINA closes.

**Sanity check** — open PowerShell / Terminal and run:

```bash
curl http://localhost:1888/v2/api/version
```

You should see a JSON response with the API version. If you get *connection refused*, the plugin isn't running.

### Step 2 — Get this MCP server

```bash
git clone https://github.com/michelebergo/nina_mcp_server.git
cd nina_mcp_server
```

Pick **one** of the two install styles:

**Option A — `uv` (recommended, no manual venv):**

```bash
# Install uv if you don't have it:
# Windows (PowerShell):  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux:         curl -LsSf https://astral.sh/uv/install.sh | sh

# Nothing else to do — the MCP client config below will install deps on first run.
```

**Option B — `venv` + `pip`:**

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Verify the server runs

Before plugging into an AI client, run the server standalone and confirm it boots:

**Option A (uv):**
```bash
uv run --with fastmcp,fastapi,uvicorn,pydantic,aiohttp,requests,python-dotenv,structlog,python-dateutil,pytz fastmcp run nina_advanced_mcp.py
```

**Option B (venv):**
```bash
python nina_advanced_mcp.py
```

You should see log lines like:

```
Starting with configuration:
NINA_HOST: localhost
NINA_PORT: 1888
LOG_LEVEL: INFO
IMAGE_SAVE_DIR: <your home>/Desktop/NINA_Images
```

Press `Ctrl+C` to stop. The server is now ready to be launched on demand by an MCP client.

> If NINA + Advanced API are running on the **same machine** as this server, **no environment variables are required** — the defaults (`localhost:1888`) just work.

### Step 4 — Connect an MCP client

Pick the client you use:

<details>
<summary><b>Claude Desktop</b></summary>

Edit your `claude_desktop_config.json`:

- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json` &nbsp;→&nbsp; `C:\Users\<YOU>\AppData\Roaming\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

> 💡 **How to find this folder if the path looks like gibberish:**
> - **Windows**: press `Win + R`, paste `%APPDATA%\Claude` and press Enter — File Explorer opens the right folder. (The `AppData` folder is hidden by default; this shortcut bypasses that.)
> - **macOS**: in Finder, press `Cmd + Shift + G` and paste `~/Library/Application Support/Claude`.
> - The simplest way to get there from inside Claude Desktop itself: open Claude → **Settings → Developer → Edit Config**. It opens the file directly with your default editor and creates it if missing.

Create the file if it doesn't exist, then paste:

```json
{
  "mcpServers": {
    "nina": {
      "command": "uv",
      "args": [
        "run",
        "--with", "fastmcp,fastapi,uvicorn,pydantic,aiohttp,requests,python-dotenv,structlog,python-dateutil,pytz",
        "fastmcp", "run",
        "C:/full/path/to/nina_mcp_server/nina_advanced_mcp.py"
      ],
      "env": {
        "NINA_HOST": "localhost",
        "NINA_PORT": "1888",
        "LOG_LEVEL": "INFO",
        "IMAGE_SAVE_DIR": "C:/Users/<YOU>/Desktop/NINA_Images"
      }
    }
  }
}
```

Replace `C:/full/path/to/nina_mcp_server/` with the actual repo path on disk. Use **forward slashes** even on Windows (Windows accepts them in JSON; backslashes would need to be escaped as `\\`).

> 💡 **How to get the exact path:** in File Explorer, open the `nina_mcp_server` folder, click in the address bar — it shows the full path. Copy it, then **replace every `\` with `/`** in your JSON.

**Restart Claude Desktop fully** (it's not enough to close the window):
- **Windows**: right-click the tray icon (bottom-right, may be under the `^` arrow) → *Quit*. Then relaunch.
- **macOS**: menu bar *Claude → Quit Claude* (`Cmd+Q`). Then relaunch.

After relaunching, look at the bottom of the chat input box. You should see a small 🔌 icon and a 🔨 hammer icon — click the hammer to see the list of `nina_*` tools available. If you don't see anything, jump to [Troubleshooting](#-troubleshooting).

If you used Option B (venv), use this instead:

```json
{
  "mcpServers": {
    "nina": {
      "command": "C:/full/path/to/nina_mcp_server/.venv/Scripts/python.exe",
      "args": ["C:/full/path/to/nina_mcp_server/nina_advanced_mcp.py"],
      "env": { "NINA_HOST": "localhost", "NINA_PORT": "1888" }
    }
  }
}
```

</details>

<details>
<summary><b>NINA AI Assistant plugin</b></summary>

If you're already using the [AI Assistant plugin](https://github.com/michelebergo/nina.plugin.aiassistant) inside NINA, you have **two options**:

1. **Built-in (recommended)** — just enable the **MCP - NINA Equipment Control** section in the plugin's options. It talks to the Advanced API *directly* over HTTP, no Python server needed. You can skip Steps 2–4 of this guide.

2. **External Python server (this repo)** — in the plugin options, scroll to **🔧 External MCP Server (Advanced)** and set:
   - **Python Path** = path to `python.exe` (your venv's, if you used Option B)
   - **MCP Server Script Path** = full path to `nina_advanced_mcp.py`

   The plugin spawns this server on demand. Useful if you want to extend tools in Python without rebuilding the C# plugin.

</details>

<details>
<summary><b>Any other MCP client</b></summary>

This server uses **stdio transport** (the FastMCP default). Configure your client to launch `python nina_advanced_mcp.py` (or the `uv run --with ...` variant) and communicate over stdin/stdout. Environment variables: see the [Configuration reference](#-configuration-reference) below.

</details>

### Step 5 — Talk to it

In your MCP client, start with a low-risk query:

> "What's connected to NINA right now?"

The agent should call a tool like `nina_get_status`. **The first time it calls each tool, Claude Desktop pops up an approval dialog** — review what it wants to do, then click *Allow* (or *Allow for this chat*). This is by design: every tool requires your consent the first time.

If the response is sensible (lists your equipment with their connection state), the whole pipeline works. Try progressively more ambitious queries:

1. *"Connect the camera."*  → tests **write** access to NINA
2. *"Cool the camera to -10°C."*  → tests **long-running** operations
3. *"Take a 5-second test exposure."*  → tests **capture + file save**

> ⚠️ **Stop here if anything fails before you're confident.** Don't ask the agent to slew, dither, or run sequences until status reads + connects work reliably.

---

## ⚙️ Configuration reference

All settings come from environment variables. Defaults are sensible for a single-machine setup.

| Variable          | Default                              | Description |
|-------------------|--------------------------------------|-------------|
| `NINA_HOST`       | `localhost`                          | Hostname or IP where NINA + Advanced API is running |
| `NINA_PORT`       | `1888`                               | Port configured in the Advanced API plugin settings |
| `LOG_LEVEL`       | `INFO`                               | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `IMAGE_SAVE_DIR`  | `~/Desktop/NINA_Images`              | Where captured-image tools save FITS files |

**Two ways to provide them**, depending on how you launch the server:

- **Launched by an MCP client** (Claude Desktop, AI Assistant plugin): put them in the `env: { ... }` block of the client's config. This is the authoritative source.
- **Launched manually for testing**: copy [`.env.example`](.env.example) to `.env` next to `nina_advanced_mcp.py` and edit the values — `python-dotenv` loads it automatically. The MCP-client-supplied `env` overrides `.env`.

> If NINA is on a **different machine** (e.g. observatory PC), set `NINA_HOST` to its IP and make sure the Advanced API port is reachable through the firewall.

---

## 💬 Usage examples

Once connected, just talk to your AI client in natural language. The agent will pick the right tool, ask follow-ups if it needs them, and report back what happened.

**Status & connection:**
- *"What's connected right now?"*
- *"Connect the camera, mount, filter wheel, and guider."*
- *"Is the mount parked?"*

**Imaging:**
- *"Cool the camera to -10°C and wait until it's stable."*
- *"Take a 30-second exposure with the Ha filter and plate-solve it."*
- *"Show me the last image you captured."*

**Targeting:**
- *"Slew to M31, center it, and start guiding."*
- *"Start the sequence named 'NGC 7000 Ha'."*
- *"Pause the current sequence, change the target to M42, and resume."*

**Diagnostics:**
- *"What was the HFR of the last 5 frames?"*
- *"Why did the last autofocus run fail? Show me the event history."*

Refer to [`API_COVERAGE_ANALYSIS.md`](API_COVERAGE_ANALYSIS.md) for the full list of supported NINA API endpoints.

---

## 🛠️ Tool coverage

- **176 MCP tools** wrapping the entire NINA Advanced API surface
- **150/150 endpoints** of Advanced API v2.2.13 implemented (**100%**)
- **21/21 endpoint groups** complete

Categories:

| | |
|---|---|
| **Equipment** | Cameras · Mounts · Focusers · Filter Wheels · Domes · Rotators · Flat Panels · Guiders (PHD2) · Weather Stations · Safety Monitors · Switches |
| **Imaging** | Capture · Cooling · Subsample · Image history · Prepared images · Thumbnails · Plate solving · Live stacking |
| **Sequencing** | Load / start / stop / pause / edit / state · Target setting · Skip / reset |
| **Framing & planning** | Coordinates · Moon separation · Rotation calculation · Framing assistant |
| **Flats automation** | Sky flats · Auto-brightness · Auto-exposure · Trained darks · Trained flats |
| **System** | Profile management · Application control · Plugin settings · Event/log history · WebSocket events |

See [`API_COVERAGE_ANALYSIS.md`](API_COVERAGE_ANALYSIS.md) for the per-endpoint matrix.

---

## ❓ FAQ

<details>
<summary><b>Do I need to keep NINA open all the time?</b></summary>

Yes. The Advanced API plugin lives inside NINA — when NINA quits, the REST endpoint goes down, and every MCP tool call starts failing. Keep NINA running for the duration of your session.

</details>

<details>
<summary><b>Do I need to manually start the MCP server before opening Claude / NINA AI Assistant?</b></summary>

No. Both clients **launch this Python process automatically** when they need it (and shut it down when they don't). You only run the server manually for [Step 3 — Verify the server runs](#step-3--verify-the-server-runs) sanity check.

</details>

<details>
<summary><b>Can NINA run on a different PC than Claude Desktop?</b></summary>

Yes — common observatory setup. Run NINA + Advanced API plugin on the observatory PC; run Claude Desktop + this MCP server on your laptop.

In your `claude_desktop_config.json`, set:
```json
"env": {
  "NINA_HOST": "192.168.1.42",   // observatory PC's IP
  "NINA_PORT": "1888"
}
```

Make sure:
- The observatory PC's firewall allows inbound TCP on port `1888`.
- Both machines are on the same LAN (or you've set up a VPN / SSH tunnel).
- The Advanced API plugin's *"Bind to localhost only"* option (if present) is **disabled** — otherwise it only accepts connections from the same machine.

Test from your laptop: `curl http://192.168.1.42:1888/v2/api/version` should return JSON.

</details>

<details>
<summary><b>How do I update the MCP server later?</b></summary>

```bash
cd nina_mcp_server
git pull
```

If you're using Option A (`uv run --with ...`), nothing else to do — dependencies are resolved on each launch. If you're using Option B (venv):

```bash
# Reactivate the venv:
.\.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt --upgrade
```

</details>

<details>
<summary><b>Do I have to approve every tool call?</b></summary>

The first time the agent calls each distinct tool in a given chat, Claude Desktop asks for your approval. You can pick *Allow once* or *Allow for this chat*. There's a global "Allow all tools from this server" toggle in *Settings → Developer*, but I recommend keeping per-tool approvals on for any tool that **moves the mount**, **starts a sequence**, or **deletes files** until you trust the setup.

</details>

<details>
<summary><b>What if I want to use the server with a different agent (Cursor, Cline, etc.)?</b></summary>

Any MCP-compliant client over **stdio** works. The launch command is the same (`uv run --with ... fastmcp run nina_advanced_mcp.py` or `python nina_advanced_mcp.py`), only the config-file format differs. Consult your client's docs for where to put the `mcpServers` block.

</details>

<details>
<summary><b>Where are the server logs?</b></summary>

`logs/nina_advanced_api.log`, created next to `nina_advanced_mcp.py` on first launch. Useful when an MCP-client-launched server fails silently — the log captures the startup banner and every request.

</details>

<details>
<summary><b>Should I enable the Advanced API plugin's "Use Access Token" option?</b></summary>

**For first-time setup, leave it off.** It adds a token-header requirement to every API call and the current MCP server doesn't pass the token. Once you have everything working and you're exposing the API beyond localhost (e.g. remote observatory), enable it and configure your firewall — but expect to also need a code change in `nina_advanced_mcp.py` to forward the token, which isn't supported out of the box yet.

</details>

---

## 🐛 Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl http://localhost:1888/v2/api/version` → *connection refused* | NINA not running, or Advanced API plugin disabled / on a different port | Start NINA. Verify plugin is enabled and the port matches `NINA_PORT`. |
| Server starts but logs say `NINA_PORT=1888` while your plugin uses a different port | Stale env var | Update the `env` block in your MCP client config (or your `.env`); restart the client. |
| Claude Desktop doesn't show the **nina** server | JSON syntax error in `claude_desktop_config.json` | Validate with `python -m json.tool < claude_desktop_config.json`. Quote all paths. |
| Tools appear but every call returns *connection error* | NINA closed mid-session, or firewall blocking the port | Restart NINA. Re-run the `curl` sanity check. Allow inbound on `1888` if remote. |
| Capture tools succeed but no FITS file appears | `IMAGE_SAVE_DIR` points somewhere the NINA process can't write | Use an existing folder writable by NINA; check the path in the server logs at startup. |
| `uv run --with ...` slow on every launch | Deps re-resolved every time | Switch to Option B (venv) and point the MCP client at the venv's `python.exe` directly. |
| Path-related errors on Windows | Backslashes in JSON | Use forward slashes everywhere in `claude_desktop_config.json` (`C:/Users/...`). |
| Claude Desktop changes don't take effect | Closed the window but didn't quit the app | Fully quit Claude (tray-icon → Quit on Windows; `Cmd+Q` on macOS), then relaunch. |
| JSON file looks fine but doesn't load | Smart quotes (`"` `"`) from a word processor | Open in VS Code / Notepad++. Replace every fancy quote with a plain `"`. Validate at https://jsonlint.com/. |
| `uv: command not found` after install | Terminal was open before uv was installed | Close ALL terminal windows and open a new one — PATH only refreshes for new shells. |
| Windows: `running scripts is disabled on this system` | PowerShell ExecutionPolicy blocks the uv installer | Use the `powershell -ExecutionPolicy ByPass -c "..."` form shown in the uv install section. |

Server logs live at `logs/nina_advanced_api.log` next to `nina_advanced_mcp.py`. Tail them while reproducing the issue:

```bash
# Windows PowerShell
Get-Content .\logs\nina_advanced_api.log -Wait -Tail 50
# macOS / Linux
tail -f logs/nina_advanced_api.log
```

Still stuck? Open an [issue](https://github.com/michelebergo/nina_mcp_server/issues) with: server log excerpt, `claude_desktop_config.json` (redact tokens), NINA version, Advanced API plugin version, OS.

---

## 📁 Repository contents

```
nina_mcp_server/
├── nina_advanced_mcp.py        # The MCP server (FastMCP + 176 @mcp.tool functions)
├── nina_help.json              # Per-tool human-readable help, surfaced to the agent
├── requirements.txt            # Python dependencies
├── API_COVERAGE_ANALYSIS.md    # Endpoint-by-endpoint coverage of Advanced API v2.2.13
├── test_weather.py             # Standalone test for the weather-station tool
├── LICENSE                     # MIT
└── README.md
```

---

## 🤝 Contributing

PRs welcome. For new tools / endpoint coverage, please:

1. Add the tool to `nina_advanced_mcp.py` following the existing `@mcp.tool` pattern.
2. Add a human-readable description to `nina_help.json`.
3. Update `API_COVERAGE_ANALYSIS.md`.
4. Test against a real NINA instance before opening the PR.

## 📜 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- [N.I.N.A.](https://nighttime-imaging.eu/) — the core astrophotography software
- [NINA Advanced API](https://github.com/christian-photo/ninaAPI) by [christian-photo](https://github.com/christian-photo) — the HTTP API this server wraps
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP framework
- Original concept inspired by [PaDev1/Nina_advanced_api_mcp](https://github.com/PaDev1/Nina_advanced_api_mcp)

## 🔗 Related projects

- [NINA AI Assistant plugin](https://github.com/michelebergo/nina.plugin.aiassistant) — integrated chat panel inside NINA, uses this server (or the built-in HTTP path)
- [Touch'N'Stars](https://github.com/Touch-N-Stars/Touch-N-Stars) — mobile/web NINA control
- [NINA Plugins index](https://nighttime-imaging.eu/plugins/)

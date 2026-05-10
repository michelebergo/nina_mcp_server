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
- [Quick start (5 minutes)](#-quick-start-5-minutes)
  - [Step 1 — Install NINA Advanced API plugin](#step-1--install-nina-advanced-api-plugin-inside-nina)
  - [Step 2 — Get this MCP server](#step-2--get-this-mcp-server)
  - [Step 3 — Verify the server runs](#step-3--verify-the-server-runs)
  - [Step 4 — Connect a client](#step-4--connect-an-mcp-client)
- [Configuration reference](#-configuration-reference)
- [Usage examples](#-usage-examples)
- [Tool coverage](#-tool-coverage)
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

Replace `C:/full/path/to/nina_mcp_server/` with the actual repo path on disk. Use **forward slashes** even on Windows. Restart Claude Desktop. You should see a 🔌 indicator showing **nina** as a connected MCP server, and a hammer 🔨 icon showing the available tools.

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

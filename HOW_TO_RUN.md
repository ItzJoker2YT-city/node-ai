# How to run Node

Node is a tiny experimental chat model with a React interface and a local Python/PyTorch backend. The trained checkpoint is already included: **you do not need to train it before chatting**.

## Requirements

- Windows or Linux, 64-bit.
- Python 3.10–3.13 (3.11 or 3.12 recommended) and pip.
- A CPU supported by PyTorch; 4 cores and 8 GB RAM are the intended starting setup.
- Internet for the initial dependency installation. Chat runs locally afterward.
- Git for cloning, or download the repository ZIP instead.
- Node.js 20+ is optional, only for rebuilding the React interface.

## 1. Get the project

```bash
git clone https://github.com/ItzJoker2YT-city/node-ai.git
cd node-ai
```

Alternatively use GitHub's **Code → Download ZIP** (or download the supplied Node-Desktop ZIP), extract the project folder onto your **Desktop**, and open a terminal in that extracted folder. Commands below must run in the folder containing `backend`, `data`, and `requirements.txt`.

## 2A. Windows — PowerShell

Check Python:

```powershell
py --version
```

Create an isolated environment, install CPU PyTorch, and start Node:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe backend/server.py
```

No virtual-environment activation is required, so PowerShell execution-policy changes are not needed. If `py` is not recognized, install Python from python.org, enable the launcher/PATH options, and reopen the terminal.

## 2B. Linux — terminal

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python backend/server.py
```

If virtual-environment creation fails on Debian/Ubuntu, install `python3-venv` using your package manager, then retry. Other distributions have their own package names.

## 3. Open the interface

Visit **http://127.0.0.1:8765** in your own browser. Keep the terminal open and the computer awake. Stop the server with **Ctrl+C**.

Try short prompts such as:

- `hi`
- `hello`
- `what is your name?`
- `can you help me?`

Answers are sampled and can vary. Successful greeting examples overlap with the training data and do not demonstrate general intelligence. Basic facts or programming answers may still be wrong.

The app sends recent conversation turns, but the model only sees the last **256 UTF-8 bytes**. Short follow-ups may work; longer context and unseen names are unreliable.

On future runs, dependencies are already installed:

Windows:

```powershell
.\.venv\Scripts\python.exe backend/server.py
```

Linux:

```bash
.venv/bin/python backend/server.py
```

## Editing the React frontend (optional)

The prebuilt `dist/index.html` already works. To change and rebuild the UI:

```bash
npm install
npm run build
```

Restart or refresh the local app after rebuilding. Do not open `dist/index.html` directly through `file://`; serve it with the Python backend so API calls work.

## Continuing training (optional)

Back up `data/node.pt` first and **stop the web server** to avoid simultaneous writes. Then:

Windows:

```powershell
.\.venv\Scripts\python.exe backend/worker.py train --steps 100000 --seconds 240 --resume
```

Linux:

```bash
.venv/bin/python backend/worker.py train --steps 100000 --seconds 240 --resume
```

This continues the saved model weights for approximately four minutes, then runs qualitative reply checks. For the 4-hour CPU Colab path with automatic checkpoint selection and baseline regression guards, use the notebook linked in README.md. The distributed checkpoint omits optimizer state, so the first continuation starts a fresh optimizer; later checkpoints save it. The `--steps` value is an upper bound, not a prediction of work completed. Training may end at either limit. Afterward restart the server.

**Warning:** the UI's **Train from scratch** button does not resume. It initializes random weights and can replace your trained checkpoint after confirmation.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `No module named torch` | Install requirements with the exact `.venv` Python command above, and launch with that same Python. |
| Address/port already in use | Stop the other Node server using port 8765. |
| No checkpoint | Confirm `data/node.pt` was included in the clone/extraction. |
| Backend offline | Keep the Python server running and open its HTTP address, not the HTML file directly. |
| Replies are gibberish | This is a very small model, not a commercial coding assistant. More data and compute may be required. |
| Long prompt seems ignored | The model sees only the last 256 UTF-8 bytes of its input. A long system prompt cannot add knowledge. |

## Public access and uptime

A public GitHub repository stores the code; **it does not host this Python backend or keep the model online**. GitHub Pages cannot run this backend. Reliable remote access requires a persistent compute host and an authenticated, rate-limited deployment. The included server is loopback-only and intended for local use. Do not expose it directly to the public internet.

Only load trusted checkpoint files. Never automatically execute generated code. Review the dataset notices and limitations in README.md.

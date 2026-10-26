# Node private dashboard

An authenticated, single-owner conversation lab. The original local app remains available through `backend/server.py`; this dashboard runs separately on loopback port 8000.

## Start

```bash
npm ci
npm run build:dashboard
python backend/dashboard.py
```

First startup creates owner-only `dashboard_state/auth.json` and `dashboard_state/initial-login.txt`. Read the latter in your private server terminal to get the `admin` username and randomly generated temporary password. **Do not share that file or commit it.** Sign in and replace the password (14 characters minimum) before using any workspace APIs. The initial plaintext credential file is deleted when the password changes.

The process binds to **http://127.0.0.1:8000**, not HTTPS. Terminate TLS at a trusted reverse proxy or Cloudflare Tunnel and forward `https://ai.runhost.bond` to `http://127.0.0.1:8000`. For that deployment:

```bash
NODE_PUBLIC_ORIGIN=https://ai.runhost.bond PORT=8000 python backend/dashboard.py
```

The proxy must preserve the public Host header. If the tunnel service is accidentally set to `https://localhost:8000`, change its origin service to **HTTP** on `localhost:8000`; the public hostname remains HTTPS. Do not expose the original unauthenticated `backend/server.py` on the public domain. The public domain is explicitly allowed by `NODE_PUBLIC_ORIGIN`; other hosts and origins are rejected.

If CPU dependencies are installed under a separate target directory, include it in `PYTHONPATH` when launching the server. Torch is only imported by training/inference subprocesses, not the HTTP process.

## Monitor the already-running experiment

The dashboard does not stop or replace an existing training process. Configure these before startup:

```bash
export NODE_EXTERNAL_OUTPUT=/root/node-ai-training/colab_outputs
export NODE_EXTERNAL_SCRIPT=/root/node-ai-training/colab/train_colab.py
export NODE_EXTERNAL_LOG=/root/node-training.log
```

That run is shown as **Current terminal run**, with live artifacts, progress, and evaluation results. It is monitor-only because its original process did not start with the dashboard stop-file option. New dashboard-launched runs support graceful stop-and-save. Process discovery also blocks duplicate training after dashboard restart.

## Features

Beyond sign-in and starting training from the browser:

1. Live progress and step counts.
2. Elapsed time and estimated time remaining.
3. Training, held-out, baseline, and selected-best losses.
4. Baseline/current/best rule-based conversation-check points.
5. Auto-refreshing logs with a text filter (latest 24 KB).
6. Run history with active, complete, stopped, and interrupted states.
7. Duration presets from one minute to four hours.
8. CPU batch-size selection from 1 to 8.
9. Graceful stop-and-save for dashboard-managed runs.
10. Authenticated checkpoint and JSON report downloads.
11. Non-destructive checkpoint selection for chat; restore original selection.
12. Expandable before/after conversation evaluations.
13. UTF-8 text upload and dataset editor (1 KB–8 MB).
14. Dataset size, line and dialogue-block counts; text export.
15. Automatic dated backup before replacing a corpus.
16. Live CPU count, load, memory, disk, Python, and dependency information.
17. Multi-turn chat playground using real checkpoint outputs.
18. Conversation export, new-chat reset, and response copying.
19. Output-length and temperature controls plus a saved system-prompt editor.
20. Responsive desktop/mobile layout with light, dark, and device themes.

Also included: sign-out, forced first-login password change, session revocation after password change, and a workspace activity journal that excludes passwords and chat prompts.

## Training and safety

- New runs continue **the original `data/node.pt`**, with a fresh optimizer. Selecting a checkpoint for chat does not change the training starting point.
- Run data is isolated under `dashboard_state/runs/<run-id>/weights`; existing checkpoints are never overwritten.
- Candidate checkpoints cannot exceed the starting held-out loss or lose points on any baseline conversation case. Eligible candidates rank by rule-based score, then loss. This is not a guarantee against general conversational degradation.
- Dataset replacement is blocked during active training. Validation data is never edited through the dashboard.
- Only artifacts created by this project are selectable; arbitrary checkpoint uploads and shell execution are not exposed.
- Stop requests finish the current evaluation operation, then save and evaluate final weights. They are not instant.
- Conversations stay only in page memory unless exported. The system prompt/theme are saved in that browser's local storage.
- Server suspension or deletion can lose local data. Download artifacts before the environment expires.
- Node Tiny still sees only the latest **256 bytes**. Longer training does not create long-term conversational memory.

## Security model

Single-owner password authentication uses scrypt hashes, random session IDs, HttpOnly/SameSite cookies (Secure on the configured HTTPS domain), eight-hour expiry, per-process login throttling, explicit Host/Origin checks, CSRF tokens, and a restrictive CSP. Sessions expire on dashboard restart. Password changes revoke all sessions. There is no registration or password-recovery endpoint: retain private server access. This is a small private lab dashboard, not a hardened multi-tenant hosting platform. Keep the host/proxy patched and consider additional access controls for sensitive deployments.

## Test

```bash
npm test
python -m unittest discover -s tests -p 'test_*.py'
```

Python tests cover checkpoint selection, notebook syntax, login/cookies, host/origin/CSRF checks, private artifact access, input validation, and an isolated training/duplicate-block/stop/download lifecycle.

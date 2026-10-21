# Node AI — experimental CPU conversation model

This repository includes your Node model's trained weights, React interface, and local CPU backend. No pretrained model was substituted. A focused 240-second continuation produced coherent replies to several practiced greetings. It is NOT a generally fluent assistant: a basic variable-definition check was incorrect.

## Quick start

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for complete Windows/Linux instructions, troubleshooting, optional training, and frontend rebuilding.

```bash
git clone https://github.com/ItzJoker2YT-city/node-ai.git
cd node-ai
```

## Windows PowerShell

Open this folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe backend/server.py
```

## Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python backend/server.py
```

Open http://127.0.0.1:8765. Leave the computer awake and server running while using it. This avoids the temporary sandbox's interruptions. Keep the local server private; do not expose it directly to the internet.

A prebuilt React interface is included. Node.js is needed only to edit/rebuild it (`npm install`, then `npm run build`).

## What is included

- Roughly 4.9 million parameters; 6 blocks; width 256; 4 heads; 256-byte context.
- A trained `data/node.pt` checkpoint containing model/config only. Optimizer state is omitted from this portable copy; future continuation can restore the learned weights, not the exact optimizer trajectory.
- 38 KB focused conversational training corpus and a separate held-out English validation file.
- Backend support for continuing from a trusted checkpoint (`--resume`) and timed training (`--seconds`). The interface's **Train from scratch** button still initializes random weights and replaces the checkpoint after confirmation. Do not click it merely to chat.
- Training/evaluation receipts in `training-checks.json`. Greeting checks overlap with the training examples and do not establish generalization or reliability.

The model generates actual sampled text, not scripted greeting fallbacks. Replies can vary, be incorrect, or become garbled. Only the last 256 UTF-8 bytes of input are visible to this tiny model, including your message. Long system prompts cannot be followed reliably.

## Training data and licensing

General dialogue and validation examples were filtered from OpenAssistant OASST1 (Apache-2.0):
https://huggingface.co/datasets/OpenAssistant/oasst1

The dataset license and manifests are in `data/`. The focused dataset is modified: it combines authored greeting examples with a small sample of filtered dialogue. Automated filtering is not a comprehensive safety/privacy audit.

## Validation limits

Python syntax, CPU training, checkpoint load, timed stop, continuation from saved weights, and actual generation were tested in Linux. The final focused run lasted 240.1 seconds and completed 860 steps. Windows setup instructions were not executed on Windows. No production security, model safety, multilingual capability, or coding-quality guarantee is made.

Only load your own trusted checkpoint files. Review generated code; this application does not execute generated code or provide coding-agent tools.


## Hosting

Publishing this repository does not deploy the AI. The Python process must run on your computer or a persistent compute host. Temporary sandbox links do not provide guaranteed uptime.

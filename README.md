# Node AI — experimental CPU conversation model

This repository includes your Node model's trained weights, React interface, and local CPU backend. No pretrained model was substituted. The included checkpoint has completed an additional 600-second conversation-training run (2,189 steps), continuing the earlier focused greeting model. The repo now also includes a 4-hour Colab continuation path with automatic checkpoint selection and baseline regression guards. It is NOT a generally fluent assistant: practiced greetings can work while new phrasings and unseen names can fail.

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
- An expanded conversation-training corpus and a separate held-out English validation file. The corpus blends authored short dialogues, longer follow-up conversations, memory/name-recall practice, uncertainty honesty, and filtered general conversation.
- Backend support for continuing from a trusted checkpoint (`--resume`) and timed training (`--seconds`). The interface's **Train from scratch** button still initializes random weights and replaces the checkpoint after confirmation. Do not click it merely to chat.
- Training/evaluation receipts in `training-checks.json`. Greeting checks overlap with the training examples and do not establish generalization or reliability.

The model generates actual sampled text, not scripted greeting fallbacks. Replies can vary, be incorrect, or become garbled. Only the last 256 UTF-8 bytes of input are visible to this tiny model, including your message. Long system prompts cannot be followed reliably.

## Training data and licensing

General dialogue and validation examples were filtered from OpenAssistant OASST1 (Apache-2.0):
https://huggingface.co/datasets/OpenAssistant/oasst1

The dataset license and manifests are in `data/`. The focused dataset is modified: it combines authored greeting examples with a small sample of filtered dialogue. Automated filtering is not a comprehensive safety/privacy audit.

## Validation limits

Python syntax, CPU training, checkpoint load, timed stop, continuation from saved weights, and actual generation were tested in Linux. The latest continuation lasted 600.3 seconds and completed 2,189 steps. Windows setup instructions were not executed on Windows. No production security, model safety, multilingual capability, or coding-quality guarantee is made.

Only load your own trusted checkpoint files. Review generated code; this application does not execute generated code or provide coding-agent tools.


## Hosting

Publishing this repository does not deploy the AI. The Python process must run on your computer or a persistent compute host. Temporary sandbox links do not provide guaranteed uptime.


## Latest conversation update

The interface now sends recent user/model turns instead of only the latest message. Workspace notices are excluded, and the request is bounded. The model still sees only the last **256 UTF-8 bytes**, so most longer conversation history is unavailable. This is not long-term memory.

Actual qualitative checks are saved in `training-checks.json`. Greetings and a practiced name-recall example worked. An unseen name (Maya) was incorrectly recalled as Sam, and a novel greeting received an unrelated answer. These failures show memorization and poor generalization; the model is not a dependable conversational assistant.

## Put the project on your Desktop

Download the repository ZIP (or the supplied Node-Desktop ZIP), then extract the project folder onto your computer's Desktop. Open a terminal inside that folder and follow [HOW_TO_RUN.md](HOW_TO_RUN.md). A remote assistant cannot directly copy files into your local Desktop.


## Conversation-input tests

With Node.js installed, run `npm test` to check that recent turns are retained, workspace notices are excluded, and requests are capped. These are interface tests, not model-quality benchmarks.


Checkpoint-selection and notebook checks (requires the Python dependencies):

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Google Colab CPU experiment

[Open the 4-hour CPU conversation notebook in Colab](https://colab.research.google.com/github/ItzJoker2YT-city/node-ai/blob/main/colab/Node_30_minute_conversation.ipynb). Select a CPU runtime (no hardware accelerator), then run the cells in order. The notebook explicitly uses `--device cpu --batch-size 2` and continues the included checkpoint for 14,400 seconds. No GPU or paid upgrade is requested.

The script records the execution device, saves periodic weights, and compares before/after conversation checks. `best.pt` only accepts candidates with held-out loss no worse than the starting checkpoint and no lost points on any baseline conversation case. Eligible candidates rank by rule-based conversation score first and held-out validation loss second; `final.pt` is still kept separately. These keyword-based checks are imperfect and partly overlap training; they do not guarantee better conversation. More training can still overfit, and fluent replies are not guaranteed. The model still only sees the latest 256 bytes (not a full long conversation). Training lasts about four hours, plus evaluation overhead. Colab may terminate earlier; runtime-local results can be lost. Existing output folders are never overwritten. Download results, then disconnect/delete the runtime. This workflow does not mount Google Drive or purchase compute. CUDA remains an optional script flag for separately authorized future experiments, but the notebook does not use it.

## Private web dashboard

For the login-protected port-8000 dashboard, browser training controls, live run monitoring, checkpoint downloads, and chat playground, see [DASHBOARD.md](DASHBOARD.md). Run `npm run build:dashboard` then `python backend/dashboard.py`. The original local interface remains unchanged.

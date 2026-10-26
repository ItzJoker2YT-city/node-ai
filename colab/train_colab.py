"""Timed Colab conversation continuation; CPU by default; CUDA only when explicitly requested.

No pretrained model is downloaded. Continue the repository's own Node checkpoint.
Select the best checkpoint using held-out loss plus basic conversation-quality checks instead of assuming longer training is better.
"""
import argparse
import json
import math
import os
import signal
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import torch
import torch.nn.functional as F
from model import NodeTiny


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def save_weights(path, model):
    # CPU-only tensors make results portable to the original Windows/Linux app.
    payload = {'config': model.config, 'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    temporary = Path(path).with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def normalize_text(text):
    return ' '.join(text.lower().split())


def has_any(text, phrases):
    lowered = normalize_text(text)
    return any(phrase in lowered for phrase in phrases)


def prompt_checks():
    return [
        {
            'case': 'practiced greeting',
            'prompt': 'User: hi\nNode:',
            'required_any': [['hi', 'hello'], ['help', 'working on', 'what can i', 'conversation']],
        },
        {
            'case': 'practiced identity',
            'prompt': 'User: what is your name?\nNode:',
            'required_any': [['node']],
            'forbidden': ['chatgpt'],
        },
        {
            'case': 'new greeting phrasing',
            'prompt': 'User: Hi there! Can we have a conversation?\nNode:',
            'required_any': [['yes', 'sure', 'hello', 'hi'], ['conversation', 'talk', 'help', 'what would you like']],
        },
        {
            'case': 'held-out name follow-up',
            'prompt': 'User: my name is Zephyra\nNode: Nice to meet you, Zephyra.\nUser: what is my name?\nNode:',
            'required_any': [['zephyra']],
            'forbidden': ['sam'],
        },
        {
            'case': 'practiced factual question',
            'prompt': 'User: What is the difference between rain and snow?\nNode:',
            'required_any': [['rain'], ['snow'], ['liquid', 'water'], ['ice', 'frozen']],
        },
        {
            'case': 'follow-up question',
            'prompt': 'User: I had a bad day.\nNode:',
            'required_any': [['sorry', 'that sounds', 'rough', 'hard day'], ['what happened', 'want to talk', 'help']],
        },
        {
            'case': 'uncertainty honesty',
            'prompt': "User: If you are not sure about something, what should you do?\nNode:",
            'required_any': [['not sure', "don't know", 'be honest', 'say so', 'say that i am not sure']],
        },
    ]


def should_update_best(candidate_loss, candidate_quality, current_loss, current_quality,
                       baseline_loss, baseline_quality):
    """Heuristic safeguards, not a guarantee of general conversational quality."""
    if not math.isfinite(candidate_loss) or candidate_loss > baseline_loss:
        return False
    # Never trade away a baseline case to gain points on unrelated prompts.
    if any(candidate['score'] < original['score']
           for candidate, original in zip(candidate_quality['cases'], baseline_quality['cases'])):
        return False
    return (candidate_quality['passed'], -candidate_loss) > (current_quality['passed'], -current_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=14400)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--stop-file', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=ROOT / 'colab_outputs')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 14400 or not 1 <= args.batch_size <= 128:
        parser.error('Duration must be 1–14400 seconds and batch size 1–128.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No Colab GPU assigned. Select Runtime > Change runtime type > GPU, then reconnect. No CPU fallback was started.')
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(3, os.cpu_count() or 1))
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise RuntimeError('Output directory is not empty. Download the old results and use a new --output directory; nothing was overwritten.')
    checkpoint = torch.load(ROOT / 'data' / 'node.pt', map_location='cpu', weights_only=True)
    model = NodeTiny(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    context = model.config['context']

    def load_bytes(name):
        data = (ROOT / 'data' / name).read_bytes()
        if len(data) <= context:
            raise RuntimeError(f'{name} is too short for {context}-byte context.')
        return torch.tensor(list(data), dtype=torch.uint8, device=device)

    train = load_bytes('corpus.txt')
    validation = load_bytes('validation.txt')
    offsets = torch.arange(context, device=device)
    # Fix validation windows for comparable baseline/after checks.
    generator = torch.Generator(device='cpu').manual_seed(2026)
    val_starts = torch.randint(0, len(validation) - context, (32,), generator=generator).to(device)
    val_indices = val_starts[:, None] + offsets[None, :]
    val_x = validation[val_indices].long()
    val_y = validation[val_indices + 1].long()

    def evaluate():
        model.eval()
        losses = []
        with torch.inference_mode():
            for i in range(0, len(val_x), args.batch_size):
                x, y = val_x[i:i+args.batch_size], val_y[i:i+args.batch_size]
                loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
                losses.append((float(loss), len(x)))
        return sum(value * count for value, count in losses) / sum(count for _, count in losses)

    def reply(prompt):
        # Sampling checks must not reset the training RNG/dropout stream.
        devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1234)
            return sample_reply(prompt)

    def sample_reply(prompt):
        model.eval()
        ids = torch.tensor([list(prompt.encode('utf-8')[-context:]) or [10]], dtype=torch.long, device=device)
        output = []
        with torch.inference_mode():
            for _ in range(128):
                logits = model(ids[:, -context:])[:, -1] / 0.7
                values, indices = torch.topk(logits, 30)
                token = indices.gather(-1, torch.multinomial(F.softmax(values, dim=-1), 1))
                output.append(int(token.item()))
                ids = torch.cat([ids, token], dim=1)
                if b'\nUser:' in bytes(output):
                    output = list(bytes(output).split(b'\nUser:', 1)[0])
                    break
        text = bytes(output).decode('utf-8', errors='replace')
        return ''.join(c for c in text if c in '\n\t\r' or c.isprintable())

    checks = prompt_checks()

    def evaluate_conversation_quality():
        reports = []
        total_passed = 0
        total_possible = 0
        for spec in checks:
            output = reply(spec['prompt'])
            passed = []
            failed = []
            stripped = output.strip()
            base_checks = {
                'non_empty': bool(stripped),
                'no_role_leak': 'User:' not in output and 'Node:' not in output,
                'printable_length': len(stripped) >= 4,
            }
            for label, ok in base_checks.items():
                (passed if ok else failed).append(label)
            for idx, options in enumerate(spec.get('required_any', []), 1):
                ok = has_any(output, options)
                (passed if ok else failed).append(f'required_any_{idx}')
            for phrase in spec.get('forbidden', []):
                ok = phrase not in normalize_text(output)
                (passed if ok else failed).append(f'forbidden:{phrase}')
            score = len(passed)
            possible = len(passed) + len(failed)
            total_passed += score
            total_possible += possible
            reports.append({
                'case': spec['case'],
                'prompt': spec['prompt'],
                'output': output,
                'score': score,
                'max_score': possible,
                'passed_checks': passed,
                'failed_checks': failed,
            })
        return {
            'passed': total_passed,
            'possible': total_possible,
            'ratio': total_passed / total_possible if total_possible else 0.0,
            'cases': reports,
        }

    baseline = evaluate()
    before_quality = evaluate_conversation_quality()
    best = baseline
    best_quality = before_quality
    save_weights(args.output / 'best.pt', model)
    save_weights(args.output / 'before.pt', model)
    atomic_json(args.output / 'baseline_quality.json', {'validation_loss': baseline, 'quality': before_quality})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    started = time.monotonic()
    wall_start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    step = 0
    last_evaluation = last_save = last_log = last_quality = 0.0
    validation_loss = baseline
    quality_report = before_quality
    loss_value = None
    hardware = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    def report(state):
        value = {
            'state': state,
            'started_at': wall_start,
            'elapsed_seconds': round(time.monotonic()-started, 1),
            'target_seconds': args.seconds,
            'step': step,
            'device': str(device),
            'hardware': hardware,
            'batch_size': args.batch_size,
            'context_bytes': context,
            'training_loss': loss_value,
            'baseline_validation_loss': baseline,
            'validation_loss': validation_loss,
            'best_validation_loss': best,
            'baseline_conversation_score': before_quality['passed'],
            'baseline_conversation_possible': before_quality['possible'],
            'conversation_score': quality_report['passed'],
            'conversation_possible': quality_report['possible'],
            'best_conversation_score': best_quality['passed'],
            'best_conversation_possible': best_quality['possible'],
        }
        atomic_json(args.output / 'progress.json', value)
        print(json.dumps(value), flush=True)
        return value

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    report('training')
    while time.monotonic() - started < args.seconds:
        if stop_requested or (args.stop_file and args.stop_file.exists()):
            stop_requested = True
            break
        elapsed = time.monotonic() - started
        progress = min(1.0, elapsed / args.seconds)
        learning_rate = (1e-5 + (1e-4-1e-5) * 0.5 * (1+math.cos(math.pi*progress))) * min(1.0, (step+1)/50)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        model.train()
        starts = torch.randint(0, len(train) - context, (args.batch_size,), device=device)
        indices = starts[:, None] + offsets[None, :]
        x, y = train[indices].long(), train[indices+1].long()
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss. Existing best checkpoint is retained.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        loss_value = float(loss.detach())
        elapsed = time.monotonic() - started
        if elapsed - last_evaluation >= 60:
            validation_loss = evaluate()
            last_evaluation = elapsed
        if elapsed - last_quality >= 180:
            # Loss and conversation checks must evaluate the exact same weights.
            validation_loss = evaluate()
            quality_report = evaluate_conversation_quality()
            if should_update_best(validation_loss, quality_report, best, best_quality, baseline, before_quality):
                best = validation_loss
                best_quality = quality_report
                save_weights(args.output / 'best.pt', model)
            last_quality = elapsed
        if elapsed - last_save >= 300:
            save_weights(args.output / 'latest.pt', model)
            last_save = elapsed
        if elapsed - last_log >= 30:
            report('training')
            last_log = elapsed

    save_weights(args.output / 'latest.pt', model)
    save_weights(args.output / 'final.pt', model)
    validation_loss = evaluate()
    quality_report = evaluate_conversation_quality()
    if should_update_best(validation_loss, quality_report, best, best_quality, baseline, before_quality):
        best = validation_loss
        best_quality = quality_report
        save_weights(args.output / 'best.pt', model)
    run = report('stopped' if stop_requested else 'training_complete')
    final_quality = quality_report
    best_checkpoint = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best_checkpoint['model'])
    best_quality_report = evaluate_conversation_quality()
    atomic_json(args.output / 'evaluation.json', {
        'run': run,
        'before_quality': before_quality,
        'after_final_quality': final_quality,
        'after_best_quality': best_quality_report,
        'selection_policy': 'Candidates must not exceed baseline validation loss or lose points on any baseline conversation case. Eligible candidates rank by total conversation-check points, then lower validation loss. The original baseline remains best.pt unless a candidate qualifies.',
        'note': 'Rule-based checks are weak proxies, not proof of normal conversation or a guarantee against degradation. Some prompts overlap training. Inspect outputs on fresh conversations before deploying. Context is limited to 256 bytes.'
    })
    # Stop only the job; no runtime deletion, Drive mounting, account changes or purchasing.
    report('stopped' if stop_requested else 'complete')


if __name__ == '__main__':
    main()

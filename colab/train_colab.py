"""Timed Colab conversation continuation; CPU by default; CUDA only when explicitly requested.

No pretrained model is downloaded. Continue the repository's own Node checkpoint.
Select the best validation checkpoint instead of assuming longer training is better.
"""
import argparse
import json
import math
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, default=ROOT / 'colab_outputs')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 1800 or not 1 <= args.batch_size <= 128:
        parser.error('Duration must be 1–1800 seconds and batch size 1–128.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No Colab GPU assigned. Select Runtime > Change runtime type > GPU, then reconnect. No CPU fallback was started.')
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(3, os.cpu_count() or 1))
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
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
        model.eval()
        torch.manual_seed(1234)
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

    prompts = [
        ('practiced greeting', 'User: hi\nNode:'),
        ('practiced identity', 'User: what is your name?\nNode:'),
        ('new greeting phrasing', 'User: Hi there! Can we have a conversation?\nNode:'),
        ('unseen name follow-up', 'User: my name is Maya\nNode: Nice to meet you, Maya.\nUser: what is my name?\nNode:'),
        ('unseen factual question', 'User: What is the difference between rain and snow?\nNode:'),
    ]
    before = [{'case': label, 'prompt': text, 'output': reply(text)} for label, text in prompts]
    baseline = evaluate()
    best = baseline
    save_weights(args.output / 'best.pt', model)
    save_weights(args.output / 'before.pt', model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    started = time.monotonic()
    wall_start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    step = 0
    last_evaluation = last_save = last_log = 0.0
    validation_loss = baseline
    loss_value = None
    hardware = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    def report(state):
        value = {'state': state, 'started_at': wall_start, 'elapsed_seconds': round(time.monotonic()-started, 1),
                 'target_seconds': args.seconds, 'step': step, 'device': str(device), 'hardware': hardware,
                 'batch_size': args.batch_size, 'context_bytes': context, 'training_loss': loss_value,
                 'baseline_validation_loss': baseline, 'validation_loss': validation_loss, 'best_validation_loss': best}
        atomic_json(args.output / 'progress.json', value)
        print(json.dumps(value), flush=True)
        return value

    report('training')
    while time.monotonic() - started < args.seconds:
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
        if elapsed - last_evaluation >= 30:
            validation_loss = evaluate()
            if validation_loss < best:
                best = validation_loss
                save_weights(args.output / 'best.pt', model)
            last_evaluation = elapsed
        if elapsed - last_save >= 60:
            save_weights(args.output / 'latest.pt', model)
            last_save = elapsed
        if elapsed - last_log >= 20:
            report('training')
            last_log = elapsed

    save_weights(args.output / 'final.pt', model)
    validation_loss = evaluate()
    if validation_loss < best:
        best = validation_loss
        save_weights(args.output / 'best.pt', model)
    run = report('training_complete')
    final_replies = [{'case': label, 'prompt': text, 'output': reply(text)} for label, text in prompts]
    best_checkpoint = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best_checkpoint['model'])
    best_replies = [{'case': label, 'prompt': text, 'output': reply(text)} for label, text in prompts]
    atomic_json(args.output / 'evaluation.json', {'run': run, 'before': before, 'after_final': final_replies,
        'after_best_validation': best_replies, 'note': 'Greeting checks overlap training. Unseen prompts are qualitative checks, not a validated fluency benchmark. More training can overfit. The best validation checkpoint can remain the original baseline.'})
    # Stop only the job; no runtime deletion, Drive mounting, account changes or purchasing.
    report('complete')


if __name__ == '__main__':
    main()

"""Subprocess worker for training and inference; torch is loaded only here."""
import argparse
import math
import subprocess
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F
from model import NodeTiny, CONFIG

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CHECKPOINT = DATA / 'node.pt'
PROGRESS = DATA / 'progress.json'
STOP = DATA / 'stop.flag'

def write_progress(**values):
    temp = PROGRESS.with_suffix('.tmp')
    temp.write_text(json.dumps(values), encoding='utf-8')
    os.replace(temp, PROGRESS)

def save(model, optimizer=None):
    temp = CHECKPOINT.with_suffix('.tmp')
    payload={'config': model.config, 'model': model.state_dict()}
    if optimizer is not None: payload['optimizer']=optimizer.state_dict()
    torch.save(payload, temp)
    os.replace(temp, CHECKPOINT)

def train(steps, seconds=0, resume=False):
    torch.set_num_threads(3)
    torch.manual_seed(42)
    raw = (DATA / 'corpus.txt').read_bytes()
    if len(raw) < 1024:
        raise ValueError('At least 1,024 UTF-8 bytes are required.')
    # Use uint8 storage; convert just the sampled minibatch to long indices.
    tokens = torch.tensor(list(raw), dtype=torch.uint8)
    validation_path = DATA / 'validation.txt'
    if validation_path.exists():
        training = tokens
        validation = torch.tensor(list(validation_path.read_bytes()), dtype=torch.uint8)
    else:
        split = int(len(tokens) * 0.9)
        training, validation = tokens[:split], tokens[split:]
    context = min(CONFIG['context'], len(validation) - 1)
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=True) if resume else None
    model=NodeTiny(checkpoint['config'] if checkpoint else None)
    if checkpoint: model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    optimizer_restored=bool(checkpoint and 'optimizer' in checkpoint)
    if optimizer_restored: optimizer.load_state_dict(checkpoint['optimizer'])
    total = sum(p.numel() for p in model.parameters())
    started = time.time()
    started_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started))
    last_saved = started
    step = 0
    loss_value = val_value = None
    def report(state, **extra):
        write_progress(state=state, step=step, steps=steps, loss=loss_value,
                       val_loss=val_value, parameters=total,
                       elapsed_seconds=round(time.time()-started, 1),
                       duration_limit_seconds=seconds or None, started_at=started_at,
                       context_bytes=model.config['context'], objective='English conversation',
                       resumed_from_checkpoint=resume, optimizer_restored=optimizer_restored,
                       learning_rate=optimizer.param_groups[0]['lr'], **extra)
    def batch(source):
        positions = torch.randint(0, len(source) - context, (2,))
        x = torch.stack([source[int(i):int(i)+context] for i in positions]).long()
        y = torch.stack([source[int(i)+1:int(i)+context+1] for i in positions]).long()
        return x, y
    report('training')
    for step in range(1, steps+1):
        if seconds and time.time()-started>=seconds:
            step -= 1
            save(model, optimizer)
            report('complete', completed_reason='duration_limit')
            return
        if STOP.exists():
            # If at least one step ran, preserve useful partial work atomically.
            step -= 1
            if step:
                save(model, optimizer)
            report('stopped')
            return
        if seconds:
            fraction=min(1.0,(time.time()-started)/seconds)
            lr=(3e-5+0.5*(3e-4-3e-5)*(1+math.cos(math.pi*fraction)))*min(1.0,step/100)
            for group in optimizer.param_groups: group['lr']=lr
        model.train()
        x, y = batch(training)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss; checkpoint was not updated.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach())
        if step == 1 or step % 10 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vals=[]
                for _ in range(3):
                    vx, vy = batch(validation)
                    vals.append(float(F.cross_entropy(model(vx).reshape(-1,256), vy.reshape(-1))))
                val_value = sum(vals)/len(vals)
            report('training')
        if step % 100 == 0 or time.time()-last_saved>=60:
            save(model, optimizer)
            last_saved=time.time()
    save(model, optimizer)
    report('complete', completed_reason='step_limit')

def generate():
    torch.set_num_threads(3)
    request = json.loads(sys.stdin.read())
    # Checkpoints must be created by this application; never load untrusted .pt files.
    ckpt = torch.load(Path(os.environ.get('NODE_MODEL_PATH', CHECKPOINT)), map_location='cpu', weights_only=True)
    model = NodeTiny(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    prompt = request['prompt'].encode('utf-8')
    ids = torch.tensor([list(prompt[-model.config['context']:]) or [10]], dtype=torch.long)
    output = []
    with torch.inference_mode():
        for _ in range(min(256, max(1, int(request.get('max_tokens',120))))):
            temperature = max(0.2, min(1.5, float(request.get('temperature', 0.8))))
            logits = model(ids[:, -model.config['context']:])[:, -1] / temperature
            values, indices = torch.topk(logits, 40)
            selected = torch.multinomial(F.softmax(values, dim=-1), 1)
            token = indices.gather(-1, selected)
            output.append(int(token.item()))
            ids = torch.cat((ids, token), dim=1)
            if b'\nUser:' in bytes(output):
                output=list(bytes(output).split(b'\nUser:',1)[0])
                break
    text = bytes(output).decode('utf-8', errors='replace')
    # Remove terminal/control characters while keeping whitespace useful for code.
    text = ''.join(c for c in text if c in '\n\r\t' or c.isprintable())
    print(json.dumps({'text':text}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','generate'])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seconds', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    DATA.mkdir(exist_ok=True)
    if args.mode == 'generate':
        generate()
    else:
        try:
            train(args.steps, args.seconds, args.resume)
            if args.seconds >= 60 and json.loads(PROGRESS.read_text()).get('state') == 'complete':
                checks=[]
                for prompt in ['hi', 'hello', 'what is your name?', 'how are you?', 'can you help me?', 'what is a variable?']:
                    result=subprocess.run([sys.executable, str(Path(__file__)), 'generate'], input=json.dumps({'prompt':'User: '+prompt+'\nNode:', 'max_tokens':128}), text=True, encoding='utf-8', capture_output=True, timeout=120)
                    try: answer=json.loads(result.stdout).get('text','') if result.returncode==0 else '(generation failed)'
                    except ValueError: answer='(invalid generation result)'
                    checks.append({'prompt':prompt,'output':answer})
                (DATA/'conversation-eval.json').write_text(json.dumps({'note':'Raw qualitative checks; not a fluency score. Some prompts match training greetings.', 'checks':checks},indent=2),encoding='utf-8')
        except Exception as e:
            previous = {}
            try:
                previous = json.loads(PROGRESS.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
            previous.update(state='error', error=str(e))
            write_progress(**previous)
            raise

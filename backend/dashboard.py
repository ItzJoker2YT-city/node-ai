"""Authenticated single-owner dashboard. Bind to loopback behind an HTTPS proxy."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get('NODE_DASHBOARD_STATE', ROOT / 'dashboard_state'))
DATA = ROOT / 'data'
PORT = int(os.environ.get('PORT', '8000'))
PUBLIC_ORIGIN = os.environ.get('NODE_PUBLIC_ORIGIN', '').rstrip('/')
EXTERNAL = Path(os.environ['NODE_EXTERNAL_OUTPUT']) if os.environ.get('NODE_EXTERNAL_OUTPUT') else None
EXTERNAL_SCRIPT = Path(os.environ['NODE_EXTERNAL_SCRIPT']).resolve() if os.environ.get('NODE_EXTERNAL_SCRIPT') else None
TRAINER = ROOT / 'colab' / 'train_colab.py'
LOCK = threading.RLock()
GENERATE_LOCK = threading.Lock()
SESSIONS = {}
FAILURES = {}
CHILDREN = {}
MAX_DATA = 8 * 1024 * 1024
SESSION_SECONDS = 8 * 3600
FILES = {'before.pt', 'best.pt', 'latest.pt', 'final.pt', 'evaluation.json', 'progress.json'}
AUTH = {}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()


def initialize():
    global AUTH
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE, 0o700)
    (STATE / 'runs').mkdir(exist_ok=True, mode=0o700)
    (STATE / 'backups').mkdir(exist_ok=True, mode=0o700)
    path = STATE / 'auth.json'
    if not path.exists():
        password = secrets.token_urlsafe(24)
        salt = secrets.token_hex(16)
        atomic_json(path, {'username': 'admin', 'salt': salt, 'hash': password_hash(password, salt), 'must_change': True})
        fd = os.open(STATE / 'initial-login.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(f'Username: admin\nTemporary password: {password}\nChange this password after signing in.\n')
        print('Initial credentials saved in dashboard_state/initial-login.txt (owner-only).', flush=True)
    AUTH = read_json(path)
    if not all(k in AUTH for k in ('username', 'salt', 'hash', 'must_change')):
        raise RuntimeError('Invalid authentication configuration; refusing to start.')


def audit(action, detail=''):
    with LOCK:
        with (STATE / 'audit.jsonl').open('a') as out:
            out.write(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'action': action, 'detail': detail}) + '\n')


def run_path(run_id):
    if run_id == 'current' and EXTERNAL:
        return EXTERNAL
    if not re.fullmatch(r'run-[0-9]{8}T[0-9]{6}-[a-f0-9]{6}', run_id):
        raise Error('Unknown run.', 404)
    path = STATE / 'runs' / run_id
    if not path.is_dir():
        raise Error('Unknown run.', 404)
    return path


def active_jobs():
    jobs = []
    for child in list(CHILDREN.values()):
        child.poll()  # Reap finished child processes.
    # Also detect the pre-existing run; never launch a duplicate after server restart.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
            if len(args) < 2 or not any('python' in Path(a).name for a in args[:1]):
                continue
            script = Path(args[1]).resolve()
            if script not in {TRAINER.resolve(), EXTERNAL_SCRIPT}:
                continue
            if '--output' in args:
                output = Path(args[args.index('--output') + 1]).resolve()
            else:
                output = script.parents[1] / 'colab_outputs'
            if EXTERNAL and output == EXTERNAL.resolve():
                jobs.append({'id': 'current', 'pid': int(proc.name), 'stoppable': False})
            elif output.name == 'weights' and output.parent.parent == (STATE / 'runs').resolve():
                jobs.append({'id': output.parent.name, 'pid': int(proc.name), 'stoppable': True})
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return jobs


def tail(path, limit=24000):
    try:
        with Path(path).open('rb') as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(0, size-limit))
            return file.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''


def runs():
    paths = [(p.name, p) for p in sorted((STATE / 'runs').glob('run-*'), reverse=True) if p.is_dir()]
    if EXTERNAL and EXTERNAL.exists():
        paths.insert(0, ('current', EXTERNAL))
    active = {job['id']: job for job in active_jobs()}
    result = []
    for run_id, path in paths[:100]:
        artifacts = path if run_id == 'current' else path / 'weights'
        progress = read_json(artifacts / 'progress.json')
        if run_id in active:
            progress['state'] = 'training' if progress.get('state') != 'stopping' else 'stopping'
        elif progress.get('state') in ('training', 'training_complete', 'stopping'):
            progress['state'] = 'interrupted'
        elif not progress:
            meta = read_json(path / 'meta.json')
            progress = {'state': 'error' if run_id not in active else 'starting', **meta}
        result.append({'id': run_id, 'progress': progress, 'active': run_id in active,
                       'stoppable': active.get(run_id, {}).get('stoppable', False),
                       'files': [{'name': name, 'bytes': (artifacts / name).stat().st_size}
                                 for name in sorted(FILES) if (artifacts / name).is_file()]})
    return result


def system_info():
    memory = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable'):
                memory[key] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass
    disk = shutil.disk_usage(ROOT)
    return {'cpu_count': os.cpu_count(), 'load': list(os.getloadavg()), 'memory': memory,
            'disk': {'total': disk.total, 'free': disk.free}, 'python': sys.version.split()[0],
            'torch_available': importlib.util.find_spec('torch') is not None}


class Error(Exception):
    def __init__(self, message, code=400):
        self.code = code
        super().__init__(message)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, fmt, *args):
        # Do not log request bodies, cookie values, or query strings.
        print(f'{self.command} {urlparse(self.path).path}', flush=True)

    def origins(self):
        return {f'http://localhost:{PORT}', f'http://127.0.0.1:{PORT}'} | ({PUBLIC_ORIGIN} if PUBLIC_ORIGIN else set())

    def check_host(self):
        allowed = {urlparse(origin).netloc for origin in self.origins()}
        if self.headers.get('Host') not in allowed:
            raise Error('Invalid host.', 403)

    def send(self, code, value, content_type='application/json; charset=utf-8', cookie=None, filename=None):
        raw = json.dumps(value).encode() if content_type.startswith('application/json') else value
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
        hashes = []
        if content_type.startswith('text/html'):
            for script in re.findall(rb'<script>(.*?)</script>', raw, re.S):
                hashes.append("'sha256-" + base64.b64encode(hashlib.sha256(script).digest()).decode() + "'")
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' " + ' '.join(hashes) + "; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if cookie:
            self.send_header('Set-Cookie', cookie)
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(raw)

    def session(self, optional=False):
        jar = SimpleCookie()
        try:
            jar.load(self.headers.get('Cookie', ''))
        except Exception:
            pass
        token = jar['node_session'].value if 'node_session' in jar else ''
        with LOCK:
            value = SESSIONS.get(token)
            if value and value['expires'] > time.time():
                return token, value
            SESSIONS.pop(token, None)
        if not optional:
            raise Error('Sign in to continue.', 401)
        return None, None

    def cookie(self, token, expired=False):
        secure = '; Secure' if PUBLIC_ORIGIN and (self.headers.get('Host') == urlparse(PUBLIC_ORIGIN).netloc or self.headers.get('Origin') == PUBLIC_ORIGIN) else ''
        return f'node_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={0 if expired else SESSION_SECONDS}{secure}'

    def do_GET(self):
        try:
            self.check_host()
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path in ('/', '/index.html'):
                return self.send(200, (ROOT / 'dist' / 'dashboard.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/health':
                return self.send(200, {'ok': True})
            if path == '/api/session':
                _, session = self.session(optional=True)
                return self.send(200, {'authenticated': bool(session), 'username': AUTH['username'] if session else None,
                                       'csrf': session['csrf'] if session else None, 'must_change': AUTH['must_change'] if session else False})
            self.session()
            if AUTH['must_change']:
                raise Error('Change the temporary password first.', 403)
            if path == '/api/dashboard':
                listing = runs()
                return self.send(200, {'runs': listing, 'system': system_info(),
                                      'dataset_bytes': (DATA / 'corpus.txt').stat().st_size,
                                      'selection': read_json(STATE / 'selection.json', {'run': 'original', 'file': 'node.pt'}),
                                      'audit': [json.loads(line) for line in tail(STATE / 'audit.jsonl', 16000).splitlines()[1:] if line.startswith('{')][-25:]})
            if path == '/api/logs':
                run_id = query.get('run', ['current'])[0]
                directory = run_path(run_id)
                log = Path(os.environ.get('NODE_EXTERNAL_LOG', '/root/node-training.log')) if run_id == 'current' else directory / 'train.log'
                return self.send(200, {'text': tail(log)})
            if path == '/api/evaluation':
                return self.send(200, read_json(output_path(query.get('run', ['current'])[0]) / 'evaluation.json'))
            if path == '/api/dataset':
                text = (DATA / 'corpus.txt').read_text(encoding='utf-8')
                return self.send(200, {'text': text, 'bytes': len(text.encode()), 'lines': len(text.splitlines()),
                                       'dialogues': len([part for part in text.split('\n\n') if part.strip()]),
                                       'backups': len(list((STATE / 'backups').glob('*.txt')))})
            if path == '/api/download':
                run_id, filename = query.get('run', [''])[0], query.get('file', [''])[0]
                if filename not in FILES:
                    raise Error('Unknown artifact.', 404)
                file = output_path(run_id) / filename
                if not file.is_file():
                    raise Error('Artifact not ready.', 404)
                return self.send(200, file.read_bytes(), 'application/octet-stream', filename=filename)
            raise Error('Not found.', 404)
        except Error as exc:
            self.send(exc.code, {'error': str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
            self.send(500, {'error': 'Dashboard request failed. See server log.'})

    def body(self):
        self.connection.settimeout(15)
        if not self.headers.get('Content-Type', '').startswith('application/json'):
            raise Error('JSON required.', 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= MAX_DATA * 6 + 1024:
                raise Error('Request too large.', 413)
            value = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeError):
            raise Error('Invalid JSON.')
        if not isinstance(value, dict):
            raise Error('JSON object required.')
        return value

    def do_POST(self):
        try:
            self.check_host()
            if self.headers.get('Origin') not in self.origins() or self.headers.get('Sec-Fetch-Site') == 'cross-site':
                raise Error('Cross-origin request denied.', 403)
            path = urlparse(self.path).path
            body = self.body()
            if path == '/api/login':
                now = time.time()
                ip = self.client_address[0]
                username, password = body.get('username'), body.get('password')
                if not isinstance(username, str) or not isinstance(password, str) or len(password) > 256:
                    raise Error('Invalid sign-in values.')
                with LOCK:
                    recent = [t for t in FAILURES.get(ip, []) if now-t < 900]
                    if len(recent) >= 10:
                        raise Error('Too many sign-in attempts. Wait 15 minutes.', 429)
                    # Reserve before hashing so parallel attempts cannot bypass the limit.
                    FAILURES[ip] = recent + [now]
                valid = hmac.compare_digest(password_hash(password, AUTH['salt']), AUTH['hash'])
                if not (hmac.compare_digest(username.encode(), AUTH['username'].encode()) and valid):
                    raise Error('Incorrect username or password.', 401)
                token = secrets.token_urlsafe(32)
                with LOCK:
                    FAILURES.pop(ip, None)
                    for old in list(SESSIONS):
                        if SESSIONS[old]['expires'] <= now:
                            SESSIONS.pop(old)
                    if len(SESSIONS) >= 32:
                        SESSIONS.pop(next(iter(SESSIONS)))
                    SESSIONS[token] = {'csrf': secrets.token_urlsafe(24), 'expires': now + SESSION_SECONDS}
                audit('Signed in')
                return self.send(200, {'ok': True}, cookie=self.cookie(token))
            token, session = self.session()
            if not hmac.compare_digest(self.headers.get('X-CSRF-Token', ''), session['csrf']):
                raise Error('Invalid request token. Reload the page.', 403)
            if path == '/api/logout':
                with LOCK:
                    SESSIONS.pop(token, None)
                return self.send(200, {'ok': True}, cookie=self.cookie('', True))
            if path == '/api/password':
                old, new = body.get('current'), body.get('password')
                if not isinstance(old, str) or not isinstance(new, str) or not 14 <= len(new) <= 256 or len(old) > 256:
                    raise Error('Use a new password containing 14–256 characters.')
                if old == new:
                    raise Error('Choose a different password.')
                with LOCK:
                    if not hmac.compare_digest(password_hash(old, AUTH['salt']), AUTH['hash']):
                        raise Error('Current password is incorrect.', 401)
                    salt = secrets.token_hex(16)
                    AUTH.update(salt=salt, hash=password_hash(new, salt), must_change=False)
                    atomic_json(STATE / 'auth.json', AUTH)
                    (STATE / 'initial-login.txt').unlink(missing_ok=True)
                    SESSIONS.clear()
                audit('Password changed; sessions revoked')
                return self.send(200, {'ok': True}, cookie=self.cookie('', True))
            if AUTH['must_change']:
                raise Error('Change the temporary password first.', 403)
            if path == '/api/train':
                seconds, batch = body.get('seconds'), body.get('batch_size', 2)
                if type(seconds) is not int or not 60 <= seconds <= 14400 or type(batch) is not int or not 1 <= batch <= 8:
                    raise Error('Duration: 60–14,400 seconds. Batch size: 1–8.')
                with LOCK:
                    if active_jobs():
                        raise Error('A training job is already active. No duplicate was started.', 409)
                    if importlib.util.find_spec('torch') is None:
                        raise Error('CPU PyTorch is not installed.', 503)
                    if shutil.disk_usage(ROOT).free < 250 * 1024 * 1024:
                        raise Error('At least 250 MB of free disk is required.', 409)
                    run_id = time.strftime('run-%Y%m%dT%H%M%S', time.gmtime()) + '-' + secrets.token_hex(3)
                    directory = STATE / 'runs' / run_id
                    directory.mkdir(mode=0o700)
                    atomic_json(directory / 'meta.json', {'target_seconds': seconds, 'batch_size': batch, 'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
                    with (directory / 'train.log').open('w') as log:
                        child = subprocess.Popen([sys.executable, str(TRAINER), '--seconds', str(seconds), '--device', 'cpu',
                                                  '--batch-size', str(batch), '--output', str(directory / 'weights'),
                                                  '--stop-file', str(directory / 'stop.flag')], cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
                    # The script requires an empty output folder. Keep metadata/logs outside it.
                    CHILDREN[run_id] = child
                    atomic_json(directory / 'process.json', {'pid': child.pid})
                audit('Training started', run_id)
                return self.send(200, {'run': run_id})
            if path == '/api/stop':
                run_id = body.get('run', '')
                directory = run_path(run_id)
                with LOCK:
                    job = next((job for job in active_jobs() if job['id'] == run_id), None)
                    if not job:
                        raise Error('This run is not active.', 409)
                    if not job['stoppable']:
                        raise Error('The existing terminal-launched run is monitor-only. Future dashboard runs support graceful stopping.', 409)
                    (directory / 'stop.flag').touch()
                audit('Graceful stop requested', run_id)
                return self.send(200, {'ok': True})
            if path == '/api/dataset':
                text = body.get('text')
                if not isinstance(text, str) or not 1024 <= len(text.encode('utf-8')) <= MAX_DATA:
                    raise Error('Dataset must contain 1 KB–8 MB of UTF-8 text.')
                with LOCK:
                    if active_jobs():
                        raise Error('Wait until training finishes before changing its dataset.', 409)
                    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + secrets.token_hex(3)
                    shutil.copy2(DATA / 'corpus.txt', STATE / 'backups' / (stamp + '.txt'))
                    temp = DATA / 'corpus.dashboard.tmp'
                    temp.write_text(text, encoding='utf-8')
                    os.replace(temp, DATA / 'corpus.txt')
                audit('Dataset replaced with backup', stamp)
                return self.send(200, {'ok': True})
            if path == '/api/select':
                run_id, filename = body.get('run'), body.get('file')
                if run_id == 'original' and filename == 'node.pt':
                    selected = DATA / 'node.pt'
                else:
                    if filename not in {'best.pt', 'before.pt', 'latest.pt', 'final.pt'}:
                        raise Error('Choose an application checkpoint.')
                    selected = output_path(run_id) / filename
                if not selected.is_file():
                    raise Error('Checkpoint is not ready.', 404)
                atomic_json(STATE / 'selection.json', {'run': run_id, 'file': filename})
                audit('Chat checkpoint selected', f'{run_id}/{filename}')
                return self.send(200, {'ok': True})
            if path == '/api/generate':
                prompt, count, temperature = body.get('prompt'), body.get('max_tokens', 120), body.get('temperature', 0.7)
                if not isinstance(prompt, str) or not 1 <= len(prompt) <= 20000 or type(count) is not int or not 1 <= count <= 256:
                    raise Error('Invalid prompt or generation length.')
                if type(temperature) not in (int, float) or not 0.2 <= temperature <= 1.5:
                    raise Error('Temperature must be 0.2–1.5.')
                if not GENERATE_LOCK.acquire(blocking=False):
                    raise Error('One chat response is already generating.', 409)
                try:
                    choice = read_json(STATE / 'selection.json', {'run': 'original', 'file': 'node.pt'})
                    selected = DATA / 'node.pt' if choice['run'] == 'original' else output_path(choice['run']) / choice['file']
                    env = dict(os.environ, NODE_MODEL_PATH=str(selected), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    result = subprocess.run([sys.executable, str(ROOT / 'backend' / 'worker.py'), 'generate'],
                                            input=json.dumps({'prompt': prompt, 'max_tokens': count, 'temperature': temperature}),
                                            text=True, capture_output=True, timeout=120, cwd=ROOT, env=env)
                    if result.returncode:
                        raise Error('Generation failed. Check the selected checkpoint.', 500)
                    return self.send(200, json.loads(result.stdout))
                finally:
                    GENERATE_LOCK.release()
            raise Error('Not found.', 404)
        except Error as exc:
            self.send(exc.code, {'error': str(exc)})
        except subprocess.TimeoutExpired:
            self.send(504, {'error': 'CPU generation timed out. Try fewer tokens.'})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
            self.send(500, {'error': 'Dashboard request failed. See server log.'})


def output_path(run_id):
    directory = run_path(run_id)
    return directory if run_id == 'current' else directory / 'weights'


if __name__ == '__main__':
    initialize()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    server.daemon_threads = True
    print(f'Authenticated Node dashboard: http://127.0.0.1:{PORT}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

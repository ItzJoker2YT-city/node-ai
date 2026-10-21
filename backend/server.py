"""Loopback-only web server. No cloud API, arbitrary file access, or code execution tool."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import threading
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
PORT = 8765
LOCK = threading.Lock()
PROCESS = None
GENERATING = False
WORKER = ROOT / 'backend' / 'worker.py'
ENV = dict(os.environ, OMP_NUM_THREADS='3', MKL_NUM_THREADS='3')
MAX_BYTES = 8 * 1024 * 1024

class APIError(Exception):
    def __init__(self, message, code=400):
        self.code = code
        super().__init__(message)

def running():
    return PROCESS is not None and PROCESS.poll() is None

def status():
    result = {'state':'idle','step':0,'steps':0,'val_loss':None}
    try:
        result.update(json.loads((DATA/'progress.json').read_text(encoding='utf-8')))
    except (OSError, ValueError):
        pass
    if running():
        result['state']='training'
    elif result['state']=='training':
        result.update(state='error', error='Training exited unexpectedly. See data/train.log.')
    result.update(checkpoint=(DATA/'node.pt').exists(),
                  dataset_bytes=(DATA/'corpus.txt').stat().st_size if (DATA/'corpus.txt').exists() else 0,
                  torch_available=importlib.util.find_spec('torch') is not None)
    return result

class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, kind='application/json; charset=utf-8'):
        content=json.dumps(body).encode('utf-8') if kind.startswith('application/json') else body
        self.send_response(code)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('X-Frame-Options','DENY')
        self.end_headers()
        self.wfile.write(content)

    def valid_host(self):
        return self.headers.get('Host','') in {f'127.0.0.1:{PORT}',f'localhost:{PORT}'}

    def do_GET(self):
        if not self.valid_host():
            return self.reply(403, {'error':'Invalid host.'})
        if self.path == '/api/status':
            return self.reply(200,status())
        if self.path in ('/','/index.html'):
            path=ROOT/'dist'/'index.html'
            if not path.exists():
                return self.reply(503,{'error':'Build the frontend using npm run build first.'})
            return self.reply(200,path.read_bytes(),'text/html; charset=utf-8')
        return self.reply(404,{'error':'Not found.'})

    def do_POST(self):
        global PROCESS, GENERATING
        try:
            if not self.valid_host():
                raise APIError('Invalid host.',403)
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'}:
                raise APIError('Cross-origin request denied.',403)
            if self.headers.get('Sec-Fetch-Site')=='cross-site':
                raise APIError('Cross-site request denied.',403)
            if not self.headers.get('Content-Type','').startswith('application/json'):
                raise APIError('JSON requests only.',415)
            length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>MAX_BYTES*6+1024:
                raise APIError('Invalid request size.',413)
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict):
                raise APIError('Expected a JSON object.')
            if self.path=='/api/dataset':
                text=body.get('text','')
                if not isinstance(text,str): raise APIError('Text must be a string.')
                raw=text.encode('utf-8')
                if not 1024<=len(raw)<=MAX_BYTES:
                    raise APIError('Dataset must contain 1,024 bytes to 8 MB of UTF-8 text.')
                with LOCK:
                    if running(): raise APIError('Stop training before replacing the dataset.',409)
                    temp=DATA/'corpus.tmp'
                    temp.write_bytes(raw)
                    os.replace(temp,DATA/'corpus.txt')
                return self.reply(200,{'bytes':len(raw)})
            if self.path=='/api/train':
                steps=body.get('steps',200)
                if isinstance(steps,bool) or not isinstance(steps,int) or not 10<=steps<=100000:
                    raise APIError('Steps must be an integer from 10 to 100,000.')
                resume=body.get('resume',False)
                if not isinstance(resume,bool): raise APIError('resume must be true or false.')
                if resume and not (DATA/'node.pt').exists(): raise APIError('No saved checkpoint to resume.')
                seconds=body.get('seconds',0)
                if isinstance(seconds,bool) or not isinstance(seconds,int) or not 0<=seconds<=3600:
                    raise APIError('Duration must be an integer from 0 to 3,600 seconds.')
                with LOCK:
                    if running() or GENERATING: raise APIError('Node is busy. Try again after the current task.',409)
                    if importlib.util.find_spec('torch') is None:
                        raise APIError('PyTorch is not installed. Follow the README setup instructions.',503)
                    if not (DATA/'corpus.txt').exists(): raise APIError('Save a dataset first.')
                    (DATA/'stop.flag').unlink(missing_ok=True)
                    (DATA/'progress.json').write_text(json.dumps({'state':'training','step':0,'steps':steps}),encoding='utf-8')
                    with open(DATA/'train.log','w',encoding='utf-8') as log:
                        PROCESS=subprocess.Popen([sys.executable,str(WORKER),'train','--steps',str(steps),'--seconds',str(seconds)]+(['--resume'] if resume else []),cwd=ROOT,stdout=log,stderr=log,env=ENV)
                return self.reply(200,{'started':True})
            if self.path=='/api/stop':
                with LOCK:
                    if not running(): raise APIError('No training run is active.',409)
                    (DATA/'stop.flag').touch()
                return self.reply(200,{'stopping':True})
            if self.path=='/api/generate':
                prompt=body.get('prompt','')
                if not isinstance(prompt,str) or not 1<=len(prompt)<=20000:
                    raise APIError('Prompt must contain 1–20,000 characters.')
                count=body.get('max_tokens',120)
                if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=256:
                    raise APIError('max_tokens must be an integer from 1 to 256.')
                with LOCK:
                    if running() or GENERATING: raise APIError('Node is busy. Wait for the current task.',409)
                    if not (DATA/'node.pt').exists(): raise APIError('No checkpoint. Train Node first.',409)
                    GENERATING=True
                try:
                    result=subprocess.run([sys.executable,str(WORKER),'generate'],input=json.dumps({'prompt':prompt,'max_tokens':count}),text=True,encoding='utf-8',capture_output=True,timeout=180,cwd=ROOT,env=ENV)
                    if result.returncode:
                        # Technical traceback goes only to the local server console.
                        print(result.stderr,file=sys.stderr)
                        raise APIError('Generation failed. Check the server console.',500)
                    answer=json.loads(result.stdout)
                finally:
                    with LOCK: GENERATING=False
                return self.reply(200,answer)
            raise APIError('Not found.',404)
        except APIError as e:
            self.reply(e.code,{'error':str(e)})
        except (ValueError,TypeError):
            self.reply(400,{'error':'Invalid JSON or request values.'})
        except subprocess.TimeoutExpired:
            self.reply(504,{'error':'CPU generation exceeded 180 seconds. Try a shorter generation.'})
        except Exception as e:
            print(repr(e),file=sys.stderr)
            self.reply(500,{'error':'Local server error. See the server console.'})

if __name__=='__main__':
    print(f'Node is available at http://127.0.0.1:{PORT}',flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if running():
            (DATA/'stop.flag').touch()
            try: PROCESS.wait(timeout=15)
            except subprocess.TimeoutExpired: PROCESS.terminate()
        server.server_close()

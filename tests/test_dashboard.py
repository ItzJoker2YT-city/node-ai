"""HTTP integration tests. Training tests use temporary output directories only."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='node-dashboard-test-')
        cls.state = Path(cls.temp.name)
        cls.password = secrets.token_urlsafe(24)
        salt = secrets.token_hex(16)
        hashed = hashlib.scrypt(cls.password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        (cls.state/'auth.json').write_text(json.dumps({'username':'admin','salt':salt,'hash':hashed,'must_change':False}))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); cls.port=sock.getsockname()[1]
        cls.origin=f'http://127.0.0.1:{cls.port}'
        env=dict(os.environ,PORT=str(cls.port),NODE_DASHBOARD_STATE=str(cls.state),NODE_PUBLIC_ORIGIN='https://ai.runhost.bond')
        env.pop('NODE_EXTERNAL_OUTPUT',None);env.pop('NODE_EXTERNAL_SCRIPT',None)
        cls.log=open(cls.state/'server.log','w')
        cls.server=subprocess.Popen([sys.executable,str(ROOT/'backend/dashboard.py')],env=env,stdout=cls.log,stderr=cls.log)
        for _ in range(100):
            try:
                if cls.request('GET','/api/health')[0]==200:break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('Test server did not start')

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate();cls.server.wait(timeout=10);cls.log.close();cls.temp.cleanup()

    @classmethod
    def request(cls,method,path,body=None,cookie=None,csrf=None,headers=None):
        h={'Origin':cls.origin,'Content-Type':'application/json'}
        if cookie:h['Cookie']=cookie
        if csrf:h['X-CSRF-Token']=csrf
        h.update(headers or {})
        conn=http.client.HTTPConnection('127.0.0.1',cls.port,timeout=30)
        conn.request(method,path,body=json.dumps(body) if body is not None else None,headers=h)
        response=conn.getresponse();raw=response.read();status=response.status;rh=dict(response.getheaders());conn.close()
        value=json.loads(raw) if rh.get('Content-Type','').startswith('application/json') else raw
        return status,value,rh

    def setUp(self):
        status,_,headers=self.request('POST','/api/login',{'username':'admin','password':self.password})
        self.assertEqual(status,200)
        self.cookie=headers['Set-Cookie'].split(';')[0]
        status,self.session,_=self.request('GET','/api/session',cookie=self.cookie)
        self.csrf=self.session['csrf']

    def test_unauthorized_api_and_artifacts_are_private(self):
        for path in ['/api/dashboard','/api/dataset','/api/logs?run=current','/api/download?run=current&file=best.pt']:
            self.assertEqual(self.request('GET',path)[0],401)

    def test_login_cookie_and_security_headers(self):
        status,_,h=self.request('POST','/api/login',{'username':'admin','password':self.password},headers={'Host':'ai.runhost.bond','Origin':'https://ai.runhost.bond'})
        self.assertEqual(status,200)
        self.assertIn('HttpOnly',h['Set-Cookie']);self.assertIn('Secure',h['Set-Cookie']);self.assertIn('SameSite=Strict',h['Set-Cookie'])
        status,html,h=self.request('GET','/')
        self.assertEqual(status,200);self.assertIn("sha256-",h['Content-Security-Policy']);self.assertEqual(h['X-Frame-Options'],'DENY')

    def test_host_origin_and_csrf_are_enforced(self):
        self.assertEqual(self.request('GET','/api/health',headers={'Host':'attacker.invalid'})[0],403)
        self.assertEqual(self.request('POST','/api/logout',{},self.cookie,self.csrf,{'Origin':'https://attacker.invalid'})[0],403)
        self.assertEqual(self.request('POST','/api/logout',{},self.cookie)[0],403)

    def test_reads_dataset_and_real_server_status(self):
        status,d,_=self.request('GET','/api/dashboard',cookie=self.cookie)
        self.assertEqual(status,200);self.assertGreater(d['dataset_bytes'],1024);self.assertGreater(d['system']['disk']['free'],0)
        status,d,_=self.request('GET','/api/dataset',cookie=self.cookie)
        self.assertEqual(status,200);self.assertEqual(d['bytes'],len(d['text'].encode()))

    def test_validation_and_artifact_traversal(self):
        self.assertEqual(self.request('POST','/api/train',{'seconds':0},self.cookie,self.csrf)[0],400)
        self.assertEqual(self.request('POST','/api/dataset',{'text':'tiny'},self.cookie,self.csrf)[0],400)
        self.assertEqual(self.request('GET','/api/download?run=../../&file=auth.json',cookie=self.cookie)[0],404)
        self.assertEqual(self.request('POST','/api/select',{'run':'../../','file':'best.pt'},self.cookie,self.csrf)[0],404)

    def test_logout_revokes_session(self):
        self.assertEqual(self.request('POST','/api/logout',{},self.cookie,self.csrf)[0],200)
        self.assertEqual(self.request('GET','/api/dashboard',cookie=self.cookie)[0],401)

    def test_password_change_requires_current_password(self):
        self.assertEqual(self.request('POST','/api/password',{'current':'wrong','password':secrets.token_urlsafe(24)},self.cookie,self.csrf)[0],401)

    def test_z_managed_training_and_graceful_stop(self):
        try:import torch
        except ImportError:self.skipTest('CPU torch not installed')
        status,d,_=self.request('POST','/api/train',{'seconds':60,'batch_size':2},self.cookie,self.csrf)
        self.assertEqual(status,200,d);run=d['run']
        self.assertEqual(self.request('POST','/api/train',{'seconds':60,'batch_size':2},self.cookie,self.csrf)[0],409)
        self.assertEqual(self.request('POST','/api/dataset',{'text':'abc\n'*400},self.cookie,self.csrf)[0],409)
        self.assertEqual(self.request('POST','/api/stop',{'run':run},self.cookie,self.csrf)[0],200)
        for _ in range(150):
            _,d,_=self.request('GET','/api/dashboard',cookie=self.cookie)
            record=next(r for r in d['runs'] if r['id']==run)
            if not record['active'] and record['progress'].get('state')=='stopped':break
            time.sleep(.2)
        else:self.fail('Managed training did not stop gracefully')
        self.assertIn('best.pt',[f['name'] for f in record['files']])
        self.assertEqual(self.request('GET','/api/evaluation?run='+run,cookie=self.cookie)[0],200)
        self.assertEqual(self.request('POST','/api/select',{'run':run,'file':'best.pt'},self.cookie,self.csrf)[0],200)
        self.assertEqual(self.request('GET','/api/download?run='+run+'&file=best.pt',cookie=self.cookie)[0],200)

class ProcessDiscoveryTests(unittest.TestCase):
    def test_relative_terminal_training_path_is_detected(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('dashboard_process_test', ROOT/'backend/dashboard.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'colab').mkdir()
            script = root/'colab/train_colab.py'
            script.write_text('import time; time.sleep(30)')
            module.EXTERNAL_SCRIPT = script.resolve()
            module.EXTERNAL = root/'colab_outputs'
            process = subprocess.Popen([sys.executable,'colab/train_colab.py'],cwd=root)
            try:
                time.sleep(.1)
                jobs = module.active_jobs()
                self.assertTrue(any(job['id']=='current' and job['pid']==process.pid for job in jobs))
            finally:
                process.terminate();process.wait(timeout=5)

if __name__=='__main__':unittest.main()

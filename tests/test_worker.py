"""Bounded worker CPU checks, no GPU/model weights. Run with python -B FILE."""
import ast
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('engine_worker',Path(__file__).resolve().parents[1]/'engine_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class Worker(unittest.TestCase):
    def test_help_no_torch(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(Path(w.__file__)),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--counts',r.stdout)
        self.assertNotIn('torch',sys.modules)

    def test_requests_and_single_user_contract(self):
        base=dict(type='generate',id='one',prompt='hello',max_tokens=3,temperature=0)
        self.assertEqual(w.validate_request(base,32,'greedy')['mode'],'greedy')
        for mutation in [dict(max_tokens=True),dict(temperature=1),dict(mode='dspark'),dict(stop=['']),
                         dict(model='other'),dict(prompt=''),dict(max_tokens=33),dict(messages=[dict(role='system',content='x')])]:
            with self.subTest(mutation=mutation),self.assertRaises(w.RequestError):
                w.validate_request(dict(base,**mutation),32,'greedy')
        chat=dict(type='generate',id='one',messages=[dict(role='user',content='hello')])
        self.assertEqual(w.validate_request(chat,1024,'lookup')['prompt'],'hello')

    def test_literal_stop_is_not_leaked_across_chunks(self):
        b=w.StopBuffer(['STOP','END'])
        self.assertEqual(b.feed('hello S'),'hello ')
        self.assertEqual(b.feed('TO'),'')
        self.assertEqual(b.feed('Pignored'),'')
        self.assertTrue(b.stopped);self.assertEqual(b.finish(),'')
        c=w.StopBuffer(['STOP']);self.assertEqual(c.feed('S'),'');self.assertEqual(c.finish(),'S')

    def test_reset_preserves_allocations_and_clears_every_request_tensor(self):
        class Tensor:
            def __init__(self):self.value='prior'
            def zero_(self):self.value=0
            def fill_(self,v):self.value=v
        names=('window','ck_codes','ck_scales','ik_codes','ik_scales','kv_state','score_state')
        layers=[SimpleNamespace(attn=SimpleNamespace(**{n:Tensor() for n in names})) for _ in range(2)]
        old=[id(getattr(l.attn,n)) for l in layers for n in names]
        model=SimpleNamespace(position=9,main_hidden='prior',state='prior',layers=layers,
            prefill_metadata='prior',engram=SimpleNamespace(hash=SimpleNamespace(cache=Tensor()),jobs=['completed']),
            _clear_verification=lambda:None)
        for unused in range(2):
            w.reset_model(model,lambda:SimpleNamespace(compress_kv=None))
            self.assertEqual(model.position,0);self.assertIsNone(model.main_hidden)
            self.assertFalse(hasattr(model,'prefill_metadata'));self.assertEqual(model.engram.jobs,[])
            self.assertEqual(old,[id(getattr(l.attn,n)) for l in layers for n in names])
            for layer in layers:
                for name in names:self.assertEqual(getattr(layer.attn,name).value,-float('inf') if name=='score_state' else 0)
            self.assertEqual(model.engram.hash.cache.value,0)

    def test_uniform_prior_is_optional_and_public_ready_is_minimal(self):
        tree=ast.parse(Path(w.__file__).read_bytes())
        self.assertNotIn('from ds41.generate',Path(w.__file__).read_text())
        self.assertNotIn('import eva',Path(w.__file__).read_text())
        ready=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':
                if any(k.arg=='type' and isinstance(k.value,ast.Constant) and k.value.value=='ready' for k in node.keywords):
                    ready.append({k.arg for k in node.keywords})
        self.assertEqual(ready,[{'type','model','context_limit','max_output_tokens','modes','startup_s'}])

    def test_cancel_before_activation_remains_queued(self):
        protocol=w.Protocol(io.StringIO());inbox=w.Inbox(protocol)
        stdin=SimpleNamespace(buffer=io.BytesIO(b'{"type":"cancel","id":"queued"}\n'))
        with patch.object(w.sys,'stdin',stdin):inbox.reader()
        self.assertIn('queued',inbox.cancelled)

    def test_lookup_callback_is_post_commit_and_original_tokens_match(self):
        # Pure original runtime module: import it without importing model/Torch.
        source=Path(__file__).resolve().parents[1]/'runtime/ds41/speculate.py'
        s=importlib.util.spec_from_file_location('lookup_original',source)
        original=importlib.util.module_from_spec(s);s.loader.exec_module(original)
        adapted,identity=w.streaming_lookup(original)
        self.assertTrue(identity['original_proposal_verification_and_commit_AST_unchanged'])
        class Tensor:
            def __init__(self,values):self.values=values
        torch=ModuleType('torch');torch.tensor=lambda values,**kwargs:Tensor(values);torch.long='long'
        policy=ModuleType('ds41.lookup_policy')
        policy.LookupProposalPolicy=SimpleNamespace(from_environment=lambda:SimpleNamespace(enabled=False))
        class Predictions:
            def __init__(self,values):self.values=values
            def argmax(self,*args):return self
            def reshape(self,*args):return self
            def tolist(self):return self.values
        class Model:
            def __init__(self):self.position=3;self.commits=[];self.cache=SimpleNamespace(cpu_backend=None,stats={'calls':0})
            def snapshot_state(self,pos0,token_count):return dict(pos0=pos0)
            def forward_logits_all(self,tensor,pos):
                self.cache.stats['calls']+=1;self.position=pos+len(tensor.values[0])
                return Predictions([i+1 for i in tensor.values[0]])
            def commit_prefix(self,saved,count):self.commits.append(count);self.position=saved['pos0']+count
        # Force proposals so the true original commit/truncation path executes.
        original.lookup_draft=lambda *args:[99,100]
        adapted,unused=w.streaming_lookup(original)
        models=[Model(),Model()];seen=[]
        with patch.dict(sys.modules,{'torch':torch,'ds41.lookup_policy':policy}):
            a=original.decode_lookup(models[0],[1,2,3],7,5,device='cpu',depth=2,commit_strategy='truncate')
            b=adapted(models[1],[1,2,3],7,5,device='cpu',depth=2,commit_strategy='truncate',on_tokens=lambda rows:seen.extend(rows))
        self.assertEqual(a[0],b[0]);self.assertEqual(seen,b[0][1:]);self.assertEqual(models[0].commits,models[1].commits)
        self.assertEqual(models[0].position,models[1].position)


if __name__=='__main__':unittest.main()

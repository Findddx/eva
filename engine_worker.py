"""Persistent single-owner BF16 engine worker. JSONL on stdout; logs on stderr.

Serving deliberately omits the separate per-request parity/prefix harness. It
does not transfer a benchmark rate or certify this new persistent lifecycle.
No model, Torch, weights or CUDA are imported by --help/--preflight.
"""
import argparse
import ast
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

GIB = 1024**3
MAX_LINE = 2*1024**2
BUILDERS = (('attention','ds41.attn_cuda','_ext'), ('hc','ds41.hc_cuda','_ext'),
    ('quant','ds41.quant_cuda','_ext'), ('kv_prepare','ds41.kv_prepare_cuda','_extension'),
    ('mean_square','ds41.canonical_mean_square1024','build_backend'),
    ('sparse','ds41.canonical_sparse1024','build_backend'), ('rope','ds41.canonical_rope1024','build_backend'),
    ('native512','ds41.moe_coop512','build_backend'), ('native3072','ds41.moe_coop3072','build_backend'))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    require(Path(path).stat().st_size <= 64*1024**2, 'Oversized metadata')
    return json.loads(Path(path).read_bytes())


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')


def fixed_environment(args):
    value = dict(DS41_HC_RESIDUAL_BF16='1', DS41_EDIT_CANDIDATE='1', DS41_LAYER_PROMOTION='0',
        DS41_LAYER_PROMOTION_MAX_SWAPS='1', DS41_LAYER_PROMOTION_HOST_BACKING='0',
        DS41_LARGE_KV_PREPARE='1', DS41_FULL_COMPRESSED_KV='1', DS41_MOE_COOP3072='1',
        DS41_MOE_COOP2048='0', DS41_MOE_COOP512='1', DS41_VERIFY_MAX_ROWS='1024',
        DS41_HC_VERIFY_GRAPHS='1024', DS41_HC_VERIFY_SHARED_POOL='1', DS41_HC_VERIFY_IO_ARENA='1',
        DS41_FAST_POINTER_BIND='1', DS41_ROUTE_SNAPSHOT='0', DS41_ROUTE_AWARE_CHUNKS='1',
        DS41_STRICT_BATCH='1', DS41_STRICT_DENSE='1', DS41_FUSED_HC='0',
        DS41_DENSE_CANONICAL_TILE='32', DS41_DENSE_M2='1', DS41_FUSED_KV_PREP='1',
        DS41_NO_GRAPH='0', DS41_CAPTURE_STREAM_SHARED='1', DS41_DEMAND_H2D_PIPELINE='1',
        DS41_STABLE_LINEAR_GRAPHS='0', DS41_NEXT_CHUNK_PREFETCH='0', DS41_CANONICAL_MEAN_SQUARE='1',
        DS41_CANONICAL_SPARSE_BATCH='1', DS41_CANONICAL_ROPE_BATCH='1', DS41_CPU_EMBEDDING='1',
        DS41_LOOKUP_POLICY='both', DS41_EXPERT_DIRECT_IO='1', DS41_ALLOW_MISSING_ENGRAM='0',
        DS41_HOST_RESERVE_GIB='8', DS41_HOST_RECLAIM_GIB='1', EXL3_INT8_GEMV='2',
        EXL3_INT8_GEMV_MAX_K='0', EXL3_MOE_COOP_WIDE='0', EXL3_MOE_COOP_KSPLIT='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='4', MAX_JOBS='2',
        TORCH_CUDA_ARCH_LIST='12.0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True,pinned_max_round_threshold_mb:0',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,pinned_max_round_threshold_mb:0')
    value.update(DS41_MODEL=str(args.model), DS41_PACK=str(args.pack),
        DS41_DENSE_CACHE=str(args.dense_cache), DS41_ENGRAM_MANIFEST=str(args.engram_manifest),
        TORCH_EXTENSIONS_DIR=str(args.jit_dir))
    return value


def preflight(args):
    source = args.source.resolve(strict=True)
    require((source/'ds41/model.py').is_file() and (source/'engine/experiments/canonical1024/manifest.json').is_file(),
        'Need the complete core ds41 + engine/canonical1024 source')
    inputs = [args.model/'config.json', args.model/'tokenizer.json', args.model/'exl3_k_map.json',
              args.model/'model.safetensors.index.json', args.pack,
              args.dense_cache/'model.safetensors.index.json', args.engram_manifest]
    if args.counts is not None:
        inputs.append(args.counts)
    for p in inputs:
        require(p.is_file() and not p.is_symlink(), 'Missing/unsafe input: '+str(p))
    require(sha(args.model/'tokenizer.json') == 'c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b',
        'Require the tested complete1017 tokenizer')
    manifest = read_json(args.engram_manifest)
    require(manifest.get('format') == 'ranma_mxfp8_gguf', 'Require complete MXFP8 Engram; ablation is forbidden')
    first = Path(manifest['index_path'])
    if not first.is_absolute():
        first = args.engram_manifest.parent/first
    require('-00001-of-00007.gguf' in first.name, 'Require seven Engram pieces')
    for i in range(1, 8):
        piece = first.with_name(first.name.replace('-00001-of-00007.gguf',f'-{i:05d}-of-00007.gguf'))
        require(piece.is_file() and piece.stat().st_size > 0, 'Missing Engram piece: '+str(piece))
    require(args.context in (8192,16384,32768,65536,131072) and 1 <= args.max_output_tokens <= 8192,
        'Choose a context capacity from8K..128K and an output limit from1..8192')
    files = sorted([*(source/'ds41').rglob('*.py'), *(source/'ds41/csrc').rglob('*'), *(source/'engine').rglob('*')])
    digest = hashlib.sha256()
    members = 0
    for p in files:
        if p.is_file() and '__pycache__' not in p.parts:
            digest.update(p.relative_to(source).as_posix().encode()); digest.update(p.read_bytes()); members += 1
    return dict(core_source_sha256=digest.hexdigest(), source_files=members,
        model_source_sha256=sha(source/'ds41/model.py'), worker_sha256=sha(__file__),
        numerical_target='fixed_native_dense_m32_hc_bf16_residual_v1',
        no_request_parity_harness=True, no_new_persistent_runtime_certificate=True,
        no_weights_read=True, no_Torch_imported='torch' not in sys.modules)


class Protocol:
    def __init__(self, stream):
        self.stream, self.lock = stream, threading.Lock()

    def send(self, value):
        with self.lock:
            self.stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'); self.stream.flush()


class RequestError(ValueError):
    pass


class StopRequest(Exception):
    pass


def validate_request(value, maximum, default_mode):
    if type(value) is not dict:
        raise RequestError('Request must be a JSON object')
    identifier = value.get('id')
    if type(identifier) is not str or not 1 <= len(identifier) <= 128:
        raise RequestError('Request id must be a nonempty string at most128chars')
    if value.get('model','eva') != 'eva' or value.get('type') != 'generate':
        raise RequestError('Only generate for model eva is supported')
    if 'messages' in value:
        messages = value['messages']
        if ('prompt' in value or type(messages) is not list or len(messages) != 1
            or type(messages[0]) is not dict or set(messages[0]) != {'role','content'}
            or messages[0]['role'] != 'user'):
            raise RequestError('unsupported_messages: exactly one user message; no system/multiturn/tools')
        prompt = messages[0]['content']
    else:
        prompt = value.get('prompt')
    if type(prompt) is not str or not prompt.strip() or len(prompt.encode('utf8')) > 1024**2:
        raise RequestError('Prompt must be nonempty UTF8 text at most1MiB')
    count = value.get('max_tokens',512)
    if type(count) is not int or not 1 <= count <= maximum:
        raise RequestError('max_tokens outside worker limit')
    if type(value.get('temperature',0)) not in (int,float) or value.get('temperature',0) != 0:
        raise RequestError('Only deterministic temperature0 is supported')
    mode = value.get('mode',default_mode)
    if mode not in ('greedy','lookup'):
        raise RequestError('mode must be greedy or lookup')
    stops = value.get('stop',[])
    if type(stops) is str:
        stops = [stops]
    if (type(stops) is not list or len(stops)>4
        or any(type(s) is not str or not 1 <= len(s) <= 128 for s in stops)):
        raise RequestError('stop must contain at most4 nonempty literal strings, max128chars each')
    return dict(id=identifier,prompt=prompt,max_tokens=count,mode=mode,stops=stops)


class StopBuffer:
    """Do not leak a partial stop prefix into the streamed response."""
    def __init__(self, stops):
        self.stops, self.pending, self.stopped = stops, '', False

    def feed(self, text):
        if self.stopped:
            return ''
        self.pending += text
        found = [self.pending.find(s) for s in self.stops if s in self.pending]
        if found:
            cut = min(found); result = self.pending[:cut]; self.pending=''; self.stopped=True
            return result
        hold = max([0]+[n for s in self.stops for n in range(1,len(s)) if self.pending.endswith(s[:n])])
        if hold:
            result, self.pending = self.pending[:-hold], self.pending[-hold:]
        else:
            result, self.pending = self.pending, ''
        return result

    def finish(self):
        result, self.pending = ('' if self.stopped else self.pending), ''
        return result


def streaming_lookup(module):
    """Original decode_lookup AST plus one callback after committed emission.

    No proposal, logit, snapshot, verification, EOS or commit node is changed.
    Cancellation from the callback occurs only after the original commit path.
    """
    path = Path(module.__file__)
    tree = ast.parse(path.read_bytes())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decode_lookup']
    require(len(functions)==1, 'Missing original lookup generator')
    node = functions[0]
    original = ast.dump(node,include_attributes=False)
    node.name = 'decode_lookup_with_stream'
    node.args.args.append(ast.arg(arg='on_tokens'))
    node.args.defaults.append(ast.Constant(value=None))
    hits = 0
    for i, statement in enumerate(list(node.body)):
        if not isinstance(statement,ast.While):
            continue
        for j, item in enumerate(list(statement.body)):
            if (isinstance(item,ast.Expr) and isinstance(item.value,ast.Call)
                and ast.unparse(item.value.func)=='generated.extend'
                and len(item.value.args)==1 and isinstance(item.value.args[0],ast.Name)
                and item.value.args[0].id=='emitted'):
                statement.body.insert(j+1,ast.parse('if on_tokens is not None:\n    on_tokens(list(emitted))').body[0]); hits += 1
    require(hits==1, 'Original committed emission boundary changed')
    ast.fix_missing_locations(node)
    namespace = dict(vars(module))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[node.name], dict(original_source_sha256=sha(path),
        original_function_AST_sha256=hashlib.sha256(original.encode()).hexdigest(),
        callback='after original generated.extend(emitted), following original rollback/commit',
        original_proposal_verification_and_commit_AST_unchanged=True)


def reset_model(model, State):
    """Call only after all compute/copy streams and Engram jobs have drained."""
    model._clear_verification()
    model.position, model.main_hidden, model.state = 0, None, State()
    if hasattr(model,'prefill_metadata'):
        del model.prefill_metadata
    for layer in model.layers:
        attn = layer.attn
        for name in ('window','ck_codes','ck_scales','ik_codes','ik_scales','kv_state'):
            tensor = getattr(attn,name,None)
            if tensor is not None:
                tensor.zero_()
        score = getattr(attn,'score_state',None)
        if score is not None:
            score.fill_(-float('inf'))
    if model.engram is not None:
        model.engram.hash.cache.zero_()
        model.engram.jobs.clear()


class CheckedTarget:
    """Delegate original target methods; never replace Transformer.forward."""
    def __init__(self, model, torch):
        self.model,self.torch=model,torch

    def __getattr__(self,name):
        return getattr(self.model,name)

    def forward_logits_all(self,*args,**kwargs):
        result=self.model.forward_logits_all(*args,**kwargs)
        require(bool(self.torch.isfinite(result).all().item()),'Nonfinite target verification logits')
        hidden=self.model.main_hidden
        if hidden is not None:
            require(bool(self.torch.isfinite(hidden).all().item()),'Nonfinite target verification hidden')
        return result


class Engine:
    def __init__(self,args,protocol,identity):
        self.args,self.protocol,self.identity = args,protocol,identity
        self.model=self.cache=None; self.failed=False
        self.thread=threading.get_ident()

    def stage(self,stage):
        self.protocol.send(dict(type='status',stage=stage,message=stage))

    def initialize(self):
        started=time.perf_counter(); self.stage('loading_dependencies')
        import torch
        import numpy as np
        from tokenizers import Tokenizer
        from ds41.args import ModelArgs
        from ds41.store import WeightStore
        from ds41.cache import ExpertCache
        from ds41.model import Transformer,State
        from ds41 import speculate
        self.torch,self.np,self.State = torch,np,State
        require(torch.cuda.is_available() and torch.cuda.device_count()==1
            and torch.cuda.get_device_capability(0)==(12,0),'One visible SM120 GPU required')
        self.stream = torch.cuda.current_stream(0)
        self.binaries={}
        self.stage('loading_cuda_extensions')
        for label,name,entry in BUILDERS:
            extension=getattr(importlib.import_module(name),entry)()
            path=Path(extension.__file__).resolve(strict=True)
            require(path.is_relative_to(self.args.jit_dir),'CUDA module escaped explicit JIT')
            self.binaries[label]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        self.tokenizer=Tokenizer.from_file(str(self.args.model/'tokenizer.json'))
        self.tokenizer_sha=sha(self.args.model/'tokenizer.json')
        cfg=ModelArgs.from_config(str(self.args.model/'config.json'));cfg.max_seq_len=self.args.context
        self.eos=read_json(self.args.model/'config.json').get('eos_token_id',1)
        require(type(self.eos) is int,'Need integer normalEOS')
        self.stage('loading_model')
        self.cache=ExpertCache(str(self.args.pack),read_json(self.args.model/'exl3_k_map.json'),64,16000)
        store=WeightStore(str(self.args.model))
        self.model=Transformer(cfg,store,self.cache,'cuda')
        require(self.model.engram is not None and getattr(self.model,'_hc_residual_bf16',False),'Full Engram/BF16 target required')
        self.cache.initialize_native3072(self.model.device,volatile_experts=384)
        torch.cuda.synchronize();store.release();del store;gc.collect();torch.cuda.empty_cache()
        self.base_counts=(np.ones_like(self.cache.counts) if self.args.counts is None
            else np.load(self.args.counts,allow_pickle=False))
        require(self.base_counts.shape==self.cache.counts.shape
            and np.issubdtype(self.base_counts.dtype,np.integer) and not np.any(self.base_counts<0),'Invalid routing prior')
        self.cache.counts=self.base_counts.copy();self.cache.stage_limit=384
        self.cache.validate_native3072_placement()
        free,_=torch.cuda.mem_get_info()
        gpu_budget=max(0,int(free-self.args.vram_headroom_gib*GIB))
        maximum=max(row[1] for row in self.cache.slots)
        require(gpu_budget>=384*maximum,'Insufficient free GPU for384active slots and headroom')
        ram_budget=int(self.args.ram_gib*GIB)
        require(available_bytes() >= ram_budget+int(self.args.host_headroom_gib*GIB),
            'Insufficient available RAM for the requested expert cache and host headroom')
        self.cache.ram_budget_bytes=ram_budget;self.cache.minimum_available_bytes=int(self.args.host_headroom_gib*GIB)
        self.cache.warm_unseen=False;self.cache.configure_host_allocator_reserve(int(self.args.host_allocator_reserve_gib*GIB))
        frozen,ram=self.cache.ranked_sets(gpu_budget-384*maximum,ram_budget)
        self.stage('preloading_experts');self.cache.preload(frozen,ram)
        self.cache.validate_native3072_placement()
        self.lookup,self.lookup_identity=streaming_lookup(speculate)
        self.drain();self.cache.ram_admission_enabled=False
        self.startup_s=time.perf_counter()-started
        self.check_health()
        print(json.dumps(dict(source_identity=self.identity,extensions=self.binaries,
            startup_s=self.startup_s,frozen_experts=len(self.cache.frozen),RAM_prior='uniform' if self.args.counts is None else 'explicit',
            diagnostic_harness_per_request=False,persistent_lifecycle_GPU_verified=False)),file=sys.stderr,flush=True)
        self.protocol.send(dict(type='ready',model='eva',context_limit=self.args.context,
            max_output_tokens=self.args.max_output_tokens,modes=['greedy','lookup'],startup_s=self.startup_s))

    def check_health(self):
        require(not self.failed and threading.get_ident()==self.thread,'Failed worker or wrong CUDA owner thread')
        require(self.torch.cuda.current_stream(0).cuda_stream==self.stream.cuda_stream,'CUDA stream changed')
        self.cache._check_placement_health()
        require(not self.cache._active_keys,'Active expert lease at request boundary')
        status=self.cache.demand_h2d_pipeline_status()
        require(not status['failed'] and not status['active'] and status['quarantined_calls']==0,'Pipeline not quiescent')

    def drain(self):
        if self.model is not None and self.model.engram is not None:
            for job in self.model.engram.jobs:
                job.result()
        self.torch.cuda.synchronize()
        if self.cache is not None:
            for future in list(self.cache._pending.values()):
                future.result()
            self.cache._drain_prefetch()
        self.torch.cuda.synchronize()

    def request(self,request,cancel):
        from ds41.prompts import encode_prompt
        from tokenizers.decoders import DecodeStream
        torch=self.torch; started=time.perf_counter(); identifier=request['id']
        ids=encode_prompt(self.tokenizer,request['prompt'],format='chat',thinking=False,max_user_tokens=0)
        if not ids or len(ids)+request['max_tokens']>self.args.context:
            raise RequestError('Complete prompt/output exceed reserved context; no truncation')
        self.check_health();self.drain();reset_model(self.model,self.State)
        self.cache.counts=self.base_counts.copy();self.cache.ram_admission_enabled=False;self.cache.reset_stats()
        output=[];first_token_s=None;first_text_s=None;decode_start=None
        decoder=DecodeStream(skip_special_tokens=True);stop=StopBuffer(request['stops'])
        chunks=[];decoded_chunks=[];reason='length';placement_s=0.;adapter_cycles=[]
        def publish(text):
            nonlocal first_text_s
            if text:
                if first_text_s is None:first_text_s=time.perf_counter()-started
                chunks.append(text);self.protocol.send(dict(type='delta',id=identifier,text=text))
        def emitted(tokens):
            nonlocal first_token_s,reason
            if cancel.is_set():
                reason='cancelled';raise StopRequest()
            for token in tokens:
                output.append(int(token))
                if first_token_s is None:first_token_s=time.perf_counter()-started
                part=decoder.step(self.tokenizer,int(token))
                if part:
                    decoded_chunks.append(part);publish(stop.feed(part))
                if stop.stopped:
                    reason='stop';raise StopRequest()
                if int(token)==self.eos:reason='stop'
        try:
            tokens=torch.tensor([ids],device='cuda',dtype=torch.long)
            self.protocol.send(dict(type='status',id=identifier,stage='prefill',message='prefill'))
            for offset in range(0,len(ids),128):
                if cancel.is_set():reason='cancelled';raise StopRequest()
                logits=self.model.forward(tokens[:,offset:offset+128].contiguous(),offset,full_logits=False)
                require(bool(torch.isfinite(logits).all().item()),'Nonfinite target logits')
                if self.model.main_hidden is not None:
                    require(bool(torch.isfinite(self.model.main_hidden).all().item()),'Nonfinite BF16 main hidden')
            first=int(logits.argmax(-1).item());del logits,tokens
            emitted([first])
            if output[-1]!=self.eos and len(output)<request['max_tokens']:
                self.drain();before=time.perf_counter()
                if self.args.prompt_reseed:
                    prompt_counts=self.cache.counts.astype(self.np.int64)-self.base_counts.astype(self.np.int64)
                    require(not self.np.any(prompt_counts<0) and int(prompt_counts.sum())==len(ids)*40*6,'Prompt route count mismatch')
                    self.cache.reseed_gpu_from_prompt(prompt_counts,max_loads=1024,max_swaps=1024)
                    self.cache.reseed_ram_from_prompt(prompt_counts,max_loads=1024,objective='bytes')
                self.drain();placement_s=time.perf_counter()-before
                self.cache.reset_stats();self.cache.ram_admission_enabled=False
                decode_start=time.perf_counter()
                if request['mode']=='lookup':
                    from ds41.edit_candidate_runtime import EditCandidateSession
                    from ds41.lookup_policy import configuration
                    session=EditCandidateSession(request['prompt'],lambda text:self.tokenizer.encode(text,add_special_tokens=False).ids,
                        tokenizer_sha256=self.tokenizer_sha,lookup_identity=configuration())
                    generated,records,adapter_cycles=self.lookup(CheckedTarget(self.model,torch),ids,first,request['max_tokens'],
                        depth=1023,eos=self.eos,ignore_eos=False,commit_strategy='truncate',
                        edit_candidate_session=session,on_tokens=emitted)
                    require(generated==output,'Streamed committed tokens differ from original lookup result')
                    session.verify_metadata()
                else:
                    while len(output)<request['max_tokens'] and output[-1]!=self.eos:
                        if cancel.is_set():reason='cancelled';raise StopRequest()
                        logits=self.model.forward(torch.tensor([[output[-1]]],device='cuda',dtype=torch.long),len(ids)+len(output)-1)
                        require(bool(torch.isfinite(logits).all().item()),'Nonfinite target logits')
                        if self.model.main_hidden is not None:
                            require(bool(torch.isfinite(self.model.main_hidden).all().item()),'Nonfinite BF16 main hidden')
                        nxt=int(logits.argmax(-1).item());del logits
                        emitted([nxt])
        except StopRequest:
            pass
        self.drain();self.check_health()
        end=time.perf_counter();decode_s=0. if decode_start is None else end-decode_start
        full=self.tokenizer.decode(output,skip_special_tokens=True)
        current=''.join(decoded_chunks)
        require(full.startswith(current),'Streaming decoder changed an already emitted prefix')
        if not stop.stopped:
            publish(stop.feed(full[len(current):]));publish(stop.finish())
        if cancel.is_set():reason='cancelled'
        elif stop.stopped or (output and output[-1]==self.eos):reason='stop'
        measured=max(0,len(output)-1)
        self.protocol.send(dict(type='done',id=identifier,finish_reason=reason,fatal=False,
            usage=dict(prompt_tokens=len(ids),completion_tokens=len(output)),
            metrics=dict(first_token_s=first_token_s,first_text_s=first_text_s,
                decode_s=decode_s,decode_tokens=measured,decode_tok_s=measured/decode_s if measured and decode_s>0 else None,
                e2e_s=end-started,prompt_placement_s=placement_s,
                timing_scope='server wall; TTFT includes reset+prefill; decode excludes first token/prompt placement and includes streaming callbacks',
                mode=request['mode'])))

    def close(self):
        errors=[]
        if self.model is not None:
            try:self.drain()
            except BaseException as error:errors.append(error)
            from ds41.canonical_attention import close as close_attention
            for fn in (lambda:close_attention(self.model),self.model.close):
                try:fn()
                except BaseException as error:errors.append(error)
        if self.cache is not None:
            try:self.cache.close()
            except BaseException as error:errors.append(error)
        if self.model is not None and self.model.engram is not None:
            try:self.model.engram.close()
            except BaseException as error:errors.append(error)
        if errors:
            raise RuntimeError('Owner close failed; retain worker until process teardown: '+repr(errors[0]))


class Inbox:
    def __init__(self,protocol):
        self.protocol=protocol;self.items=queue.Queue(maxsize=2)
        self.lock=threading.Lock();self.active=None;self.cancel=threading.Event();self.stopping=threading.Event();self.cancelled=set()

    def reader(self):
        while not self.stopping.is_set():
            line=sys.stdin.buffer.readline(MAX_LINE+1)
            if not line:
                self.stopping.set();self.cancel.set();break
            if len(line)>MAX_LINE:
                self.protocol.send(dict(type='error',id=None,code='request_too_large',message='JSONL limit2MiB',fatal=False));continue
            try:
                value=json.loads(line)
                if type(value) is dict and value.get('type')=='cancel':
                    with self.lock:
                        if value.get('id')==self.active:self.cancel.set()
                        elif type(value.get('id')) is str:
                            if len(self.cancelled)>=32:self.cancelled.pop()
                            self.cancelled.add(value['id'])
                    continue
                if type(value) is dict and value.get('type')=='shutdown':
                    self.stopping.set();self.cancel.set();break
                self.items.put_nowait(value)
            except (ValueError,queue.Full) as error:
                self.protocol.send(dict(type='error',id=value.get('id') if 'value' in locals() and isinstance(value,dict) else None,
                    code='invalid_request',message=str(error),fatal=False))


def main(argv=None):
    sys.dont_write_bytecode=True
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','pack','dense-cache','engram-manifest','jit-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--counts',type=Path,help='Optional nonnegative route histogram; absent uses uniform initial prior')
    p.add_argument('--context',type=int,default=131072)
    p.add_argument('--max-output-tokens',type=int,default=4096)
    p.add_argument('--mode',choices=('greedy','lookup'),default='greedy')
    p.add_argument('--prompt-reseed',type=int,choices=(0,1),default=1)
    p.add_argument('--vram-headroom-gib',type=float,default=1.8)
    p.add_argument('--ram-gib',type=float,default=64)
    p.add_argument('--host-headroom-gib',type=float,default=8)
    p.add_argument('--host-allocator-reserve-gib',type=float,default=1)
    p.add_argument('--preflight',action='store_true')
    args=p.parse_args(argv)
    for name in ('source','model','pack','dense_cache','engram_manifest','counts','jit_dir'):
        if getattr(args,name) is not None:
            setattr(args,name,getattr(args,name).expanduser().resolve(strict=False))
    require(1.8<=args.vram_headroom_gib<=4,'Headroom must be in1.8..4GiB')
    require(4<=args.ram_gib<=1024 and 2<=args.host_headroom_gib<=1024
        and 0<args.host_allocator_reserve_gib<args.ram_gib,'Invalid host memory budgets')
    identity=preflight(args)
    if args.preflight:
        print(json.dumps(dict(passed=True,identity=identity,no_GPU=True)));return 0
    require(sys.platform=='linux' and sys.version_info[:2]==(3,12),'Actual worker requires Linux/Python3.12')
    require('torch' not in sys.modules,'Environment must be set before importing Torch')
    for name in list(os.environ):
        if name.startswith(('DS41_','EXL3_')) or name in ('NVCC_APPEND_FLAGS','NVCC_PREPEND_FLAGS','CUDAFLAGS','CXXFLAGS','TORCH_NVCC_FLAGS'):
            os.environ.pop(name)
    os.environ.update(fixed_environment(args));sys.path.insert(0,str(args.source))
    args.jit_dir.mkdir(parents=True,exist_ok=True)
    require(available_bytes()>=(args.ram_gib+args.host_headroom_gib+2)*GIB,
        'Insufficient available RAM for the requested cache, host headroom and loading overhead')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    require(processes.returncode==0 and not processes.stdout.strip(),'GPU has a running compute process; wait for it to finish')
    Path('/proc/self/oom_score_adj').write_text('500\n')
    protocol_stream=os.fdopen(os.dup(sys.stdout.fileno()),'w',encoding='utf8',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno());sys.stdout=sys.stderr
    protocol=Protocol(protocol_stream);engine=Engine(args,protocol,identity);inbox=Inbox(protocol)
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *unused:(inbox.stopping.set(),inbox.cancel.set()))
    exitcode=0;fatal_identifier=None
    try:
        import torch
        with torch.inference_mode():
            engine.initialize()
            thread=threading.Thread(target=inbox.reader,daemon=True);thread.start()
            while not inbox.stopping.is_set():
                try:value=inbox.items.get(timeout=.25)
                except queue.Empty:continue
                try:request=validate_request(value,args.max_output_tokens,args.mode)
                except RequestError as error:
                    protocol.send(dict(type='error',id=value.get('id') if isinstance(value,dict) else None,
                        code='invalid_request',message=str(error),fatal=False));continue
                fatal_identifier=request['id']
                with inbox.lock:
                    inbox.active=request['id'];inbox.cancel.clear()
                    if request['id'] in inbox.cancelled:
                        inbox.cancelled.discard(request['id']);inbox.cancel.set()
                try:engine.request(request,inbox.cancel)
                except RequestError as error:
                    protocol.send(dict(type='error',id=request['id'],code='invalid_request',message=str(error),fatal=False))
                finally:
                    with inbox.lock:inbox.active=None
    except BaseException as error:
        engine.failed=True;exitcode=1
        protocol.send(dict(type='error',id=fatal_identifier,code='engine_failed',message='Engine failed; see the private server log.',fatal=True))
        import traceback;traceback.print_exc(file=sys.stderr)
    finally:
        inbox.stopping.set();inbox.cancel.set()
        try:engine.close()
        except BaseException as error:
            print(repr(error),file=sys.stderr,flush=True)
            protocol.send(dict(type='error',id=None,code='close_failed',message='Engine cleanup failed; see the private server log.',fatal=True));exitcode=1
    return exitcode


if __name__=='__main__':
    raise SystemExit(main())

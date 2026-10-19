"""Stdlib source/provenance checks shared by isolated1024 wrappers."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    value=json.loads((ROOT/'manifest.json').read_bytes());seen=set()
    require(value.get('schema')==1 and value.get('production_integration') is False,'Wrong independent manifest')
    require(not any(p.is_symlink() for p in ROOT.rglob('*')),'Symlink in independent experiment')
    for row in value['files']:
        name=row['path'];relative=Path(name)
        require(name not in seen and not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,'Unsafe/duplicate manifest member')
        path=ROOT/relative;require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],
            'Experimental source changed: '+name);seen.add(name)
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=ROOT/'manifest.json'}
    require(actual==seen,'Missing/unmanifested independent source');return value


def checked(source,component):
    manifest();source=Path(source)
    require(source.is_absolute() and source.is_dir() and not source.is_symlink(),'Explicit ordinary frozen source required')
    source=source.resolve(strict=True);p=json.loads((ROOT/'csrc'/component/'provenance.json').read_bytes())
    parent=p['parent512']
    for key in ('wrapper','binding','kernel','provenance'):
        row=parent[key];path=source/row['path']
        require(path.is_file() and not path.is_symlink() and sha(path)==row['sha256'],'Pinned original512 source changed: '+key)
    header=(ROOT/'csrc'/component/p['kernel_filename']).read_text('utf8')
    require(sha(ROOT/'csrc'/component/p['kernel_filename'])==p['source_header_sha256'],'New1024 header identity changed')
    require(hashlib.sha256(header.replace(p['new_namespace'],p['old_namespace']).encode()).hexdigest()==parent['kernel']['LF_sha256'],
        '1024 float body differs from namespace-only original512')
    binding=(ROOT/'csrc'/component/'binding.cu').read_text('utf8')
    for before,after in reversed(p['binding_literal_replacements']):
        require(after in binding,'1024 integer/namespace binding patch changed');binding=binding.replace(after,before)
    require(hashlib.sha256(binding.encode()).hexdigest()==parent['binding']['LF_sha256'],'1024 binding exceeds declared integer/namespace delta')
    path=source/parent['wrapper']['path'];spec=importlib.util.spec_from_file_location('explicit_parent512_'+component,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.source_metadata()  # Original provenance also restores its M1 ancestor.
    require(original.get('max_queries',original.get('max_rows'))==512,'Reference is not an actual512 primitive')
    return p,original


def metadata(source,component,wrapper,cpp_flags,cuda_flags):
    p,parent=checked(source,component)
    digest=hashlib.sha256()
    for path in sorted((ROOT/'csrc'/component).iterdir()):
        require(path.is_file() and not path.is_symlink(),'Special independent source entry')
        digest.update(path.name.encode());digest.update(path.read_bytes())
    recipe=dict(source_sha256=digest.hexdigest(),wrapper_sha256=sha(wrapper),common_sha256=sha(__file__),
        experiment_manifest_sha256=sha(ROOT/'manifest.json'),parent512_source_metadata=parent,
        cpp_flags=list(cpp_flags),cuda_flags=list(cuda_flags))
    build=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    value=dict(parent,**recipe,build_sha256=build,independent_1024_prototype=True,parent_GPU_proof_inherited=False,
        production_integration=False,GPU_math_validated=False,full_model_GPU_validated=False,performance_measured=False,
        float_source_delta='Namespace only; original512 body reverses byte-for-byte, including original M1 provenance')
    value.pop('independent_512_prototype',None)
    value['standalone_capacity_delta']=dict(max_queries_or_rows=1024,namespace_only_float_change=True,
        old_proof_inherited=False,production_owner_modified=False)
    if component=='mean':value.update(max_rows=1024,max_float_conversion_bytes=1024*20480*4)
    else:
        value.update(max_queries=1024)
        if component=='sparse':value['max_kv_rows']=127+1024*513
    return value

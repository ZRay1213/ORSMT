"""Execute a host-approved build_model() script in a bounded child process.

Process isolation is NOT a security sandbox. Untrusted generated programs must
run inside the host's container/sandbox. No code executes in the caller process.
The builder must return a live supported model BEFORE solving it.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from .milp_codegen import model_to_dict
from .translator import ModelTranslator


def capture_builder(source, *, python=sys.executable, timeout=30, max_output_bytes=8*1024*1024):
    if not isinstance(source,str) or not source.strip() or timeout<=0 or max_output_bytes<=0:
        raise ValueError('invalid builder or capture budget')
    with tempfile.TemporaryDirectory(prefix='orsmt-capture-') as folder:
        script=Path(folder)/'builder.py';output=Path(folder)/'captured.json'
        script.write_text(source,encoding='utf-8')
        env=os.environ.copy()
        root=str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH']=root+os.pathsep+env.get('PYTHONPATH','')
        command=[str(python),'-m','orsmt.capture','--worker',str(script),str(output),str(max_output_bytes)]
        process=subprocess.Popen(command,cwd=folder,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
            process.wait()
            raise TimeoutError('native builder exceeded capture deadline') from None
        if not output.exists() or output.stat().st_size>max_output_bytes:
            raise ValueError('native capture failed or exceeded output budget')
        envelope=json.loads(output.read_text(encoding='utf-8'))
        if process.returncode or envelope.get('status')!='ok':
            raise ValueError('native capture failed: '+str(envelope.get('error','worker exit')))
        translated=ModelTranslator().translate(envelope['formulation'],format='dict',allow_llm=False)
        return translated.model, {'execution':'host_approved_builder_subprocess','security_sandbox':False,
                                 'native_translation':envelope['translation'],'timeout_seconds':timeout}


def worker(script,output,max_output_bytes):
    try:
        if os.name=='posix':
            import resource
            resource.setrlimit(resource.RLIMIT_FSIZE,(max_output_bytes,max_output_bytes))
        import runpy
        namespace=runpy.run_path(script,run_name='__orsmt_builder__')
        builder=namespace.get('build_model')
        if not callable(builder):raise ValueError('Python artifact must expose build_model()')
        translated=ModelTranslator().translate(builder(),allow_llm=False)
        result={'status':'ok','formulation':model_to_dict(translated.model),'translation':translated.metadata()}
    except Exception as exc:
        result={'status':'error','error':type(exc).__name__+': '+str(exc)}
    payload=json.dumps(result,ensure_ascii=False,default=str)
    if len(payload.encode())>max_output_bytes:raise ValueError('capture exceeds byte limit')
    Path(output).write_text(payload,encoding='utf-8')
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=3,required=True)
    args=p.parse_args();raise SystemExit(worker(args.worker[0],args.worker[1],int(args.worker[2])))

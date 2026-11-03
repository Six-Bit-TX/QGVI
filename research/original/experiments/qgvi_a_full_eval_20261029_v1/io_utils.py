"""Atomic, content-bound experiment records. No provider credentials here."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
def identity():
    fields=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()
    return {'pid':os.getpid(),'start_ticks':int(fields[19])}
def require(condition,message):
    if not condition:raise ValueError(message)

class Heartbeat:
    def __init__(self,path,data):
        self.path=Path(path);self.data=data;self.stop=threading.Event();self.worker=identity()
    def tick(self):write(self.path,{**self.data,'updated_at':now(),'heartbeat_at':now(),'worker':self.worker})
    def loop(self):
        while not self.stop.wait(5):self.tick()
    def __enter__(self):
        self.tick();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self.data
    def __exit__(self,typ,value,trace):
        self.stop.set();self.thread.join(timeout=2)
        self.data.update(status='failed' if typ else 'completed',active_query=None);self.tick()

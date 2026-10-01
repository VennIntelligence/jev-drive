import concurrent.futures,subprocess,pathlib,json,time,shutil
root=pathlib.Path(__file__).resolve().parent
url="https://cn.download.nvidia.com/XFree86/Linux-x86_64/580.159.03/NVIDIA-Linux-x86_64-580.159.03.run"
size=398016015
chunk=4194304
(root/'parts').mkdir(exist_ok=True)
log=(root/'download-events.jsonl').open('a',buffering=1)
def download(i):
 start=i*chunk; end=min(size,start+chunk)-1
 p=root/'parts'/str(i)
 if not p.exists() or p.stat().st_size!=end-start+1:
  subprocess.run(['curl','-fsSL','--max-time','180','--retry','2','--range',f'{start}-{end}','-o',str(p),url],check=True)
 assert p.stat().st_size==end-start+1
 log.write(json.dumps({'t':time.time(),'part':i,'bytes':p.stat().st_size})+'\n')
 return i
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 for n,i in enumerate(pool.map(download,range((size+chunk-1)//chunk)),1): print(f'{n}/95 parts done',flush=True)
with (root/'NVIDIA-Linux-x86_64-580.159.03-complete.run').open('wb') as dest:
 for i in range((size+chunk-1)//chunk):
  with (root/'parts'/str(i)).open('rb') as src: shutil.copyfileobj(src,dest)
print('Complete',flush=True)

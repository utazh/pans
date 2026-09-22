from pathlib import Path
import json,subprocess,time,os,platform
ROOT=Path(__file__).resolve().parents[1]
def snapshot(name,phase):
 def command(args):
  z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20)
  return {'exit_code':z.returncode,'output':z.stdout}
 d={'name':name,'phase':phase,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'loadavg':os.getloadavg(),'uname':list(platform.uname()),'cpu_count':os.cpu_count(),
    'gpu':command(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,pstate,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem','--format=csv']),
    'memory':Path('/proc/meminfo').read_text(),
    'filesystems':command(['df','-h',str(ROOT)]),
    'storage':command(['lsblk','-o','NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS'])}
 with (ROOT/'audit/v25_timing_environment.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
 return d
if __name__=='__main__':snapshot('setup','before_formal_timing')

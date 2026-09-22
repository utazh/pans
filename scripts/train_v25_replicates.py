
import argparse
from v25_common import ROOT,run,train_cmd
p=argparse.ArgumentParser();p.add_argument('--G',type=int,required=True);p.add_argument('--R',type=int,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
for s in [25,26]:run(f'v25_g{a.G}_r{a.R}_s{s}',train_cmd(a.G,a.R,s),a.gpu)
print('REPLICATES COMPLETE',a.R,flush=True)

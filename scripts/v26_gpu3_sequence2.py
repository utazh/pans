from v26_common import *
run('v26_prefix_p1_index_retry',[PY,'-u','scripts/build_selector_index.py','--source-pcache-dir',str(ROOT/'runtime_data/v26_p1_pcache_gpu3'),'--store-root',str(ROOT/'runtime_data/v26_prefix_p1'),'--output-dir',str(ROOT/'assets/v26_selector_p1'),'--tasks','sst2,subj,trec,rte','--source-task-order','sst2,subj,trec,rte','--model-path',MODEL],3)
run('v26_p1_training_worker',[PY,'-u','scripts/v26_training_worker.py','--gpu','3','--prefix','p1','--budgets','25'],3)
run('v26_quality_gpu3',[PY,'-u','scripts/v26_quality_worker.py','--gpu','3'],3)

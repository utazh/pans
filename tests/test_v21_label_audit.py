from contiguous_fuxian.sharedkv_v21.audit import mean_label_scores
def test_mean_label_scores_matches_runner_with_unequal_token_lengths():
 scores=mean_label_scores({'ABBR':[-4.,-0.1],'DESC':[-3.]})
 assert max(scores,key=scores.get)=='ABBR'
 assert max({'ABBR':-4.1,'DESC':-3.},key={'ABBR':-4.1,'DESC':-3.}.get)=='DESC'

def test_shadow_conversion_replaces_loader_copy():
 import torch
 from types import SimpleNamespace
 from contiguous_fuxian.sharedkv_v21.runtime import shadow_resolve
 from contiguous_fuxian.sharedkv.runtime import RequestState
 l=SimpleNamespace(_loaded={3:(torch.ones(5,2,4),torch.ones(5,2,4))})
 l.resolve=lambda lid:l._loaded[lid]
 cfg={'shadow_load':True,'shadow_buffers':{}}
 state=RequestState([0,1,2,2],{},'reference')
 shadow_resolve(state,l,3,cfg,torch.bfloat16,'cpu')
 assert l._loaded[3][0].dtype==torch.bfloat16
 assert l._loaded[3][0].data_ptr()==cfg['shadow_buffers'][3][0].data_ptr()
 assert state.bank=={}

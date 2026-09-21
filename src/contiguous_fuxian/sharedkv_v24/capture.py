"""Attach to the existing scorer, retaining all candidates and NO gold labels.

No hook on decoder internals is needed. This is calibration, not formal timing.
"""
from __future__ import annotations
from pathlib import Path
import torch
from .records import FORMAT,digest_ids,json_digest,save_new,validate


class ScoreCollector:
    def __init__(self,root,split):
        if split not in ('fit','validation'):raise ValueError('Explicit data split required')
        self.root=Path(root);self.split=split
    def install(self,runtime,runner):
        if runtime.mode!='identity':raise ValueError('Capture only the clean same-backend identity')
        previous=runner.greedy_flexgen_completion
        def completion(*args,**kw):
            if args or kw.get('max_tokens')!=1:raise ValueError('Keyword arguments and max_tokens=1 required')
            candidates=kw.get('label_continuations')
            if not candidates:raise ValueError('Actual label-continuation scorer must be active')
            model,loader=kw['model'],kw['loader']
            geom,sources,readers,_=runtime._prepare(model,loader,kw['impress_selection_period_size'])
            if readers or sources!=list(range(geom['layers'])):raise ValueError('Teacher must have native layer KV')
            scores=kw.get('label_token_logprobs_out')
            if scores is None:scores={};kw['label_token_logprobs_out']=scores
            result=previous(*args,**kw)
            order=[str(x) for x in candidates]
            token_ids={str(k):[int(x) for x in v] for k,v in candidates.items()}
            token_lp={k:[float(x) for x in scores[k]] for k in order}
            query=[int(x) for x in kw['query_token_ids']]
            selected=[list(map(int,x)) for x in loader._selected_by_layer]
            row={'format':FORMAT,'split':self.split,'task':loader._info.task,
                 'query_hash':digest_ids(query),'query_token_ids':query,
                 'token_hash':loader._info.token_hash,'prefix_tokens':loader.prefix_tokens,
                 'geometry':geom,'period':kw['impress_selection_period_size'],
                 'keep_ratio':float(loader.keep_ratio),'model_identity':loader._selector_index.payload['model_identity'],
                 'teacher_sources':sources,'teacher_backend':runtime.backend,
                 'teacher_layout':runtime.layout,'label_order':order,'label_ids':token_ids,
                 'teacher_token_logprobs':token_lp,
                 'teacher_scores':{k:sum(v)/len(v) for k,v in token_lp.items()},
                 'score_rule':'mean_continuation_logprob','selection_by_layer':selected,
                 'selection_hash':json_digest(selected),'phase':'calibration_not_timing'}
            validate(row);save_new(self.root/row['task']/(row['query_hash']+'.pt'),row)
            return result
        runner.greedy_flexgen_completion=completion

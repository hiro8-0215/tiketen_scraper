"""Compare policy thresholds on purged future blocks; never overwrite policy.json."""
import argparse
import itertools
import json
import sys
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(); p.add_argument('--folder',type=Path,required=True)
    args=p.parse_args()
    base=Path(__file__).resolve().parents[1]/'buy_timing_model'
    sys.path.insert(0,str(base))
    from config import GRID, PROFILES
    from data_loader import load_oof
    from decision import apply_policy, summarize
    from train_policy import _mean_regret
    frame=load_oof(args.folder/'demand_predictions.csv',args.folder/'alternative_predictions.csv')
    existing=json.loads((base/'artifacts/policy.json').read_text(encoding='utf-8'))['profiles']
    result={}
    for h,part in frame.groupby('horizon_days'):
        times=sorted(part.landmark_at.unique())
        if len(times)<3:
            result[str(h)]={'status':'insufficient_evidence','reason':'Need three prediction dates for policy selection and held-out evaluation'}
            continue
        boundary=pd.Timestamp(times[-1])
        fit=part[part.landmark_at+pd.Timedelta(days=int(h))<boundary]
        test=part[part.landmark_at>=boundary]
        if any(x.ticket_id.nunique()<100 or x.loc[x.true_state.eq(1),'ticket_id'].nunique()<40 or x.true_alternative.nunique()<2 for x in [fit,test]):
            result[str(h)]={'status':'insufficient_evidence','reason':'Too few independent outcomes after horizon purge'}
            continue
        profiles={}
        for name,penalties in PROFILES.items():
            if str(int(h)) not in existing[name]: continue
            keys=list(GRID)
            candidates=[dict(zip(keys,values)) for values in itertools.product(*(GRID[k] for k in keys))]
            best=min(candidates,key=lambda candidate:_mean_regret(fit,candidate,penalties))
            profiles[name]={'candidate_parameters':best,
                'candidate_test':summarize(apply_policy(test,best),penalties),
                'original_test':summarize(apply_policy(test,existing[name][str(int(h))]),penalties),
                'always_buy':summarize(test.assign(action='buy_now'),penalties),
                'always_wait':summarize(test.assign(action='wait'),penalties)}
        result[str(h)]={'status':'compared_not_promoted','profiles':profiles}
    (args.folder/'policy_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()

"""Evidence-aware metrics and purged calibration comparisons; no model promotion."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score


def probabilities(frame,kind):
    if kind=='demand':
        return frame[['p_active','p_sold','p_deleted']].to_numpy(float)
    values=frame.p_alternative.to_numpy(float)
    return np.column_stack([1-values,values])


def score(y,p):
    labels=np.arange(p.shape[1])
    result={'rows':len(y),'accuracy':float((p.argmax(1)==y).mean()),
        'log_loss':float(log_loss(y,p,labels=labels)),
        'brier_mean_columns':float(np.mean((np.eye(p.shape[1])[y]-p)**2)),
        'class_counts':{str(i):int((y==i).sum()) for i in labels},
        'mean_probabilities':p.mean(0).tolist()}
    for i in labels:
        if 0<(y==i).sum()<len(y):
            result[f'class_{i}_roc_auc']=float(roc_auc_score(y==i,p[:,i]))
            result[f'class_{i}_pr_auc']=float(average_precision_score(y==i,p[:,i]))
    return result


def adjust(p,prior,temperature,shrink):
    logits=np.log(np.clip(p,1e-12,1))/temperature
    logits-=logits.max(axis=1,keepdims=True)
    values=np.exp(logits); values/=values.sum(axis=1,keepdims=True)
    return (1-shrink)*values+shrink*prior


def compare(frame,kind,horizon):
    """Three disjoint dates: estimate prior, choose correction, untouched test."""
    times=sorted(frame.landmark_at.unique())
    if len(times)<3:
        return {'status':'insufficient_evidence','reason':'Need at least three independent prediction dates for fit/tune/test'}
    target='true_state' if kind=='demand' else 'true_alternative'
    tune_at,test_at=pd.Timestamp(times[-2]),pd.Timestamp(times[-1])
    fit=frame[(frame.landmark_at+pd.Timedelta(days=horizon)<tune_at)]
    tune=frame[(frame.landmark_at>=tune_at)&(frame.landmark_at+pd.Timedelta(days=horizon)<test_at)]
    test=frame[frame.landmark_at>=test_at]
    k=3 if kind=='demand' else 2
    for name,part in [('fit',fit),('tune',tune),('test',test)]:
        # Count independent tickets per class, not repeated daily rows.
        counts=part.drop_duplicates('ticket_id').groupby(target).size()
        if any(counts.get(i,0)<40 for i in range(k)):
            return {'status':'insufficient_evidence','reason':f'{name}: fewer than 40 independent tickets per class after horizon purge','counts':{str(key):int(n) for key,n in counts.items()}}
    counts=np.bincount(fit[target].to_numpy(int),minlength=k)+1
    prior=counts/counts.sum()
    candidates=[]
    for temperature in [0.5,1.,2.,4.]:
        for shrink in [0.,0.25,0.5,0.75,1.]:
            p=adjust(probabilities(tune,kind),prior,temperature,shrink)
            candidates.append((log_loss(tune[target],p,labels=np.arange(k)),temperature,shrink))
    _,temperature,shrink=min(candidates)
    original=probabilities(test,kind)
    y=test[target].to_numpy(int)
    return {'status':'compared_not_promoted','fit_rows':len(fit),'tune_rows':len(tune),'test_rows':len(test),
        'test_start':str(test_at),'candidate':{'temperature':temperature,'shrink_to_past_prior':shrink,'prior':prior.tolist()},
        'original':score(y,original),'candidate_test':score(y,adjust(original,prior,temperature,shrink)),
        'past_constant_baseline':score(y,np.tile(prior,(len(test),1)))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args()
    report={'evaluation_version':'snapshot_evidence_v2','training_executed':False,'models':{},
        'limitations':['Manual metadata uses current values. Snapshot availability is retrospective.',
                      'Repeated observations are not independent samples.',
                      'Missing collection coverage blocks negative alternative labels and reliable buy-policy evaluation.']}
    for kind in ['demand','alternative']:
        parts=[]
        for path in args.folder.glob(f'*/{kind}_all.csv'):
            try:
                f=pd.read_csv(path,parse_dates=['landmark_at'])
            except pd.errors.EmptyDataError:
                continue
            if not f.empty: parts.append(f)
        target='true_state' if kind=='demand' else 'true_alternative'
        section={'horizons':{}}
        report['models'][kind]=section
        if not parts:
            section.update(status='insufficient_evidence',reason='No fresh source listings')
            continue
        f=pd.concat(parts,ignore_index=True)
        valid=f[f[target].ge(0)].copy()
        valid.to_csv(args.folder/(kind+'_predictions.csv'),index=False)
        for h,all_rows in f.groupby('horizon_days'):
            rows=all_rows[all_rows[target].ge(0)]
            value={'prediction_rows':len(all_rows),'labelled_rows':len(rows),'unknown_rows':int(all_rows[target].lt(0).sum()),'unique_tickets':rows.ticket_id.nunique(),'events':rows.event_id.nunique()}
            if len(rows):
                y=rows[target].to_numpy(int); p=probabilities(rows,kind)
                baseline=np.zeros_like(p); baseline[:,0]=1
                value['metrics']=score(y,p)
                value['always_active_or_no_arrival']=score(y,baseline)
                value['calibration_comparison']=compare(rows,kind,int(h))
                value['all_classes_observed']=len(np.unique(y))==p.shape[1]
            else:
                value['status']='insufficient_evidence'
            section['horizons'][str(h)]=value
    paths=[args.folder/(k+'_predictions.csv') for k in ['demand','alternative']]
    if all(p.exists() for p in paths):
        a,b=[pd.read_csv(p) for p in paths]
        keys=['ticket_id','landmark_at','horizon_days']
        joined=a.merge(b[keys+['true_alternative']],on=keys,validate='one_to_one')
        if len(joined) and joined.true_state.eq(1).any() and joined.true_alternative.nunique()==2:
            subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('worker.py')),'--kind','buy','--output',str(args.folder)],check=True)
            report['buy_timing']=json.loads((args.folder/'buy_report.json').read_text(encoding='utf-8'))
            subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('policy_compare.py')),'--folder',str(args.folder)],check=True)
            report['policy_comparison']=json.loads((args.folder/'policy_comparison.json').read_text(encoding='utf-8'))
        else:
            report['buy_timing']={'status':'insufficient_evidence','aligned_rows':len(joined),'reason':'Need observed sold outcomes and both alternative classes; no policy tuning on this sample'}
    else:
        report['buy_timing']={'status':'insufficient_evidence','reason':'No aligned certified labels'}
    (args.folder/'strict_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

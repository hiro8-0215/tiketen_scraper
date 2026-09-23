"""One entry for new-period validation of three frozen decision models. No fitting."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parent
MODELS=[BASE/'demand_state_model/artifacts/demand_state.joblib',
        BASE/'alternative_arrival_model/artifacts/alternative_arrival.joblib',
        BASE/'buy_timing_model/artifacts/policy.json',
        ROOT/'hybrid_AI_model16/artifacts/model16.joblib']
MANUAL_MASTERS = sorted((ROOT / '手動_data').glob('master_*.csv'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def latest_snapshot():
    def key(p):
        n=tuple(map(int,p.name.removeprefix('data_').split('_')))
        return n if len(n)==3 else (datetime.now().year,*n)
    paths=[p for p in (ROOT/'tiketen_date_data').glob('data_*') if p.is_dir() and any(p.glob('*_master.csv'))]
    return max(paths,key=key)


def snapshot_audit(old,new):
    def load(folder):
        frames=[]
        for path in folder.glob('*_master.csv'):
            f=pd.read_csv(path,dtype=str).fillna('')
            if f.empty:
                continue
            f['logical_id']=f.event_id+'|'+f.created_at_unix.str.replace(r'\.0$','',regex=True)
            f.loc[f.created_at_unix.eq(''),'logical_id']='ticket:'+f.ticket_id
            frames.append(f)
        return pd.concat(frames,ignore_index=True).drop_duplicates('logical_id')
    a,b=load(old),load(new)
    joined=a[['logical_id','status']].merge(b[['logical_id','status']],on='logical_id',suffixes=('_old','_new'))
    old_sold=set(a.loc[a.status.eq('sold'),'logical_id'])
    new_sold=set(b.loc[b.status.eq('sold'),'logical_id'])
    incoming=b[~b.logical_id.isin(a.logical_id)]
    return {'baseline':str(old),'new_snapshot':str(new),'new_ticket_rows':len(incoming),
            'new_ticket_status_counts':incoming.status.value_counts().to_dict(),
            'new_sold_id_count':len(new_sold-old_sold),
            'transitions':{f'{a}->{b}':int(n) for (a,b),n in joined.groupby(['status_old','status_new']).size().items()},
            'sold_collection_requires_review':not bool(new_sold-old_sold)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--prepare',action='store_true',help='Generate missing semantics and refresh Model16 price predictions; no training')
    p.add_argument('--check',action='store_true',help='Check frozen models and load coverage without inference or generation')
    args=p.parse_args()
    if args.check and args.prepare:
        p.error('--check and --prepare cannot be combined')
    snapshot=(args.data_dir or latest_snapshot()).resolve()
    reports=[json.loads((BASE/f'{kind}/artifacts/training_report.json').read_text(encoding='utf-8'))
             for kind in ['demand_state_model','alternative_arrival_model']]
    start=max(pd.Timestamp(r['observation_cutoff']) for r in reports)
    baseline=json.loads((ROOT/'hybrid_AI_model16/artifacts/incremental_baseline.json').read_text(encoding='utf-8'))
    start=max(start,pd.Timestamp(baseline['training_cutoff']))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in MODELS}
    manual_hashes={str(p.relative_to(ROOT)):sha(p) for p in MANUAL_MASTERS}
    output=HERE/'artifacts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    manifest={'snapshot':str(snapshot),'evaluation_start_exclusive':str(start),'model_hashes':hashes,
              'manual_master_hashes':manual_hashes,
              'training_executed':False,'status':'running',
              'limitation':'Retrospective reconstruction from latest saved fields/current manual metadata; not a timestamped live prediction trial.',
              'data_audit':snapshot_audit(Path(reports[0]['snapshot_dir']),snapshot)}
    save(output/'manifest.json',manifest)
    def run(command,cwd,stage):
        print('RUN: '+' '.join(map(str,command)),flush=True)
        with (output/'run.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen(list(map(str,command)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  encoding='utf-8',errors='replace')
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            if proc.wait():
                raise RuntimeError(f'Stage failed: {stage}. See {output / "run.log"}')
    display_results = False
    try:
        if args.prepare:
            for folder,script in [('semantic_data_builder','extract_semantic_json.py'),('model16_price_bridge','build_fair_price_cache.py')]:
                run([sys.executable,'-X','utf8',BASE/folder/script,'--data-dir',snapshot],BASE/folder,folder)
        for kind in ['demand','alternative']:
            command=[sys.executable,'-X','utf8',HERE/'worker.py','--kind',kind,'--snapshot',snapshot,
                     '--start',str(start),'--output',output]
            if args.check:
                command.append('--check')
            run(command,HERE,kind)
        if not args.check:
            run([sys.executable,'-X','utf8',HERE/'worker.py','--kind','buy','--output',output],HERE,'buy')
        after={str(p.relative_to(ROOT)):sha(p) for p in MODELS}
        if hashes!=after:
            raise RuntimeError('Saved models changed during evaluation; results cannot be accepted')
        manifest['status']='checked' if args.check else 'completed'
        # Persist the terminal state before any viewer reads the manifest.
        # Result rendering is presentation only and must never invalidate an
        # otherwise completed model evaluation.
        save(output/'manifest.json',manifest)
        if not args.check:
            save(HERE/'artifacts/latest.json',{'folder':str(output)})
            display_results = True
        print(f'Completed. Results: {output}')
    except Exception as exc:
        manifest['status']='failed'
        manifest['error']=str(exc)
        raise
    finally:
        save(output/'manifest.json',manifest)
    if display_results:
        viewer = subprocess.run(
            [sys.executable,'-X','utf8',HERE/'show_results.py','--folder',output],
            cwd=HERE,
        )
        if viewer.returncode:
            print(
                f'WARNING: validation completed, but result display failed '
                f'(exit={viewer.returncode}). Run the result-view launcher.',
                flush=True,
            )


if __name__=='__main__':
    # Launchers now use snapshot evidence; retain the old implementation only
    # for reading historical artifacts, never as the default evaluation path.
    from strict_run import main as strict_main
    strict_main()

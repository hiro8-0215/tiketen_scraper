"""Audit, frozen snapshot validation, then optional chronological comparison."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from observation import read_snapshot, snapshot_end, audit
from run import ROOT, BASE, HERE, MODELS, sha, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--data-dir',type=Path)
    args=p.parse_args()
    cutoff=max(pd.Timestamp(json.loads((BASE/f'{k}/artifacts/training_report.json').read_text(encoding='utf-8'))['observation_cutoff']) for k in ['demand_state_model','alternative_arrival_model'])
    price_baseline=json.loads((ROOT/'hybrid_AI_model16/artifacts/incremental_baseline.json').read_text(encoding='utf-8'))
    cutoff=max(cutoff,pd.Timestamp(price_baseline['training_cutoff']))
    snapshots=[]
    for folder in (ROOT/'tiketen_date_data').glob('data_*'):
        if folder.is_dir() and any(folder.glob('*_master.csv')):
            end=snapshot_end(read_snapshot(folder))
            if end > cutoff:
                snapshots.append((end,folder))
    snapshots.sort(key=lambda item:(item[0],str(item[1])))
    if args.data_dir:
        limit=snapshot_end(read_snapshot(args.data_dir))
        snapshots=[item for item in snapshots if item[0]<=limit]
    output=HERE/'artifacts'/('strict_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    hashes={str(path.relative_to(ROOT)):sha(path) for path in MODELS}
    manifest={'status':'running','evaluation_version':'snapshot_evidence_v2','training_executed':False,
        'model_hashes':hashes,'training_cutoff':str(cutoff),'pairs':[],
        'limitations':['Current manual metadata is used; historical versions are unavailable.',
                      'No collection log means no certified negative alternative-arrival labels.',
                      'Comparability requires equal recorded semantic fields; differs from legacy unknown-wildcard labels.']}
    save(output/'manifest.json',manifest)
    save(output/'collection_audit.json',audit([path for _,path in snapshots]))
    for path in (ROOT/'手動_data').glob('master_*.csv'):
        dest=output/'manual_masters'/path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path,dest)
    manifest['manual_hashes']={str(p):sha(p) for p in (ROOT/'手動_data').glob('master_*.csv')}
    manifest['snapshot_hashes']={str(p):sha(p) for _,folder in snapshots for p in folder.glob('*_master.csv')}
    def execute(script,*parameters):
        command=[sys.executable,'-X','utf8',str(script),*map(str,parameters)]
        print('RUN: '+' '.join(command),flush=True)
        with (output/'run.log').open('a',encoding='utf-8') as log:
            proc=subprocess.Popen(command,cwd=script.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8',errors='replace')
            for line in proc.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            if proc.wait():
                raise RuntimeError(f'Stage failed: {script.name}; see {output}')
    try:
        if len(snapshots)<2:
            manifest['status']='insufficient_evidence'
            manifest['reason']='Need two snapshots after the model training cutoff'
        elif args.check:
            manifest['status']='checked'
        else:
            for (_,source),(_,target) in zip(snapshots[:-1],snapshots[1:]):
                pair=output/(source.name+'__'+target.name)
                pair.mkdir()
                if args.prepare:
                    for snapshot in [source,target]:
                        execute(BASE/'semantic_data_builder/extract_semantic_json.py','--data-dir',snapshot)
                prices=pair/'fair_prices.csv'
                execute(BASE/'model16_price_bridge/build_fair_price_cache.py','--data-dir',source,'--output',prices)
                for kind in ['demand','alternative']:
                    execute(HERE/'snapshot_worker.py','--kind',kind,'--source',source,'--target',target,'--prices',prices,'--output',pair/(kind+'_all.csv'))
                manifest['pairs'].append(str(pair))
            execute(HERE/'strict_summary.py','--folder',output)
            manifest['status']='completed'
        if any(sha(ROOT/name)!=value for name,value in hashes.items()):
            raise RuntimeError('Model changed during evaluation')
        if any(sha(Path(name))!=value for name,value in manifest['manual_hashes'].items()):
            raise RuntimeError('Manual metadata changed during evaluation')
        if any(sha(Path(name))!=value for name,value in manifest['snapshot_hashes'].items()):
            raise RuntimeError('Snapshot changed during evaluation')
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        save(output/'manifest.json',manifest)
    if manifest['status']=='completed':
        save(HERE/'artifacts/latest.json',{'folder':str(output)})
    print(json.dumps({'status':manifest['status'],'results':str(output)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

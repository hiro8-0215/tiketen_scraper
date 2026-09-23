"""Event collection evidence, independent of other model folders."""
import json
import numpy as np
import pandas as pd


def load_coverage(folder):
    records=[]
    for path in folder.glob('observation_*.jsonl'):
        records.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
    return records


def observed_windows(records,event,moments,deadlines):
    times=np.array(sorted(set(pd.Timestamp(r['observed_at']).to_datetime64() for r in records
        if r.get('complete') is True and r.get('event_id')==event)),dtype='datetime64[ns]')
    valid=np.zeros(len(moments),dtype=bool)
    if len(times)<2: return valid
    gaps=np.r_[0,np.cumsum(np.diff(times)>np.timedelta64(2,'h'))]
    left=np.searchsorted(times,moments,side='right')-1
    right=np.searchsorted(times,deadlines,side='left')
    within=(left>=0)&(right<len(times))
    valid[within]=gaps[left[within]]==gaps[right[within]]
    return valid

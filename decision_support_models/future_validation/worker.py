"""Isolated inference workers: existing model imports never share config modules."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def evaluate_model(kind, snapshot, start, output, check=False):
    folder = BASE / ('demand_state_model' if kind == 'demand' else 'alternative_arrival_model')
    sys.path.insert(0, str(folder))
    from data_loader import load_tickets
    from timeline import build_landmarks, observation_cutoff
    from features import add_market_features
    from modeling import enforce_monotonic_horizons
    model_file = folder / 'artifacts' / ('demand_state.joblib' if kind == 'demand' else 'alternative_arrival.joblib')
    payload = joblib.load(model_file)
    if pd.Timestamp(start) < pd.Timestamp(payload['observation_cutoff']):
        raise ValueError('Evaluation starts before the fitted model observation cutoff')
    horizons = sorted(map(int, payload['selected_features']))
    tickets = load_tickets(snapshot)
    cutoff = observation_cutoff(tickets)
    if cutoff <= pd.Timestamp(start):
        raise ValueError('No observations after the saved models training cutoff')
    if kind == 'demand' and ('fair_price' not in tickets or tickets.fair_price.isna().any()):
        raise ValueError('Complete Model16 fair prices required; run the full future-validation launcher')
    info = {'training_executed':False, 'rows_loaded':len(tickets), 'horizons':horizons,
            'start_exclusive':str(start), 'observation_end':str(cutoff)}
    if check:
        write_json(output / (kind+'_check.json'), info)
        return
    # Keep actual first_observed_at for age and arrivals. Only limit expansion
    # to dates strictly after every saved model's training period.
    tickets.attrs['trusted_temporal_start_at'] = str(pd.Timestamp(start)+pd.Timedelta(seconds=1))
    landmarks = build_landmarks(tickets, horizons=horizons, cutoff=cutoff)
    landmarks = landmarks[landmarks.landmark_at.gt(pd.Timestamp(start))].copy()
    if landmarks.empty:
        write_json(output / (kind+'_report.json'), dict(info, eligible_rows=0))
        return
    frame = add_market_features(landmarks, tickets)
    columns = [c for c in ('ticket_id','event_id','landmark_at','price','fair_price',
                           'market_prior_sold_median','market_price_median') if c in frame]
    parts = []
    for horizon in horizons:
        selected = payload['selected_features'][str(horizon)]
        x = frame[selected['numeric'] + selected['categorical']]
        part = frame[columns].copy()
        part['horizon_days'] = horizon
        # fold is an alignment field in the policy loader, never represented
        # as OOF evidence: all these predictions use frozen final models.
        part['fold'] = -1
        if kind == 'demand':
            from modeling import aligned_probabilities, apply_temperature
            p = aligned_probabilities(payload['models'][str(horizon)], x)
            p = apply_temperature(p, payload['temperatures'][str(horizon)])
            part[['p_active','p_sold','p_deleted']] = p
            part['true_state'] = frame[f'state_{horizon}d'].to_numpy()
        else:
            from modeling import predict_positive_probability, calibrate
            p = predict_positive_probability(payload['models'][str(horizon)], x)
            part['p_alternative'] = calibrate(payload['calibrators'][str(horizon)],p)
            part['true_alternative'] = frame[f'alternative_{horizon}d'].to_numpy()
            part[f'potential_savings_{horizon}d'] = frame[f'potential_savings_{horizon}d'].to_numpy()
        parts.append(part)
    all_predictions = enforce_monotonic_horizons(pd.concat(parts,ignore_index=True))
    target = 'true_state' if kind == 'demand' else 'true_alternative'
    observed = all_predictions[all_predictions[target].ge(0)].copy()
    observed.to_csv(output / (kind+'_predictions.csv'),index=False,encoding='utf-8-sig')
    report = dict(info, eligible_rows=len(observed), censored_rows=int(all_predictions[target].lt(0).sum()),
                  evaluation_kind='retrospective_new_period_frozen_models', metrics={}, class_counts={})
    for h in horizons:
        part = observed[observed.horizon_days.eq(h)]
        report['class_counts'][str(h)] = part[target].value_counts().to_dict()
        if part.empty:
            report['metrics'][str(h)] = None
        elif kind == 'demand':
            from modeling import probability_metrics
            report['metrics'][str(h)] = probability_metrics(part[target].to_numpy(int),part[['p_active','p_sold','p_deleted']].to_numpy(float))
            report.setdefault('sold_validation_supported',{})[str(h)] = bool(part[target].eq(1).any())
        else:
            from modeling import metrics
            report['metrics'][str(h)] = metrics(part[target].to_numpy(int),part.p_alternative.to_numpy(float))
    write_json(output / (kind+'_report.json'), report)


def evaluate_policy(output):
    sys.path.insert(0,str(BASE/'buy_timing_model'))
    from data_loader import load_oof
    from decision import apply_policy,summarize
    from config import PROFILES
    policy = json.loads((BASE/'buy_timing_model/artifacts/policy.json').read_text(encoding='utf-8'))
    paths = [output/'demand_predictions.csv',output/'alternative_predictions.csv']
    if any(not p.exists() or pd.read_csv(p,nrows=1).empty for p in paths):
        write_json(output/'buy_report.json',{'training_executed':False,'rows':0,'reason':'No jointly evaluable demand/alternative rows'})
        return
    frame = load_oof(*paths)
    report = {'training_executed':False,'evaluation_kind':'retrospective_policy_loss_proxy',
              'warning':'Regret uses fixed penalty assumptions, not realized purchasing profit. Sold collection uncertainty also affects this score.',
              'profiles':{}}
    predictions = []
    for name, horizons in policy['profiles'].items():
        report['profiles'][name] = {}
        for horizon, params in horizons.items():
            part = frame[frame.horizon_days.eq(int(horizon))]
            if part.empty:
                report['profiles'][name][horizon] = None
                continue
            chosen = apply_policy(part,params)
            score = summarize(chosen,PROFILES[name])
            score['always_buy'] = summarize(part.assign(action='buy_now'),PROFILES[name])
            score['always_wait'] = summarize(part.assign(action='wait'),PROFILES[name])
            report['profiles'][name][horizon] = score
            predictions.append(chosen.assign(profile=name))
    if predictions:
        pd.concat(predictions,ignore_index=True).to_csv(output/'buy_predictions.csv',index=False,encoding='utf-8-sig')
    write_json(output/'buy_report.json',report)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['demand','alternative','buy'],required=True)
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--start')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.kind=='buy':
        evaluate_policy(args.output)
    else:
        evaluate_model(args.kind,args.snapshot,args.start,args.output,args.check)

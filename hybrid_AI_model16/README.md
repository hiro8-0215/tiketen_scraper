# Model16 — 全価格共通・nestedグローバルアンサンブル

Model16は、価格帯別モデルや行ごとのexpert切替を使用しません。すべてのチケットへ
同一の固定予測式を適用し、Model13と同じクレンジング済みsold 7,313件で評価します。

## 構成

- `lgbm_log_mae`: One-Hot表特徴＋LLM JSON意味特徴、log-L1
- `lgbm_raw_mape`: 同一表特徴、全行共通の `1 / price` 学習重みによるMAPE expert
- `catboost_raw_mae`: native categorical＋Ordered boosting、raw MAE
- `bert_ridge`: BERTを木へ直結せず、PCA 16/32/48/64/96次元＋Ridgeで容量制限
- 最終統合は全行共通の非負・総和1の固定重みだけを使用

`delivery_method`は1,335種類あるため、LightGBMでは希少カテゴリをまとめたOne-Hot、
CatBoostでは順序付きカテゴリ統計として扱います。さらに配送経路・時期を価格ラベルなしの
決定規則で正規化します。曜日・月・開演時刻などの数値コードは、連続値とカテゴリの両方で
表現し、周期性をsin/cosでも与えます。定数列は目的変数を見ずに除外します。現在は入力96列、
LightGBMのfold-local One-Hot後616列で、BERTは64次元固定をやめて内側foldだけで次元数を選択します。

Qwenは使用しません。ordered Qwen単体MAEは8,612円で、Model15では最適化後もMAEを
261円悪化させました。また既存OOFをnested内側foldへ流用すると厳密な評価を壊します。

## 評価方式

- 外側5-fold: 完全な未使用評価行
- 内側4-fold: 各expertのOptuna、BERT次元、固定blend重みを決定
- 重複説明文は必ず同じfold
- 主指標は全件MAEで学習前に固定
- blendは最良MAE expertよりMAPEが0.10ポイント超悪化しない制約
- 価格帯別の学習・ルーティング・採用判断は禁止

外側foldごとの結果はatomic checkpointへ保存されます。中断後は完了foldとOptuna studyを
再利用します。2種類のLightGBM間でも同じfold-local One-Hot行列を共有し、Optunaを並列化せず、各モデルが
CPU全コアを効率的に利用します。Qwen/BERT抽出やGPU処理はありません。

## 実行

```powershell
cd C:\Users\hero\Documents\tiketen_scraper\hybrid_AI_model16
python run_model16.py
```

厳密なnested評価のため約2,100回の小規模tree fitを行います。目安は4～10時間ですが、
CatBoostの条件とCPU状況により変動します。こちらの構築作業では学習を実行していません。

## 成果物

- `artifacts/evaluation_model16.json`
- `artifacts/oof_predictions_model16.csv`
- `artifacts/candidate_comparison.csv`
- `artifacts/model16.joblib`
- `artifacts/model16_optuna.db`

学習後の推論は `inference.load_and_predict(df, bert_embeddings)` を使用します。
最終BERT重みが0の場合は埋め込みを省略でき、正の重みならModel15と同じBERT v3埋め込みを渡します。

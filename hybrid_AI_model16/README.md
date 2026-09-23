# Model16 — 全価格共通・nestedグローバルアンサンブル

Model16は、価格帯別モデルや行ごとのexpert切替を使用しません。すべてのチケットへ
同一の固定予測式を適用し、Model13と同じクレンジング規則を通過したsoldデータで評価します。
対象件数は最新マスタへ追随し、2026-09-04時点では9,353件です。

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

Qwenの価格予測値やQwen OOFは使用しません。価格を渡さずに抽出したJSON意味特徴だけは
表特徴として使用します。ordered Qwen単体MAEは8,612円で、Model15では価格予測として
混合すると最適化後もMAEを261円悪化させました。また既存OOFをnested内側foldへ流用すると
厳密な評価を壊します。

## 評価方式

- 外側5-fold: 完全な未使用評価行
- 内側4-fold: 各expertのOptuna、BERT次元、固定blend重みを決定
- 重複説明文は必ず同じfold
- 主指標は全件MAEで学習前に固定
- blendは最良MAE expertよりMAPEが0.10ポイント超悪化しない制約
- 価格帯別の学習・ルーティング・採用判断は禁止

外側foldごとの結果はatomic checkpointへ保存されます。中断後は完了foldとOptuna studyを
再利用します。2種類のLightGBM間でも同じfold-local One-Hot行列を共有し、Optunaを並列化せず、各モデルが
CPU全コアを効率的に利用します。学習本体は主にCPUを使い、事前の差分JSON意味抽出と
BERT埋め込み生成だけNVIDIA GPUを使います。

## 実行

VS Codeの「実行とデバッグ」では、次の4項目だけでModel16を完結できます。

1. `[10 価格][確認] Model16 差分・GPU確認（生成なし）`
   - 現在の対象件数、意味特徴/BERTの未生成件数、fold、CUDA、空き容量を確認します。
2. `[10 価格][準備] Model16 意味・BERT・fold差分更新（学習なし）`
   - Model16に不足する行だけ生成し、foldを最新対象へ合わせます。Model15/16は学習しません。
3. `[10 価格][一括] Model16 準備 → 全学習 → 結果`
   - 2の準備を再開可能な状態で済ませ、厳密チェック後にModel16だけを学習して結果を開きます。
4. `[10 価格][結果] Model16 グラフ・指標表示`
   - 保存済み結果だけを表示します。

### 更新データで完全に0から学習する場合

- `[10 価格][0から開始] Model16 全4予測器を完全新規学習`
  - 不足する意味特徴・BERTだけを準備した後、旧Optuna study・旧fold checkpointを
    一切使わず、4予測器と固定blendを別領域で新規学習します。
  - 正常完了したときだけ正式Model16へ切り替え、従来成果物は
    `artifacts_history/production_*` に退避します。
- 中断・PC再起動後は `[10 価格][再開/未開始なら新規] Model16 全学習` を使います。
  未開始で選んだ場合は自動的に0から開始します。
  `0から開始`をもう一度選ぶと、その未完了分も退避して本当に最初からになります。

意味特徴とBERT埋め込みは価格ラベルを学習したモデルではないため差分再利用します。
LightGBM、CatBoost、Ridge、Optuna、fold評価、blendはすべて新規です。

## 追加データの外部評価（再学習なし）

毎回Model16を再学習する必要はありません。現在の学習済みモデルと学習対象を一度固定し、
その後に増えたsoldだけを未使用データとして評価できます。

1. 全学習直後に `[10 価格][追加評価][基準固定] 現在Model16を固定（学習なし）` を1回だけ実行します。
   - `[10 価格][一括]` の正常完了時にも自動で基準を更新します。
2. 新しいデータを配置し、`[00 データ][更新] マスタデータ` と必要な手動項目の補完を行います。
3. 通常は `[10 価格][追加評価][完全] 意味差分 → 新規sold評価（再学習なし）` を実行します。
   - 新規説明文のJSON意味特徴だけを差分生成し、固定済みModel16で予測・評価します。
   - Qwen、LightGBM、CatBoost、BERTの再学習は行いません。現在の最終BERT重みは0なのでBERT生成も不要です。
4. すぐ確認したい場合は `[10 価格][追加評価][高速] 新規sold評価（生成・学習なし）` を使います。
   - 意味特徴を生成せず、既存キャッシュで直ちに評価します。意味特徴の充足率も結果へ記録します。

評価対象は基準固定後に新しく加わった論理出品です。なかでも `first_observed_at` が学習締切より
後の行を厳密な将来評価として別集計します。基準以前から掲載され、あとでsoldになった行は混ぜずに
参考値として分けます。厳密な将来評価が500件以上たまるまでは傾向確認とし、500件到達後に
再学習の要否を判断します。

追加評価の成果物は `artifacts/incremental_evaluation/` に保存されます。

- `baseline.json`: 固定した学習基準とモデルの指紋
- `baseline_population.csv.gz`: 学習時点の論理出品集合
- `latest_predictions.csv`: 基準後のsold予測
- `latest_report.json`: 全追加分と厳密な将来分の指標
- `history.csv`: 評価実行ごとの推移

コマンドで一括実行する場合:

```powershell
cd C:\Users\hero\Documents\tiketen_scraper\hybrid_AI_model16
python run_model16.py
```

意味特徴とBERTの実体は過去モデルと共有可能なtarget-freeキャッシュのため、互換性を保って
`hybrid_AI_model15/artifacts`へ保存されます。ただし利用者がModel15のローンチを先に実行する
必要はありません。Model16の「準備」または「一括」が必要な処理を自動で行います。

厳密なnested評価のため約2,100回のtree fitを行います。旧7,313件の実測は約48.5時間で、
最新9,353件の全再学習は約55～75時間が目安です。これとは別に、初回の差分意味特徴・BERT
準備へ約2～6時間を見込みます。CatBoostの条件、CPU負荷、GPU速度により変動します。
こちらの構築作業では学習を実行していません。

## 成果物

- `artifacts/evaluation_model16.json`
- `artifacts/oof_predictions_model16.csv`
- `artifacts/candidate_comparison.csv`
- `artifacts/model16.joblib`
- `artifacts/model16_optuna.db`

学習後の推論は `inference.load_and_predict(df, bert_embeddings)` を使用します。
最終BERT重みが0の場合は埋め込みを省略でき、正の重みならModel15と同じBERT v3埋め込みを渡します。

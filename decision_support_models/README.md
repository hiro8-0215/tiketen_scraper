# チケット意思決定支援モデル

データはプロジェクト共通の `tiketen_date_data/` と `手動_data/` を読みますが、
モデルコード・設定・検証・推論・成果物は種類ごとのフォルダ内で完結します。

```text
decision_support_models/
├─ demand_state_model/          継続・確認済みsold・deletedの確率
├─ alternative_arrival_model/  より安い比較可能な新規出品の出現確率
└─ buy_timing_model/            上記確率と妥当価格を統合する買い時判断
```

各フォルダは `README.md`、`config.py`、データ処理、学習、推論、評価、
`preflight.py`、単体テスト、`artifacts/` を持ちます。フォルダ間でPython
モジュールをimportしません。統合時は、他モデルのOOF／推論結果をデータファイルとして受け渡します。

Codexによる構築作業では学習を実行しません。

## フォルダ構成

```text
decision_support_models/
├─ README.md
├─ semantic_data_builder/       全ticket用の価格非依存LLM JSON抽出
├─ model16_price_bridge/        Model16の全ticket価格キャッシュ生成
├─ demand_state_model/
│  ├─ config.py / data_loader.py / timeline.py / features.py / modeling.py
│  ├─ train.py / inference.py / evaluate.py / preflight.py / run_training.py
│  ├─ tests/
│  └─ artifacts/
├─ alternative_arrival_model/
│  ├─ config.py / data_loader.py / timeline.py / features.py / modeling.py
│  ├─ train.py / inference.py / evaluate.py / preflight.py / run_training.py
│  ├─ tests/
│  └─ artifacts/
└─ buy_timing_model/
   ├─ config.py / data_loader.py / decision.py
   ├─ train_policy.py / inference.py / evaluate.py / preflight.py
   ├─ prepare_inputs.py / run_training.py / tests/
   ├─ inputs/
   └─ artifacts/
```

## 実行順

全工程を一度に実行する場合は、VS Codeの「実行とデバッグ」から
`[20 一括][自動] 完了stateから再開` を選択します。入力スナップショットまたは
コードのfingerprintが変わった場合は、完了stateを引き継がず全工程を実行します。
一括ランナーの安全条件と再開方法は `pipeline_runner/README.md` に記載しています。

1. `semantic_data_builder/preflight.py`
2. `semantic_data_builder/extract_semantic_json.py`（全説明文の意味データ生成）
3. `model16_price_bridge/build_fair_price_cache.py`
4. `demand_state_model/preflight.py`、その後に同フォルダの `run_training.py`
5. `alternative_arrival_model/preflight.py`、その後に同フォルダの `run_training.py`
6. `buy_timing_model/prepare_inputs.py`
7. `buy_timing_model/preflight.py`、その後に同フォルダの `run_training.py`

個々の `README.md` にコマンドと出力を記載しています。`preflight.py` はfitを
行わず、`run_training.py` だけが明示的に学習を開始します。

通常は先に`[20 一括][確認] パイプライン dry-run（学習なし）`で実行予定を確認します。
LLM意味特徴とModel 16価格キャッシュを維持して需要以降だけを再学習するときは、
`[20 一括][再学習] 需要 → 代替 → 買い時`を使用します。全ローンチの分類と
安全区分は`.vscode/README.md`に記載しています。

意味特徴は全説明文を覆わない限り需要・代替モデルが学習を拒否します。各期間で
LLM JSONあり／なしを同一fold比較し、log lossが実質改善した場合だけ自動採用します。
Qwen価格回帰とBERT埋め込みは既存評価で悪化したため使用しません。

`data_8_6`と`data_8_26`は、実体が2026-07-28で止まり、同時刻に3,801件が
一括で`deleted`化された異常snapshotです。需要・代替・買い時の正式な学習にも、
現在掲載中チケットへの推論にも使用しません。一括ランナーは既定でこれを拒否します。
修正版スクレイパーからlistingを含む新しいsnapshotを取得し、欠測期間の終了状態を
推測で補わずに再構築してから再学習します。

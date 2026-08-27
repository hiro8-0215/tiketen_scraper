# 需要状態モデル

各ランドマークから1・3・7日以内の状態を、`active`（その期間を超えて掲載継続）、
`sold`（販売確認）、`deleted`（消失）の3クラスで予測します。deletedを売却扱い
しないため、他媒体での販売や出品取消の不確実性を残したモデルです。

全ticketを覆う価格非依存LLM JSONから、座席階層、列位置、当選経路、名義、
本人確認、配布、視界、FC初期、ランダム条件を入力候補にします。期間ごとに同一foldの
`tabular / semantic` アブレーションを行い、log loss改善かつBrier非悪化の場合だけ
意味特徴を最終モデルへ採用します。

## ファイル

- `data_loader.py`: 最新スナップショットと手動マスタの読込
- `timeline.py`: 日次ランドマークと将来ラベル
- `features.py`: ランドマーク時点までの市場特徴
- `modeling.py`: LightGBM、時系列分割、確率校正、指標
- `train.py` / `inference.py` / `evaluate.py`: 学習・推論・OOF検証
- `preflight.py`: 学習しない事前検査
- `tests/`: 合成データ単体テスト
- `artifacts/`: このモデルだけの学習済み成果物

Model16価格を使う場合、`artifacts/fair_price_all_tickets.csv` に
`ticket_id,fair_price` を用意します。soldのみを覆うキャッシュは、欠損自体が
正解を漏らすため拒否されます。全チケットを覆わない限り価格キャッシュは使いません。

```powershell
cd C:\Users\hero\Documents\tiketen_scraper\decision_support_models\demand_state_model
python preflight.py
python run_training.py
python inference.py
python evaluate.py
```

`preflight.py` は読込検査だけです。学習は `train.py` または `run_training.py` を
明示的に実行した場合だけ始まります。

前処理表とOOF foldは`artifacts/`へ指紋付き・atomicに保存されます。中断後に同じ入力と
コードで再実行した場合だけ再利用されます。価格0は状態を示す収集エラーにならないよう、
行を残したまま欠損化し、各foldのtraining側中央値で補完します。

事前に `semantic_data_builder/extract_semantic_json.py` を完了させてください。

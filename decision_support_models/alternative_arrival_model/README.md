# 安価な代替出品の出現モデル

現在の出品より `max(2,000円, 5%)` 以上安く、同一イベント・枚数・券種・
名義種別が一致する新規出品が1・3・7日以内に現れる確率を予測します。
価格変更履歴を必要とせず、`first_observed_at` の新規流入を学習対象にします。

需要モデルと同じ全ticket用LLM JSON意味特徴を候補にし、期間別の同一fold
アブレーションでlog loss・Brier・PR-AUCを守って改善した場合だけ採用します。
座席階層・列位置・視界条件が双方で判明して矛盾する候補は、安くても比較可能な
代替には数えません。片方が不明な場合は除外せず、過剰なデータ削減を防ぎます。

フォルダ内に読込、時系列ラベル、特徴、LightGBM学習、推論、OOF評価、
事前検査、単体テスト、成果物をすべて持ちます。他モデルのPythonコードは
importしません。

```powershell
cd C:\Users\hero\Documents\tiketen_scraper\decision_support_models\alternative_arrival_model
python preflight.py
python run_training.py
python inference.py
python evaluate.py
```

`preflight.py` は学習しません。比較可能条件と最低節約額は `config.py` で固定し、
検証時にも同じ定義を使用します。

前処理表とOOF foldは`artifacts/`へ指紋付き・atomicに保存されます。価格0では節約額を
定義できないため、0円の基準出品と候補出品は監査件数を残して価格比較母集団から除外します。

事前に `semantic_data_builder/extract_semantic_json.py` を完了させてください。

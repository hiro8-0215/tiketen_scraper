# Model 14 — sold価格特化・高精度版

Model 14 predicts the transaction price conditional on a ticket having sold.
It intentionally learns from sold rows only. Unsold/deleted/listing rows are not
negative examples because sale probability is outside this model's objective.

## Model 13からの方針

- 売却済みで正の価格を持つ行は、説明文の長さやイベント内価格順位で落とさない。
- Model 13の豊富な表特徴、説明文特徴、ドメイン特徴を維持する。
- 出品時点より前に成立した同一公演の件数・平均・中央値をas-of特徴として使う。
- `sold_at`、売れるまでの日数、最終観測時刻、最終売却率はモデル入力にしない。
- Qwen価格回帰は5-fold LoRAで完全OOF予測を生成する。
- 完全一致説明文は必ず同じfoldに置き、重複による精度水増しを防ぐ。
- BERTは生埋め込みだけ先に取得し、PCAは各foldの学習行だけでfitする。
- LightGBMと、インストール済みならCatBoostをOOFで比較・ブレンドする。

## 実行順序

```powershell
python run_all.py
```

または各段階を個別に実行します。

```powershell
python make_folds.py
python 1_extract_bert.py
python 2_train_qwen_oof.py
python 3_train_meta.py
```

Qwenはfold単位でも実行できます。中断・再開やGPU時間の分割に使います。

```powershell
python 2_train_qwen_oof.py --fold 0
python 2_train_qwen_oof.py --fold 1
python 2_train_qwen_oof.py --fold 2
python 2_train_qwen_oof.py --fold 3
python 2_train_qwen_oof.py --fold 4
```

最終成果物は`artifacts/`に保存されます。`evaluation.json`と
`oof_predictions.csv`が主評価、`model14_meta.joblib`が全soldで再学習した
後段モデルです。新規推論時は5個のQwen fold adapterの平均を入力します。

## 評価の意味

主評価は「売却済みチケットの既知市場内価格推定」です。Model 13に近い用途を
維持しつつ、Qwenが検証行の価格を学習済みというリークだけを除去しています。
未知公演価格の評価や売れる確率の予測は別モデル・別指標として扱います。

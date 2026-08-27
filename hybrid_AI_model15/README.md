# Model 15 — 意味特徴復活・価格帯バイアス補正版

## 最終メタモデル再選択（Qwen再実行なし）

修正済みQwen OOFの評価により、Qwen/BERTを常にLightGBMへ入力する構成が
MAEを悪化させることが判明しました。現在の最終版は、次の4特徴構成と3学習方式を
120-trialの共同Optuna探索で比較し、同一5-fold OOFで最良だった構成だけを
本番モデルへ保存します。

- 特徴構成: full / qwen_only / bert_only / tabular_only
- 学習方式: log_l1 / log_weighted / raw_huber
- 価格重みの指数とraw Huberのalphaも固定せず、選択された方式の中で最適化
- LLM JSON意味特徴は全構成で維持し、選択後に同条件の除外比較を実施
- 既存Qwen OOFとBERTキャッシュを読み取るだけで、Qwen学習・推論・BERT抽出は行わない
- LightGBMはCPU全論理コアを利用し、Optunaは逐次実行して過剰なCPU競合を避ける

実行するのは次のエントリーポイントだけです。

```powershell
python retrain_meta_model15.py
```

VS Codeでは
`[Model15最終修正版] 特徴構成＋学習方式を自動選択（Qwen再実行なし）`
を選択してください。

## 完了済み学習の修復

初回完了版には、長さ別 sampler で並べ替えた Qwen 予測を元の ticket 行へ代入する不具合がありました。
既存の5 fold `best_adapter`は再利用できるため、Qwenを約28時間再学習せずに修復できます。

```powershell
python repair_model15.py
```

修復経路は ordered FP16 Qwen 再推論、Qwen OOF 品質検査、修正済み後段モデル学習、結果表示を行います。
原因と全修正は `MODEL15_ACCURACY_REPAIR_REPORT.md` を参照してください。

Model 15はModel 14の分析結果を反映しつつ、学習母集団をModel 13と同等の
意味のあるクリーンデータへ戻した精度改善モデルです。BERTはticket IDと入力文ハッシュが
一致するModel15キャッシュだけを再利用し、Qwen価格モデルはクレンジング後のデータで5-fold OOF再学習します。

## 改善点

- LLM JSONの座席階層・列位置・FC初期・ランダム条件を復活
- 名義状態・本人確認・当選経路・配布方法・視界条件の拡張schemaを追加
- `price_estimate`は明示的に禁止し、価格ラベル由来リークを防止
- 正規表現ドメイン特徴、BERT、Qwen OOF、as-of相場とJSON意味特徴を統合
- Model 14のlog-L1に加え、価格weight付きlog-L2とraw-price Huberを学習
- 高価格ほどraw-priceモデルを強くする固定tail blendを評価
- JSON意味特徴なしの同一foldアブレーションを自動生成
- クレンジング後soldではMAE・MdAPE・WMAPEを主指標とし、MAPEは参考値にする
- Model 13と同じ説明文5文字未満・取引ノイズ語・2,000～150,000円外・公演内下位5%／上位2%の除外を適用
- Model 14の全sold用Qwen OOFは流用せず、クレンジング後データだけで再学習
- BERTはticket IDと説明文・タグのhashが一致する行だけ再利用し、新規・更新行をGPU抽出
- 4特徴構成×3学習方式を120-trialの共同Optuna探索で自動選択し、fold前処理行列を全trialで再利用

## 実行

完全版は、既存Model 13のJSONを初期値として読み込み、Model 13クレンジング後のsold説明文をModel 15の拡張schemaで再抽出してからQwen OOFと統合モデルを学習します。再抽出は4件batch（OOM時は自動縮小）で専用NVIDIA GPUを使用し、20件ごとに保存して再開できます。

```powershell
python run_all.py
```

学習を開始せずデータ・GPU・空き容量・依存ライブラリだけ確認する場合:

```powershell
python preflight.py
```

`run_all.py` は Step 0 → 新規行BERT抽出 → clean fold作成 → 全意味JSON再抽出 → clean Qwen OOF学習 → 統合学習 → 結果表示を順に実行します。すでに存在するBERT行、Qwen15で抽出済みの説明文、完了済みfoldは再処理しません。

学習やLLM抽出を行わず、Model15の準備だけを確認したい場合は、スクリプトを実行せずこのフォルダとVS CodeのModel15実行構成を使用してください。

拡張されたModel 15 schemaで全説明文を再抽出する場合:

```powershell
python bootstrap_artifacts.py
python 1_extract_bert.py
python make_folds.py
python 1_extract_semantic_json.py --refresh-legacy --batch-size 4
python 2_train_qwen_oof.py
python 2_train_model15.py
python view_results.py
```

`1_extract_semantic_json.py`は価格を生成せず、出品時点で説明文から分かる意味だけを
JSON化します。中断時は20件ごとに保存され、再実行で続きから開始します。

## 学習方式・実行計画

1. 最新snapshotを選び、Model 13クレンジングを適用
2. ticket IDと入力文hashが一致するBERTを再利用し、新規・更新行だけ抽出
3. 重複説明文を同じ側へ置く5-foldを作成
4. 価格を含まないLLM意味JSONを4件batchで抽出
5. Qwen 7Bを4bit NF4、5 epochs、5-fold OOFで学習
6. fold-local補完・OneHot・BERT PCAを作成
7. 前処理済みfold行列を使い回し、4特徴構成×3学習方式をOptuna 120 trialsで共同探索
8. 価格重み指数とraw Huber alphaも条件付きで最適化
9. 12構成の最良trialを同一OOFで再評価し、最小MAEの健全な候補を自動選択
10. 選択候補と同じ条件でLLM JSON列だけを外し、追加効果を測定
11. 選択した特徴構成・学習方式・反復数を全クリーンデータで最終学習

Qwenは実効batch 32を維持し、専用NVIDIA GPUの90%を上限にします。CPU/disk offloadを検出した場合は停止し、VRAM不足時は物理batchを4から2へ下げてgradient accumulationを増やします。BERTと意味JSONもOOM時にbatchを自動調整し、途中成果物はatomic保存します。

実行前チェックは専用VRAM空き8GiB、システムRAM空き16GiB、Qwenキャッシュ存在時でもディスク空き5GiBを最低条件とします。

OOF値はモデル選択にも使う内部推定です。最終的な無偏り精度は、モデル構成を固定した後の未使用の将来snapshotで確認します。

## 成果物

- `artifacts/evaluation_model15.json`
- `artifacts/oof_predictions_model15.csv`
- `artifacts/candidate_comparison.csv`
- `artifacts/model15.joblib`

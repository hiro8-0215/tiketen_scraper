# Model15 精度低下の追加監査と修正

> 2026-08-10追記: ordered Qwen修復後、BERTなし（MAE 6,293円）と
> Qwenなし（6,360円）が固定全統合モデル（6,497円）を上回った。このためv3では、
> 4特徴構成×3学習方式を120-trialで共同探索し、アブレーションを含む最良候補と
> 同一構成を本番保存する。実行入口は `retrain_meta_model15.py` で、Qwen再実行は不要。

## 結論

Model15 の保存済み評価は、Qwen 自体の学習失敗ではなく、長さ別 sampler で並べ替えられた
予測配列を元の ticket 行へそのまま代入したことが最大の原因だった。学習ログの検証
log-MSE は fold ごとに 0.17～0.48 だったが、保存後の行で再計算すると 0.81～1.10、
Qwen 単体 MAE は 19,086 円になっていた。

保存済み 5 fold の `best_adapter` は検証 loss が正常に計算された状態で保存されている。
そのため約28時間の Qwen adapter 学習を繰り返さず、元の行順で再推論して OOF を修復できる。

## 確認した原因と修正

1. **Qwen OOF の行順ずれ（重大）**
   - Transformers 5.13 は `train_sampling_strategy=group_by_length` を評価 sampler にも適用する。
   - `OrderedEvalTrainer` を追加し、学習だけ長さ別、評価・予測は `SequentialSampler` に固定した。
   - fold log-MSE と全体 MAE の品質ゲートを追加し、同じ事故を自動停止する。

2. **bf16 の回帰値解像度不足**
   - 7,313 行に対して Qwen 出力が 81 種類しかなかった。Model14 の FP16 は 632 種類だった。
   - 4bit NF4 は維持し、計算 dtype・学習 autocast・修復推論を FP16 に変更した。

3. **raw Huber の定数化**
   - 旧 raw Huber は全件をほぼ 34,620～35,227 円と予測し、MAE 16,220 円だった。
   - raw price を 10,000 円単位に正規化して Huber を学習し、出力時に円へ戻す。
   - 予測分散と R² の健全性検査を追加し、壊れた raw expert と blend を primary 候補から除外する。

4. **early stopping 指標の不一致**
   - 旧実装は最終目的が円 MAE なのに LightGBM の既定 log-L2 等で停止していた。
   - 全候補を変換後の円 MAE で early stopping するよう変更した。

5. **Model13 の有効な交互作用特徴が欠落**
   - `bante_x_baseprice` と `surikae_x_bante` を復活した。

6. **不安定・極端に疎な直接入力**
   - `seller_rating` は72.9%欠損、`row_number`は99.78%欠損、`block_rank`は99.85%欠損、
     `ticket_count_offered`は97.33%欠損だった。
   - Model13 と同様、これらは raw データから消さず LightGBM の直接列だけから除外した。

7. **意味 JSON の不要な parse error**
   - greedy regex ではなく最初の完全な JSON object を decoder で読むよう変更した。
   - legacy boolean の文字列 `"false"` が true になる変換も修正した。

8. **成果物の版管理不足**
   - `pipeline_version` と `qwen_oof_schema_version` を追加した。
   - 旧評価・旧モデル・順序未修復 Qwen を後段学習と結果表示が拒否する。

9. **寄与分析不足**
   - LLM JSON なしに加え、Qwen なし・BERT なしを同じ fold で評価する。
   - 次回レポートには各特徴の MAE 改善額を保存する。

## 修復手順

VS Code の次の構成を実行する。

`[Model15修正版] 既存adapterから全修復（Qwen再学習なし）`

内部では次だけを実行する。

1. 修復用 preflight
2. 旧成果物を `artifacts/pre_order_repair/` へ退避
3. 保存済み 5 fold adapter の順序固定 FP16 推論
4. 修正済み LightGBM/Optuna 後段学習
5. 新評価表示

`2_train_qwen_oof.py` は呼ばないため、Qwen adapter の約28時間の学習は繰り返さない。

## 比較上の注意

Model13 公称値は旧 snapshot の 6,049 行で、Qwen 学習内サンプルを含む楽観的な特徴と
通常の row KFold を使用している。Model15 の7,313行・重複group OOFとは完全な同条件ではない。
修復後の値を Model15 の正式結果とし、最終的な無偏り性能は未使用の将来 snapshot でも確認する。

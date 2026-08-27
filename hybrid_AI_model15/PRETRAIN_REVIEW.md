# Model15 実行前最終レビュー

> 2026-08-09追記: 完了後監査でQwen OOFの行順不整合を発見し、`model15_repaired_v2`へ修正しました。
> この事前レポートより `MODEL15_ACCURACY_REPAIR_REPORT.md` を優先してください。

確認日: 2026-08-06

## 判定

コード・データ契約・依存ライブラリは学習開始可能な状態。ただし、Cドライブ空きが
3.0GiBしかないため、`preflight.py` は安全基準5GiBで停止する。学習開始前に最低2GiB、
できれば5GiB以上を追加で確保する。

## 最新データ

- snapshot: `data_8_6`
- 読み込み行: 63,190
- sold: 9,654
- 説明文5文字以上: 9,484
- 取引ノイズ語除外後: 8,165
- 2,000～150,000円: 7,820
- 公演内下位5%・上位2%除外後: 7,316
- ticket ID重複除外後の学習対象: 7,313
- 公演数: 21
- 一意説明文: 6,057

## 学習方式

- Qwen 2.5 7B Instruct、4bit NF4 QLoRA
- 5 folds、5 epochs、物理batch 4、gradient accumulation 8、実効batch 32
- duplicate description groupをfold間で分離
- Qwen価格特徴は完全OOFで作成
- BERTはraw 768次元を保持し、各foldの学習側だけでPCA 64次元化
- 表データの補完・OneHotも各foldの学習側だけでfit
- Model13式sqrt(price)重みを復活
- log-L1、重み付きlog-L2、raw Huber、固定tail blendを比較
- LightGBMはOptuna 50 trialsでMAEを最適化
- LLM JSON列だけを外したアブレーションを同一foldで評価

## 効率・資源管理

- CUDA device 0がWindows上の `NVIDIA RTX A2000 12GB`であることをpreflightで確認
- CUDAモデルのCPU/disk offloadを禁止
- 専用VRAM使用上限90%、Qwen学習に必要な空きVRAM8GiBを事前確認
- Qwen OOM時は物理batch 4→2、gradient accumulation 8→16として実効batchを維持
- BERT・意味JSONもbatch OOM時に自動縮小
- BERTはticket IDだけでなく入力文hash一致時のみ再利用
- Qwenは入力・ラベル・fold・prompt versionのglobal fingerprint一致時のみ再開
- Optunaはmeta特徴全体のfingerprint別SQLite studyとして50 trialまで再開
- BERT/PCA/OneHotのfold行列をOptuna全trialで再利用
- JSON、BERT、fold、Qwen OOF、最終成果物はatomic replacementで保存

## 検証結果

- Python構文: OK
- VS Code `launch.json`: OK、Model15項目10件
- `git diff --check`: OK
- 依存バージョンとTrainingArguments引数互換性: OK
- 単体テスト: 8件中7件成功、1件skip
- skip理由: クレンジング後fold/Qwen成果物はまだ学習していないため。想定どおり。
- 学習、LLM抽出、BERTモデル読込は未実行

## 実行前に残る条件

1. Cドライブ空きを5GiB以上にする。
2. `[Model15] 実行前チェック（学習なし）`が成功することを確認する。
3. 成功後に`[Model15] 全学習（意味特徴復活・高額帯補正）`を開始する。
4. OOFはOptuna・候補選択にも使う内部推定なので、構成固定後の未使用の将来snapshotで最終精度を確認する。

新規チケットを直接入力する本番推論CLI/APIは学習パイプラインとは別工程であり、現時点では未実装。

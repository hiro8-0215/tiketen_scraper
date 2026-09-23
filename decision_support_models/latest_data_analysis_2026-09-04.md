# 2026-09-04 最新データ監査

## 統合状態

- 対象: `tiketen_date_data/data_9_4`
- master: 13ファイル
- raw rows: 7,764
- logical rows: 7,296
- listing: 2,684
- sold: 2,960
- deleted: 1,652
- snapshot quality gate: PASS

## data_9_3との差分

- 新規logical listings: 280
- 前回までのlogical listings欠落: 0
- listing -> sold: 80
- listing -> deleted: 171
- listing継続: 2,464
- 修正後に直接観測したsold累計: 778

## 修正確認

- shareCode回転の残存: 0
- 状態矛盾duplicate: 0
- missing ID / invalid status / invalid time: 0
- 最新時刻のdeleted: 3
- 最大deleted spike: 47
- 旧時刻不明sold: 2,067（価格学習に保持、時間教師から隔離）
- 信頼済みsold時刻の最大同時件数: 38

## 学習準備

- 信頼できる観測期間: 7.17日（7日条件達成）
- Model13相当クレンジング後sold: 9,353
- Model16用データ量: 十分
- Model15 BERT/fold cache: 現在の9,353行と不一致
- unique descriptions: 5,680
- semantic coverage: 535 / 5,680（9.42%）
- 推定差分LLM抽出: 5,145 descriptions
- demand semantic missing rows: 4,753

## 結論

データ構造、ID回転、削除判定、売却時刻の新規記録はいずれも正常。7日ホライズンの最低観測期間も満たした。学習前に、未処理説明文の差分意味抽出、差分BERT抽出、現在母集団でのfold再作成が必要。旧モデル成果物をそのまま流用した学習は禁止する。

この監査では学習・LLM抽出を実行していない。

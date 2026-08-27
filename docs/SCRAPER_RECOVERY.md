# 収集再開時のデータ切替

## 既存masterをそのまま継続しない

現在の`data/*_master.csv`には2026-07-20以降の一括deleted障害が残っています。
修正版scraperを同じmasterへ実行すると、現在activeの行はlistingへ復元できますが、
欠測期間中に消えた行の本当の終了時刻は復元できません。フォルダ名だけ更新して
学習データにすることは禁止します。

GitHubへ反映するときは、旧`data/`を履歴用の別名で保存してから、新しい`data/`には
`targets.json`だけを置き、修正版scraperでmasterを新規生成します。Git履歴にも旧データが
残るため、削除ではなくデータ世代の切替として扱えます。

推奨例:

```text
data_legacy_corrupted_20260728/  旧master・snapshot（学習禁止）
data/                            修正版で新規収集する現行世代
  targets.json
  snapshot/
  market_snapshot/
```

`.github/workflows/scrape.yml`は`data/`だけをcommitするため、現行世代を引き続き`data/`に
する構成ならワークフローの保存先変更は不要です。旧データを別名へ移す最初のcommitだけは
`git add data/ data_legacy_corrupted_20260728/`の両方を対象にします。

## モデル再開条件

- 新snapshotにlistingが存在する
- 観測最終日とsnapshot名が一致する
- 単一時刻への大量deleted集中がない
- 需要・代替モデルの最長7日ラベルに対し、最低7日以上の連続観測がある
- 正式な時系列評価では、複数foldを作れるよう3〜4週間以上のクリーン観測を推奨

`decision_support_models/pipeline_runner/run_all.py --dry-run`が成功するまでは学習を
開始しません。LLM意味特徴キャッシュは結果ラベルを使わないため再利用できます。

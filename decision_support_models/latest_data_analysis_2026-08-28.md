# data_8_28 監査・モデル影響レポート

## 結論

`data_8_28`は価格モデルの後日評価と今後の継続観測の開始点には利用できるが、
そのまま需要・代替出品・買い時モデルの再学習には利用してはいけない。
shareCode更新を削除と誤認した行と、初回取得時に過去soldへ仮のsold_atを付けた行が
含まれるためである。

## データ監査

|項目|結果|
|---|---:|
|masterファイル|13|
|raw行数|7,430|
|raw ticket_idユニーク数|7,036|
|論理出品数（event_id + created_at_unix）|5,090|
|複数shareCodeを持つ論理出品|1,946|
|raw listing / sold / deleted|3,238 / 2,060 / 2,132|
|論理統合後 listing / sold / deleted|2,974 / 1,940 / 176|
|最新時刻のraw deleted|993|
|最新時刻の論理統合後deleted|2|

最新時刻のdeleted 993件のうち991件は同じ論理IDのlistingが同時点に存在した。
877件は価格も同じ、114件は価格が変わっていた。需要減少ではなく、編集時の
shareCode更新と価格変更を別ticketとして扱ったことが原因である。

sold_atは2026-08-27 07:17:28に1,572件、05:10:56に361件が集中し、論理統合後でも
この2時刻に1,815件が残る。初回API取得以前に既にsoldだった取引へ取得時刻を付けた
もので、売却時刻の教師ラベルとしては利用できない。

## Model16価格モデル

最新snapshotのclean sold 1,682件に対する凍結済みModel16の後日評価は、
MAE 8,399.25円、Median AE 6,124.92円、MAPE 31.59%、RMSE 11,408.25円、
R² 0.6568だった。既存OOF ticket_idとの重複は0件である。

従来OOFのMAE約5,889円より悪く、分布変化が確認された。しかし最新日だけで
再学習すると7,313件から1,682件へ縮小するため不適切である。

全13 historical snapshotを論理IDで統合するとclean sold 8,500件、24イベント、
10グループになる。従来学習より1,187件（16.2%）多いため、次回Model16再学習は
この履歴統合母集団を使用する。

## LLM意味特徴

最新データの4,477種類の説明文に対し、既存cacheで利用可能なのは1,040件
（23.23%）、未抽出は3,437件である。現在説明文に対応するparse errorは4件
（0.09%）で品質閾値内だが、coverageは未完了である。

## 実装した修正と再学習方針

1. `event_id + created_at_unix`を安定IDとしてshareCode・価格変更を追跡する。
2. 初回取得済みsoldは`sold_at_source=historical_unknown`としてsold_atを作らない。
3. listingからsoldへの実観測だけ`transition_observed`を付与する。
4. snapshot監査と全decision loaderを論理ID統合へ変更する。
5. 同時刻では直接sold、直接listing、推定deletedの順に信頼する。
6. 需要・代替モデルは最低7日のclean observationがなければ学習を拒否する。
7. Model16の次回学習はlatest-onlyでなくhistorical sold unionを使う。
8. semantic preflightのparse-error率を現在説明文だけで計算する。

現在のModel16 artifactは暫定価格推論に継続利用できる。需要・代替・買い時は再学習せず、
修正後スクレイパーで最低7日、安定評価には14〜28日収集する。その後、未抽出3,437説明文を
差分抽出し、需要→代替→買い時の順に再学習する。旧需要・代替artifactは旧pipeline version
なので現在の意思決定には使用しない。rawの`data_8_28`は監査証跡として変更せず保持する。

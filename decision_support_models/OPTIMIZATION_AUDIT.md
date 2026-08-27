# 意思決定モデル 高速化・精度監査

## 精度を変えない高速化

- 日次ランドマークをticket×日付のPython辞書ループからNumPy配列展開へ変更。
- 市場状態をevent単位の配列計算へ変更し、全ticket表の反復filterを廃止。
- 代替出品ラベルはticket単位で比較可能候補を一度だけ絞り、3期間で再利用。
- 同一horizonの時系列splitをtabular/semanticで共有。
- 入力CSV、手動マスタ、意味特徴、Model 16価格、関連コードのSHA-256が一致する場合だけ
  前処理済み学習表を再利用。
- 各OOF foldは入力、split index、特徴列、LightGBM設定、学習コードの指紋が一致する場合だけ
  未校正確率を再利用。確率校正は再開時も時系列順に再計算する。
- LightGBMは従来どおり全論理CPUを使用。GPU化はヒストグラム計算と結果を変えるため既定にしない。
- 60秒ごとのheartbeatを追加し、長いfit中も停止と誤認しないようにした。

実データ先頭200ticketで旧実装と全市場特徴の値が一致することを確認した。

| モデル | 行数 | 旧前処理 | 新前処理 | 倍率 | 値 |
|---|---:|---:|---:|---:|---|
| 需要 | 5,643 | 23.896秒 | 0.682秒 | 35.06倍 | 一致 |
| 代替出品 | 5,661 | 98.234秒 | 0.973秒 | 100.92倍 | 一致 |

この倍率は前処理だけの小規模実測であり、全学習時間には27回ずつのLightGBM fitが残るため、
全体が同じ倍率で短縮されることを意味しない。

## 精度・評価監査で修正した事項

1. 価格0の1,961件
   - demand: validな状態ラベルは保持し、価格だけ欠損へ変更して各foldの学習データだけで中央値補完。
   - alternative: 節約額を定義できず0円候補も無効なため、価格比較母集団から監査除外。
   - 0円の大半がdeletedだったため、0円をそのまま特徴にすると状態を直接示す疑似リークになっていた。
2. semantic採否の評価リーク
   - 最終productionモデルの採否は全OOF比較を維持。
   - 買い時モデルへ渡すOOFは、各foldより前のOOFだけでtabular/semanticを選択するよう変更。
3. demand foldの欠損クラス
   - 固定3列とLightGBM内部クラスの対応ずれを修正。fold内の実クラス数で学習し、
     元のactive/sold/deleted列へ明示的に再配置する。
4. sklearn警告
   - 前処理済み行列をLightGBM Boosterへ直接渡し、feature-name警告を発生させず同じ確率を取得。
5. 公演後の代替ランドマーク
   - 全ラベルが未観測`-1`になる公演後行は生成しない。学習対象行は変えず計算量だけ削減。
6. 途中破損
   - model、OOF、report、cache、policyは可能な箇所を一時ファイルからのatomic replaceへ統一。
   - キャッシュとfold checkpointは指紋・shape・有限値を検証する。
7. 初期foldがactiveだけになる時系列分割
   - 固定20%の初期学習期間にsold/deletedが存在せず、全件実行時に1日需要fold 0が停止した。
   - 最初の境界だけを、各クラスがLightGBMの`min_child_samples`以上揃う最初の時点まで
     前進させる。40%以降の境界、4fold、重複group分離、horizon purgeは維持する。
   - 全1/3/7日horizonの行数、クラス数、時系列purge、重複group分離を、最初のfitより前に
     全件学習表で監査する。後半horizonで数時間後に停止することを防ぐ。
8. 競合リスク確率の浮動小数点境界誤差
   - 単調性投影後の`1 - p_sold - p_deleted`が、丸め誤差で`-2.22e-16`になる場合があり、
     全fit完了後の`log_loss`集計が停止していた。
   - 0～1へのclipと行単位の再正規化を投影時・評価時の両方に追加した。モデルのfit、順位、
     実用的な確率値は変更しない。
   - 評価・保存だけの修正でOOFを再学習しないよう、checkpoint互換性をfit policy単位に分離した。

## 維持した学習条件

- 1・3・7日horizon
- 4時系列fold
- tabular/semanticの両方を同一foldで学習するアブレーション
- demand 700 trees、alternative 650 trees
- 全特徴、LLM JSON意味特徴、Model 16適正価格
- LightGBMハイパーパラメータ、最終全行fit、時系列purge、確率校正

## 現時点で残るデータ上の制約

- deletedは確認済みsoldではなく、他媒体販売・取消・掲載終了を含む。
- 価格変更履歴がないため、同一出品の値下げ行動は学習できない。
- `data_8_6`の最終状態にはlistingがなく、学習・OOF評価は可能だが現在出品への推論には
  listingを含む新しいスナップショットが必要。
- Model 16は価格目的ではOOFを使うが、需要モデルの各時系列foldの内側で再学習した価格モデルではない。
  したがって「価格モデルまで含む完全nested temporal評価」と主張する場合は、別途非常に重い検証が必要。

## 検証済みコマンド

```powershell
python -m unittest discover -s decision_support_models\demand_state_model\tests -v
python -m unittest discover -s decision_support_models\alternative_arrival_model\tests -v
python -m unittest discover -s decision_support_models\buy_timing_model\tests -v
python decision_support_models\pipeline_runner\run_all.py --from-stage demand --dry-run
```

本学習はCodex側では実行していない。

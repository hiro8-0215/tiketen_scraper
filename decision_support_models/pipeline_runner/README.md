# 意思決定モデル一括ランナー

Model 16を価格推定の大本として、次の工程を一回の実行で順番に進めます。

1. 全ticketのQwen意味特徴抽出（既存キャッシュから再開）
2. 完成済みModel 16から全ticket価格キャッシュを生成
3. 需要状態モデルの学習・評価
4. 安価な代替出品モデルの学習・評価
5. 2モデルのOOFを買い時モデルへコピー
6. 買い時方針の学習・評価

VS Codeの「実行とデバッグ」から
`[意思決定] 全工程一括実行（Model16必須・再開対応）` を選択します。

LLM意味特徴とModel 16価格キャッシュが完成済みで、需要モデルから再開する場合は
`[意思決定] 高速化版: 需要以降を一括実行（既存LLM/Model16再利用）` を選択します。

途中で失敗・停止した場合は、同じ項目をもう一度実行してください。同じデータと
コードで正常終了済みの工程はスキップされます。入力データまたはコードが変わると、
自動的に新しい実行として扱います。

需要・代替モデルは前処理表と各OOF foldにも個別の検証済みcheckpointを持つため、
工程の途中で停止した場合も、入力・コード・split・特徴・設定が一致する計算だけを再利用します。

## 安全条件

既定では完成済みModel 16を要求します。

- `hybrid_AI_model16/artifacts/model16.joblib`

`demand_state_model/artifacts/fair_price_all_tickets.csv` がない場合は、意味特徴抽出後に
自動生成します。Model 16の学習対象だったsold行にはOOF予測、それ以外には最終ensemble
の予測を使います。部分的な価格キャッシュはsold状態を漏らすため拒否します。

`--from-stage demand` のように価格生成工程を飛ばし、かつ価格キャッシュがない状態で
市場中央値フォールバックを明示的に検証する場合だけ次を使用できます。

```powershell
python decision_support_models\pipeline_runner\run_all.py --allow-price-fallback
```

これはModel 16を大本にした本番学習とは見なしません。

一括ランナーは、モデル確認より先に入力スナップショットを監査します。次のいずれかを
検出した場合は、抽出・学習を開始せず停止します。

- フォルダの日付より観測最終日が3日以上古い
- `listing`が0件
- 任意の単一観測時刻に大量の`deleted`が集中している（過去の汚染も検出）
- ticket ID、日時、statusに構造上の異常がある

過去データの診断だけを意図する場合は`--allow-historical-snapshot`で鮮度・listing・
一括deleted検査のみ明示的に解除できます。未知statusなどの構造エラーは解除できません。
このオプションで作った需要・代替・買い時の出力は、現在の判断に使用できません。

## 補助オプション

```powershell
# 重い処理を行わず実行順・Model 16接続だけ確認
python decision_support_models\pipeline_runner\run_all.py --dry-run

# 問題のある過去snapshotを診断目的で確認（本番学習・現在推論には使用不可）
python decision_support_models\pipeline_runner\run_all.py --dry-run --allow-historical-snapshot

# すべて再実行
python decision_support_models\pipeline_runner\run_all.py --force

# 指定工程以降を実行
python decision_support_models\pipeline_runner\run_all.py --from-stage demand

# Qwenの初期バッチサイズを標準8から変更
python decision_support_models\pipeline_runner\run_all.py --batch-size 2
```

ログは `pipeline_runner/logs/`、再開状態は `pipeline_runner/artifacts/` に保存されます。

# Model 16 全ticket価格ブリッジ

完成済みModel 16を、需要・買い時モデルが読み込める全ticket価格キャッシュへ変換します。

- Model 16のクレンジング済み学習行: 保存済みOOF予測を使用
- それ以外のsold/deleted/listing行: Model 16最終ensembleで推論
- 意味特徴: 全ticket用の価格非依存Qwen JSONを使用
- `semantic_source`はstatusと無関係になるよう統一
- Model 16の目的変数を使った再学習は行わない

出力は `demand_state_model/artifacts/fair_price_all_tickets.csv` です。一括ランナーでは
意味特徴抽出の直後、需要モデル学習の直前に自動実行されます。

キャッシュにはOOF全体のMAEと絶対誤差80%点も保存します。需要モデル側は現在、
互換性のため `ticket_id,fair_price` を使用します。

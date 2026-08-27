# Model16 実行前監査

監査日: 2026-08-10

## 修正済み

- 低カードinalityの数値コード（曜日、月、開演時刻、座席・ランダム・同行・当選種別）を、数値だけでなくカテゴリとしても扱うようにした。
- 曜日・月・開演時刻へsin/cos周期特徴を追加した。
- 現在のsnapshotで全件同値の列を、目的価格を参照せず除外するようにした。
- 2種類のLightGBMが同一inner foldの補完・One-Hot結果を共有し、前処理の二重実行をなくした。
- dataset fingerprintへ特徴名と順序を追加し、特徴定義変更後に古いOptuna studyやfold checkpointを誤再利用しないようにした。
- Windows上で無効なNPZ checkpointを開いたまま上書きしないよう、読込後に必ずcloseし、shape・有限値・固定重みとの一致も検証するようにした。
- ほぼ0のexpert重みを除去し、無意味な微小BERT重みで推論時の埋め込みが必須にならないようにした。
- target、予測、blend、前処理行列のNaN・infinityと不正な重みをfail-fastで検出するようにした。

## 検証結果

- Model13相当soldクレンジング: 63,190 → 7,313件
- モデル入力: 数値71列 + カテゴリ25列 = 96列
- LightGBM One-Hot後: 616列、疎行列527,719 non-zero
- CatBoost: 7,313行 × 96列のPool構築成功
- BERT: 7,313行 × 768次元、ticket ID・説明文hash・fold行順が一致
- duplicate descriptionの外側fold跨ぎなし
- 禁止特徴、重複特徴名、数値infinityなし
- Python compile成功、単体テスト11件成功、preflight成功

監査ではLightGBM、CatBoost、Ridgeの学習は実行していない。

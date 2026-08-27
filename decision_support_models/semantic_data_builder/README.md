# 全チケット用LLM意味データ生成

最新snapshotに存在する全説明文を、価格・status・sold_atを一切渡さず、Qwen 2.5
7BでModel15実績schemaの意味特徴へ変換します。出力はモデルコードではなく、
共有入力データとして `C:\Users\hero\Documents\tiketen_scraper\semantic_feature_data`
へ保存します。

既存Model15キャッシュは、Qwen15・schema一致・説明文SHA-256一致の場合だけ再利用します。
legacy・parse error・未登録説明文は再抽出対象です。100件ごとにatomic保存し、再実行時は
未処理hashから続行します。CUDA device 0へ固定し、CPU/disk offloadを拒否します。

Qwenには長いJSONを自由生成させず、同じ9項目を短い固定順序の整数配列として回答
させます。出力は決定的に列名付きschemaへ戻します。初回の形式解析に失敗した行だけ
強化プロンプトで一度再試行し、なお失敗した応答は `parse_failures.jsonl` に保存します。
既存の成功行は保持し、`parse_error`行だけを再実行します。

標準バッチサイズは8です。VRAM不足時は自動的に半減して同じbatchを再試行します。

```powershell
cd C:\Users\hero\Documents\tiketen_scraper\decision_support_models\semantic_data_builder
python preflight.py
python extract_semantic_json.py
python preflight.py
```

旧方式の実行を停止後に同じコマンドを実行する場合、`--reset`は付けないでください。
既存成功キャッシュを再利用し、失敗・未処理行だけを新方式で処理します。

## parse_errorだけを修復

通常抽出が終了した後、Antigravityの実行メニューから
`[意思決定] Step 1R: LLM parse_errorのみ修復（成功済み保持）` を実行します。
成功済み特徴は変更せず、保存済みの初回・再試行応答から安全に復旧できる
形式エラーだけを修復し、下流モデルと同じ1%品質ゲートを確認します。

通常抽出または全工程ランナーの実行中は、ファイル競合を避けるため起動を拒否します。
修復後は全工程一括実行を再度選択してください。`--reset`は不要です。

Codexは抽出を実行しません。`preflight.py` もモデルをロードせず、fit・生成を行いません。

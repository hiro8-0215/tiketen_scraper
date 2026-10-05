# Google Drive自動バックアップ設定

GitHub Actionsから認証付きGoogle Apps Scriptへmaster CSVを送り、指定したGoogle Drive
フォルダの`data_月_日/`へ保存します。同じ日・同じファイル名は置き換えます。

## 1. Apps Scriptを作成

1. <https://script.google.com/>で「新しいプロジェクト」を作成します。
2. `Code.gs`の内容を貼り付けて保存します。
3. 左側の「プロジェクトの設定」→「スクリプト プロパティ」に次を追加します。
   - `PARENT_FOLDER_ID`: 保存先DriveフォルダURLの`folders/`以降
   - `UPLOAD_TOKEN`: 推測できないランダム文字列
4. 右上「デプロイ」→「新しいデプロイ」→種類「ウェブアプリ」を選択します。
5. 「次のユーザーとして実行」は自分、「アクセスできるユーザー」は全員にします。
   URL自体は公開されますが、`UPLOAD_TOKEN`が一致しない要求は拒否されます。
6. 初回認可後、表示された`https://script.google.com/macros/s/.../exec`を控えます。

トークンはローカルで次のように生成できます。

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## 2. GitHub Secretsを登録

GitHubリポジトリの`Settings` → `Secrets and variables` → `Actions` →
`New repository secret`で次の2つを登録します。

- `GDRIVE_WEBAPP_URL`: Apps Scriptの`/exec` URL
- `GDRIVE_UPLOAD_TOKEN`: Apps Scriptの`UPLOAD_TOKEN`と同じ値

フォルダIDやトークン、URLをPython・YAMLへ直接書かないでください。

## 3. 動作確認

GitHubの`Actions` → `Daily Archive to Google Drive` → `Run workflow`を実行します。
成功後、Drive保存先に当日名のフォルダと`*_master.csv`があることを確認します。
以降は`Ticket Scraper`ワークフローが成功すると自動実行されます。

HTTP 404などの一時エラーは10/20/40/60秒間隔で再試行し、なお失敗したファイルは
他のファイルを保存してから60秒後・120秒後に再試行します。全体の時間予算は15分です。
全ファイルの保存が完了した場合のみ成功を報告します。原因と運用上の限界は
[`一時障害と復旧対応`](../docs/actions_transient_recovery.md)を参照してください。

### 変更履歴JSONLも保存する場合

`ticket_changes_YYYYMMDD_001.jsonl`などの分割ファイルを受け付ける更新済み`Code.gs`を
貼り付け、「デプロイを管理」から新しいバージョンへ再デプロイしてください。
同じデプロイの更新ならURLは維持できます。新しいデプロイを作った場合は
`GDRIVE_WEBAPP_URL`も更新します。

再デプロイ後にGitHub Actions変数`GDRIVE_INCLUDE_JSONL=true`を設定すると、観測ログ・
変更履歴の全part・匿名売却在庫をmaster CSVと一緒に送ります。旧名の変更ログも
アップロード前に分割します。変更履歴は1ファイル30 MiB以下で、アップローダーの
35 MiB上限を下回ります。すでにJSONL保存が有効な場合は次回実行前に再デプロイが必要です。
既定ではmasterのみを保存します。

データ形式の互換性と移行手順は[`Actions復旧とログ分割`](../docs/actions_log_rotation.md)を
参照してください。過去に保存した旧名ログと分割ログを同時に集計すると重複するため、
読み込み対象は移行後のファイル集合にそろえてください。

## 4. 旧Apps Scriptの無効化

旧コードは認証なしの公開URLだったため、Apps Scriptの「デプロイを管理」から旧デプロイを
アーカイブし、新しい認証付きデプロイだけを有効にしてください。

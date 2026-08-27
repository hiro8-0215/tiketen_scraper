# Ticketen Scraping Project

チケテン（Ticketen.jp）から指定したアーティストのチケット情報を自動収集し、販売状況・売り切れ状況・相場データをCSV化するスクレイピングプロジェクトです。

## リポジトリ方針

このリポジトリには収集・モデル構築のコード、設定、テストを保存します。スクレイパーの
継続状態である`data/`はGitHub Actionsが更新し、同時にGoogle Driveへ日付別バックアップ
します。学習用snapshot、学習済みモデル、OOF、LLMキャッシュ、認証JSONはGitHubへ
置かず、Google Driveまたはローカルで管理します。Google Drive自動保存の設定は
`google_drive_apps_script/README.md`を参照してください。

`decision_support_models/`にはModel 16を価格基準として使用する需要・代替出品・買い時
モデルがあり、`hybrid_AI_model13`〜`hybrid_AI_model16`には価格モデルの世代別コードが
あります。学習前にはsnapshot品質ゲートが実行されます。

## 特徴
1. **ハイブリッド収集による高速化**: 
   - 隠しAPI（`tickets/all`）を利用し、出品中チケットだけでなく**過去の売り切れ・非表示チケットも高速に一括取得**します。
   - ブラウザ操作（Playwright）は「完全新規の出品中チケット」の詳細情報（出品者名、備考欄の全文など）を取得する初回のみ稼働するため、日々の定期実行（差分更新）は数分で完了します。
2. **GitHub Actions対応**:
   - 高速化により、GitHub Actionsの無料枠（月2000分）の範囲内で1日8回以上の定期実行が可能です。
3. **耐障害性**:
   - ページ読み込み時の無限ロード対策として5秒の自動タイムアウトを実装。
   - ブラウザ操作のたびにデータを逐次保存するため、途中で強制終了してもデータを失いません。
4. **相場スナップショット**:
   - 公演日時ごとに「総チケット数」「出品数」「売り切れ数」「平均・最低・最高価格」を自動集計し保存します。

## ディレクトリ構成
```text
ticketen_scraping/
├── scraper.py            # メインのスクレイピングプログラム
├── measure_run.py        # 実行時間を計測するためのラッパースクリプト
├── requirements.txt      # 必要なPythonパッケージ
├── README.md             # 本ドキュメント
└── data/
    ├── targets.json      # 取得対象のアーティストスラッグ（ID）リスト
    ├── ticket_master.csv # チケット全件が蓄積されるマスターデータベース
    ├── snapshot/         # 月別の全件バックアップCSV
    └── market_snapshot/  # 公演ごとの相場・統計データCSV
```

## セットアップ手順
1. パッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```
2. Playwright（ブラウザ自動化ツール）のブラウザをインストール
   ```bash
   playwright install chromium
   ```

## 使い方

### 対象アーティストの変更
`data/targets.json` に取得したいアーティストのスラッグ（URLの一部）を配列で指定します。
```json
[
    "snow-man",
    "sixtones",
    "naniwa-danshi"
]
```
※スラッグは実際のチケテンのURL `https://ticketen.jp/performers/〇〇` の「〇〇」の部分です。

Ticketen側のURL識別子が履歴CSVで使うグループ名と異なる場合は、保存名を
`name`、現在の取得元識別子を`source`として分離できます。これによりURL変更後も
master CSV名と学習時のグループIDは変わりません。

```json
[
  {"name": "b-and-zai", "source": "bandzai"}
]
```

### プログラムの実行
以下のコマンドでスクレイピングを開始します。
```bash
python scraper.py
```
実行時間の計測を行いたい場合は以下を実行してください。
```bash
python measure_run.py
```

## GitHub Actionsでの運用例

現在のワークフローは二段階で実行します。最初の`SCRAPE_MODE=api`で対象アーティスト
全件のAPI状態を先に取得して全masterを保存し、その後の`SCRAPE_MODE=details`で
未取得の出品者・備考詳細を25分の範囲で差分補完します。これにより初回実行でも、
詳細取得の時間切れによって後半アーティストのmasterが作られないことを防ぎます。
プロジェクトのルートに `.github/workflows/scrape.yml` を作成することで、指定したスケジュールで自動実行できます。

```yaml
name: Scheduled Scraping

on:
  schedule:
    # 毎日3時間おき（1日8回）実行する例 (UTC指定のため注意)
    - cron: '0 0,3,6,9,12,15,18,21 * * *'
  workflow_dispatch: # 手動実行用

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      
      - name: Set up Python
        uses: actions/setup-python@v7
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          playwright install chromium
          
      - name: Run Scraper
        run: python scraper.py
        
      - name: Commit and Push changes
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add data/
          # 変更がある場合のみコミット
          git diff --quiet && git diff --staged --quiet || git commit -m "Update ticket data"
          git push
```

## 各ファイルの解説
- **`scraper.py`**
  - **`fetch_all_tickets`**: API経由で非表示（売り切れ）を含む全チケットの基本情報を取得。
  - **`parse_ticket_details`**: Playwrightでブラウザを開き、APIだけでは取得できない詳細データ（出品者評価や備考欄など）を取得。
  - **`save_snapshots`**: Pandasを使用して、月別の生データと公演ごとの相場サマリーを作成します。

- **`ticket_master.csv` (主なカラム)**
  - `ticket_id`: チケットの一意なID
  - `status`: `listing`（出品中）、`sold`（売り切れ）、`deleted`（削除・取り下げ）
  - `price`: チケット価格
  - `quantity`: 枚数
  - `raw_description`: 出品者の備考テキスト

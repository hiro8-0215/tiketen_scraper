# 買い時判断モデル

需要状態モデルと安価な代替出品モデルのOOF予測を、CSVデータとして統合します。
他モデルのPythonコードや学習済みオブジェクトはimportしません。

LLMはここへ直接入力しません。LLMが需要・代替出品の未使用時系列評価を改善した場合、
改善後の確率だけがOOF CSV経由でこの方針層へ渡るため、意味特徴の二重利用を防ぎます。

判断には次を使います。

- 現在価格の妥当価格からの割安率
- soldとdeletedを分けて学習した消失確率
- より安い比較可能出品が期間内に現れる確率

`safety`、`balanced`、`savings` は価格帯の分割ではなく、買い逃しと節約の
どちらを重視するかという利用者側の損失設定です。価格帯別モデルにはしません。
時系列前半で方針を選び、後半30%を最終検証として残します。

```powershell
cd C:\Users\hero\Documents\tiketen_scraper\decision_support_models\buy_timing_model
python prepare_inputs.py
python preflight.py
python run_training.py
python inference.py --demand <需要予測CSV> --alternative <代替出現予測CSV>
```

これは価格変更の因果効果を推定するモデルではありません。現状データでは
同一ticket_idの値下げ履歴がないため、判断結果は「観測可能な出品流入と消失に
基づく方針」として評価します。

# ============================================================
# [Model 13] ドメイン知識パーサー (description_parser.py)
# ============================================================
# ジャニーズチケット転売の raw_description から、価格に直結する
# ドメイン固有の情報を正規表現で構造化して抽出するモジュール。
#
# BERTが捉えられない「番手＝+3万円」「ランダム＝−1万円」のような
# 価格ルールを、人間のドメイン知識としてコードに落とし込む。
#
# [Model 13 追加]
# 「安い理由」特徴量を追加:
#   - 制作開放 (is_seisaku_kaihou)
#   - 見切れ/注釈付き統合 (is_mikire_chuushaku)
#   - バラ売り (is_bara)
#   - 急ぎ/投げ売り (is_urgent_sale)
#   - 定価以下キーワード (is_teikaware_keyword)
# ============================================================

import re
import numpy as np
import pandas as pd


# ============================================================
# 1. 番手（bante） — 最重要特徴量
# ============================================================
# 出品者が複数名義を持っている場合、何番目に良い席を渡すかの順番。
# 1番手（＝最良席確約）が最も高額。番手が大きいほど残り物リスク。

def extract_bante(text: str) -> tuple:
    """
    番手（何番手か）と総名義数を抽出する。

    Returns:
        (bante: int or NaN, total_meigi: int or NaN)

    表記揺れの例:
        「2名義中2番手」「3名義中1番手」「2番目」
        「単独」「重複無し」「1名義のみ」「QR毎」
    """
    if not isinstance(text, str) or not text:
        return (np.nan, np.nan)

    bante = np.nan
    total_meigi = np.nan

    # --- パターン1: 「N名義中M番手」の明示的な表記 ---
    m = re.search(r'(\d+)\s*名義\s*中\s*(\d+)\s*番(?:手|目)', text)
    if m:
        total_meigi = int(m.group(1))
        bante = int(m.group(2))
        return (bante, total_meigi)

    # --- パターン2: 単純な「M番手」「M番目」 ---
    m = re.search(r'(\d+)\s*番(?:手|目)', text)
    if m:
        bante = int(m.group(1))

    # --- パターン3: 「単独名義」「1名義のみ」「重複なし（名義が1つ）」→ 1番手相当 ---
    if re.search(r'単独(?:名義)?|1\s*名義\s*(?:のみ|だけ)|重複\s*(?:無し|なし|ナシ)|重複\s*(?:無|な)し', text):
        if np.isnan(bante):
            bante = 1
        if np.isnan(total_meigi):
            total_meigi = 1

    # --- パターン4: 「QR毎」「QRごと」→ 1番手相当（名義ごとにQRが独立） ---
    if re.search(r'QR\s*(?:毎|ごと|各)', text):
        if np.isnan(bante):
            bante = 1

    # --- 総名義数の追加抽出（「N名義所持」「N名義持ち」） ---
    if np.isnan(total_meigi):
        m2 = re.search(r'(\d+)\s*名義\s*(?:所持|持ち|保有|ある)', text)
        if m2:
            total_meigi = int(m2.group(1))

    # --- 「複数名義所持」→ 総名義数は不明だが2以上と推定 ---
    if np.isnan(total_meigi) and re.search(r'複数\s*名義', text):
        total_meigi = -1  # 「複数だが正確な数は不明」を示す特殊値

    return (bante, total_meigi)


# ============================================================
# 2. ランダム配布（random_type）
# ============================================================
# 座席を出品者が選ばず、ランダムに配布するタイプ。
# 少人数ランダム（当たりやすい）と大人数ランダム（ギャンブル）がある。

def extract_random_type(text: str) -> int:
    """
    ランダム配布の種別を抽出する。

    Returns:
        0: ランダムでない
        1: 少人数ランダム（当たりの確率が高め）
        2: 通常ランダム（標準的）
    """
    if not isinstance(text, str) or not text:
        return 0

    # 少人数ランダム（価値が高い）
    if re.search(r'少人数\s*(?:の\s*)?ランダム', text):
        return 1

    # 通常ランダム
    if re.search(r'ランダム\s*(?:配布|で\s*お渡し|でお配り|にて)', text):
        return 2

    # 「ランダム」単体でも（「ランダムエラー」「ランエラ」は除外）
    if re.search(r'ランダム', text) and not re.search(r'ランダム\s*エラー|ランエラ', text):
        return 2

    return 0


# ============================================================
# 3. すり替え（surikae）
# ============================================================
# 良い席を出品者が抜いて、悪い席を渡すリスクへの保証。

def extract_surikae(text: str) -> int:
    """
    すり替えなし保証の有無を抽出する。

    Returns:
        1: すり替えなし（保証あり → 高価格傾向）
        0: 保証なし or 記載なし
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(r'(?:重複\s*)?すり替え\s*(?:無し|なし|ナシ|無)', text):
        return 1

    return 0


# ============================================================
# 4. 同行タイプ（doukou_type）
# ============================================================
# 入場方法による安全性の差。QR毎が最も安全（独立入場可能）。

def extract_doukou_type(text: str) -> int:
    """
    同行・入場方法の種別を抽出する。

    Returns:
        3: QR毎・ログイン情報譲渡（最安全、自分のスマホで独立入場）
        2: 同行者登録可能（事前登録、比較的安全）
        1: 同行/同時入場（標準、出品者と一緒）
        0: 記載なし
    """
    if not isinstance(text, str) or not text:
        return 0

    # QR毎、名義ごと、ログイン情報譲渡（最安全）
    if re.search(r'QR\s*(?:毎|ごと|各)|名義\s*(?:ごと|毎)|ログイン情報|アカウント(?:ごと|毎|譲渡)', text):
        return 3

    # 同行者登録可能
    if re.search(r'同行者\s*登録\s*(?:可能|可|済)', text):
        return 2

    # 同行 / 同時入場（標準）
    if re.search(r'同行(?!者\s*登録)|同時\s*入場', text):
        return 1

    return 0

# ============================================================
# 4.5. ゲート情報 (gate_info) - Model 11追加
# ============================================================
def extract_gate_info(text: str) -> str:
    """
    ゲート情報を抽出する（Lゲート、入場口C、11ゲート など）。
    アリーナ濃厚などのプレミアム要素を判定するためのカテゴリ変数。
    """
    if not isinstance(text, str) or not text:
        return "UNKNOWN"

    # Lゲート、Jゲートなどアルファベットゲート
    m1 = re.search(r'([A-Za-z])\s*ゲート', text)
    if m1:
        return m1.group(1).upper() + "ゲート"
    
    # 入場口C、入場口Nなど
    m2 = re.search(r'入場口\s*([A-Za-z0-9]+)', text)
    if m2:
        return "入場口" + m2.group(1).upper()
        
    # 11ゲートなど数字ゲート
    m3 = re.search(r'(\d+)\s*ゲート', text)
    if m3:
        return m3.group(1) + "ゲート"
        
    return "UNKNOWN"


# ============================================================
# 5. 当選種別（tousen_type）
# ============================================================
# チケットの取得経路。初期当選が最も信頼性が高い。

def extract_tousen_type(text: str) -> int:
    """
    当選種別を抽出する。

    Returns:
        3: 初期当選（FC先行の本命）
        2: 復活当選（キャンセル分の再抽選）
        1: 制作開放/解放（関係者枠放出、見切れ席リスク）
        0: 記載なし or 不明
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(r'初期\s*当選|初期\s*選', text):
        return 3

    if re.search(r'復活\s*当選', text):
        return 2

    if re.search(r'制作\s*(?:開放|解放)', text):
        return 1

    return 0


# ============================================================
# 6. 座席情報（構造化）
# ============================================================

def extract_seat_type(text: str) -> int:
    """
    座席種別を抽出する。

    Returns:
        2: アリーナ
        1: スタンド
        0: 不明
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(r'アリーナ', text):
        return 2
    if re.search(r'スタンド', text):
        return 1

    return 0


def extract_row_number(text: str) -> float:
    """
    列番号を抽出する。

    Returns:
        列番号（int）。不明はNaN。
        「最前列」は 1 として返す。
    """
    if not isinstance(text, str) or not text:
        return np.nan

    # 「最前列」「最前」→ 1列目
    if re.search(r'最前(?:列)?', text):
        return 1.0

    # 「N列目」「N列」
    m = re.search(r'(\d{1,2})\s*列(?:目)?', text)
    if m:
        row = int(m.group(1))
        if 1 <= row <= 80:  # 妥当な範囲のみ
            return float(row)

    return np.nan


def extract_block_rank(text: str) -> float:
    """
    ブロックのランク（アルファベット順）を抽出する。
    Aブロック=1, Bブロック=2, Cブロック=3 ...

    Returns:
        ブロックランク（int）。不明はNaN。
    """
    if not isinstance(text, str) or not text:
        return np.nan

    m = re.search(r'([A-Za-z])\s*(?:ブロック|ブロ)', text)
    if m:
        letter = m.group(1).upper()
        return float(ord(letter) - ord('A') + 1)

    return np.nan


def extract_is_front_row(text: str) -> int:
    """最前列・前方フラグ"""
    if not isinstance(text, str) or not text:
        return 0
    if re.search(r'最前(?:列)?|前方', text):
        return 1
    return 0


# ============================================================
# 7. 名義の信頼性
# ============================================================

def extract_meigi_trust(text: str) -> dict:
    """
    名義に関する信頼性指標を抽出する。

    Returns:
        dict: {
            "is_valid_term": int,      # 有効期限内 (0/1)
            "has_shimosanketa": int,    # 下3桁提示可 (0/1)
            "is_honnin_taiou": int,    # 本人確認対応 (0/1)
            "henbo_ari": int,          # 変更ボタンあり (0/1)
            "is_gaitou_meigi": int,    # 該当名義 (0/1)
        }
    """
    if not isinstance(text, str) or not text:
        return {
            "is_valid_term": 0,
            "has_shimosanketa": 0,
            "is_honnin_taiou": 0,
            "henbo_ari": 0,
            "is_gaitou_meigi": 0,
        }

    result = {}

    # 有効期限内
    result["is_valid_term"] = int(bool(re.search(r'有効\s*期限\s*内|期限\s*内', text)))

    # 下3桁提示可
    result["has_shimosanketa"] = int(bool(re.search(r'下\s*[3三]\s*桁\s*(?:提示|開示)', text)))

    # 本人確認対応
    result["is_honnin_taiou"] = int(bool(re.search(r'本人\s*確認\s*(?:対応|可)|本確\s*(?:対応|可)', text)))

    # 変更ボタンあり（同行者変更が可能）
    result["henbo_ari"] = int(bool(re.search(r'変更\s*ボタン\s*(?:あり|有)|変ボ\s*(?:有|あり)', text)))

    # 該当名義
    result["is_gaitou_meigi"] = int(bool(re.search(r'該当\s*名義', text)))

    return result


# ============================================================
# 8. その他の価格影響因子
# ============================================================

def extract_misc_features(text: str) -> dict:
    """
    その他の価格に影響する条件を抽出する。

    Returns:
        dict: {
            "is_chakuburo_nashi": int,  # 着ブロ指定なし (0/1)
            "is_ren_error_nashi": int,  # ランダムエラー経験なし (0/1)
            "has_refund_policy": int,   # 公演中止返金あり (0/1)
            "desc_length": int,         # 説明文の文字数
            "num_conditions": int,      # 条件項目の多さ（箇条書き数）
            "is_s_seat": int,           # S席確約 (0/1)
            "ticket_count_offered": float,  # 出品枚数（2連中1枚 等）
        }
    """
    if not isinstance(text, str) or not text:
        return {
            "is_chakuburo_nashi": 0,
            "is_ren_error_nashi": 0,
            "has_refund_policy": 0,
            "desc_length": 0,
            "num_conditions": 0,
            "is_s_seat": 0,
            "ticket_count_offered": np.nan,
        }

    result = {}

    # 着ブロ指定なし（ブロック指定をしていない → 公平な抽選）
    result["is_chakuburo_nashi"] = int(bool(
        re.search(r'着\s*ブロ\s*(?:指定\s*(?:なし|無し|なし)|希望\s*(?:なし|無し))', text)
    ))

    # ランダムエラー経験なし（入場エラーの実績なし → 安全）
    result["is_ren_error_nashi"] = int(bool(
        re.search(r'(?:ランダム\s*)?エラー\s*(?:経験\s*)?(?:なし|無し|無)|ランエラ\s*(?:なし|無)', text)
    ))

    # 返金ポリシー
    result["has_refund_policy"] = int(bool(
        re.search(r'公演\s*中止\s*(?:の\s*場合\s*)?返金|中止\s*以外\s*(?:返金\s*)?不可', text)
    ))

    # 説明文の文字数
    result["desc_length"] = len(text)

    # 条件項目の多さ（箇条書きマーカーの数）
    markers = re.findall(r'[・★※☆▪►●■]|^[\-\*]', text, re.MULTILINE)
    result["num_conditions"] = len(markers)

    # S席
    result["is_s_seat"] = int(bool(re.search(r'[Ss]\s*席', text)))

    # 出品枚数（「2連中1枚」「4連中2枚」など）
    m = re.search(r'(\d+)\s*連\s*(?:中|の\s*うち)\s*(\d+)\s*枚', text)
    if m:
        result["ticket_count_offered"] = float(m.group(2))
    else:
        result["ticket_count_offered"] = np.nan

    return result


# ============================================================
# 9. [Model 13 追加] 制作開放/解放枠 (is_seisaku_kaihou)
# ============================================================
# 制作開放席は見切れリスクが高く、定価以下で取引される主要因。
# 検証結果: 204件, 平均価格 18,361円 (非該当 34,280円), -46.4%

def extract_seisaku_kaihou(text: str) -> int:
    """
    制作開放/解放枠のフラグを抽出する。

    Returns:
        1: 制作開放/解放枠
        0: 該当なし
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(r'制作\s*(?:開放|解放|枠)', text):
        return 1

    return 0


# ============================================================
# 10. [Model 13 追加] 見切れ/注釈付き統合 (is_mikire_chuushaku)
# ============================================================
# 見切れ席（5件, -75.2%）と注釈付き席（7件, -67.8%）を統合。
# 制作開放と重複する場合もあるが、独立した情報として有用。

def extract_mikire_chuushaku(text: str) -> int:
    """
    見切れ席・注釈付き席のフラグを抽出する。

    Returns:
        1: 見切れ or 注釈付き席
        0: 該当なし
    """
    if not isinstance(text, str) or not text:
        return 0

    # 見切れ席
    if re.search(r'見切れ|見えにくい|見えづらい|視界不良|視界が悪|柱\s*(?:あり|有|が)', text):
        return 1

    # 注釈付き席
    if re.search(r'注釈\s*付[きけ]|注釈\s*席|※\s*座席\s*の\s*特性|ステージ\s*(?:の\s*)?一部\s*(?:が\s*)?見え', text):
        return 1

    return 0


# ============================================================
# 11. [Model 13 追加] バラ売り (is_bara)
# ============================================================
# 連番のうち1枚だけの出品。1人参戦用→需要低→低価格化。
# 検証結果: 91件, 平均価格 26,934円 (非該当 33,810円), -20.3%

def extract_bara(text: str) -> int:
    """
    連番バラ売りのフラグを抽出する。

    Returns:
        1: バラ売り
        0: 該当なし
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(
        r'バラ(?:売り|\s*で)|'
        r'(?:1|１)\s*枚\s*(?:のみ|だけ|単独)|'
        r'(?:1|１)\s*枚\s*(?:で\s*)?(?:の\s*)?出品|'
        r'(?:連番\s*)?(?:の\s*)?(?:うち|中)\s*(?:1|１)\s*枚',
        text
    ):
        return 1

    return 0


# ============================================================
# 12. [Model 13 追加] 急ぎ/投げ売り (is_urgent_sale)
# ============================================================
# 急遽行けなくなった等の事情による安値出品の傾向。
# 検証結果: 1,316件, 平均価格 30,474円 (非該当 34,574円), -11.9%

def extract_urgent_sale(text: str) -> int:
    """
    急ぎ・投げ売り傾向のフラグを抽出する。

    Returns:
        1: 急ぎ/投げ売り傾向あり
        0: 該当なし
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(
        r'急[いぎ]|至急|'
        r'(?:どなた|誰)\s*か|'
        r'(?:お\s*)?(?:譲り|引[きけ]取[りっ])\s*(?:先|手)\s*(?:を\s*)?(?:探|さが)|'
        r'行[けか]な[いく]なっ|'
        r'(?:仕事|急用|体調)\s*(?:の\s*)?(?:都合|ため|により)',
        text
    ):
        return 1

    return 0


# ============================================================
# 13. [Model 13 追加] 定価以下キーワード (is_teikaware_keyword)
# ============================================================
# 「定価以下」「半額」等のキーワード。
# 検証結果: 52件, 平均価格 12,098円 (非該当 33,905円), -64.3%

def extract_teikaware_keyword(text: str) -> int:
    """
    定価以下・安価キーワードのフラグを抽出する。

    Returns:
        1: 定価以下キーワードあり
        0: 該当なし
    """
    if not isinstance(text, str) or not text:
        return 0

    if re.search(
        r'定価\s*(?:以下|割れ|未満|より\s*安)|'
        r'定価\s*(?:の\s*)?\d+\s*割|'
        r'半額|お安く',
        text
    ):
        return 1

    return 0


# ============================================================
# メインの一括抽出関数
# ============================================================

def parse_description(df: pd.DataFrame, text_col: str = "raw_description") -> pd.DataFrame:
    """
    DataFrameの raw_description カラムから、ジャニーズ転売の
    ドメイン知識に基づく構造化特徴量を一括抽出する。

    Args:
        df: raw_description を含む DataFrame
        text_col: テキストカラム名

    Returns:
        df: 構造化特徴量が追加された DataFrame
    """
    df = df.copy()
    texts = df[text_col].fillna("")

    print(f"\n  [ドメイン知識パーサー] {len(texts)} 件の説明文を構造化中...")

    # --- 1. 番手 ---
    bante_results = texts.apply(extract_bante)
    df["bante"]       = bante_results.apply(lambda x: x[0])
    df["total_meigi"]  = bante_results.apply(lambda x: x[1])

    # --- 2. ランダム ---
    df["random_type"] = texts.apply(extract_random_type)

    # --- 3. すり替え ---
    df["surikae_nashi"] = texts.apply(extract_surikae)

    # --- 4. 同行タイプ ---
    df["doukou_type"] = texts.apply(extract_doukou_type)

    # --- 5. 当選種別 ---
    df["tousen_type"] = texts.apply(extract_tousen_type)

    # --- 6. 座席情報 ---
    df["seat_type"]     = texts.apply(extract_seat_type)
    df["row_number"]    = texts.apply(extract_row_number)
    df["block_rank"]    = texts.apply(extract_block_rank)
    df["is_front_row"]  = texts.apply(extract_is_front_row)
    df["gate_info"]     = texts.apply(extract_gate_info)

    # --- 7. 名義の信頼性 ---
    trust_results = texts.apply(extract_meigi_trust)
    trust_df = pd.DataFrame(trust_results.tolist(), index=df.index)
    df = pd.concat([df, trust_df], axis=1)

    # --- 8. その他 ---
    misc_results = texts.apply(extract_misc_features)
    misc_df = pd.DataFrame(misc_results.tolist(), index=df.index)
    df = pd.concat([df, misc_df], axis=1)

    # --- 9. [Model 13] 制作開放 ---
    df["is_seisaku_kaihou"] = texts.apply(extract_seisaku_kaihou)

    # --- 10. [Model 13] 見切れ/注釈付き統合 ---
    df["is_mikire_chuushaku"] = texts.apply(extract_mikire_chuushaku)

    # --- 11. [Model 13] バラ売り ---
    df["is_bara"] = texts.apply(extract_bara)

    # --- 12. [Model 13] 急ぎ/投げ売り ---
    df["is_urgent_sale"] = texts.apply(extract_urgent_sale)

    # --- 13. [Model 13] 定価以下キーワード ---
    df["is_teikaware_keyword"] = texts.apply(extract_teikaware_keyword)

    # --- [Model 13] 安い理由の複合スコア ---
    df["negative_keyword_count"] = (
        df["is_seisaku_kaihou"] +
        df["is_mikire_chuushaku"] +
        df["is_bara"] +
        df["is_urgent_sale"] +
        df["is_teikaware_keyword"]
    )

    # --- 抽出結果のサマリー ---
    n = len(df)
    print(f"    番手あり:       {df['bante'].notna().sum():>5} 件 ({df['bante'].notna().sum()/n*100:>5.1f}%)")
    print(f"    ランダム:       {(df['random_type'] > 0).sum():>5} 件 ({(df['random_type'] > 0).sum()/n*100:>5.1f}%)")
    print(f"    すり替えなし:   {df['surikae_nashi'].sum():>5} 件 ({df['surikae_nashi'].sum()/n*100:>5.1f}%)")
    print(f"    同行タイプあり: {(df['doukou_type'] > 0).sum():>5} 件 ({(df['doukou_type'] > 0).sum()/n*100:>5.1f}%)")
    print(f"    当選種別あり:   {(df['tousen_type'] > 0).sum():>5} 件 ({(df['tousen_type'] > 0).sum()/n*100:>5.1f}%)")
    print(f"    座席情報あり:   {(df['seat_type'] > 0).sum():>5} 件 ({(df['seat_type'] > 0).sum()/n*100:>5.1f}%)")
    print(f"    ゲート情報あり: {(df['gate_info'] != 'UNKNOWN').sum():>5} 件 ({(df['gate_info'] != 'UNKNOWN').sum()/n*100:>5.1f}%)")
    print(f"    列番号あり:     {df['row_number'].notna().sum():>5} 件 ({df['row_number'].notna().sum()/n*100:>5.1f}%)")
    print(f"    有効期限内:     {df['is_valid_term'].sum():>5} 件 ({df['is_valid_term'].sum()/n*100:>5.1f}%)")
    print(f"    下3桁提示可:    {df['has_shimosanketa'].sum():>5} 件 ({df['has_shimosanketa'].sum()/n*100:>5.1f}%)")
    print(f"    --- [Model 13 追加] 安い理由の特徴量 ---")
    print(f"    制作開放:       {df['is_seisaku_kaihou'].sum():>5} 件 ({df['is_seisaku_kaihou'].sum()/n*100:>5.1f}%)")
    print(f"    見切れ/注釈:    {df['is_mikire_chuushaku'].sum():>5} 件 ({df['is_mikire_chuushaku'].sum()/n*100:>5.1f}%)")
    print(f"    バラ売り:       {df['is_bara'].sum():>5} 件 ({df['is_bara'].sum()/n*100:>5.1f}%)")
    print(f"    急ぎ/投げ売り:  {df['is_urgent_sale'].sum():>5} 件 ({df['is_urgent_sale'].sum()/n*100:>5.1f}%)")
    print(f"    定価以下KW:     {df['is_teikaware_keyword'].sum():>5} 件 ({df['is_teikaware_keyword'].sum()/n*100:>5.1f}%)")
    print(f"  [ドメイン知識パーサー] 完了 — 構造化特徴量を追加しました（Model 13: 安い理由 5特徴量追加）")

    return df

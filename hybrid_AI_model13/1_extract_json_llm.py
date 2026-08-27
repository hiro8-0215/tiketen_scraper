# ============================================================
# [Model 12] Qwenによる構造化データ(JSON)抽出
# ============================================================
import os
import sys
import json
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 日本語出力のためのエンコーディング設定
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import (
    LLM_MODEL_ID, OUTPUT_DIR, DATA_DIR,
    TARGET_COLUMN
)
from data_loader import load_raw_data, clean_data

def get_unique_descriptions(df):
    """
    重複を排除した一意の説明文リストを作成する。
    処理時間を短縮するため、全く同じ説明文は1回だけLLMに推論させる。
    """
    unique_descs = df["raw_description"].dropna().unique()
    return unique_descs

def build_prompt(description):
    system_prompt = """あなたは優秀なチケット査定士です。
ユーザーから提供されたチケットの説明文を読み取り、以下のJSONフォーマットで厳密に回答してください。
JSON以外のテキスト（解説や挨拶など）は一切含めないでください。

【出力フォーマット】
{
  "seat_level": "アリーナ" または "スタンド" または "不明",
  "row_position": "前方" または "中列" または "後方" または "不明",
  "is_fc_early": true または false,
  "is_random": true または false,
  "price_estimate": 0
}

【判定基準のヒント】
- seat_level: 「アリーナ」「アリーナA」「アリーナB」などはアリーナ。「スタンド」「1階席」「2階席」「バルコニー」などはスタンド。記載なしは不明。
- row_position: 「最前列」「1列〜5列」「一桁列」「アリーナ最前ブロック」などは前方。「10列前後」「中段」などは中列。「後方」「最後列」「天井席」などは後方。
- is_fc_early: 「FC先行」「最速」「初期当選」「名義」などのワードがあればtrue。
- is_random: 「ランダム」「同行」「番手」「入場後座席選択」「すり替え」「重複」などはtrue。
- price_estimate: あなたが予想する適正転売価格（整数・円単位）。不明な場合は0。
"""
    user_prompt = f"以下の説明文を分析してJSONを出力してください。\n\n【説明文】\n{description}\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages


def extract_features():
    # 引数パーサー
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="既存の抽出結果を削除して最初からやり直す")
    parser.add_argument("--retry-errors", action="store_true", help="抽出失敗(raw_response_error)のエントリのみ再処理する")
    args = parser.parse_args()

    print(f"[*] モデル12: 構造化データ抽出を開始します")
    
    # データ読み込みとクレンジング
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw, keep_all_status=True)
    
    unique_descs = get_unique_descriptions(df_clean)
    print(f"[*] 一意な説明文の数: {len(unique_descs)} 件")
    
    # 抽出済み結果を保存するJSONファイルのパス
    output_path = os.path.join(OUTPUT_DIR, "llm_extracted_features.json")
    
    # --reset: 既存ファイルを削除
    if args.reset and os.path.exists(output_path):
        os.remove(output_path)
        print("[*] --reset: 既存の抽出結果を削除しました。")
    
    # 既存の結果を読み込み（レジューム機能）
    extracted_dict = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                extracted_dict = json.load(f)
            print(f"[*] 既存の抽出結果を {len(extracted_dict)} 件読み込みました。")
        except:
            print("[!] 既存ファイルの読み込みに失敗しました。新規作成します。")
    
    # --retry-errors: エラーエントリのみ再処理
    if args.retry_errors:
        error_keys = [k for k, v in extracted_dict.items() if "raw_response_error" in v]
        for k in error_keys:
            del extracted_dict[k]
        print(f"[*] --retry-errors: {len(error_keys)} 件のエラーエントリを再処理対象にしました。")
    
    # まだ抽出していない説明文だけをフィルタ
    remaining_descs = [d for d in unique_descs if str(d) not in extracted_dict]
    print(f"[*] LLMで新規に処理する件数: {len(remaining_descs)} 件")
    
    if len(remaining_descs) == 0:
        print("[*] 全ての説明文が処理済みです！")
        return
    
    # モデルのロード (Transformers + BitsAndBytes)
    print(f"[*] Qwenモデル({LLM_MODEL_ID})をロードしています...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # 1件ずつ逐次推論（バッチ推論はパディングで出力が壊れるため廃止）
    success_count = 0
    error_count = 0
    
    try:
        for i, desc in enumerate(tqdm(remaining_descs, desc="LLM Extraction")):
            # プロンプト作成とトークナイズ（1件ずつ、パディング不要）
            prompt = tokenizer.apply_chat_template(
                build_prompt(desc), tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to("cuda")
            
            # 推論（1件ずつ）
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # デコード（入力プロンプト部分を切り落とす）
            input_len = inputs["input_ids"].shape[1]
            output_ids = outputs[0][input_len:]
            response_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # JSONパース
            try:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response_text
                    
                parsed = json.loads(json_str)
                extracted_dict[str(desc)] = parsed
                success_count += 1
            except Exception as e:
                extracted_dict[str(desc)] = {
                    "seat_level": "不明",
                    "row_position": "不明",
                    "is_fc_early": False,
                    "is_random": False,
                    "price_estimate": 0,
                    "raw_response_error": response_text[:200]
                }
                error_count += 1
                    
            # 10件ごとにこまめにセーブ
            if (i + 1) % 10 == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(extracted_dict, f, ensure_ascii=False, indent=2)
                    
    except KeyboardInterrupt:
        print("[!] 中断されました。そこまでの結果を保存します。")
        
    finally:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_dict, f, ensure_ascii=False, indent=2)
        print(f"[*] 抽出完了。全 {len(extracted_dict)} 件の結果を保存しました: {output_path}")
        print(f"    成功: {success_count} 件, エラー: {error_count} 件")

if __name__ == "__main__":
    extract_features()

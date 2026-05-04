#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_subsidies.py — 批次匯入各縣市早療補助 PDF 到 MySQL

流程：
  1. 用 pdfplumber 抽取每份 PDF 全文
  2. 用 LLM (gpt-4o-mini) 結構化提取欄位
  3. 寫入 SubsidyProgram 表

用法：
  python scripts/import_subsidies.py --pdf-dir "C:/Users/88696/Desktop/edu_sys/早療補助"

前置需求：
  pip install pdfplumber openai
  環境變數 OPENAI_API_KEY
"""

import os
import sys
import json
import glob
import argparse
import pdfplumber

# 載入 .env
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path)

# Flask app context
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, db, SubsidyProgram


EXTRACT_PROMPT = """你是一位資料整理助理。以下是某縣市的「早期療育補助計畫」官方文件全文。
請從中提取以下欄位，以 JSON 格式回傳（所有值為字串）：

{
  "city": "縣市名稱（如：臺北市）",
  "eligibility": "補助對象與資格條件（完整列出）",
  "subsidy_items": "補助項目說明（交通費、療育訓練費等，列出涵蓋哪些治療類型）",
  "transport_fee": "交通補助金額與規則（如：每次250元，同日同處限一次）",
  "training_cap": "一般戶每月補助上限（如：每月最高4000元）",
  "low_income_cap": "低收入戶每月補助上限（如：每月最高6000元）",
  "excluded_items": "不補助的項目（如：掛號費、門診、評估等）",
  "required_docs": "申請應備文件（條列）",
  "apply_deadline": "申請期限說明",
  "apply_where": "申請方式與受理窗口",
  "notes": "其他重要注意事項（如：不得重複領取、審核時間等）"
}

請只回傳 JSON，不要加 markdown 格式或其他文字。如果某欄位文件中未提及，填「未明確載明」。

===== 文件全文 =====
"""


def extract_pdf_text(pdf_path: str) -> str:
    """用 pdfplumber 抽取 PDF 全文"""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def llm_extract_fields(full_text: str, city_hint: str) -> dict:
    """呼叫 LLM 做結構化欄位提取"""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  [ERROR] 未設定 OPENAI_API_KEY，無法進行 LLM 提取")
        return {}

    client = OpenAI(api_key=api_key)

    prompt = EXTRACT_PROMPT + full_text[:8000]  # 限制長度避免超出 context

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是資料整理助理，只回傳 JSON。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # 清理可能的 markdown 包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)
        # 確保 city 欄位正確（以檔名為準）
        data["city"] = city_hint
        return data

    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON 解析失敗: {e}")
        print(f"  [RAW] {raw[:300]}")
        return {"city": city_hint}
    except Exception as e:
        print(f"  [ERROR] LLM 呼叫失敗: {e}")
        return {"city": city_hint}


def import_one(pdf_path: str, dry_run: bool = False) -> bool:
    """匯入單一 PDF"""
    filename = os.path.basename(pdf_path)
    city = filename.replace(".pdf", "").replace(".PDF", "")
    print(f"\n{'='*50}")
    print(f"  處理：{filename} → {city}")

    # 1. 抽取全文
    full_text = extract_pdf_text(pdf_path)
    if not full_text or len(full_text) < 50:
        print(f"  [SKIP] PDF 文字過少（{len(full_text)} 字），可能是掃描檔")
        return False

    print(f"  抽取文字：{len(full_text)} 字")

    # 2. LLM 結構化提取
    fields = llm_extract_fields(full_text, city)
    if not fields:
        print(f"  [SKIP] LLM 提取失敗")
        return False

    print(f"  LLM 提取完成：{len(fields)} 個欄位")

    # LLM 有時回傳 list（如 required_docs），統一轉為字串
    for key, val in fields.items():
        if isinstance(val, list):
            fields[key] = "\n".join(str(v) for v in val)

    if dry_run:
        print(f"  [DRY RUN] 欄位預覽：")
        for k, v in fields.items():
            preview = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
            print(f"    {k}: {preview}")
        return True

    # 3. 寫入 MySQL（先刪除舊的同城市記錄）
    existing = SubsidyProgram.query.filter_by(city=city).first()
    if existing:
        db.session.delete(existing)
        print(f"  刪除舊記錄：{city}")

    record = SubsidyProgram(
        city=fields.get("city", city),
        eligibility=fields.get("eligibility"),
        subsidy_items=fields.get("subsidy_items"),
        transport_fee=fields.get("transport_fee"),
        training_cap=fields.get("training_cap"),
        low_income_cap=fields.get("low_income_cap"),
        excluded_items=fields.get("excluded_items"),
        required_docs=fields.get("required_docs"),
        apply_deadline=fields.get("apply_deadline"),
        apply_where=fields.get("apply_where"),
        notes=fields.get("notes"),
        full_text=full_text,
        source_file=filename,
    )
    db.session.add(record)
    db.session.commit()
    print(f"  [OK] 寫入成功：{city}")
    return True


def main():
    parser = argparse.ArgumentParser(description="匯入各縣市早療補助 PDF 到 MySQL")
    parser.add_argument(
        "--pdf-dir",
        default=r"C:\Users\88696\Desktop\edu_sys\早療補助",
        help="PDF 資料夾路徑",
    )
    parser.add_argument("--dry-run", action="store_true", help="只預覽提取結果，不寫入 DB")
    parser.add_argument("--city", type=str, default=None, help="只處理指定縣市（如：臺北市）")
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    if not os.path.isdir(pdf_dir):
        print(f"[ERROR] 找不到資料夾：{pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.PDF")))

    if args.city:
        pdf_files = [f for f in pdf_files if args.city in os.path.basename(f)]

    print(f"找到 {len(pdf_files)} 份 PDF")

    with app.app_context():
        # 確保表存在
        db.create_all()

        success = 0
        for pdf_path in pdf_files:
            if import_one(pdf_path, dry_run=args.dry_run):
                success += 1

        print(f"\n{'='*50}")
        print(f"完成：{success}/{len(pdf_files)} 份匯入成功")
        if args.dry_run:
            print("（dry-run 模式，未實際寫入資料庫）")


if __name__ == "__main__":
    main()

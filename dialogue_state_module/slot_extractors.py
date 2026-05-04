# slot_extractors.py
"""
Slot 值提取器 — 從使用者輸入中提取特定槽位值

已有元件直接複用：
  - region     → utils/region_extractor.py (extract_region)
  - domain_focus → DomainRouter top_domain
  - child_age   → DB Child.birth_date
  - report_range → 預設最近兩份

本模組負責尚未有提取器的槽位：
  - ability_focus  → 比對臨床能力關鍵字
  - school_type    → 關鍵字比對
  - time_range     → 時間表達式提取
"""
from __future__ import annotations

import re
from typing import Optional


# ============================================================================
# ability_focus：從輸入中提取能力/領域關鍵字
# ============================================================================

# 常見的能力名稱（對應報告中的 subdomain / 訓練方向）
_ABILITY_KEYWORDS = {
    "口語表達": ["口語", "說話", "表達", "講話", "語言表達"],
    "語言理解": ["語言理解", "聽懂", "理解指令", "聽指令"],
    "構音": ["構音", "發音", "咬字"],
    "精細動作": ["精細", "手部", "握筆", "剪刀", "穿珠", "扣鈕扣"],
    "粗大動作": ["粗大", "跑跳", "平衡", "走路", "爬樓梯", "體能"],
    "認知": ["認知", "思考", "概念", "邏輯", "分類", "配對"],
    "社交": ["社交", "人際", "互動", "同儕", "交朋友", "社會"],
    "情緒": ["情緒", "自我調節", "挫折", "哭鬧", "發脾氣"],
    "生活自理": ["自理", "穿衣", "吃飯", "如廁", "刷牙", "盥洗"],
    "注意力": ["注意力", "專注", "分心", "坐不住"],
    "感覺統合": ["感統", "感覺統合", "觸覺", "前庭", "本體覺"],
}


def extract_ability_focus(text: str) -> Optional[str]:
    """從使用者輸入中提取能力焦點"""
    text_lower = text.strip()
    for ability, keywords in _ABILITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return ability
    return None


# ============================================================================
# school_type：學校類型
# ============================================================================

_SCHOOL_KEYWORDS = {
    "公幼": ["公幼", "公立幼兒園", "公立幼稚園"],
    "私幼": ["私幼", "私立幼兒園", "私立幼稚園"],
    "準公共幼兒園": ["準公共", "準公幼"],
    "非營利幼兒園": ["非營利"],
    "國小": ["國小", "小學", "國民小學"],
    "國小特教班": ["特教班", "特教"],
    "學前特教": ["學前特教", "早療班"],
    "在家自學": ["在家", "自學"],
    "托嬰中心": ["托嬰", "托育"],
}


def extract_school_type(text: str) -> Optional[str]:
    """從使用者輸入中提取學校類型"""
    text_lower = text.strip()
    for school, keywords in _SCHOOL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return school
    return None


# ============================================================================
# time_range：時間表達式
# ============================================================================

_TIME_PATTERNS = [
    (r"去年", "去年"),
    (r"前年", "前年"),
    (r"上次", "上次"),
    (r"(\d+)\s*個月前", None),   # 動態：「3個月前」
    (r"(\d+)\s*年前", None),     # 動態：「1年前」
    (r"(\d{3,4})\s*年\s*(\d{1,2})\s*月", None),  # 「2025年3月」或「114年3月」
    (r"(\d{1,2})\s*月", None),   # 「3月」
    (r"最近", "最近"),
    (r"上一次", "上一次"),
]


def extract_time_range(text: str) -> Optional[str]:
    """從使用者輸入中提取時間範圍表達式"""
    text = text.strip()

    # 先嘗試具體模式
    m = re.search(r"(\d+)\s*個月前", text)
    if m:
        return f"{m.group(1)}個月前"

    m = re.search(r"(\d+)\s*年前", text)
    if m:
        return f"{m.group(1)}年前"

    m = re.search(r"(\d{3,4})\s*年\s*(\d{1,2})\s*月", text)
    if m:
        return f"{m.group(1)}年{m.group(2)}月"

    # 再嘗試固定關鍵字
    for pattern, label in _TIME_PATTERNS:
        if label and re.search(pattern, text):
            return label

    return None

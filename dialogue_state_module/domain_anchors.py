# 定義各領域及其對應的 anchor，用於 dialogue state tracking 中的 domain routing。

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# 預設設定檔路徑（dialogue_state_module/config/domain_anchors.json）
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_DOMAIN_ANCHORS_JSON_PATH = os.path.join(_CONFIG_DIR, "domain_anchors.json")

def load_domain_anchors(
    config_path: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    從 JSON 載入領域與錨點（domains, overview_anchors, domain_anchors）。
    """
    path = config_path or DEFAULT_DOMAIN_ANCHORS_JSON_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        domains = list(data.get("domains") or [])
        overview_anchors = list(data.get("overview_anchors") or [])
        raw = data.get("domain_anchors") or {}
        domain_anchors = {k: list(v) for k, v in raw.items() if isinstance(v, list)}
        
        if not domains or not domain_anchors:
            raise ValueError("Empty domains or anchors in JSON")
            
        return (domains, overview_anchors, domain_anchors)
    except Exception as e:
        print(f"[DomainAnchors] Error loading {path}: {e}")
        # 極簡備用方案，確保系統不崩潰
        fallback_domains = ["認知功能"]
        fallback_overview = ["整體狀況"]
        fallback_anchors = {"認知功能": ["認知發展"]}
        return (fallback_domains, fallback_overview, fallback_anchors)

# 模組初始化時自動載入
DOMAINS, OVERVIEW_ANCHORS, DOMAIN_ANCHORS = load_domain_anchors()

@dataclass(frozen=True) # frozen: 不可變
class DomainConfig:
    # 絕對門檻：機率 >= 這個才算 active
    active_prob_th: float = 0.30

    # 相對門檻：機率 >= top1 * active_ratio_th 才算 active（避免只剩 top1）
    active_ratio_th: float = 0.60

    # 最少保留幾個 active domain（保底）
    min_active_domains: int = 1

    # 最多保留幾個（避免太多造成檢索爆炸，通常 3~5）
    max_active_domains: int = 5

    # margin 太小表示模糊（可當作後續 follow-up / clarify 訊號）
    margin_low_th: float = 0.10

DEFAULT_DOMAIN_CONFIG = DomainConfig()

def validate_domain_anchors( #確認 domains 與 anchors 是否一致
    domains: List[str] = DOMAINS,
    anchors: Dict[str, List[str]] = DOMAIN_ANCHORS,
) -> Tuple[bool, List[str]]:
    """
    確保 domains 與 anchors 一致，避免後面 encode/score 時 KeyError。
    回傳 (ok, errors)。
    """
    errors: List[str] = []

    domain_set = set(domains or [])
    anchor_keys = set(anchors.keys() if anchors else [])

    missing_in_anchors = sorted(list(domain_set - anchor_keys))
    extra_in_anchors = sorted(list(anchor_keys - domain_set))

    if missing_in_anchors:
        errors.append(f"Missing anchors for domains: {missing_in_anchors}")
    if extra_in_anchors:
        errors.append(f"Anchors provided but domain not listed: {extra_in_anchors}")

    for d in domains:
        sentences = anchors.get(d, [])
        if not isinstance(sentences, list) or len(sentences) == 0:
            errors.append(f"Anchor sentences is empty/invalid for domain: {d}")
        else:
            for i, sentence in enumerate(sentences):
                if not isinstance(sentence, str) or not sentence.strip():
                    errors.append(f"Anchor sentence {i} is empty/invalid for domain: {d}")

    return (len(errors) == 0), errors
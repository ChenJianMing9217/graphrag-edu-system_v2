# dst_policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DSTPolicyConfig:
    # ---- 主決策：四象限（C vs Multi-topic continuity）----
    C_high_th: float = 0.55          # Context similarity 高/低門檻
    MT_high_th: float = 0.50         # Multi-topic continuity 高/低門檻（用 overlap/continue 聚合後的 MT）

    # ---- 輔助檢索（縮放 / 保險）----
    # 簡化版：只用 entropy 判斷模糊度
    entropy_high_th: float = 0.7   # Entropy 高門檻（用於判斷模糊度）

    # ---- flow 輸出（可選：若你需要 shift_soft）----
    flow_soft_when_one_high: bool = True  # C 或 MT 只有一個高時輸出 shift_soft，否則用 continue
    MT_soft_th: float = 0.30              # MT 介於 soft~high 時可視為「弱延續」
    C_soft_th: float = 0.45               # C 介於 soft~high 時可視為「弱延續」
    
    # ---- 模糊延續配置（簡化版）----
    enable_ambiguous_continuation: bool = True  # 是否啟用模糊延續
    ambiguous_continuation_entropy_th: float = 0.7  # 模糊判斷的熵值門檻（簡化版：只用 entropy）
    ambiguous_continuation_min_overlap: float = 0.5  # 調整後的最小overlap分數

    # ---- Memory Agent 雙軌制開關 ----
    enable_memory_agent: bool = True  # True=使用 RL 神經網路決策，False=回退到手寫 Threshold 規則


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _is_ambiguous(normalized_entropy: Optional[float], cfg: DSTPolicyConfig) -> bool:
    """
    簡化版：只用 entropy 判斷模糊度
    """
    if normalized_entropy is None:
        return False
    return normalized_entropy >= cfg.entropy_high_th


def compute_MT(topic_continue: bool, topic_overlap: float) -> float:
    """
    Multi-topic continuity scalar in [0,1].
    - overlap 本來就應該在 [0,1]，但這裡保險 clamp
    - topic_continue=True 作為加成 (+0.2)，不覆蓋 overlap 的連續強度
      → 強延續 (overlap=0.95) 和弱延續 (overlap=0.55) 在四象限中可區分
    """
    ov = _clamp01(float(topic_overlap))
    bonus = 0.2 if topic_continue else 0.0
    return _clamp01(ov + bonus)


def predicted_flow_from_C_MT(
    C: float,
    MT: float,
    cfg: DSTPolicyConfig,
) -> str:
    """
    Flow 只由 C + MT 決定（主張：延續不靠 D）。
    - continue：C 高 & MT 高（最強延續）
    - shift_soft：只有一個高（或弱延續），代表同主題池切換/承接不強
    - shift_hard：兩者都低
    """
    C = float(C)
    MT = float(MT)

    C_hi = C >= cfg.C_high_th
    MT_hi = MT >= cfg.MT_high_th

    if C_hi and MT_hi:
        return "continue"

    if not cfg.flow_soft_when_one_high:
        return "continue" if (C_hi or MT_hi) else "shift_hard"

    # soft region
    C_soft = C >= cfg.C_soft_th
    MT_soft = MT >= cfg.MT_soft_th

    if (C_hi or MT_hi) or (C_soft and MT_soft):
        return "shift_soft"

    return "shift_hard"


def decide_policy(
    *,
    C: float,
    normalized_entropy: Optional[float],
    topic_continue: bool,
    topic_overlap: float,
    is_multi_domain: bool,
    cfg: DSTPolicyConfig,
    query_len_norm: float = 0.5, # 新增：字數長度特徵
    task_label: Optional[str] = None,
    detected_region: Optional[str] = None,
) -> Tuple[str, bool, str, str]:
    """
    簡化版四象限決策：用 (C, MT) 決定「延續與否/大方向」
    輔助檢索決策：只用 ambig 決定「能不能縮小檢索」與「要不要雙路/反問」

    回傳：
      C_level, ambig, policy_case, action
    """
    C = float(C)
    MT = compute_MT(topic_continue, topic_overlap)

    ambig = _is_ambiguous(normalized_entropy, cfg)
    
    # [NEW] 極端領域跳轉保險 (Hard TV Override)
    # 如果領域跳轉非常大 (overlap < 0.2)，且字數又不長 (query_len_norm < 0.35)
    # 那這題絕對是模糊的（例如：那這部分呢？ 那要做啥？）
    if topic_overlap < 0.2 and query_len_norm < 0.35:
        ambig = True
    
    # [NEW] 文字相似度幻覺保險
    if topic_overlap < 0.1 and C >= cfg.C_high_th:
        ambig = True

    # ---- levels ----
    C_level = "high" if C >= cfg.C_high_th else "low"
    MT_level = "high" if MT >= cfg.MT_high_th else "low"

    # ---- main quadrant ----
    C_hi = (C_level == "high")
    MT_hi = (MT_level == "high")
    can_narrow = not ambig  # 簡化：只用模糊判斷

    policy_case = f"C{C_level[0].upper()}_MT{MT_level[0].upper()}"

    # ---- action selection (primary by C/MT; helper by ambig) ----
    # Q1: C high, MT high => strongest continuation
    if C_hi and MT_hi:
        if can_narrow:
            action = "NARROW_GRAPH"
            policy_case += "_NARROW"
        else:
            action = "CONTEXT_FIRST"
            policy_case += "_CTX"

    # Q2: C high, MT low => text looks like continuation, but topic pool doesn't
    elif C_hi and (not MT_hi):
        # 這種情況：不要硬縮小，先靠上下文延續
        action = "CONTEXT_FIRST"
        policy_case += "_CTX"

    # Q3: C low, MT high => topic pool continues, but semantic continuity weak (often domain alternation)
    elif (not C_hi) and MT_hi:
        # 若不模糊：可「在主題池內擴域」找（避免過度縮小）
        if can_narrow:
            action = "WIDE_IN_DOMAIN"
            policy_case += "_WIDE"
        else:
            action = "CONTEXT_FIRST"
            policy_case += "_CTX"

    # Q4: C low, MT low => likely hard shift / unclear
    else:
        # 若不模糊：可能是同領域但新問題（仍可 in-domain wide）
        if can_narrow:
            action = "WIDE_IN_DOMAIN"
            policy_case += "_WIDE"
        else:
            action = "DUAL_OR_CLARIFY"
            policy_case += "_DUAL"

    # ---- 特殊任務觸發 (Task H: 轉介與在地資源 / Task K: 補助與福利) ----
    if task_label in ("H", "K"):
        if not detected_region:
            action = "LOCAL_RESOURCE_CLARIFY"
            policy_case = f"TASK_{task_label}_CLARIFY"
        else:
            action = "LOCAL_RESOURCE_SEARCH"
            policy_case = f"TASK_{task_label}_SEARCH"

    # ---- modifiers for debugging / downstream logic ----
    if ambig:
        policy_case += "_AMBIG"
    if is_multi_domain:
        policy_case += "_MD"

    return C_level, ambig, policy_case, action


def action_to_predicted_flow(action: str) -> str:
    """
    Map retrieval action to semantic flow type.
    - NARROW_GRAPH / CONTEXT_FIRST => "continue"
    - WIDE_IN_DOMAIN => "shift_soft"
    - DUAL_OR_CLARIFY => "shift_hard"
    """
    action = str(action).strip().upper()
    
    if action in ("NARROW_GRAPH", "CONTEXT_FIRST"):
        return "continue"
    elif action == "WIDE_IN_DOMAIN":
        return "shift_soft"
    elif action == "LOCAL_RESOURCE_SEARCH":
        return "continue"
    elif action == "LOCAL_RESOURCE_CLARIFY":
        return "continue"
    else:  # DUAL_OR_CLARIFY or unknown
        return "shift_hard"
# 根據多領域主題分布，追蹤主題是否延續

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


# 控制 MultiTopicTracker 是否輸出詳細除錯訊息
MT_DEBUG_VERBOSE: bool = False


def l1_renormalize(d: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(v, 0.0) for v in d.values()))
    if s <= 0:
        return {}
    return {k: float(max(v, 0.0) / s) for k, v in d.items()}


def cosine_sim_dist(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Cosine similarity between two sparse distributions on the same domain set.
    Return in [0,1] if all entries are non-negative.
    """
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def total_variation_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    計算兩個概率分布的 Total Variation 距離
    
    TV(P, Q) = 0.5 * sum(|P(i) - Q(i)|) for all i
    
    Args:
        a: 第一個分布（已 L1-normalized）
        b: 第二個分布（已 L1-normalized）
    
    Returns:
        TV 距離 [0, 1]，0 表示完全相同，1 表示完全不同
    """
    all_keys = sorted(set(a.keys()) | set(b.keys()))
    if not all_keys:
        return 0.0
    
    tv_sum = 0.0
    for key in all_keys:
        p_val = a.get(key, 0.0)
        q_val = b.get(key, 0.0)
        tv_sum += abs(p_val - q_val)
    
    return 0.5 * tv_sum


@dataclass(frozen=True)
class MultiTopicConfig:
    """
    - decay_factor: 主題記憶更新時，舊記憶的保留比例（0.7 表示保留 70%）
    - overlap_th: 多少 overlap 以上算延續（你在 policy 用）
    - similarity_th: 分布 cosine 相似度門檻（弱 backup，用來救一些極端情況）
    - min_conf_for_similarity: 信心太低時不要用 similarity rule（避免亂救）
    """
    decay_factor: float = 0.7
    overlap_th: float = 0.5

    similarity_th: float = 0.97
    min_conf_for_similarity: float = 0.10

    # 強切換判斷門檻（僅用 TV 距離）
    hard_shift_tv_threshold: float = 0.6  # TV 距離超過此值視為強切換

    # TV 判斷延續強度用門檻
    tv_strong_th: float = 0.2  # TV <= tv_strong_th → 強延續
    tv_shift_th: float = 0.5   # TV >= tv_shift_th → 偏切換


@dataclass
class MultiTopicState:
    memory_dist: Dict[str, float] = field(default_factory=dict)
    prev_dist: Dict[str, float] = field(default_factory=dict)
    prev_raw_top_domain: Optional[str] = None
    prev_active_domains: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.memory_dist = {}
        self.prev_dist = {}
        self.prev_raw_top_domain = None
        self.prev_active_domains = []


class MultiTopicTracker:
    """
    追蹤「多領域主題池」是否延續。
    輸入應該是 domain_router 的 dist（完整 10 維機率分布）。
    """

    def __init__(self, cfg: MultiTopicConfig):
        self.cfg = cfg
        self.state = MultiTopicState()

    def _update_memory(self, cur_dist: Dict[str, float]) -> None:
        # EMA decay update on full dist keys
        mem = dict(self.state.memory_dist)
        for k, cur_v in cur_dist.items():
            mem_v = mem.get(k, 0.0)
            mem[k] = self.cfg.decay_factor * mem_v + (1.0 - self.cfg.decay_factor) * float(cur_v)
        self.state.memory_dist = l1_renormalize(mem)

    def compute_topic_overlap(
        self, 
        mem_dist: Dict[str, float],  # 記憶分布（歷史趨勢，不包含本次）
        cur_dist: Dict[str, float],
        prev_active_domains: Optional[List[str]] = None,  # 保留參數以保持接口兼容，但不使用
        cur_active_domains: Optional[List[str]] = None,   # 保留參數以保持接口兼容，但不使用
    ) -> float:
        """
        簡化版：只用 TV 距離計算主題重疊分數 (MT)
        
        返回 [0, 1] 的分數，1.0 表示完全相似，0.0 表示完全不同
        """
        mem = l1_renormalize(mem_dist)
        cur = l1_renormalize(cur_dist)
        
        if not mem or not cur:
            return 0.0

        # 計算 TV 距離並轉換為相似度
        tv_distance = total_variation_distance(mem, cur)
        dist_similarity = 1.0 - tv_distance  # TV 距離轉換為相似度
        
        return float(max(0.0, min(1.0, dist_similarity)))

    def check_topic_continuation(
        self,
        *,
        cur_dist: Dict[str, float],
        cur_raw_top_domain: str,
        confidence: float,
        cur_active_domains: Optional[List[str]] = None,
        prev_active_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        純觀察者模式：只用 TV 距離計算 MT 和判斷強切換。
        不接受 is_ambiguous，所有判斷僅基於分布距離。

        回傳：
        - topic_continue: bool
        - topic_overlap: float (0..1) - MT 分數（基於 TV 距離）
        - dist_similarity: float
        - reason: str
        - memory_top_domain / cur_top_domain
        """
        cur_dist = l1_renormalize(cur_dist)

        # first turn
        if not self.state.prev_dist:
            # 第一次出現主題：視為「強延續起點」，記憶直接等於本輪
            self.state.memory_dist = dict(cur_dist)
            self.state.prev_dist = dict(cur_dist)
            self.state.prev_raw_top_domain = cur_raw_top_domain
            if cur_active_domains:
                self.state.prev_active_domains = list(cur_active_domains)

            continuation_mode = "strong"
            reason = "first_turn"

            if MT_DEBUG_VERBOSE:
                print(
                    f"[MultiTopicTracker] tv_distance=0.000, "
                    f"active_domain_coverage=0.000, "
                    f"topic_continue=True, reason={reason}"
                )

            return {
                "topic_continue": True,
                "reason": reason,
                "topic_overlap": 0.0,
                "dist_similarity": 0.0,
                "active_domain_coverage": 0.0,
                "continuation_mode": continuation_mode,
                "prev_raw_top_domain": None,
                "cur_raw_top_domain": cur_raw_top_domain,
                "is_hard_shift": False,
                "tv_distance": 0.0,
            }

        # 使用未更新的 mem（歷史趨勢，不包含本次）
        mem_before_update = dict(self.state.memory_dist)
        prev = self.state.prev_dist

        # 保存更新前的 prev_raw_top_domain（真正的上一輪值）
        prev_raw_top_domain_before_update = self.state.prev_raw_top_domain

        # 使用 state 中的 prev_active_domains（如果沒有傳入）
        if prev_active_domains is None:
            prev_active_domains = self.state.prev_active_domains
        
        # 計算 TV 距離（用於判斷是否強切換和計算 MT）
        tv_distance = total_variation_distance(
            l1_renormalize(mem_before_update), 
            l1_renormalize(cur_dist)
        )

        # TV 距離判斷是否強切換
        is_hard_shift = tv_distance >= self.cfg.hard_shift_tv_threshold

        # 計算綜合 MT 分數（使用未更新的 mem）
        topic_overlap = self.compute_topic_overlap(
            mem_before_update,  # 使用未更新的 mem（歷史趨勢）
            cur_dist,
            prev_active_domains=prev_active_domains,
            cur_active_domains=cur_active_domains,
        )

        dist_similarity = cosine_sim_dist(prev, cur_dist)

        # 計算 active_domains 覆蓋度（Jaccard）——目前僅供觀察，不參與判斷
        if prev_active_domains and cur_active_domains:
            prev_set = set(prev_active_domains)
            cur_set = set(cur_active_domains)
            inter = prev_set & cur_set
            union = prev_set | cur_set
            active_domain_coverage = float(len(inter) / len(union)) if union else 0.0
        else:
            active_domain_coverage = 0.0

        # get tops for debug
        mem_top = max(mem_before_update.items(), key=lambda x: x[1])[0] if mem_before_update else None
        cur_top = max(cur_dist.items(), key=lambda x: x[1])[0] if cur_dist else None

        # Decision：根據 TV 把情況切成 強延續 / 軟延續 / 切換（純觀察者，不受外部 flag 影響）
        tv = float(tv_distance)
        continuation_mode = "soft"  # 預設視為軟延續 / 一般情況

        # TV 主導強/軟/切換判斷
        # Case 1：TV 低 → 強延續
        if tv <= self.cfg.tv_strong_th:
            topic_continue = True
            topic_overlap = max(topic_overlap, max(self.cfg.overlap_th, 0.7))
            reason = "tv_strong"
            continuation_mode = "strong"

        # Case 2：TV 高 → 偏切換
        elif tv >= self.cfg.tv_shift_th:
            topic_continue = False
            topic_overlap = min(topic_overlap, 0.2)
            reason = "tv_shift"
            continuation_mode = "shift"

        # Case 3：介於強與切換之間 → 軟延續 / 一般切換
        else:
            if mem_top is not None and cur_top is not None and cur_top == mem_top:
                topic_continue = True
                reason = "top_domain_match"
            elif topic_overlap >= self.cfg.overlap_th:
                topic_continue = True
                reason = "high_topic_overlap"
            elif dist_similarity >= self.cfg.similarity_th and confidence >= self.cfg.min_conf_for_similarity:
                topic_continue = True
                reason = "high_dist_similarity"
            else:
                topic_continue = False
                reason = "topic_shift"

        # 調試輸出：觀察 TV 與 active_domains 覆蓋度（僅在 verbose 模式下啟用）
        if MT_DEBUG_VERBOSE:
            print(
                f"[MultiTopicTracker] tv_distance={tv_distance:.3f}, "
                f"active_domain_coverage={active_domain_coverage:.3f}, "
                f"topic_continue={topic_continue}, reason={reason}"
            )

        # 更新記憶：強切換時重置，否則 EMA 更新（純觀察者，不受外部 flag 影響）
        if is_hard_shift:
            # 強切換：重置 mem，只記本輪
            self.state.memory_dist = dict(cur_dist)
            reason += " (mem_reset)"
        else:
            # 正常更新：EMA
            self._update_memory(cur_dist)

        # update prev
        self.state.prev_dist = dict(cur_dist)
        self.state.prev_raw_top_domain = cur_raw_top_domain
        if cur_active_domains:
            self.state.prev_active_domains = list(cur_active_domains)

        return {
            "topic_continue": topic_continue,
            "reason": reason,
            "topic_overlap": float(topic_overlap),  # 這是綜合 MT 分數
            "dist_similarity": float(dist_similarity),
            "active_domain_coverage": float(active_domain_coverage),
            "continuation_mode": continuation_mode,
            "memory_top_domain": mem_top,
            "cur_top_domain": cur_top,
            "prev_raw_top_domain": prev_raw_top_domain_before_update,  # 使用更新前的值（真正的上一輪）
            "cur_raw_top_domain": cur_raw_top_domain,
            "is_hard_shift": is_hard_shift,  # 新增：是否強切換
            "tv_distance": float(tv_distance),  # 新增：TV 距離
        }

    def reset(self) -> None:
        self.state.reset()

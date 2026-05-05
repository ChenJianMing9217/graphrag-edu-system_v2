# semantic_flow_module_v2.py
# 清晰、模塊化的語義流程追蹤系統

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

from .embedding import TextEncoder, score_overview_similarity
from .domain_router import DomainRouter, DomainResult
from .context_similarity import ContextSimilarity, ContextSimConfig
from .multi_topic_tracker import MultiTopicTracker, MultiTopicConfig
from .dst_policy import DSTPolicyConfig, decide_policy
from .task_scope_classifier import TaskScopeClassifier, PredictResult
from .utils.region_extractor import extract_region
from .feature_v2 import extract_memory_features_v2  # Shadow log: 8d 新特徵


# 控制 DST 模組是否輸出詳細除錯訊息
DST_DEBUG_VERBOSE: bool = True


# ============================================================================
# 結果數據結構
# ============================================================================

@dataclass
class DomainAnalysis: # 領域分析結果
    top_domain: str
    top_prob: float
    entropy: float
    distribution: Dict[str, float] = field(default_factory=dict)
    active_domains: List[str] = field(default_factory=list)
    active_domain_probs: Dict[str, float] = field(default_factory=dict)
    is_multi_domain: bool = False
    fused_distribution: Optional[Dict[str, float]] = None  # 模糊延續融合後的分布


@dataclass
class ContextAnalysis: # 上下文相似度分析
    similarity_score: float  # C
    source: str  # "first_turn" | "prev_user" | "prev_bot"
    is_first_turn: bool


@dataclass
class TopicAnalysis: # 主題延續分析
    is_continuing: bool
    overlap_score: float  # MT
    reason: str
    prev_top_domain: Optional[str] = None
    cur_top_domain: Optional[str] = None
    prev_dist: Optional[Dict[str, float]] = None  # 上一輪的領域分布（更新前）
    prev_active_domains: Optional[List[str]] = None  # 上一輪的活躍領域列表（更新前）
    tv_distance: Optional[float] = None  # TV 距離（Total Variation Distance）
    active_domain_coverage: Optional[float] = None  # active_domains Jaccard 覆蓋度
    continuation_mode: Optional[str] = None  # "strong" | "soft" | "shift"


@dataclass
class PolicyDecision: # 策略決策結果
    context_level: str  # "high" | "low"
    is_ambiguous: bool
    policy_case: str  # e.g., "CH_MTH_NARROW_MD"
    retrieval_action: str  # "NARROW_GRAPH" | "CONTEXT_FIRST" | etc
    semantic_flow: str  # "continue" | "shift_soft" | "shift_hard"
    memory_action: Optional[str] = None  # "STAY" | "REFRESH"（新設計移除 CLARIFY）
    # Clarify 屬性（新設計：從 action 變 attribute）
    clarify_type: Optional[str] = None   # None | "DOMAIN_HARD" | "TASK_SOFT" | "SLOT_REGION" | "CONTEXT_MISSING"
    clarify_reason: Optional[str] = None  # 自然語言說明，供生成層參考
    anchor_turn: Optional[int] = None     # 最近一次 REFRESH 的 turn_index（debug / 繼承追蹤）
    # Log instrument（不影響決策；供離線重訓 / ablation 使用）
    memory_features: Optional[Dict[str, float]] = None  # Memory Agent 看到的 7 維 raw 特徵
    memory_probs: Optional[List[float]] = None          # [P_STAY, P_REFRESH]，Agent fallback 時為 None
    memory_features_v2: Optional[Dict[str, float]] = None  # 8d 新特徵 + 屬性（shadow log）
    agent_used: bool = False                             # 是否走 Memory Agent 路徑（False = fallback 規則）
    agent_decision_raw: Optional[str] = None             # Agent 原始輸出（在 override 前）
    fallback_reason: Optional[str] = None                # None | "low_confidence" | "exception" | "rule_path"
    overrides_fired: Optional[Dict[str, bool]] = None    # 哪些 override 規則觸發
    prev_query: Optional[str] = None                     # 上一輪 user query（供序列模型用）
    prev_task: Optional[str] = None                      # 上一輪 task_pred
    prev_task_dist: Optional[Dict[str, float]] = None    # 上一輪 task_dist


@dataclass
class FlowResult: # 完整的語義流程分析結果
    turn_index: int
    
    # 三層分析
    domain_analysis: DomainAnalysis
    context_analysis: ContextAnalysis
    topic_analysis: TopicAnalysis
    policy_decision: PolicyDecision
    
    # 可選：任務/範圍分類
    task_label: Optional[str] = None
    task_dist: Optional[Dict[str, float]] = None
    task_top_score: Optional[float] = None   # raw cosine similarity to best prototype（出域偵測用）
    task_entropy: Optional[float] = None    # 任務分布歸一化熵 [0,1]（Slot 回填偵測用）
    secondary_tasks: List[str] = field(default_factory=list)  # top-2+ tasks（多任務時）
    scope_label: Optional[str] = None
    scope_dist: Optional[Dict[str, float]] = None
    detected_region: Optional[str] = None  # 偵測到的地區（如：台北市）

    def to_dict(self) -> dict: # 轉換為字典格式（完整分析結果）
        result = {
            "turn_index": self.turn_index,
            "domain_analysis": {
                "top_domain": self.domain_analysis.top_domain,
                "top_prob": float(self.domain_analysis.top_prob),
                "entropy": float(self.domain_analysis.entropy),
                "distribution": {k: float(v) for k, v in self.domain_analysis.distribution.items()},
                "active_domains": list(self.domain_analysis.active_domains),
                "active_domain_probs": {k: float(v) for k, v in self.domain_analysis.active_domain_probs.items()},
                "is_multi_domain": self.domain_analysis.is_multi_domain,
                "top_domain": self.domain_analysis.top_domain,
            },
            "context_analysis": {
                "similarity_score": float(self.context_analysis.similarity_score),
                "source": self.context_analysis.source,
                "is_first_turn": self.context_analysis.is_first_turn,
            },
            "topic_analysis": {
                "is_continuing": self.topic_analysis.is_continuing,
                "overlap_score": float(self.topic_analysis.overlap_score),
                "reason": self.topic_analysis.reason,
                "prev_top_domain": self.topic_analysis.prev_top_domain,
                "cur_top_domain": self.topic_analysis.cur_top_domain,
                "tv_distance": float(self.topic_analysis.tv_distance) if self.topic_analysis.tv_distance is not None else None,
                "prev_dist": {k: float(v) for k, v in self.topic_analysis.prev_dist.items()} if self.topic_analysis.prev_dist else None,
                "prev_active_domains": list(self.topic_analysis.prev_active_domains) if self.topic_analysis.prev_active_domains else None,
            },
            "policy_decision": {
                "context_level": self.policy_decision.context_level,
                "is_ambiguous": self.policy_decision.is_ambiguous,
                "policy_case": self.policy_decision.policy_case,
                "retrieval_action": self.policy_decision.retrieval_action,
                "semantic_flow": self.policy_decision.semantic_flow,
                "memory_action": self.policy_decision.memory_action,
                "clarify_type": self.policy_decision.clarify_type,
                "clarify_reason": self.policy_decision.clarify_reason,
                "anchor_turn": self.policy_decision.anchor_turn,
                "memory_features": self.policy_decision.memory_features,
                "memory_probs": self.policy_decision.memory_probs,
                "memory_features_v2": self.policy_decision.memory_features_v2,
                "agent_used": self.policy_decision.agent_used,
                "agent_decision_raw": self.policy_decision.agent_decision_raw,
                "fallback_reason": self.policy_decision.fallback_reason,
                "overrides_fired": self.policy_decision.overrides_fired,
                "prev_query": self.policy_decision.prev_query,
                "prev_task": self.policy_decision.prev_task,
                "prev_task_dist": self.policy_decision.prev_task_dist,
            },
        }
        
        # 添加任務/範圍分類
        if self.task_label:
            result["task_label"] = self.task_label
        if self.task_dist:
            result["task_dist"] = {k: float(v) for k, v in self.task_dist.items()}
        if self.task_top_score is not None:
            result["task_top_score"] = float(self.task_top_score)
        if self.scope_label:
            result["scope_label"] = self.scope_label
        if self.scope_dist:
            result["scope_dist"] = {k: float(v) for k, v in self.scope_dist.items()}
        
        if self.detected_region:
            result["detected_region"] = self.detected_region

        # 添加多任務標籤（若有）
        if self.secondary_tasks:
            result["secondary_tasks"] = list(self.secondary_tasks)

        # 添加融合後的分布（如果存在）
        if self.domain_analysis.fused_distribution:
            result["domain_analysis"]["fused_distribution"] = {
                k: float(v) for k, v in self.domain_analysis.fused_distribution.items()
            }
        
        
        return result

    def to_json(self) -> str: # 轉換為 JSON 字符串
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str: # 簡潔的文本表示
        lines = [
            f"[Turn {self.turn_index}] {self.policy_decision.semantic_flow.upper()} | "
            f"{self.policy_decision.retrieval_action}",
            f"  Domain: {self.domain_analysis.top_domain} "
            f"(p={self.domain_analysis.top_prob:.3f}, entropy={self.domain_analysis.entropy:.3f})",
            f"  Context: C={self.context_analysis.similarity_score:.3f} ({self.context_analysis.source})",
            f"  Topic: continuing={self.topic_analysis.is_continuing} "
            f"(overlap={self.topic_analysis.overlap_score:.3f}, {self.topic_analysis.reason})",
            f"  Policy: {self.policy_decision.policy_case}",
        ]
        if self.task_label:
            lines.append(f"  Task: {self.task_label}")
        if self.scope_label:
            lines.append(f"  Scope: {self.scope_label}")
        if self.detected_region:
            lines.append(f"  Region: {self.detected_region}")
        return "\n".join(lines)


# ============================================================================
# 主分類器
# ============================================================================

class SemanticFlowClassifier: # 語義流程分類器
    """
    語義流程分類器 - 整合多個模組進行對話狀態追蹤
    
    流程：
    1. 領域路由：判斷涉及的領域及其機率分布
    2. 上下文相似度：計算當前輸入與歷史的相似度
    3. 主題追蹤：判斷多領域主題是否延續
    4. 策略決策：基於 C+MT 進行四象限決策，選擇檢索策略
    5. (可選) 任務/範圍分類：分類用戶的查詢類型和範圍
    """

    def __init__(
        self,
        *,
        text_encoder: TextEncoder,
        domain_router: DomainRouter,
        context_similarity: Optional[ContextSimilarity] = None,
        topic_tracker: Optional[MultiTopicTracker] = None,
        policy_cfg: Optional[DSTPolicyConfig] = None,
        enable_task_scope: bool = False,
        task_scope_clf: TaskScopeClassifier = None,
        overview_anchor_vecs: Optional[List[np.ndarray]] = None,
        overview_sim_threshold: float = 0.5,
        memory_agent: Optional[Any] = None,
    ):
        """
        初始化語義流程分類器
        
        Args:
            text_encoder: 文本編碼器
            domain_router: 領域路由器
            context_similarity: 上下文相似度計算器 (可選，使用預設配置)
            topic_tracker: 主題追蹤器 (可選，使用預設配置)
            policy_cfg: 策略配置 (可選，使用預設配置)
            enable_task_scope: 是否啟用任務/範圍分類
            task_scope_clf: 任務/範圍分類器 (如果 enable_task_scope=True 需提供)
            overview_anchor_vecs: 整體意圖錨點向量列表（用於向量比對，取代關鍵字）
            overview_sim_threshold: 整體意圖相似度門檻
            memory_agent: RL Memory Agent（可選，使用 None 則走原始 Threshold 規則）
        """
        self.text_encoder = text_encoder
        self.domain_router = domain_router

        # 初始化各子模組
        self.context_similarity = context_similarity or ContextSimilarity(
            encoder=self.text_encoder,
            cfg=ContextSimConfig(),
        )
        self.topic_tracker = topic_tracker or MultiTopicTracker(MultiTopicConfig())
        self.policy_cfg = policy_cfg or DSTPolicyConfig()

        # RL Memory Agent（雙軌制：有 Agent 時用 RL，否則走 Threshold 回退）
        self.memory_agent = memory_agent

        # 狀態追蹤
        self.turn_index = 0
        self._prev_scope: Optional[str] = None  # 上一輪 Scope（供沿用與持久化）
        self._prev_was_overview: bool = False  # 上一輪是否為整體（供整體概況規則 A/B）
        self._slot_clarify_cooldown: int = 0   # Slot 追問冷卻（避免連續追問地區）
        self._ood_clarify_cooldown: int = 0    # OOD 追問冷卻（避免連續引導）
        self._last_refresh_turn: Optional[int] = None  # 最近一次 REFRESH 的 turn_index（anchor_turn）

        # 跨輪追蹤（供 v2 特徵 + 序列模型用；不影響任何決策邏輯）
        self._prev_task: Optional[str] = None
        self._prev_task_dist: Optional[Dict[str, float]] = None
        self._prev_query: Optional[str] = None
        # 每輪 _handle_memory_and_fused_distribution 內 set，供 _decide_policy 結尾抓
        self._overrides_fired_log: Dict[str, bool] = {}

        # 整體意圖：向量比對（不用關鍵字）
        self._overview_anchor_vecs = overview_anchor_vecs if overview_anchor_vecs else []
        self._overview_sim_threshold = overview_sim_threshold

        # 任務/範圍分類
        self.enable_task_scope = enable_task_scope
        self.task_scope_clf = task_scope_clf

    def reset(self) -> None:
        """重置對話狀態"""
        self.turn_index = 0
        self._prev_scope = None
        self._prev_was_overview = False
        self._slot_clarify_cooldown = 0
        self._ood_clarify_cooldown = 0
        self._last_refresh_turn = None
        self._prev_task = None
        self._prev_task_dist = None
        self._prev_query = None
        self._overrides_fired_log = {}
        self.context_similarity.reset()
        self.topic_tracker.reset()
    
    def save_state(self, user_id: int, child_id: int, state_dir: str = "dialogue_states") -> bool:
        """
        保存當前對話狀態到文件
        
        Args:
            user_id: 用戶 ID
            child_id: 兒童 ID
            state_dir: 狀態文件保存目錄
        
        Returns:
            是否成功保存
        """
        try:
            from .state_persistence import save_dialogue_state
            return save_dialogue_state(
                user_id, child_id,
                self.context_similarity,
                self.topic_tracker,
                self.turn_index,
                state_dir,
                prev_scope=self._prev_scope,
                prev_was_overview=self._prev_was_overview,
            )
        except Exception as e:
            print(f"[DST] 保存狀態失敗: {e}")
            return False
    
    def load_state(self, user_id: int, child_id: int, state_dir: str = "dialogue_states") -> bool: # 載入前一輪的對話狀態
        """
        從文件加載對話狀態
        
        Args:
            user_id: 用戶 ID
            child_id: 兒童 ID
            state_dir: 狀態文件保存目錄
        
        Returns:
            是否成功加載
        """
        try:
            from .state_persistence import load_dialogue_state # 載入前一輪的對話狀態
            result = load_dialogue_state(
                user_id, child_id,
                self.context_similarity,
                self.topic_tracker,
                state_dir
            )
            if result is not None:
                self.turn_index = result[0]
                self._prev_scope = result[1]
                self._prev_was_overview = result[2] if len(result) > 2 else False
                return True
            return False
        except Exception as e:
            print(f"[DST] 加載狀態失敗: {e}")
            return False

    def _analyze_domain(self, user_query: str, query_vec=None) -> DomainAnalysis:
        dr: DomainResult = self.domain_router.predict(user_query, query_vec=query_vec)
        
        return DomainAnalysis(
            top_domain=dr.top_domain,
            top_prob=float(dr.top_prob),
            entropy=float(dr.entropy),
            distribution=dict(dr.dist),
            active_domains=list(dr.active_domains),
            active_domain_probs=dict(dr.active_domain_probs),
            is_multi_domain=len(dr.active_domains) >= 2,
        )

    def _analyze_context(self, user_query: str) -> ContextAnalysis: # 計算本輪與前一輪的語義相似度
        if self.turn_index == 0:
            C_info = {
                "C": self.context_similarity.cfg.neutral_first_turn,
                "source": "first_turn",
            }
        else:
            C_info = self.context_similarity.compute(user_query)
        
        C = float(C_info["C"])
        C = max(0.0, min(1.0, C))
        
        return ContextAnalysis(
            similarity_score=C,
            source=str(C_info.get("source", "")),
            is_first_turn=(self.turn_index == 0),
        )

    def _analyze_topic( # 判斷本輪是否為主題延續（純觀察者，不接受 is_ambiguous）
        self,
        domain_dist: Dict[str, float],
        top_domain: str,
        cur_active_domains: List[str],
    ) -> TopicAnalysis:
        # 在更新之前保存上一輪的分布和活躍領域（用於 agent 決策後的域回退）
        prev_dist_before_update = dict(self.topic_tracker.state.prev_dist) if self.topic_tracker.state.prev_dist else None
        prev_active_domains_before_update = list(self.topic_tracker.state.prev_active_domains) if self.topic_tracker.state.prev_active_domains else None

        topic_info = self.topic_tracker.check_topic_continuation(
            cur_dist=domain_dist,
            cur_raw_top_domain=top_domain,
            confidence=0.0,
            cur_active_domains=cur_active_domains,
            prev_active_domains=self.topic_tracker.state.prev_active_domains,
        )
        
        return TopicAnalysis(
            is_continuing=bool(topic_info.get("topic_continue", True)),
            overlap_score=float(topic_info.get("topic_overlap", 0.0)),  # 這是綜合 MT 分數
            reason=str(topic_info.get("reason", "")),
            prev_top_domain=topic_info.get("prev_raw_top_domain"),
            cur_top_domain=topic_info.get("cur_raw_top_domain"),
            prev_dist=prev_dist_before_update,  # 保存更新前的上一輪分布
            prev_active_domains=prev_active_domains_before_update,  # 保存更新前的上一輪活躍領域
            tv_distance=topic_info.get("tv_distance"),  # TV 距離
            active_domain_coverage=topic_info.get("active_domain_coverage"),
            continuation_mode=topic_info.get("continuation_mode"),
        )

    def _get_all_domains(self) -> List[str]: # 取得所有領域列表
        from .domain_anchors import DOMAINS
        return DOMAINS.copy()
    
    def _handle_memory_and_fused_distribution(
        self,
        domain: DomainAnalysis,
        topic: TopicAnalysis,
        agent_decision: Optional[str] = None,  # "STAY" / "REFRESH" / "CLARIFY" / None（規則 fallback）
        user_query: str = "",
    ) -> Tuple[bool, float, Optional[Dict[str, float]]]:
        # 重置本輪的 overrides 觸發追蹤（供 log 寫出 / ablation 分析使用，不影響決策）
        self._overrides_fired_log = {
            "rule_a_overview_high_entropy": False,
            "rule_b_prev_overview_continued": False,
            "short_followup_lock": False,
            "override_a_strong_shift_stay_to_refresh": False,
            "top_sticky_weak_signal": False,
        }
        """
        根據 agent_decision（或規則 fallback）決定本輪使用哪一輪的 active_domains / fused_distribution。
        已移除 is_ambiguous 參數，整體概況規則改用 entropy 直接判斷。

        優先順序：
          1. 規則 A/B（整體概況特殊處理，不被 Agent 覆蓋）
          2. Agent 決策（STAY / REFRESH / CLARIFY）
          3. 規則 Fallback（agent_decision is None 時，走 strong/soft/shift 邏輯）
        """
        adjusted_topic_continue = topic.is_continuing
        adjusted_topic_overlap = topic.overlap_score
        fused_distribution: Optional[Dict[str, float]] = None
        domain_handled = False

        # 用 entropy 判斷是否為高熵（僅用於整體概況規則 A/B）
        _high_entropy = domain.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th

        # ── 規則 A/B：整體概況特殊處理（最高優先，不被 Agent 覆蓋）─────────────
        if _high_entropy and domain.top_domain == "整體概況" and not self._prev_was_overview:
            self.topic_tracker.reset()
            domain_handled = True
            self._overrides_fired_log["rule_a_overview_high_entropy"] = True
            if DST_DEBUG_VERBOSE:
                print("  [整體查詢] 規則A：高熵+整體概況領域，清除記憶、啟用新對話")

        elif _high_entropy and self._prev_was_overview:
            domain_handled = True
            self._overrides_fired_log["rule_b_prev_overview_continued"] = True
            if DST_DEBUG_VERBOSE:
                print("  [整體查詢] 規則B：高熵+上一輪整體，沿用整體意圖")

        # ── Agent 決策路徑（Rules A/B 未觸發時）─────────────────────────────────
        if not domain_handled and agent_decision is not None:
            prev_dist = topic.prev_dist
            prev_active_domains = topic.prev_active_domains
            prev_top = topic.prev_top_domain
            has_history = bool(prev_dist) and self.turn_index > 0

            # ──── 短接續詞鎖定：STAY 時若 query 為短接續詞，強制沿用 prev top + active ────
            # 例：「怎麼訓練」「然後呢」「那個呢」「再來」「呢」
            # Memory Agent 看不到 query 文字，DomainRouter 對短 query 易漂移到 anchor 偏好的 domain
            # （例：「怎麼訓練」漂到「精細動作」因 anchor 多含「訓練」字）。
            # 這類 query 用戶意圖明顯是接續上輪主題，應沿用 prev_top / prev_active。
            _q = (user_query or "").strip()
            _DOMAIN_KEYWORDS = (
                "粗大", "精細", "認知", "口語", "感統", "感覺統合",
                "口腔", "情緒", "社交", "吞嚥", "說話", "語言"
            )
            _FOLLOWUP_INDICATORS = (
                "怎麼", "如何", "然後", "那", "呢", "還有", "再", "繼續",
                "接著", "請問呢", "為什麼", "是什麼", "為何"
            )
            _is_short_followup = (
                agent_decision == "STAY"
                and len(_q) <= 8
                and not any(kw in _q for kw in _DOMAIN_KEYWORDS)  # 沒明確 domain 詞
                and (any(ind in _q for ind in _FOLLOWUP_INDICATORS) or len(_q) <= 4)
            )
            if _is_short_followup and prev_active_domains and prev_top:
                self._overrides_fired_log["short_followup_lock"] = True
                if DST_DEBUG_VERBOSE:
                    print(
                        f"  [Short Followup 鎖定] q='{_q}' (短接續詞)："
                        f"top_domain {domain.top_domain} → {prev_top}, "
                        f"active {domain.active_domains} → {prev_active_domains}"
                    )
                domain.top_domain = prev_top
                domain.active_domains = list(prev_active_domains)
                domain.is_multi_domain = len(domain.active_domains) >= 2
                # 同步 raw_top 給下一輪用
                try:
                    self.topic_tracker.state.prev_raw_top_domain = prev_top
                except Exception:
                    pass

            # ──── Domain 強切換 Override：STAY → REFRESH ────
            # Memory Agent 只看 numeric features，看不到 query 文字本身的 domain 關鍵字。
            # 當用戶明確切 domain（如「粗大功能」「認知怎麼練」）時，DomainRouter top_prob 會升高，
            # 但 Memory Agent 可能仍判 STAY → 鎖死在舊 domain。
            # 觸發條件（全部成立才 override）：
            #   1. agent_decision == "STAY"
            #   2. 本輪 top_domain 不在 prev_active_domains 內（換到陌生 domain）
            #   3. 本輪 top_prob >= 0.32（query 信號明確）
            #   4. tv_distance >= 0.40（domain 分布有顯著變化）
            #   5. 有歷史（非 T0）
            _STRONG_DOMAIN_SHIFT_TOP_PROB = 0.32
            _STRONG_DOMAIN_SHIFT_TV = 0.40
            _override_to_refresh = False
            if (has_history
                    and agent_decision == "STAY"
                    and prev_active_domains
                    and domain.top_domain not in prev_active_domains
                    and float(domain.top_prob) >= _STRONG_DOMAIN_SHIFT_TOP_PROB
                    and float(topic.tv_distance or 0) >= _STRONG_DOMAIN_SHIFT_TV):
                self._overrides_fired_log["override_a_strong_shift_stay_to_refresh"] = True
                if DST_DEBUG_VERBOSE:
                    print(
                        f"  [Override STAY→REFRESH] domain 強切換："
                        f"{domain.top_domain}(prob={domain.top_prob:.2f}) "
                        f"不在 prev_active={prev_active_domains}, tv={topic.tv_distance:.2f}"
                    )
                agent_decision = "REFRESH"
                _override_to_refresh = True

            if agent_decision == "STAY":
                if has_history:
                    # 沿用上一輪 domain：active_domains / fused_distribution 回退
                    adjusted_topic_continue = True
                    adjusted_topic_overlap = max(
                        topic.overlap_score,
                        self.policy_cfg.ambiguous_continuation_min_overlap,
                    )
                    topic.is_continuing = adjusted_topic_continue
                    topic.overlap_score = adjusted_topic_overlap
                    fused_distribution = dict(prev_dist)
                    if prev_active_domains:
                        domain.active_domains = list(prev_active_domains)
                        domain.is_multi_domain = len(domain.active_domains) >= 2
                    # STAY 時 top_domain 沿用（避免短 query 漂移到不相關 domain）
                    # 條件：
                    #   (1) 本輪 top_domain 不在上輪 active 內（漂移到陌生 domain）
                    #   (2) 本輪 top_prob 不夠強（< 0.35），表示 query 本身對 domain 的信號不確定
                    #   (3) prev_top 不是「整體概況」（避免硬鎖）
                    # 若本輪 top_prob >= 0.35（用戶明確切 domain，如「粗大功能有問題嗎」），不沿用，相信本輪 DST。
                    _STRONG_TOP_PROB_TH = 0.35
                    if (prev_top
                            and prev_active_domains
                            and domain.top_domain not in prev_active_domains
                            and prev_top != "整體概況"
                            and float(domain.top_prob) < _STRONG_TOP_PROB_TH):
                        self._overrides_fired_log["top_sticky_weak_signal"] = True
                        if DST_DEBUG_VERBOSE:
                            print(f"  [Agent STAY] top_domain 弱信號漂移：{domain.top_domain}({domain.top_prob:.2f}) → 沿用 {prev_top}（不在 active 內）")
                        domain.top_domain = prev_top
                        # 同步 topic_tracker，讓下一輪的 prev_top 沿用「修正後」的值，
                        # 避免 raw_top 漂移值在多輪累積。
                        try:
                            self.topic_tracker.state.prev_raw_top_domain = prev_top
                        except Exception:
                            pass
                    elif (prev_top
                            and prev_active_domains
                            and domain.top_domain not in prev_active_domains
                            and float(domain.top_prob) >= _STRONG_TOP_PROB_TH):
                        if DST_DEBUG_VERBOSE:
                            print(f"  [Agent STAY] top_domain 強切換：{domain.top_domain}({domain.top_prob:.2f}) 信號明確，不沿用 prev_top")
                    # 記憶狀態回退，讓下一輪的 TV 計算以上一輪為基準
                    self.topic_tracker.state.memory_dist = dict(prev_dist)
                    self.topic_tracker.state.prev_dist = dict(prev_dist)
                    if prev_active_domains:
                        self.topic_tracker.state.prev_active_domains = list(prev_active_domains)
                    if DST_DEBUG_VERBOSE:
                        print(f"  [Agent STAY] 沿用上一輪 domain: {prev_top}, active: {prev_active_domains}")
                        print(
                            f"  [Agent STAY] 調整後："
                            f"topic_continue={adjusted_topic_continue}, topic_overlap={adjusted_topic_overlap:.4f}"
                        )
                else:
                    if DST_DEBUG_VERBOSE:
                        print("  [Agent STAY] 首輪或無歷史，保持本輪 domain")

            elif agent_decision == "REFRESH":
                # 切換到本輪 domain，不做融合
                adjusted_topic_continue = False
                # fused_distribution = None → 使用本輪 distribution
                if DST_DEBUG_VERBOSE:
                    print(f"  [Agent REFRESH] 切換到本輪 domain: {domain.active_domains}")

            # 新設計：Memory Agent 不再輸出 CLARIFY（已移至 clarify_type 屬性）

            domain_handled = True

        # ── 規則 Fallback（agent_decision is None）─────────────────────────────
        if not domain_handled:
            # 非整體概況：strong / soft / shift（基於 Topic Tracker 的純觀察結果）
            if domain.top_domain != "整體概況":
                mode = getattr(topic, "continuation_mode", None)
                prev_dist = topic.prev_dist
                prev_active = topic.prev_active_domains or []

                if mode == "strong":
                    if prev_active:
                        domain.active_domains = list(prev_active)
                        domain.is_multi_domain = len(domain.active_domains) >= 2
                    if prev_dist:
                        fused_distribution = dict(prev_dist)
                        domain.fused_distribution = fused_distribution

                elif mode == "soft":
                    merged = set(domain.active_domains or []) | set(prev_active or [])
                    domain.active_domains = sorted(list(merged)) if merged else list(domain.active_domains)
                    domain.is_multi_domain = len(domain.active_domains) >= 2
                    if prev_dist:
                        alpha = 0.5
                        mixed: Dict[str, float] = {}
                        keys = set(prev_dist.keys()) | set(domain.distribution.keys())
                        for k in keys:
                            mixed[k] = alpha * float(prev_dist.get(k, 0.0)) + (1.0 - alpha) * float(domain.distribution.get(k, 0.0))
                        total = sum(mixed.values())
                        if total > 0:
                            mixed = {k: v / total for k, v in mixed.items()}
                        fused_distribution = mixed
                        domain.fused_distribution = fused_distribution
                # shift：保持本輪，不做額外處理

        return adjusted_topic_continue, adjusted_topic_overlap, fused_distribution
    
    # ========================================================================
    # Clarify 決策引擎（規則式，與 Memory Agent 解耦）
    # 判斷系統是否需要在回應中加入澄清追問，不阻塞檢索主流程。
    # ========================================================================
    FOLLOWUP_MARKERS = ("對了", "剛才", "還想問", "那個", "那樣")

    def _decide_clarify(
        self,
        domain: "DomainAnalysis",
        task_label: Optional[str],
        secondary_tasks: Optional[List[str]],
        detected_region: Optional[str],
        user_query: str,
        q_len_norm: float,
        task_dist: Optional[Dict[str, float]] = None,
        task_top_score: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        依優先順序判斷 clarify_type，回傳 (clarify_type, reason)。

        優先順序（高到低）：
          1. OUT_OF_DOMAIN  task_top_score < 0.55，與所有 task prototype 相似度極低
                            → 使用者偏離早療話題，引導回早療
          2. DOMAIN_HARD    極度模糊 query，系統無法推論 domain，必須追問
          3. CONTEXT_MISSING T0 帶接續詞但無歷史，必須追問
          4. SLOT_REGION    Task H/K + 缺地區，給通用答案 + 詢問地區（含 cooldown）
          5. TASK_SOFT      多任務觸發（需 secondary task_dist >= 0.25 才觸發）
          6. None           正常回應
        """
        # 0. OUT_OF_DOMAIN：偏離早療話題（最高優先）
        #    當 query 與所有 14 個 task prototype 的最高相似度都低於門檻
        #    代表此 query 不屬於本系統涵蓋的早療範疇
        if task_top_score is not None and task_top_score < 0.55:
            if self._ood_clarify_cooldown <= 0:
                self._ood_clarify_cooldown = 2  # 連續 2 輪內不重複引導
                return (
                    "OUT_OF_DOMAIN",
                    f"task_top_score={task_top_score:.2f} 低於 OOD 門檻 0.55，疑似偏離早療話題"
                )
            # 在 cooldown 中，不重複引導，但仍可走後續分支

        # 1. DOMAIN_HARD：極度模糊
        if domain.entropy > 0.90 and q_len_norm < 0.16:  # len(query) < 8
            return "DOMAIN_HARD", f"domain_entropy={domain.entropy:.2f}, query極短"

        # 2. CONTEXT_MISSING：T0 接續詞
        if self.turn_index == 0:
            has_followup = any(m in user_query for m in self.FOLLOWUP_MARKERS)
            if has_followup:
                return "CONTEXT_MISSING", "T0 帶接續詞但無對話歷史"

        # 3. SLOT_REGION：Task H/K 缺地區（含 cooldown 避免連續追問）
        task_candidates = [task_label] + (secondary_tasks or [])
        if any(t in ("H", "K") for t in task_candidates) and not detected_region:
            if self._slot_clarify_cooldown <= 0:
                self._slot_clarify_cooldown = 3  # 冷卻 3 輪
                return "SLOT_REGION", f"Task={task_label} 缺地區資訊"
            # 在冷卻中，不重複追問

        # 4. TASK_SOFT：多任務（需 task_dist 分數門檻，避免過度觸發）
        #    修正：舊實作只要 secondary_tasks 非空就觸發，導致 50.7% turn 都觸發。
        #    新實作要求 secondary task 的 task_dist 分數 >= 0.18 才算「真正的多任務」。
        #    0.25 → 0.18：給 H/K + 個人化 query 的多任務識別更多空間，
        #    讓 strategy_mapper 的 TASK_SOFT 分支能擴展 use_sections 涵蓋個案資料。
        _TASK_SOFT_TH = 0.18
        if secondary_tasks and task_dist:
            qualified_secondary = [
                t for t in secondary_tasks
                if task_dist.get(t, 0.0) >= _TASK_SOFT_TH
            ]
            if qualified_secondary:
                return (
                    "TASK_SOFT",
                    f"多任務觸發: {task_label} + {qualified_secondary} (task_dist >= {_TASK_SOFT_TH:.2f})"
                )

        return None, None

    def _decide_policy( # 決策層
        self,
        domain: DomainAnalysis,
        context: ContextAnalysis,
        topic: TopicAnalysis,
        user_query: str,
        task_label: Optional[str] = None,
        secondary_tasks: Optional[List[str]] = None,
        detected_region: Optional[str] = None,
        task_dist: Optional[Dict[str, float]] = None,
        task_top_score: Optional[float] = None,
    ) -> PolicyDecision:
        from .dst_policy import compute_MT, predicted_flow_from_C_MT

        q_len_norm = min(float(len(user_query)) / 50.0, 1.0)

        # ========================================================================
        # 🧠 Memory Agent 優先推論（使用原始 topic.overlap_score，非 adjusted）
        # ========================================================================
        agent_decision: Optional[str] = None  # "STAY" / "REFRESH" / "CLARIFY" / None
        _agent_probs_log = None
        _agent_state_log = None
        _agent_decision_raw_log: Optional[str] = None  # Override 前的原始 Agent 決策
        _fallback_reason_log: Optional[str] = None     # Agent 為何 fallback

        if self.policy_cfg.enable_memory_agent and self.memory_agent is not None:
            try:
                # 組裝 DST 7維特徵向量（使用原始 overlap，讓 Agent 看到未經規則調整的信號）
                _agent_state_log = {
                    "entropy": float(domain.entropy),
                    "tv_distance": float(topic.tv_distance) if topic.tv_distance is not None else 0.0,
                    "topic_overlap": float(topic.overlap_score),  # RAW，未經規則調整
                    "context_sim": float(context.similarity_score),
                    "turn_index_norm": min(float(self.turn_index) / 10.0, 1.0),
                    "query_len_norm": q_len_norm,
                    "is_multi_domain": domain.is_multi_domain,
                }
                result = self.memory_agent.select_action(_agent_state_log, deterministic=True)
                _agent_probs_log = result["probs"]
                _agent_decision_raw_log = result["action_str"]  # 保留 raw 供 log

                # 信心門檻：max prob < 0.5 → 回退規則，避免冷啟動時不確定的決策影響域選擇
                if max(_agent_probs_log) >= 0.5:
                    agent_decision = result["action_str"]
                else:
                    _fallback_reason_log = "low_confidence"  # agent_decision stays None → fallback
            except Exception as e:
                print(f"  [Memory Agent] ⚠️ 推論失敗，回退到 Threshold 規則: {e}")
                agent_decision = None
                _fallback_reason_log = "exception"
        else:
            _fallback_reason_log = "agent_disabled"

        # ========================================================================
        # 🆕 Shadow log: 8d 新特徵（不影響任何決策；供 Phase 1 ablation / 重訓使用）
        # ========================================================================
        try:
            _top3_pairs = sorted(domain.distribution.items(), key=lambda kv: -kv[1])[:3]
            _top3_domains = [d for d, _ in _top3_pairs]
            _memory_features_v2_log = extract_memory_features_v2(
                user_query=user_query,
                domain_entropy=float(domain.entropy),
                cur_top_domain=domain.top_domain,
                cur_top3_domains=_top3_domains,
                prev_top_domain=topic.prev_top_domain,
                cur_task_dist=task_dist or {},
                prev_task_dist=self._prev_task_dist,
                prev_task=self._prev_task,
                tv_distance_raw=float(topic.tv_distance) if topic.tv_distance is not None else 0.0,
            )
        except Exception as _e:
            print(f"  [feature_v2] ⚠️ 抽取失敗: {_e}")
            _memory_features_v2_log = None

        # ========================================================================
        # 1. 根據 agent_decision 統一決定記憶與 fused_distribution
        # ========================================================================
        adjusted_topic_continue, adjusted_topic_overlap, fused_distribution = self._handle_memory_and_fused_distribution(
            domain=domain,
            topic=topic,
            agent_decision=agent_decision,
            user_query=user_query,
        )

        # ========================================================================
        # 2. Agent 路徑：從 agent_decision 映射 action / policy_case / memory_action
        # ========================================================================
        agent_used = False

        if agent_decision is not None:
            # 新設計：Memory Agent 僅輸出 STAY / REFRESH
            if agent_decision == "STAY":
                action = "NARROW_GRAPH"
                policy_case = "AGENT_STAY"
                memory_action = "STAY"
            elif agent_decision == "REFRESH":
                action = "WIDE_IN_DOMAIN"
                policy_case = "AGENT_REFRESH"
                memory_action = "REFRESH"
            else:
                # 保守處理：萬一舊模型仍輸出 CLARIFY（3 分類），降級為 STAY
                action = "NARROW_GRAPH"
                policy_case = "AGENT_STAY_FALLBACK"
                memory_action = "STAY"

            C_level = "high" if context.similarity_score >= self.policy_cfg.C_high_th else "low"
            ambig = False  # is_ambiguous 稍後由 clarify_type 派生
            agent_used = True

            if DST_DEBUG_VERBOSE:
                print(
                    f"  [Memory Agent] 決策={agent_decision} "
                    f"(probs={[f'{p:.3f}' for p in _agent_probs_log]}) → action={action}"
                )
                print(
                    f"  [Memory Agent] 輸入特徵: entropy={_agent_state_log['entropy']:.3f}, "
                    f"tv_dist={_agent_state_log['tv_distance']:.3f}, "
                    f"overlap={_agent_state_log['topic_overlap']:.3f}, "
                    f"ctx_sim={_agent_state_log['context_sim']:.3f}, "
                    f"turn={_agent_state_log['turn_index_norm']:.2f}, "
                    f"q_len={_agent_state_log['query_len_norm']:.2f}"
                )

        # ========================================================================
        # 📏 原始 Threshold 規則（Fallback Path：agent_decision is None）
        # ========================================================================
        if not agent_used:
            # 多任務時：H/K 覆寫需要信心門檻（與 downstream 覆寫邏輯一致）
            _fb_resource_task = task_label  # 預設用主任務
            if task_label not in ("H", "K") and secondary_tasks and task_dist:
                for _t in secondary_tasks:
                    if _t in ("H", "K") and task_dist.get(_t, 0.0) >= 0.3:
                        _fb_resource_task = _t
                        break
            C_level, ambig, policy_case, action = decide_policy(
                C=context.similarity_score,
                normalized_entropy=domain.entropy,
                topic_continue=adjusted_topic_continue,
                topic_overlap=adjusted_topic_overlap,
                is_multi_domain=domain.is_multi_domain,
                cfg=self.policy_cfg,
                query_len_norm=q_len_norm,
                task_label=_fb_resource_task,
                detected_region=detected_region,
            )
            # Fallback 規則下的推論 memory_action
            # CLARIFY 只能由 Agent 產生，Fallback 不會輸出 CLARIFY
            if action in ("NARROW_GRAPH", "CONTEXT_FIRST"):
                memory_action = "STAY"
            else:
                memory_action = "REFRESH"
            # Fallback 路徑的 ambig 始終為 False（CLARIFY 只能由 Agent 產生）
            ambig = False

        # ========================================================================
        # 計算 semantic_flow
        # ========================================================================
        if agent_used:
            # Agent 決策直接映射 semantic_flow（2 分類）
            _memory_to_flow = {"STAY": "continue", "REFRESH": "shift_hard"}
            semantic_flow = _memory_to_flow.get(memory_action, "continue")
        else:
            MT = compute_MT(adjusted_topic_continue, adjusted_topic_overlap)
            semantic_flow = predicted_flow_from_C_MT(
                C=context.similarity_score, MT=MT, cfg=self.policy_cfg
            )

        # ========================================================================
        # 特殊任務：Task H/K + 有地區 → 直接觸發 LOCAL_RESOURCE_SEARCH
        #   （缺地區的情境已由 _decide_clarify 的 SLOT_REGION 處理，不再阻塞）
        # ========================================================================
        _HK_CONFIDENCE_TH = 0.3
        _resource_task = None
        if task_label in ("H", "K"):
            _resource_task = task_label
        elif secondary_tasks and task_dist:
            for t in secondary_tasks:
                if t in ("H", "K") and task_dist.get(t, 0.0) >= _HK_CONFIDENCE_TH:
                    _resource_task = t
                    break
        if _resource_task and detected_region:
            action = "LOCAL_RESOURCE_SEARCH"
            policy_case = f"TASK_{_resource_task}_SEARCH"
            # memory_action / semantic_flow 維持原判斷，不再強制覆寫

        # ========================================================================
        # Clarify 決策（規則式，附加屬性，不阻塞檢索）
        # ========================================================================
        clarify_type, clarify_reason = self._decide_clarify(
            domain=domain,
            task_label=task_label,
            secondary_tasks=secondary_tasks,
            detected_region=detected_region,
            user_query=user_query,
            q_len_norm=q_len_norm,
            task_dist=task_dist,
            task_top_score=task_top_score,
        )

        # is_ambiguous 派生：僅當 clarify 觸發時為 True（供回應層參考）
        ambig = clarify_type in ("DOMAIN_HARD", "CONTEXT_MISSING")

        # anchor_turn：若本輪是 REFRESH 則更新錨點
        if memory_action == "REFRESH":
            self._last_refresh_turn = self.turn_index
        anchor_turn = self._last_refresh_turn

        # 如果進行了記憶融合，更新 domain 的 fused_distribution
        if fused_distribution is not None:
            domain.fused_distribution = fused_distribution

        # 更新上一輪是否為整體的狀態（供下一輪判斷規則 B）
        self._prev_was_overview = (domain.top_domain == "整體概況")

        # 修飾 policy_case（debug 標記）
        if agent_used:
            if ambig:
                policy_case += "_AMBIG"
            if domain.is_multi_domain:
                policy_case += "_MD"
        if clarify_type:
            policy_case += f"_{clarify_type}"

        # 規則 fallback 路徑也補一個 reason
        _final_fallback_reason = _fallback_reason_log if not agent_used else None
        return PolicyDecision(
            context_level=C_level,
            is_ambiguous=ambig,
            policy_case=policy_case,
            retrieval_action=action,
            semantic_flow=semantic_flow,
            memory_action=memory_action,
            clarify_type=clarify_type,
            clarify_reason=clarify_reason,
            anchor_turn=anchor_turn,
            memory_features=_agent_state_log,
            memory_probs=_agent_probs_log,
            memory_features_v2=_memory_features_v2_log,
            agent_used=agent_used,
            agent_decision_raw=_agent_decision_raw_log,
            fallback_reason=_final_fallback_reason,
            overrides_fired=dict(self._overrides_fired_log) if self._overrides_fired_log else None,
            prev_query=self._prev_query,
            prev_task=self._prev_task,
            prev_task_dist=self._prev_task_dist,
        )

    def _classify_scope(
        self,
        domain_analysis: DomainAnalysis,
        user_query: str,
    ) -> tuple:
        """
        依規則分類 Scope（要檢索多少主題分支）。
        特殊處理：
        1. 若發生領域模糊觸發對話沿用（fused_distribution 已設）→ Scope 沿用上一輪。
        2. 若首輪且本輪模糊（entropy 高）→ 預設「整體」。
        否則：
        - S_overview: 整體查詢 → 查整張圖
        - S_multi_domain: 多領域 → 查多個特定領域
        - S_domain: 單領域 → 查單一領域
        """
        # 1. 整體概況領域 → S_overview
        if domain_analysis.top_domain == "整體概況":
            label = "S_overview"
            dist = {label: 1.0}
            return label, dist

        # 2. 模糊沿用：只有在「模糊 + 已有 fused 分布」時，Scope 才沿用上一輪
        #    若只是一般 strong/soft 延續導致 fused_distribution 被設，但熵不高，仍應依本輪 active_domains 判斷
        if (
            domain_analysis.fused_distribution is not None
            and self.policy_cfg.enable_ambiguous_continuation
            and domain_analysis.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th
        ):
            label = self._prev_scope or "S_overview"
            dist = {label: 1.0}
            return label, dist

        # 3. 首輪且模糊：無上一輪可沿用，預設整體
        if self.turn_index == 0 and domain_analysis.entropy >= self.policy_cfg.ambiguous_continuation_entropy_th:
            label = "S_overview"
            dist = {label: 1.0}
            return label, dist

        # 4. 一般規則
        if len(domain_analysis.active_domains) >= 2:
            label = "S_multi_domain"
        else:
            label = "S_domain"
        dist = {label: 1.0}
        return label, dist

    def _classify_task_only(self, user_query: str, query_vec=None) -> tuple:
        """僅做 Task 分類，同時返回 secondary tasks（多任務支援）。
        Returns: (task_label, task_dist, secondary_tasks, task_top_score, task_entropy)
        """
        if not self.enable_task_scope:
            return None, None, [], None, None

        try:
            task_result = self.task_scope_clf.predict_task(user_query, query_vec=query_vec)
            task_label     = task_result.label if hasattr(task_result, "label") else str(task_result)
            task_dist      = task_result.dist  if hasattr(task_result, "dist")  else None
            task_top_score = float(task_result.score) if hasattr(task_result, "score") else None
            task_entropy   = float(task_result.entropy) if hasattr(task_result, "entropy") else None

            # 多任務偵測：top-k 方法（複用同一次 predict 的 dist，不再重新 encode）
            topk = self.task_scope_clf.predict_task_topk(user_query, k=2, query_vec=query_vec)
            secondary_tasks = topk[1:]  # 排除 top-1

            if secondary_tasks:
                print(f"[DST] 多任務偵測：主任務={task_label}，副任務={secondary_tasks}")

            return task_label, task_dist, secondary_tasks, task_top_score, task_entropy
        except Exception:
            return None, None, [], None, None

    def predict(
        self,
        user_query: str,
        assistant_reply: Optional[str] = None,
        query_vec=None,
    ) -> FlowResult:
        """
        進行完整的語義流程分析

        Args:
            user_query: 用戶輸入
            assistant_reply: 助手回應 (用於更新上下文記憶)
            query_vec: 預算好的 query embedding（可選，避免重複 encode）

        Returns:
            FlowResult: 包含所有分析層的完整結果
        """
        user_query = (user_query or "").strip()
        turn_idx = self.turn_index

        # 預算 query_vec，供後續所有子模組共用
        if query_vec is None:
            query_vec = self.text_encoder.encode(user_query)

        # 逐層分析
        domain = self._analyze_domain(user_query, query_vec=query_vec)
        context = self._analyze_context(user_query)

        # Topic Tracker 為純觀察者，不接受 is_ambiguous，誠實計算 TV/overlap
        topic = self._analyze_topic(
            domain.distribution,
            domain.top_domain,
            domain.active_domains,
        )
        # 提取地區
        detected_region = extract_region(user_query)

        # Task 分類（分類器）；Scope 分類（規則：整體/單領域/多領域）
        task_label, task_dist, secondary_tasks, task_top_score, task_entropy = self._classify_task_only(user_query, query_vec=query_vec)

        policy = self._decide_policy(
            domain, context, topic, user_query,
            task_label=task_label,
            secondary_tasks=secondary_tasks,
            detected_region=detected_region,
            task_dist=task_dist,
            task_top_score=task_top_score,
        )

        scope_label, scope_dist = self._classify_scope(domain, user_query)

        # 構建結果
        result = FlowResult(
            turn_index=turn_idx,
            domain_analysis=domain,
            context_analysis=context,
            topic_analysis=topic,
            policy_decision=policy,
            task_label=task_label,
            task_dist=task_dist,
            task_top_score=task_top_score,
            task_entropy=task_entropy,
            secondary_tasks=secondary_tasks,
            scope_label=scope_label,
            scope_dist=scope_dist,
            detected_region=detected_region,
        )

        # 供下一輪 Scope 沿用與持久化
        self._prev_scope = scope_label
        self._prev_was_overview = (domain.top_domain == "整體概況")

        # 更新記憶狀態
        # DOMAIN_HARD 時跳過 context_similarity 更新（避免極模糊 query 污染後續狀態）
        if policy.clarify_type != "DOMAIN_HARD":
            if assistant_reply is None:
                self.context_similarity.update(user_query)
            else:
                self.context_similarity.update(user_query, assistant_reply)
        else:
            if DST_DEBUG_VERBOSE:
                print("  [DOMAIN_HARD] 跳過 context_similarity 更新（避免極模糊 query 污染狀態）")

        # Cooldown 遞減（每輪 -1，下限 0）
        if self._slot_clarify_cooldown > 0:
            self._slot_clarify_cooldown -= 1
        if self._ood_clarify_cooldown > 0:
            self._ood_clarify_cooldown -= 1

        # 更新跨輪追蹤（供下一輪 v2 特徵 + 序列模型使用；不影響任何決策）
        self._prev_task = task_label
        self._prev_task_dist = task_dist
        self._prev_query = user_query

        self.turn_index += 1
        return result
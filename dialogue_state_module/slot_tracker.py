# slot_tracker.py
"""
Slot Tracker — 輕量槽位追蹤��回填偵測

設計原則：
  - Slot 是 refinement，不是阻塞：缺槽時仍可寬範圍檢索 + 回答 + 追問
  - 追問繼承靠 pending_slot：系統追問後，使用者短回答走回填而非重新分類
  - 任務每輪重判，不做跨輪 EMA
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================================================
# 任務槽位定義
# ============================================================================

TASK_SLOTS: Dict[str, List[str]] = {
    "B": ["domain_focus"],       # 分數解讀 — 需要知道哪個領域
    "C": ["domain_focus"],       # 觀察解讀 — 需要知道哪個領域
    "E": ["ability_focus"],      # 在家訓練 — 需要知道哪個能力
    "F": ["child_age"],          # 融入作息 — 需要知道小朋友年齡（可從 DB 自動取）
    "H": ["region"],             # 轉介資源 — 需要知道縣市
    "J": ["school_type"],        # 學校合作 — 需要知道學校類型
    "K": ["region"],             # 補助福利 — 需要知道縣市
    "L": ["time_range"],         # 後續追蹤 — 需要知道上次評估時間
    "N": ["report_range"],       # 進步查詢 — 需要知道比較哪幾份
}

# 各任務缺槽時的追問提示（附在回覆結尾）
SLOT_FOLLOWUP_HINTS: Dict[str, Dict[str, str]] = {
    "region": {
        "H": "方便告訴我您目前在哪個縣市嗎？這樣我可以幫您查詢附近的早療機構和資源。",
        "K": "請問您在哪個縣市呢？各縣市的補助方案不同，提供地區後我可以給您更精確的資訊。",
    },
    "domain_focus": {
        "B": "想進一步了解哪個領域的分數呢？例如語言、認知、動作等。",
        "C": "想特別看哪方面的觀察紀錄呢？",
    },
    "ability_focus": {
        "E": "想先針對哪個能力來練習呢？例如口語表達、精細動作等。",
    },
    "child_age": {
        "F": "方便告訴我小朋友目前幾歲嗎？這樣我可以建議更適合的作息安排。",
    },
    "school_type": {
        "J": "小朋友目前就讀哪種學校呢？例如公幼、私幼、國小等。",
    },
    "time_range": {
        "L": "上次評估大概是什麼時候呢？這樣我可以建議追蹤的時程。",
    },
    "report_range": {
        "N": "想比較哪幾次的評估報告呢？預設會使用最近的兩份。",
    },
}


# ============================================================================
# Slot 狀態
# ============================================================================

@dataclass
class SlotState:
    """儲存在 dialogue_states/ JSON 中的槽位狀態"""
    active_task: Optional[str] = None          # 上輪的任務標籤
    pending_slot: Optional[str] = None         # 等待回填的槽位名稱
    filled_slots: Dict[str, Any] = field(default_factory=dict)  # 已填的槽位值

    def has_pending(self) -> bool:
        return self.pending_slot is not None

    def clear(self):
        self.active_task = None
        self.pending_slot = None
        self.filled_slots = {}

    def to_dict(self) -> dict:
        return {
            "active_task": self.active_task,
            "pending_slot": self.pending_slot,
            "filled_slots": self.filled_slots,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlotState":
        if not d:
            return cls()
        return cls(
            active_task=d.get("active_task"),
            pending_slot=d.get("pending_slot"),
            filled_slots=d.get("filled_slots", {}),
        )


# ============================================================================
# Slot Tracker
# ============================================================================

@dataclass
class SlotCheckResult:
    """Slot 檢查結果"""
    task_label: str                          # 最終使用的任務標籤（可能繼承自上輪）
    slot_status: str                         # "all_filled" | "has_missing" | "no_slots"
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    followup_hint: Optional[str] = None      # 缺槽時的追問提示
    is_slot_refill: bool = False             # 本輪是否為 slot 回填（繼承了上輪任務）


class SlotTracker:

    def __init__(self):
        self.state = SlotState()

    # ------------------------------------------------------------------
    # 回填偵測（B2 步驟）
    # ------------------------------------------------------------------
    def detect_refill(
        self,
        user_input: str,
        task_label: str,
        task_entropy: float,
        task_top_prob: float,
    ) -> Optional[str]:
        """
        判斷本輪輸入是否為 slot 回填（而非新問題）。

        回填條件（同時滿足）：
          A: 輸入短（≤ 8 字）且不含問號
          B: task 分類信心低（top1 prob < 0.5 或 entropy > 0.5）

        Args:
            user_input: 使用者輸入
            task_label: 本輪分類器判定的任務
            task_entropy: 本輪 task_entropy
            task_top_prob: 本輪 task top-1 softmax 機率

        Returns:
            繼承的 task_label（回填時）或 None（跳題 / 無 pending）
        """
        if not self.state.has_pending():
            return None

        text = user_input.strip()

        # 條件 A：輸入短且無問號
        is_short_answer = len(text) <= 8 and "？" not in text and "?" not in text

        if not is_short_answer:
            # 長句或含問號 → 大概率是新問題，清除 pending
            self.state.clear()
            return None

        # 條件 B：task ��類信心低（分類器對短回答沒把握）
        low_confidence = task_top_prob < 0.5 or task_entropy > 0.5

        if low_confidence:
            # 判定為回填：繼承上輪任務
            inherited_task = self.state.active_task
            return inherited_task
        else:
            # 短句但分類器有把握，且判定跟上輪不同 → 跳題
            if task_label != self.state.active_task:
                self.state.clear()
                return None
            # 短句、信心高、跟上輪同任務 → 也視為回填
            return self.state.active_task

    # ------------------------------------------------------------------
    # 槽位檢查（B5 步驟）
    # ------------------------------------------------------------------
    def check_slots(
        self,
        task_label: str,
        available_values: Dict[str, Any],
        is_refill: bool = False,
    ) -> SlotCheckResult:
        """
        檢查任務所需的槽位是否已填滿。

        Args:
            task_label: 任務標籤 (A~N)
            available_values: 目前可用的槽位值
                例：{"region": "台北", "domain_focus": "語言", "child_age": 36, ...}
            is_refill: 本輪是否為 slot 回填

        Returns:
            SlotCheckResult
        """
        required = TASK_SLOTS.get(task_label, [])

        if not required:
            return SlotCheckResult(
                task_label=task_label,
                slot_status="no_slots",
                filled_slots={},
                missing_slots=[],
                is_slot_refill=is_refill,
            )

        filled = {}
        missing = []
        for slot_name in required:
            val = available_values.get(slot_name)
            if val is not None and val != "":
                filled[slot_name] = val
            else:
                missing.append(slot_name)

        # 追問提示（取第一個缺槽的提示）
        followup = None
        if missing:
            first_missing = missing[0]
            task_hints = SLOT_FOLLOWUP_HINTS.get(first_missing, {})
            followup = task_hints.get(task_label)

        status = "all_filled" if not missing else "has_missing"

        return SlotCheckResult(
            task_label=task_label,
            slot_status=status,
            filled_slots=filled,
            missing_slots=missing,
            followup_hint=followup,
            is_slot_refill=is_refill,
        )

    # ------------------------------------------------------------------
    # 更新 pending（G 步驟）
    # ------------------------------------------------------------------
    def update_pending(self, task_label: str, slot_result: SlotCheckResult):
        """
        根據本輪 slot 檢查結果更新 pending 狀態。
        """
        if slot_result.slot_status == "has_missing":
            self.state.active_task = task_label
            self.state.pending_slot = slot_result.missing_slots[0]
            self.state.filled_slots = slot_result.filled_slots
        else:
            # all_filled 或 no_slots → 清除
            self.state.clear()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def load_state(self, state_dict: Optional[dict]):
        self.state = SlotState.from_dict(state_dict) if state_dict else SlotState()

    def save_state(self) -> dict:
        return self.state.to_dict()

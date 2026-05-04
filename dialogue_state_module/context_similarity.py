from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .embedding import TextEncoder, cosine_sim


@dataclass(frozen=True)
class ContextSimConfig:
    """
    - neutral_first_turn: 第一輪沒有 prev 時回傳的中性值
    - bot_max_chars: 用 bot 回覆計算相似度時，最多使用多少字元（避免過長、過泛）
    - bot_keep_tail_chars: 若 bot 回覆很長，保留尾端多少字元（常包含使用者在意的重點/條列結尾）
    - use_max: True => C = max(C_user, C_bot)；False => 可改成加權平均
    """
    neutral_first_turn: float = 0.5

    bot_max_chars: int = 1200
    bot_keep_tail_chars: int = 400

    use_max: bool = True


@dataclass
class ContextSimState:
    prev_user_text: Optional[str] = None
    prev_user_vec: Optional[np.ndarray] = None

    prev_bot_text: Optional[str] = None
    prev_bot_vec: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_user_text = None
        self.prev_user_vec = None
        self.prev_bot_text = None
        self.prev_bot_vec = None


class ContextSimilarity:
    """
    C 的計算來源：
    - C_user = sim(cur_user, prev_user)
    - C_bot  = sim(cur_user, prev_bot)
    最終 C：預設取 max，並回報 source 方便 debug
    """

    def __init__(self, encoder: TextEncoder, cfg: ContextSimConfig):
        self.encoder = encoder
        self.cfg = cfg
        self.state = ContextSimState()

    def _trim_bot_text(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""

        # 太長就截斷：保留前段 + 尾端（避免只剩開頭寒暄，也避免整篇太泛）
        if len(t) <= self.cfg.bot_max_chars:
            return t

        head_len = max(self.cfg.bot_max_chars - self.cfg.bot_keep_tail_chars, 0)
        head = t[:head_len].strip()
        tail = t[-self.cfg.bot_keep_tail_chars :].strip()

        if head and tail:
            return head + "\n...\n" + tail
        return (head or tail).strip()

    def compute(self, cur_user_text: str) -> Dict[str, Any]:
        cur_user_text = (cur_user_text or "").strip()
        cur_vec = self.encoder.encode(cur_user_text)

        # 若沒有任何歷史，回傳 neutral
        if self.state.prev_user_vec is None and self.state.prev_bot_vec is None:
            return {
                "C": self.cfg.neutral_first_turn,
                "C_user": None,
                "C_bot": None,
                "source": "first_turn",
            }

        C_user: Optional[float] = None
        C_bot: Optional[float] = None

        if self.state.prev_user_vec is not None:
            C_user = cosine_sim(cur_vec, self.state.prev_user_vec)

        if self.state.prev_bot_vec is not None:
            C_bot = cosine_sim(cur_vec, self.state.prev_bot_vec)

        # 合成 C
        candidates: list[Tuple[str, float]] = []
        if C_user is not None:
            candidates.append(("prev_user", C_user))
        if C_bot is not None:
            candidates.append(("prev_bot", C_bot))

        if not candidates:
            return {
                "C": self.cfg.neutral_first_turn,
                "C_user": C_user,
                "C_bot": C_bot,
                "source": "no_prev_vec",
            }

        if self.cfg.use_max:
            source, C = max(candidates, key=lambda x: x[1])
        else:
            # 若你之後想改成加權平均，可在這裡調
            # 目前給一個合理預設：user 0.6, bot 0.4（但你現在選 max，所以不會走到）
            w_user = 0.6 if C_user is not None else 0.0
            w_bot = 0.4 if C_bot is not None else 0.0
            denom = w_user + w_bot
            if denom <= 0:
                source, C = max(candidates, key=lambda x: x[1])
            else:
                C = 0.0
                if C_user is not None:
                    C += w_user * C_user
                if C_bot is not None:
                    C += w_bot * C_bot
                C = C / denom
                source = "weighted"

        return {
            "C": float(C),
            "C_user": C_user,
            "C_bot": C_bot,
            "source": source,
        }

    def update(self, cur_user_text: str, cur_bot_text: Optional[str] = None) -> None:
        cur_user_text = (cur_user_text or "").strip()
        self.state.prev_user_text = cur_user_text
        self.state.prev_user_vec = self.encoder.encode(cur_user_text)

        if cur_bot_text is not None:
            trimmed = self._trim_bot_text(cur_bot_text)
            self.state.prev_bot_text = trimmed
            self.state.prev_bot_vec = self.encoder.encode(trimmed)

    def update_bot_only(self, cur_bot_text: str) -> None:
        """
        只更新機器人回覆，不更新用戶輸入。
        用於在用戶輸入已經更新的情況下，只添加機器人回覆。
        
        Args:
            cur_bot_text: 機器人回覆文字
        """
        trimmed = self._trim_bot_text(cur_bot_text)
        self.state.prev_bot_text = trimmed
        self.state.prev_bot_vec = self.encoder.encode(trimmed)

    def reset(self) -> None:
        self.state.reset()
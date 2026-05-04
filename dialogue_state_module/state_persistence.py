# state_persistence.py
# 對話狀態持久化模組 - 解決程式重啟後狀態丟失的問題

from __future__ import annotations

import json
import os
import base64
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from .context_similarity import ContextSimilarity, ContextSimState
from .multi_topic_tracker import MultiTopicTracker, MultiTopicState


@dataclass
class DialogueStateSnapshot:
    """對話狀態快照 - 可序列化的狀態數據"""
    # ContextSimilarity 狀態
    prev_user_text: Optional[str] = None
    prev_user_vec_base64: Optional[str] = None  # numpy array 轉 base64
    prev_bot_text: Optional[str] = None
    prev_bot_vec_base64: Optional[str] = None
    
    # MultiTopicTracker 狀態
    memory_dist: Dict[str, float] = None
    prev_dist: Dict[str, float] = None
    prev_raw_top_domain: Optional[str] = None
    prev_active_domains: list = None  # 新增
    
    # SemanticFlowClassifier 狀態（Scope 沿用用）
    turn_index: int = 0
    prev_scope: Optional[str] = None  # 上一輪的 scope_label
    prev_was_overview: bool = False  # 上一輪是否為整體（供模糊+整體兩條規則）
    
    def __post_init__(self):
        if self.memory_dist is None:
            self.memory_dist = {}
        if self.prev_dist is None:
            self.prev_dist = {}
        if self.prev_active_domains is None:
            self.prev_active_domains = []


def _numpy_to_base64(arr: Optional[np.ndarray]) -> Optional[str]:
    """將 numpy array 轉換為 base64 字符串（包含 shape 和 dtype 信息）"""
    if arr is None:
        return None
    # 保存 shape, dtype 和數據
    info = {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'data': base64.b64encode(arr.tobytes()).decode('utf-8')
    }
    return base64.b64encode(json.dumps(info).encode('utf-8')).decode('utf-8')


def _base64_to_numpy(b64_str: Optional[str]) -> Optional[np.ndarray]:
    """將 base64 字符串轉換回 numpy array"""
    if b64_str is None:
        return None
    try:
        # 解碼包含 shape 和 dtype 的信息
        info_json = base64.b64decode(b64_str.encode('utf-8')).decode('utf-8')
        info = json.loads(info_json)
        
        shape = tuple(info['shape'])
        dtype = np.dtype(info['dtype'])
        bytes_data = base64.b64decode(info['data'].encode('utf-8'))
        
        arr = np.frombuffer(bytes_data, dtype=dtype).reshape(shape)
        return arr
    except Exception as e:
        print(f"[STATE] 反序列化 numpy array 失敗: {e}")
        return None


def save_dialogue_state(
    user_id: int,
    child_id: int,
    context_sim: ContextSimilarity,
    topic_tracker: MultiTopicTracker,
    turn_index: int,
    state_dir: str = "dialogue_states",
    prev_scope: Optional[str] = None,
    prev_was_overview: bool = False,
) -> bool:
    """
    保存對話狀態到文件
    
    Args:
        user_id: 用戶 ID
        child_id: 兒童 ID
        context_sim: ContextSimilarity 實例
        topic_tracker: MultiTopicTracker 實例
        turn_index: 當前輪次
        state_dir: 狀態文件保存目錄
    
    Returns:
        是否成功保存
    """
    try:
        os.makedirs(state_dir, exist_ok=True)
        
        state = context_sim.state
        topic_state = topic_tracker.state
        
        snapshot = DialogueStateSnapshot(
            prev_user_text=state.prev_user_text,
            prev_user_vec_base64=_numpy_to_base64(state.prev_user_vec),
            prev_bot_text=state.prev_bot_text,
            prev_bot_vec_base64=_numpy_to_base64(state.prev_bot_vec),
            memory_dist=dict(topic_state.memory_dist),
            prev_dist=dict(topic_state.prev_dist),
            prev_raw_top_domain=topic_state.prev_raw_top_domain,
            prev_active_domains=list(topic_state.prev_active_domains),
            turn_index=turn_index,
            prev_scope=prev_scope,
            prev_was_overview=prev_was_overview,
        )
        
        filename = os.path.join(state_dir, f"user_{user_id}_child_{child_id}_state.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(asdict(snapshot), f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"[STATE] 保存狀態失敗 (user={user_id}, child={child_id}): {e}")
        return False


def load_dialogue_state(
    user_id: int,
    child_id: int,
    context_sim: ContextSimilarity,
    topic_tracker: MultiTopicTracker,
    state_dir: str = "dialogue_states"
) -> Optional[tuple]:
    """
    從文件加載對話狀態
    
    Args:
        user_id: 用戶 ID
        child_id: 兒童 ID
        context_sim: ContextSimilarity 實例（將被恢復狀態）
        topic_tracker: MultiTopicTracker 實例（將被恢復狀態）
        state_dir: 狀態文件保存目錄
    
    Returns:
        (turn_index, prev_scope, prev_was_overview) 或 None（失敗時）
    """
    try:
        filename = os.path.join(state_dir, f"user_{user_id}_child_{child_id}_state.json")
        
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        snapshot = DialogueStateSnapshot(**data)
        
        # 恢復 ContextSimilarity 狀態
        context_sim.state.prev_user_text = snapshot.prev_user_text
        context_sim.state.prev_user_vec = _base64_to_numpy(snapshot.prev_user_vec_base64)
        context_sim.state.prev_bot_text = snapshot.prev_bot_text
        context_sim.state.prev_bot_vec = _base64_to_numpy(snapshot.prev_bot_vec_base64)
        
        # 恢復 MultiTopicTracker 狀態
        topic_tracker.state.memory_dist = dict(snapshot.memory_dist)
        topic_tracker.state.prev_dist = dict(snapshot.prev_dist)
        topic_tracker.state.prev_raw_top_domain = snapshot.prev_raw_top_domain
        topic_tracker.state.prev_active_domains = list(snapshot.prev_active_domains)
        
        prev_scope = getattr(snapshot, "prev_scope", None)
        prev_was_overview = getattr(snapshot, "prev_was_overview", False)
        return (snapshot.turn_index, prev_scope, prev_was_overview)
    except Exception as e:
        print(f"[STATE] 加載狀態失敗 (user={user_id}, child={child_id}): {e}")
        return None


def delete_dialogue_state(
    user_id: int,
    child_id: int,
    state_dir: str = "dialogue_states"
) -> bool:
    """
    刪除對話狀態文件（例如：對話結束時）
    
    Args:
        user_id: 用戶 ID
        child_id: 兒童 ID
        state_dir: 狀態文件保存目錄
    
    Returns:
        是否成功刪除
    """
    try:
        filename = os.path.join(state_dir, f"user_{user_id}_child_{child_id}_state.json")
        if os.path.exists(filename):
            os.remove(filename)
            return True
        return False
    except Exception as e:
        print(f"[STATE] 刪除狀態失敗 (user={user_id}, child={child_id}): {e}")
        return False


def list_user_states(state_dir: str = "dialogue_states") -> list[tuple[int, int]]:
    """
    列出所有已保存的狀態文件對應的 (user_id, child_id)
    
    Args:
        state_dir: 狀態文件保存目錄
    
    Returns:
        (user_id, child_id) 元組列表
    """
    if not os.path.exists(state_dir):
        return []
    
    states = []
    for filename in os.listdir(state_dir):
        if filename.endswith("_state.json"):
            try:
                # 解析文件名: user_{user_id}_child_{child_id}_state.json
                parts = filename.replace("_state.json", "").split("_")
                if len(parts) >= 4 and parts[0] == "user" and parts[2] == "child":
                    user_id = int(parts[1])
                    child_id = int(parts[3])
                    states.append((user_id, child_id))
            except (ValueError, IndexError):
                continue
    
    return states


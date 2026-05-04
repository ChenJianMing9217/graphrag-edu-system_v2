#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topic_ontology.py
通用主題本體配置（domain-agnostic）
"""

import os
import json
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field

# 預設權重檔路徑（retrieval_module/config/task_section_weights.json）
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_WEIGHTS_JSON_PATH = os.path.join(_CONFIG_DIR, "task_section_weights.json")

DEFAULT_TASK_TO_SECTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    # section keys: assessment, observation, training, suggestion, community_resources, external_gpt
    "A": {"assessment": 0.45, "observation": 0.0,  "training": 0.35, "suggestion": 0.20, "community_resources": 0.0, "external_gpt": 0.0},
    "B": {"assessment": 0.65, "observation": 0.35, "training": 0.0,  "suggestion": 0.0,  "community_resources": 0.0, "external_gpt": 0.0},
    "C": {"assessment": 0.55, "observation": 0.45, "training": 0.0,  "suggestion": 0.0,  "community_resources": 0.0, "external_gpt": 0.0},
    "D": {"assessment": 0.50, "observation": 0.30, "training": 0.20, "suggestion": 0.0,  "community_resources": 0.0, "external_gpt": 0.0},
    "E": {"assessment": 0.0,  "observation": 0.25, "training": 0.40, "suggestion": 0.35, "community_resources": 0.0, "external_gpt": 0.0},
    "F": {"assessment": 0.0,  "observation": 0.30, "training": 0.35, "suggestion": 0.35, "community_resources": 0.0, "external_gpt": 0.0},
    "G": {"assessment": 0.45, "observation": 0.0,  "training": 0.35, "suggestion": 0.20, "community_resources": 0.0, "external_gpt": 0.0},
    "H": {"assessment": 0.0,  "observation": 0.0,  "training": 0.0,  "suggestion": 0.0,  "community_resources": 0.50, "external_gpt": 0.50},
    "I": {"assessment": 0.20, "observation": 0.0,  "training": 0.0,  "suggestion": 0.0,  "community_resources": 0.0, "external_gpt": 0.80},
    "J": {"assessment": 0.20, "observation": 0.25, "training": 0.20, "suggestion": 0.15, "community_resources": 0.0, "external_gpt": 0.20},
    "K": {"assessment": 0.0,  "observation": 0.0,  "training": 0.0,  "suggestion": 0.0,  "community_resources": 0.50, "external_gpt": 0.50},
    "L": {"assessment": 0.30, "observation": 0.0,  "training": 0.45, "suggestion": 0.25, "community_resources": 0.0, "external_gpt": 0.0},
    "M": {"assessment": 0.0,  "observation": 0.0,  "training": 0.0,  "suggestion": 0.40, "community_resources": 0.35, "external_gpt": 0.25},
    "N": {"assessment": 0.60, "observation": 0.20, "training": 0.10, "suggestion": 0.10, "community_resources": 0.0, "external_gpt": 0.0},
}


def load_task_section_weights(weights_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    從 JSON 載入 Task -> Section 權重。若路徑為 None 或讀檔失敗則回傳內建預設。
    """
    path = weights_path or DEFAULT_WEIGHTS_JSON_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULT_TASK_TO_SECTION_WEIGHTS
        return {k: dict(v) for k, v in data.items() if isinstance(v, dict)}
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return DEFAULT_TASK_TO_SECTION_WEIGHTS


@dataclass
class TopicOntology:
    """
    主題本體配置（可替換，適用於不同領域）
    
    - topic: 可對應 subdomain、產品類別、章節、疾病科別、功能模組等
    - section_type: 可對應文件段落類型、節點類型、metadata 類型
    """
    
    # Canonical topic labels（標準主題列表）
    TOPIC_LABELS: List[str] = field(default_factory=lambda: [
        "粗大動作", "精細動作", "感覺統合", "口腔動作", "吞嚥功能",
        "口語理解", "口語表達", "說話", "認知功能", "情緒行為與社會適應功能"
    ])
    
    # Topic aliases（別名映射：alias -> canonical）
    TOPIC_ALIASES: Dict[str, str] = field(default_factory=lambda: {
        "粗大": "粗大動作",
        "大動作": "粗大動作",
        "精細": "精細動作",
        "小動作": "精細動作",
        "感覺": "感覺統合",
        "統合": "感覺統合",
        "口腔": "口腔動作",
        "吞嚥": "吞嚥功能",
        "理解": "口語理解",
        "表達": "口語表達",
        "語言": "說話",
        "認知": "認知功能",
        "情緒": "情緒行為與社會適應功能",
        "社會": "情緒行為與社會適應功能",
        "行為": "情緒行為與社會適應功能"
    })
    
    # Task A–M 到 Section Type 的權重映射（依使用資料優先順序）
    # section_type: assessment=評估結果, observation=行為觀察, training=訓練方向, suggestion=具體建議
    TASK_TO_SECTION_WEIGHTS: Dict[str, Dict[str, float]] = field(default_factory=lambda: DEFAULT_TASK_TO_SECTION_WEIGHTS.copy())
    # 若提供權重檔路徑，建構時會從檔案載入並覆寫 TASK_TO_SECTION_WEIGHTS
    weights_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.weights_path:
            loaded = load_task_section_weights(self.weights_path)
            if loaded:
                object.__setattr__(self, "TASK_TO_SECTION_WEIGHTS", loaded)
    
    # Policy 超參數
    MAX_TOPICS: int = 4  # 最多同時檢索的主題數
    MIN_PER_TOPIC: int = 1  # 每個主題至少檢索的項目數
    MAIN_TOPIC_RATIO: float = 0.7  # 主要主題的配額比例（SOFT_FOCUS 時）
    
    # 不確定性閾值
    TV_TH: float = 0.15  # Topic variance threshold（主題變異閾值）
    MARGIN_TH: float = 0.1  # Margin threshold（邊際閾值）
    SHORT_TEXT_PENALTY_TH: int = 10  # 短文本懲罰閾值（字符數）
    
    def normalize_topic(self, topic_text: str) -> Optional[str]:
        """
        將主題文本標準化為 canonical label
        
        Args:
            topic_text: 原始主題文本
            
        Returns:
            Canonical topic label，如果找不到則返回 None
        """
        # 先檢查是否已經是 canonical
        if topic_text in self.TOPIC_LABELS:
            return topic_text
        
        # 檢查 alias
        if topic_text in self.TOPIC_ALIASES:
            return self.TOPIC_ALIASES[topic_text]
        
        # 模糊匹配（包含關係）
        topic_lower = topic_text.lower()
        for canonical in self.TOPIC_LABELS:
            if topic_lower in canonical.lower() or canonical.lower() in topic_lower:
                return canonical
        
        return None
    
    # 各個分類（Section）的語義描述，用於動態語義權重計算
    SECTION_SEMANTICS: Dict[str, str] = field(default_factory=lambda: {
        "assessment": "標準化評估的量化分數、測驗結果與正式報告內容。專注於數據與評分。不包含任何給家長的回饋、未來的復健課程計畫或感官觀察描述。",
        "observation": "專業人員對兒童表現的質化觀察描述、當前行為現狀與生理徵兆。不包含量化的測驗總分或未來的介入目標清單。",
        "training": "醫療或早療領域的專業介入策略、訓練加強方向與復健目標執行計畫。強調未來的導引方向與療育方案。不包含標準化測驗的原始數據或給家長的日常建議。",
        "suggestion": "給家長、照顧者的日常生活指導策略與家庭練習建議。強調執行層面與環境調整。不包含任何正式評估工具的名稱或醫療等級的介入報告。",
        "community_resources": "社會福利資源、早療機構、補助申請、復健診所、據點查詢、政府方案與社區服務內容。針對個案外的社會支持資源。",
        "external_gpt": "通用育兒知識、醫療定義、DSM-5 診斷規範、常見發展規範與衛教常識。包含社福資源查詢、外部建議或地方資源補充。非針對特定個案報告的通用專業通識。"
    })

    def get_section_weights(self, task: str) -> Dict[str, float]:
        """
        根據 task 獲取 section type 權重
        
        Args:
            task: 任務類型
            
        Returns:
            Section type 權重字典，如果找不到則返回均勻權重
        """
        if task in self.TASK_TO_SECTION_WEIGHTS:
            return self.TASK_TO_SECTION_WEIGHTS[task].copy()
        
        # 預設均勻權重
        default_sections = ["assessment", "observation", "training", "suggestion"]
        return {sec: 1.0 / len(default_sections) for sec in default_sections}

    def get_section_matching_scores(self, query_vector: List[float], text_encoder: Any) -> Dict[str, float]:
        """
        透過向量相似度，計算用戶問題與各分類語義定義的匹配分數
        """
        import numpy as np
        
        # [NEW] 預先編碼定義向量，避免重複執行
        if not hasattr(self, "_def_vectors_cache"):
            self._def_vectors_cache = {}
            for sec, definition in self.SECTION_SEMANTICS.items():
                self._def_vectors_cache[sec] = text_encoder.encode(definition)
        
        def cosine_sim(a, b):
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na <= 0 or nb <= 0: return 0.0
            return float(np.dot(a, b) / (na * nb))

        scores = {}
        # [DEBUG] 檢查 Query 向量的能量 (Norm)，若是 0 則代表 Embedding 失敗
        q_norm = np.linalg.norm(query_vector)
        # print(f"[DEBUG] TopicOntology Query Vector Norm: {q_norm:.4f}")

        for sec, def_vec in self._def_vectors_cache.items():
            similarity = cosine_sim(query_vector, def_vec)
            # 將相似度底限設為 0.01，避免全零輸入導致模型掛掉
            scores[sec] = max(0.01, similarity)
            
        # [DEBUG] 顯示計算出的語義匹配分，協助排查機率卡死問題
        print(f"    [TopicOntology] Semantic Scores for this Query: {scores}")
            
        return scores


# 預設實例（可替換；未傳 weights_path 時使用預設權重檔路徑，讀檔失敗則用內建權重）
default_ontology = TopicOntology(weights_path=DEFAULT_WEIGHTS_JSON_PATH)

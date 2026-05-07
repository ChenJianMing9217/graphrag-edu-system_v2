from typing import List, Dict, Any, Optional
import numpy as np
from .types import CandidateNode

class Reranker:
    """
    Reranks candidate nodes based on semantic, structural, and contextual signals.
    """
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder

    def rerank(
        self,
        candidates: List[CandidateNode],
        user_query: str,
        config: Dict[str, float],
        task_label: str = None,
        domain_distribution: Dict[str, float] = None,
        query_vec: Optional[np.ndarray] = None,
    ) -> List[CandidateNode]:
        if not candidates:
            return []

        # 0. Weight Clamping (防止 RL 權重坍縮)
        raw_sem = config.get("semantic_weight", 0.6)
        raw_str = config.get("structural_weight", 0.2)
        raw_ctx = config.get("context_weight", 0.2)
        semantic_weight = max(raw_sem, 0.25)
        structural_weight = min(raw_str, 0.50)
        context_weight = max(raw_ctx, 0.10)
        total = semantic_weight + structural_weight + context_weight
        semantic_weight /= total
        structural_weight /= total
        context_weight /= total

        # 1. Semantic Scoring — 複用傳入的 query_vec，batch encode 候選文本
        if query_vec is None:
            query_vec = self.text_encoder.encode(user_query)

        # Batch encode 所有候選節點（利用 cache 避免重複呼叫）
        cand_texts = [c.text for c in candidates]
        cand_vecs = self.text_encoder.encode_many(cand_texts)  # [N, D]

        for i, cand in enumerate(candidates):
            cand.score = float(query_vec @ cand_vecs[i]) * semantic_weight

            # 2. Structural Boosting (Task-based Label Boost)
            if task_label:
                boost_labels = self._get_boost_labels(task_label)
                if cand.label in boost_labels:
                    cand.score += 0.1 * structural_weight

            # 3. Path-Aware Boosting (Domain Distribution Boost)
            if domain_distribution:
                subdomain = cand.properties.get("subdomain")
                if subdomain:
                    prob = domain_distribution.get(subdomain, 0.0)
                    cand.score += (prob * 0.2) * context_weight

            # 4. Label Penalties
            cand_text = cand.text
            if "(未選用)" in cand_text or "未選取" in cand_text:
                cand.score *= 0.01
            elif "(非重點)" in cand_text:
                cand.score *= 0.3

            # 5. 臨床增強節點權重提升
            if cand.label == "ClinicalNorm":
                cand.score += 0.4

            # 6. [NEW 2026-05-07] 多來源並存策略：SQL 與 GPT 同等高權重
            #    SubsidyInfo (官方補助公告) 最高 → LocalResource (sfaa/社區)、ExternalGPT (即時 web)
            #    與 ClinicalNorm (常模) 形成 4 大互補來源,確保都進得了 prompt
            if cand.label == "SubsidyInfo":
                cand.score += 0.6   # 官方補助文件,金額/條件具體
            elif cand.label == "LocalResource":
                cand.score += 0.5   # SQL 在地機構/據點
            elif cand.label == "ExternalGPT":
                cand.score += 0.5   # 即時 web search,常含治療所名單(SQL 沒有)

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _get_boost_labels(self, task_label: str) -> List[str]:
        mapping = {
            "A": ["Summary", "Meta"],
            "B": ["Assessment", "Score"],
            "C": ["Observation", "Assessment"],
            "D": ["Assessment", "Observation"],
            "E": ["TrainingDirection", "Recommendation"],
            "F": ["TrainingDirection", "Observation"],
            "G": ["Assessment", "Recommendation"],
            # [NEW] H/K 任務 boost SQL 在地資源節點（與第 6 步骤的 label boost 互補）
            "H": ["LocalResource", "SubsidyInfo"],
            "K": ["SubsidyInfo", "LocalResource"],
            "L": ["Assessment", "TrainingDirection"],
            "M": ["Recommendation", "GeneralRecommendation"],
            "N": ["Assessment", "Score"],
        }
        return mapping.get(task_label, [])

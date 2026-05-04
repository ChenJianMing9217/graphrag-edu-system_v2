# 根據輸入文字，計算各 domain 的相似度分數，並選出 active domains

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from .embedding import TextEncoder, cosine_sim

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        temperature = 1.0
    z = x / float(temperature)
    z = z - np.max(z)  # for numerical stability
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        return np.ones_like(x) / max(1, len(x))
    return e / s


def normalized_entropy(p: np.ndarray) -> float:
    """
    回傳 [0,1]：
    - 0：非常集中（很確定某個 domain）
    - 1：非常平均（很不確定/模糊）
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p)
    h = -np.sum(p * np.log(p))
    h_max = np.log(len(p)) if len(p) > 1 else 1.0
    return float(h / h_max) if h_max > 0 else 0.0


@dataclass(frozen=True)
class DomainRouterConfig:
    """
    active_domains 的集合式門檻：
    - active_prob_th: 絕對門檻
    - active_ratio_th: 相對門檻（>= top1 * ratio）
    - min_active_domains: 至少保留幾個（保底 topK）
    - max_active_domains: 最多保留幾個（避免爆炸）
    """
    temperature: float = 0.04

    active_prob_th: float = 0.2
    active_ratio_th: float = 0.60
    min_active_domains: int = 1
    max_active_domains: int = 4

    # 關鍵字加權 (Keyword Boosting)
    keyword_boosts: Dict[str, List[str]] = field(default_factory=lambda: {
        "整體概況": ["整體", "總覽", "全部", "總結", "概況", "整份", "彙整"]
    })
    boost_value: float = 0.05


@dataclass
class DomainResult:
    dist: Dict[str, float]

    top_domain: str
    top_prob: float
    entropy: float

    active_domains: List[str]
    active_domain_probs: Dict[str, float]


class DomainRouter:
    def __init__(
        self,
        encoder: TextEncoder,
        domains: List[str],
        anchor_vecs: Dict[str, List[np.ndarray]],
        cfg: DomainRouterConfig,
    ):
        self.encoder = encoder
        self.domains = list(domains)
        self.anchor_vecs = dict(anchor_vecs)
        self.cfg = cfg

        # sanity check
        for d in self.domains:
            if d not in self.anchor_vecs:
                raise ValueError(f"Missing anchor vectors for domain: {d}")
            if not isinstance(self.anchor_vecs[d], list) or len(self.anchor_vecs[d]) == 0:
                raise ValueError(f"Anchor vectors for domain {d} must be a non-empty list")

    def score_domains(self, text: str, query_vec: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        計算用戶查詢與各領域的相似度分數。
        對於每個領域，使用 Max Pooling（取最大值）作為最終分數。
        """
        u = query_vec if query_vec is not None else self.encoder.encode(text)
        scores: List[Tuple[str, float]] = []
        
        for d in self.domains:
            # 計算與該領域所有 anchor 向量的相似度，取最大值（Max Pooling）
            anchor_vecs = self.anchor_vecs[d]
            max_sim = max(cosine_sim(u, vec) for vec in anchor_vecs)
            
            # [加權] 關鍵字 Boosting：若匹配到高意圖關鍵字，增加固定偏移量
            if d in self.cfg.keyword_boosts:
                boost_keywords = self.cfg.keyword_boosts[d]
                if any(kw in text for kw in boost_keywords):
                    max_sim += self.cfg.boost_value
                    
            scores.append((d, float(max_sim)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _select_active_domains(self, ranked: List[Tuple[str, float]], dist: Dict[str, float]) -> Tuple[List[str], Dict[str, float]]:
        top1, p_top = ranked[0][0], dist[ranked[0][0]]

        active = [
            d for d, _ in ranked
            if (dist[d] >= self.cfg.active_prob_th) or (dist[d] >= p_top * self.cfg.active_ratio_th)
        ]

        if len(active) < self.cfg.min_active_domains:
            active = [d for d, _ in ranked[: self.cfg.min_active_domains]]

        if len(active) > self.cfg.max_active_domains:
            active = active[: self.cfg.max_active_domains]

        active_probs = {d: float(dist[d]) for d in active}
        return active, active_probs

    def predict(self, text: str, query_vec: Optional[np.ndarray] = None) -> DomainResult:
        ranked = self.score_domains(text, query_vec=query_vec)
        raw_scores = np.array([s for _, s in ranked], dtype=np.float64)

        probs = softmax(raw_scores, temperature=self.cfg.temperature)
        dist = {ranked[i][0]: float(probs[i]) for i in range(len(ranked))}

        top_domain = ranked[0][0]
        top_prob = float(dist[top_domain])

        ent = normalized_entropy(np.array([dist[d] for d, _ in ranked], dtype=np.float64))

        active_domains, active_domain_probs = self._select_active_domains(ranked, dist)

        return DomainResult(
            dist=dist, # 領域分布
            top_domain=top_domain, # 最高相似度領域
            top_prob=top_prob, # 最高相似度領域機率
            entropy=ent, # 熵
            active_domains=active_domains, # 活躍領域
            active_domain_probs=active_domain_probs, # 活躍領域機率
        )

# task_scope_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import json

from dialogue_state_module.embedding import _encode_with_cache

# 預設設定檔路徑（與本模組同目錄下的 config/task_scope_prototypes.jsonl）
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
DEFAULT_PROTOTYPES_JSONL_PATH = os.path.join(_CONFIG_DIR, "task_scope_prototypes.jsonl")

def load_prototypes_from_jsonl(jsonl_path: Optional[str] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load task and scope prototypes from JSONL file.
    """
    task_prototypes: Dict[str, List[str]] = {}
    scope_prototypes: Dict[str, List[str]] = {}
    path = jsonl_path or DEFAULT_PROTOTYPES_JSONL_PATH
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                label = obj['label']
                examples = obj['examples']
                if obj['type'] == 'task':
                    task_prototypes[label] = examples
                elif obj['type'] == 'scope':
                    scope_prototypes[label] = examples
                    
        if not task_prototypes:
            raise ValueError("No task prototypes found in JSONL")
            
    except Exception as e:
        print(f"[TaskScopeClassifier] Error loading {path}: {e}")
        # 極簡備用方案，確保系統不崩潰
        task_prototypes = {"A": ["重點整理"]}
        scope_prototypes = {"S_overview": ["整體狀況"]}
    
    return task_prototypes, scope_prototypes

TASK_NAME_ZH = {
    "A": "報告總覽與閱讀順序",
    "B": "分數/量表/百分位解讀",
    "C": "臨床觀察與表現解讀",
    "D": "能力剖面（優勢/需求/優先順序）",
    "E": "在家訓練怎麼做",
    "F": "融入日常作息的練習",
    "G": "是否需要早療/成效追蹤",
    "H": "轉介與在地資源",
    "I": "報告分享/隱私與安全",
    "J": "與學校合作",
    "K": "補助/福利/申請",
    "L": "後續追蹤/再評估",
    "M": "家長情緒支持與家庭協作",
    "N": "進步查詢",
}

SCOPE_NAME_ZH = {
    "S_overview": "Overview(整體)",
    "S_domain": "Domain(單領域)",
    "S_multi_domain": "Multi-Domain(多領域)",
}

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom

def _embed_texts(embedder: Any, texts: List[str]) -> np.ndarray:
    """
    Try common embedding APIs.
    """
    if hasattr(embedder, "encode_many"):
        vec = embedder.encode_many(texts)
    elif hasattr(embedder, "encode"):
        try:
            vec = embedder.encode(texts)
        except (AttributeError, TypeError):
            if hasattr(embedder, "encode_many"):
                vec = embedder.encode_many(texts)
            else:
                raise TypeError(f"embedder.encode() does not accept list, and encode_many() not available")
    elif hasattr(embedder, "embed"):
        vec = embedder.embed(texts)
    elif callable(embedder):
        vec = embedder(texts)
    else:
        raise TypeError("Unsupported embedder: expected .encode_many/.encode/.embed or callable.")
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec[None, :]
    return vec

def _normalized_entropy(probs: np.ndarray) -> float:
    """計算 [0,1] 歸一化熵：0=非常集中，1=完全均勻"""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p)
    h = -np.sum(p * np.log(p))
    h_max = np.log(len(p)) if len(p) > 1 else 1.0
    return float(h / h_max) if h_max > 0 else 0.0


@dataclass
class PredictResult:
    label: str
    score: float
    dist: Dict[str, float]
    entropy: float = 0.0        # 任務分布歸一化熵 [0,1]
    raw_sims: Optional[Dict[str, float]] = None  # 原始 cosine similarity（供 topk 使用）

class PrototypeClassifier:
    def __init__(self, embedder: Any, prototypes: Dict[str, List[str]]):
        self.embedder = embedder
        self.prototypes = prototypes
        self.labels = list(prototypes.keys())
        self.proto_texts = [prototypes[k] for k in self.labels]

        # Build prototype vectors (mean of sentence embeddings)
        # 使用磁碟快取：將所有原型句子展平編碼，再按類別分組取平均
        all_sents = []
        label_ranges = []  # [(start, end), ...]
        for sents in self.proto_texts:
            start = len(all_sents)
            all_sents.extend(sents)
            label_ranges.append((start, len(all_sents)))

        all_embs = _encode_with_cache(self.embedder, all_sents, "task_prototypes")
        all_embs = _l2_normalize(all_embs)

        proto_vecs = []
        for start, end in label_ranges:
            proto = all_embs[start:end].mean(axis=0, keepdims=True)
            proto = _l2_normalize(proto)[0]
            proto_vecs.append(proto)
        self.proto_mat = np.stack(proto_vecs, axis=0).astype(np.float32)  # [K, D]

    def predict(self, text: str, query_vec: Optional[np.ndarray] = None) -> PredictResult:
        if query_vec is not None:
            q = _l2_normalize(query_vec.reshape(1, -1))[0]
        else:
            q = _l2_normalize(_embed_texts(self.embedder, [text]))[0]  # [D]
        sims = self.proto_mat @ q  # cosine since normalized
        best_idx = int(np.argmax(sims))
        best_label = self.labels[best_idx]
        best_score = float(sims[best_idx])

        # 原始 cosine similarity（供 topk 差值判斷）
        raw_sims = {self.labels[i]: float(sims[i]) for i in range(len(self.labels))}

        temp = 12.0
        logits = sims * temp
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        dist = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}

        entropy = _normalized_entropy(probs)

        return PredictResult(label=best_label, score=best_score, dist=dist,
                             entropy=entropy, raw_sims=raw_sims)

class TaskScopeClassifier:
    def __init__(
        self,
        embedder: Any,
        task_prototypes: Dict[str, List[str]] = None,
        prototypes_jsonl: Optional[str] = None,
    ):
        if task_prototypes is None:
            path = prototypes_jsonl or DEFAULT_PROTOTYPES_JSONL_PATH
            loaded_task, _ = load_prototypes_from_jsonl(path)
            task_prototypes = loaded_task
        
        self.task_clf = PrototypeClassifier(embedder, task_prototypes)

    def predict_task(self, text: str, query_vec: Optional[np.ndarray] = None) -> PredictResult:
        return self.task_clf.predict(text, query_vec=query_vec)

    def predict_task_topk(
        self,
        text: str,
        k: int = 2,
        max_sim_gap: float = 0.05,
        query_vec: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        返回多個 task 標籤（當問題跨多任務時）。
        使用原始 cosine similarity 差值判斷，不受 softmax 溫度影響。
        觸發條件：top1_sim - topN_sim <= max_sim_gap。
        """
        result = self.task_clf.predict(text, query_vec=query_vec)
        if not result.raw_sims:
            return [result.label]

        sorted_sims = sorted(result.raw_sims.items(), key=lambda x: x[1], reverse=True)
        top1_label, top1_sim = sorted_sims[0]

        active = [top1_label]
        for label, sim in sorted_sims[1:k]:
            if top1_sim - sim > max_sim_gap:
                break
            active.append(label)
        return active

def format_topk(dist: Dict[str, float], name_map: Dict[str, str], k: int = 2) -> str:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
    return ", ".join([f"{name_map.get(lbl, lbl)}={p:.2f}" for lbl, p in items])

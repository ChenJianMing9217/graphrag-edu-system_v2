from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

class SearchOperationType(Enum):
    SUBDOMAIN_FETCH = "SUBDOMAIN_FETCH"
    SUMMARY_FETCH = "SUMMARY_FETCH"
    META_FETCH = "META_FETCH"
    MYSQL_RESOURCE_FETCH = "MYSQL_RESOURCE_FETCH"
    CLINICAL_FETCH = "CLINICAL_FETCH"
    GPT_FETCH = "GPT_FETCH"

@dataclass
class SearchOperation:
    op_type: SearchOperationType
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

@dataclass
class SearchStrategy:
    operations: List[SearchOperation] = field(default_factory=list)
    rerank_config: Dict[str, float] = field(default_factory=dict)
    semantic_section_scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

@dataclass
class CandidateNode:
    node_id: str
    label: str
    text: str
    properties: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

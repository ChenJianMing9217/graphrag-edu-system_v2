"""
完整訊號抽取：把線上對話模組（DomainRouter / TaskScopeClassifier /
ContextSimilarity / MultiTopicTracker）真的串起來跑 1758 筆 CSV，
產出每筆 turn 的「全套」訊號 ─ 含 7d legacy + 12d v2，共 ~22 鍵。

讀: rl_pipeline/dataset/memory_decision_final_combined_1758.csv
寫: rl_pipeline/dataset/memory_v3_full_signals.jsonl
    rl_pipeline/dataset/memory_v3_full_signals.meta.json

注意：
- 資料只有 user_query，沒有 assistant_reply → context_sim 只能用 prev_user 比對
  （線上是 max(prev_user, prev_bot)），會比線上略低
- topic_overlap 採用 MultiTopicTracker 的 EMA-decay 公式（與線上一致）
- 每換一個 session_id 就 reset 所有 stateful tracker

使用：
  python rl_pipeline/scripts/build_v3_full_signals.py
"""
from __future__ import annotations
import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dialogue_state_module.embedding import (
    TextEncoder, encode_anchors, encode_overview_anchors
)
from dialogue_state_module.domain_anchors import (
    DOMAINS, DOMAIN_ANCHORS, OVERVIEW_ANCHORS
)
from dialogue_state_module.domain_router import DomainRouter, DomainRouterConfig
from dialogue_state_module.task_scope_classifier import (
    TaskScopeClassifier, load_prototypes_from_jsonl
)
from dialogue_state_module.context_similarity import (
    ContextSimilarity, ContextSimConfig
)
from dialogue_state_module.multi_topic_tracker import (
    MultiTopicTracker, MultiTopicConfig, total_variation_distance
)
from dialogue_state_module.feature_v2 import extract_memory_features_v2

CSV_PATH = ROOT / "rl_pipeline/dataset/memory_decision_final_combined_1758.csv"
OUT_PATH = ROOT / "rl_pipeline/dataset/memory_v3_full_signals.jsonl"
META_PATH = ROOT / "rl_pipeline/dataset/memory_v3_full_signals.meta.json"

LABEL_MAP = {"STAY": 0, "REFRESH": 1, "HYBRID": 0}  # CLARIFY 跳過


def init_pipeline():
    print("[Init] Loading encoder + 4 modules ...")
    encoder = TextEncoder()
    encoded_domain_anchors = encode_anchors(encoder, DOMAIN_ANCHORS, DOMAINS)
    _ = encode_overview_anchors(encoder, OVERVIEW_ANCHORS)

    domain_router = DomainRouter(
        encoder=encoder,
        domains=DOMAINS,
        anchor_vecs=encoded_domain_anchors,
        cfg=DomainRouterConfig(),
    )

    task_protos, _ = load_prototypes_from_jsonl()
    task_clf = TaskScopeClassifier(embedder=encoder, task_prototypes=task_protos)

    print(f"[Init] Done. Domains={len(DOMAINS)}, Tasks={len(task_protos)}")
    return encoder, domain_router, task_clf


def load_csv_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["turn_id"] = int(r["turn_id"])
            except (TypeError, ValueError):
                continue
            rows.append(r)
    rows.sort(key=lambda x: (x["session_id"], x["turn_id"]))
    return rows


def task_tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    if not p or not q:
        return 0.0
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def main():
    encoder, domain_router, task_clf = init_pipeline()
    rows = load_csv_rows(CSV_PATH)
    print(f"[Load] {len(rows)} rows")

    label_counts = {"STAY": 0, "REFRESH": 0, "HYBRID": 0, "CLARIFY": 0, "OTHER": 0}
    written = 0
    skipped_clarify = 0
    skipped_empty = 0
    t0 = time.time()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(OUT_PATH, "w", encoding="utf-8")

    cur_session = None
    # session-scoped trackers
    ctx_sim = ContextSimilarity(encoder=encoder, cfg=ContextSimConfig())
    topic_tracker = MultiTopicTracker(cfg=MultiTopicConfig())
    # session-scoped prev caches (for v2 features)
    prev_top_domain: Optional[str] = None
    prev_task: Optional[str] = None
    prev_task_dist: Optional[Dict[str, float]] = None
    prev_active_domains: List[str] = []
    turn_index_in_session = 0

    for idx, r in enumerate(rows):
        session_id = r["session_id"]
        turn_id = r["turn_id"]
        user_query = (r.get("user_query") or "").strip()
        label = (r.get("expected_memory_decision") or "").strip().upper()

        if session_id != cur_session:
            cur_session = session_id
            ctx_sim.reset()
            topic_tracker = MultiTopicTracker(cfg=MultiTopicConfig())
            prev_top_domain = None
            prev_task = None
            prev_task_dist = None
            prev_active_domains = []
            turn_index_in_session = 0

        if not user_query:
            skipped_empty += 1
            continue

        label_counts[label if label in label_counts else "OTHER"] = (
            label_counts.get(label, 0) + 1
        )

        if label == "CLARIFY":
            skipped_clarify += 1
            continue
        if label not in LABEL_MAP:
            continue

        turn_index_in_session += 1

        # 1) embed once
        try:
            query_vec = encoder.encode(user_query)
        except Exception as e:
            print(f"[Warn] encode fail s={session_id} t={turn_id}: {e}")
            continue

        # 2) DomainRouter
        dom_res = domain_router.predict(user_query, query_vec=query_vec)
        cur_active_domains = list(dom_res.active_domains)
        is_multi_domain = len(cur_active_domains) >= 2
        top3_domains = [d for d, _ in sorted(dom_res.dist.items(), key=lambda kv: -kv[1])[:3]]

        # 3) TaskScopeClassifier
        task_res = task_clf.predict_task(user_query, query_vec=query_vec)
        cur_task_dist = task_res.dist
        cur_task = task_res.label
        task_top1_drop = 0.0
        if prev_task and prev_task_dist and cur_task_dist:
            prev_p = float(prev_task_dist.get(prev_task, 0.0))
            cur_p = float(cur_task_dist.get(prev_task, 0.0))
            task_top1_drop = max(0.0, prev_p - cur_p)
        tv_task = task_tv_distance(prev_task_dist or {}, cur_task_dist) if prev_task_dist else 0.0

        # 4) ContextSimilarity（cur vs prev_user/prev_bot；offline 沒 bot → 只 prev_user）
        ctx_info = ctx_sim.compute(user_query)
        context_sim_val = float(ctx_info.get("C", 0.5))

        # 5) MultiTopicTracker（domain dist 序列 → topic_overlap / tv_distance_domain）
        mt_res = topic_tracker.check_topic_continuation(
            cur_dist=dict(dom_res.dist),
            cur_raw_top_domain=dom_res.top_domain,
            confidence=float(dom_res.top_prob),
            cur_active_domains=cur_active_domains,
            prev_active_domains=prev_active_domains or None,
        )
        topic_overlap = float(mt_res.get("topic_overlap", 0.0))
        tv_distance_domain = float(mt_res.get("tv_distance", 0.0))
        active_domain_coverage = float(mt_res.get("active_domain_coverage", 0.0))

        # 6) v2 12d features
        feats_v2 = extract_memory_features_v2(
            user_query=user_query,
            domain_entropy=dom_res.entropy,
            cur_top_domain=dom_res.top_domain,
            cur_top3_domains=top3_domains,
            prev_top_domain=prev_top_domain,
            cur_task_dist=cur_task_dist,
            prev_task_dist=prev_task_dist,
            prev_task=prev_task,
            tv_distance_raw=tv_task,
        )

        # 7) 7d legacy 訊號（與線上 MemoryAgent 對齊）
        q_len = len(user_query)
        feats_7d = {
            "entropy": float(dom_res.entropy),
            "tv_distance": tv_distance_domain,        # 線上是 domain TV，不是 task TV
            "topic_overlap": topic_overlap,
            "context_sim": context_sim_val,
            "turn_index_norm": min(turn_index_in_session / 10.0, 1.0),
            "query_len_norm": min(q_len / 30.0, 1.0),
            "is_multi_domain": int(is_multi_domain),
        }

        out_obj = {
            "session_id": session_id,
            "turn_id": turn_id,
            "user_query": user_query,
            "label_orig": label,
            "label": "REFRESH" if LABEL_MAP[label] == 1 else "STAY",
            "label_idx": LABEL_MAP[label],
            "features_7d_legacy": feats_7d,
            "features_v2": feats_v2,
            "ctx": {
                "top_domain": dom_res.top_domain,
                "top_prob": float(dom_res.top_prob),
                "active_domains": cur_active_domains,
                "is_multi_domain": is_multi_domain,
                "domain_entropy": float(dom_res.entropy),
                "top3_domains": top3_domains,
                "task_label": cur_task,
                "task_top_prob": float(task_res.score),
                "task_entropy": float(task_res.entropy),
                "tv_distance_task": float(tv_task),
                "tv_distance_domain": tv_distance_domain,
                "topic_overlap": topic_overlap,
                "context_sim": context_sim_val,
                "context_sim_source": ctx_info.get("source"),
                "active_domain_coverage": active_domain_coverage,
                "prev_top_domain": prev_top_domain,
                "prev_task": prev_task,
                "prev_active_domains": prev_active_domains,
                "ground_truth_domain": r.get("ability_domain"),
                "ground_truth_task": r.get("task_type"),
                "dataset_source": r.get("dataset_source"),
            },
        }
        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        written += 1

        # 8) 更新所有 prev / state
        ctx_sim.update(user_query, cur_bot_text=None)  # offline 無 bot
        prev_top_domain = dom_res.top_domain
        prev_task = cur_task
        prev_task_dist = cur_task_dist
        prev_active_domains = cur_active_domains

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(1e-6, elapsed)
            print(f"  ... {idx+1}/{len(rows)}  ({rate:.1f} rows/s, written={written})")

    fout.close()

    meta = {
        "csv_path": str(CSV_PATH),
        "out_path": str(OUT_PATH),
        "total_rows": len(rows),
        "written": written,
        "skipped_empty": skipped_empty,
        "skipped_clarify": skipped_clarify,
        "label_counts_orig": label_counts,
        "label_map": LABEL_MAP,
        "feature_keys_7d_legacy": [
            "entropy", "tv_distance", "topic_overlap", "context_sim",
            "turn_index_norm", "query_len_norm", "is_multi_domain",
        ],
        "feature_keys_12d_v2": [
            "prev_top_eq_current_raw_top", "prev_top_in_current_top3",
            "ambiguous_followup_score", "followup_kw_present",
            "switch_kw_present", "tv_distance_raw", "task_top1_drop",
            "query_len_norm",
            "domain_kw_present", "query_len_chars",
            "has_question_mark", "domain_entropy_raw",
        ],
        "elapsed_sec": time.time() - t0,
        "limitations": [
            "context_sim: offline 沒有 assistant_reply, 只比對 prev_user, 比線上略低",
            "topic_overlap / tv_distance_domain: 與線上一致 (MultiTopicTracker EMA)",
            "其他模組與線上完全一致",
        ],
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"[Done] wrote {written} samples")
    print(f"       elapsed {meta['elapsed_sec']:.1f}s")
    print(f"       label counts (orig): {label_counts}")
    print(f"[Meta] {META_PATH.name}")


if __name__ == "__main__":
    main()

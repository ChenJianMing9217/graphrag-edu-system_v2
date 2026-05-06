"""
Build v2 Memory Agent training dataset (12d features)

輸入: rl_pipeline/dataset/memory_decision_final_combined_1758.csv
輸出: rl_pipeline/dataset/memory_v2_dataset_12d.jsonl

流程：
1. 讀 CSV，依 session_id+turn_id 排序
2. 啟動 TextEncoder + DomainRouter + TaskScopeClassifier
3. 逐 session 遍歷，維護 prev_top_domain / prev_task / prev_task_dist / prev_query
4. 每筆 turn：
   - encode 一次 user_query → query_vec（domain & task 共用）
   - DomainRouter.predict(text, query_vec) → dist / top_domain / entropy
   - TaskScopeClassifier.predict_task(text, query_vec) → task_dist / task_label
   - 計算 TV distance（task dist 變化）
   - extract_memory_features_v2(...) → 12d
5. 標籤處理：CLARIFY 丟掉，HYBRID→STAY（領域延續）
6. 輸出 JSONL

使用：
  python rl_pipeline/scripts/build_v2_dataset.py
"""
from __future__ import annotations
import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

# Project root on path
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
from dialogue_state_module.feature_v2 import extract_memory_features_v2

CSV_PATH = ROOT / "rl_pipeline/dataset/memory_decision_final_combined_1758.csv"
OUT_PATH = ROOT / "rl_pipeline/dataset/memory_v2_dataset_12d.jsonl"
META_PATH = ROOT / "rl_pipeline/dataset/memory_v2_dataset_12d.meta.json"

# 標籤映射：CLARIFY 丟、HYBRID→STAY（domain 延續、retrieval 不重啟）
LABEL_MAP = {"STAY": 0, "REFRESH": 1, "HYBRID": 0}  # CLARIFY 不在此 → 自動跳過


def tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """TV(p,q) = 0.5 * sum |p_i - q_i|"""
    if not p or not q:
        return 0.0
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def init_pipeline():
    print("[Init] Loading TextEncoder + DomainRouter + TaskScopeClassifier ...")
    encoder = TextEncoder()

    encoded_domain_anchors = encode_anchors(encoder, DOMAIN_ANCHORS, DOMAINS)
    _ = encode_overview_anchors(encoder, OVERVIEW_ANCHORS)  # warm cache

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


def load_csv_rows(csv_path: Path) -> List[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["turn_id"] = int(r["turn_id"])
            except (TypeError, ValueError):
                continue
            rows.append(r)
    # 排序：先 session，再 turn
    rows.sort(key=lambda x: (x["session_id"], x["turn_id"]))
    return rows


def main():
    encoder, domain_router, task_clf = init_pipeline()
    rows = load_csv_rows(CSV_PATH)
    print(f"[Load] {len(rows)} rows from {CSV_PATH.name}")

    # 統計
    label_counts = {"STAY": 0, "REFRESH": 0, "HYBRID": 0, "CLARIFY": 0, "OTHER": 0}
    skipped_clarify = 0
    skipped_empty = 0
    written = 0
    t0 = time.time()

    # 逐 session 處理：每換一個 session 就 reset prev_*
    cur_session = None
    prev_top_domain: Optional[str] = None
    prev_task: Optional[str] = None
    prev_task_dist: Optional[Dict[str, float]] = None
    prev_query: Optional[str] = None

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = open(OUT_PATH, "w", encoding="utf-8")

    for idx, r in enumerate(rows):
        session_id = r["session_id"]
        turn_id = r["turn_id"]
        user_query = (r.get("user_query") or "").strip()
        label = (r.get("expected_memory_decision") or "").strip().upper()

        if session_id != cur_session:
            cur_session = session_id
            prev_top_domain = None
            prev_task = None
            prev_task_dist = None
            prev_query = None

        if not user_query:
            skipped_empty += 1
            continue

        label_counts[label if label in label_counts else "OTHER"] = (
            label_counts.get(label, 0) + 1
        )

        # 跳過 CLARIFY
        if label == "CLARIFY":
            skipped_clarify += 1
            continue
        if label not in LABEL_MAP:
            continue  # OTHER 也跳過

        # 1) embed
        try:
            query_vec = encoder.encode(user_query)
        except Exception as e:
            print(f"[Warn] encode fail session={session_id} turn={turn_id}: {e}")
            continue

        # 2) domain
        dom_res = domain_router.predict(user_query, query_vec=query_vec)
        # top3 (依機率排序)
        top3 = sorted(dom_res.dist.items(), key=lambda kv: -kv[1])[:3]
        top3_domains = [d for d, _ in top3]

        # 3) task
        task_res = task_clf.predict_task(user_query, query_vec=query_vec)
        cur_task_dist = task_res.dist
        cur_task = task_res.label

        # 4) TV distance（task dist 變化；無前一輪 → 0）
        tv = tv_distance(prev_task_dist or {}, cur_task_dist) if prev_task_dist else 0.0

        # 5) v2 features
        feats = extract_memory_features_v2(
            user_query=user_query,
            domain_entropy=dom_res.entropy,
            cur_top_domain=dom_res.top_domain,
            cur_top3_domains=top3_domains,
            prev_top_domain=prev_top_domain,
            cur_task_dist=cur_task_dist,
            prev_task_dist=prev_task_dist,
            prev_task=prev_task,
            tv_distance_raw=tv,
        )

        out_obj = {
            "session_id": session_id,
            "turn_id": turn_id,
            "user_query": user_query,
            "label_orig": label,                    # STAY / REFRESH / HYBRID
            "label": "REFRESH" if LABEL_MAP[label] == 1 else "STAY",
            "label_idx": LABEL_MAP[label],
            "features_v2": feats,
            "ctx": {
                "top_domain": dom_res.top_domain,
                "top_prob": float(dom_res.top_prob),
                "domain_entropy": float(dom_res.entropy),
                "top3_domains": top3_domains,
                "task_label": cur_task,
                "task_top_prob": float(task_res.score),
                "tv_distance": float(tv),
                "prev_top_domain": prev_top_domain,
                "prev_task": prev_task,
                "ground_truth_domain": r.get("ability_domain"),
                "ground_truth_task": r.get("task_type"),
                "dataset_source": r.get("dataset_source"),
            },
        }
        fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        written += 1

        # 6) 更新 prev_*
        prev_top_domain = dom_res.top_domain
        prev_task = cur_task
        prev_task_dist = cur_task_dist
        prev_query = user_query

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(1e-6, elapsed)
            print(f"  ... {idx+1}/{len(rows)}  ({rate:.1f} rows/s, written={written})")

    fout.close()

    # 寫 meta
    meta = {
        "csv_path": str(CSV_PATH),
        "out_path": str(OUT_PATH),
        "total_rows": len(rows),
        "written": written,
        "skipped_empty": skipped_empty,
        "skipped_clarify": skipped_clarify,
        "label_counts_orig": label_counts,
        "label_map": LABEL_MAP,
        "feature_keys_12d": [
            "prev_top_eq_current_raw_top",
            "prev_top_in_current_top3",
            "ambiguous_followup_score",
            "followup_kw_present",
            "switch_kw_present",
            "tv_distance_raw",
            "task_top1_drop",
            "query_len_norm",
            # 4 attributes also recorded
            "domain_kw_present",
            "query_len_chars",
            "has_question_mark",
            "domain_entropy_raw",
        ],
        "elapsed_sec": time.time() - t0,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"[Done] wrote {written} samples to {OUT_PATH.name}")
    print(f"       elapsed {meta['elapsed_sec']:.1f}s")
    print(f"       label counts (orig): {label_counts}")
    print(f"       skipped: empty={skipped_empty} CLARIFY={skipped_clarify}")
    print(f"[Meta] {META_PATH.name}")


if __name__ == "__main__":
    main()

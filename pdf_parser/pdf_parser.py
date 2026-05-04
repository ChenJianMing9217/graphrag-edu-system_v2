# analysis IEP Report PDF and output JSON
import hashlib, re, json, os, copy, pdfplumber
from collections import defaultdict
from typing import List, Dict, Any

# Pre-compiled Regex Patterns for Performance
RE_RECORD_NO = re.compile(r"病歷號碼[：\s]+([A-Za-z0-9]+)")
RE_ID_NO = re.compile(r"身份證字號[：\s]+([A-Za-z0-9]+)")
RE_NAME = re.compile(r"姓名[：\s]+([^\s0-9]+)")
RE_GENDER = re.compile(r"性別[：\s]+([男女])")
RE_BIRTH = re.compile(r"生日[：\s]+(民國[\s\d年月]+日)")
RE_AGE = re.compile(r"年齡[：\s]+(\d+\s*歲\s*\d*\s*月?)")
RE_DATE_DV = re.compile(r"醫師門診日期：.*?[\r\n]+(民國[\s\d年月]+日)")
RE_DATE_FT = re.compile(r"治療師第一項評估日期：.*?[\r\n]+(民國[\s\d年月]+日)")
RE_DATE_RC = re.compile(r"綜合報告書完成通知日期：.*?[\r\n]+(民國[\s\d年月]+日)")
RE_DATE_NR = re.compile(r"下次複評日期[：\s]*(?:.*?)[\r\n]+(民國[\s\d年月]+日)")
RE_DIAG_KIND = re.compile(r'(疑似|確定)[：:]?')
RE_BULLET_MAJOR = re.compile(r"^(\d+)[\.\s]+(.*)")
RE_BULLET_MINOR = re.compile(r"^(\d+)\)[\s]+(.*)")
RE_SCORE_PERCENTILE = re.compile(r"(?:百分位|百分等級|%ile)[：:=\s]*([<＜]?\s*[\d\.]+)|([<＜]?\s*[\d\.]+)(?:%ile|百分位|百分等級)")
RE_SCORE_DQ = re.compile(r"(?:發展商數|智商|CCS|DQ|FSIQ|IQ)[：:=\s]*([\d\.]+)")
RE_SCORE_SS = re.compile(r"(?:標準分數|SS|T分數|分量表)[：:=\s]*([\d\.]+)")
RE_RAW_SCORE = re.compile(r"原始分數[：:\s]*([\d\.]+)")
RE_SCORE_ABILITY = re.compile(r"(\d+\s*歲\s*\d*\s*個?月?)(?:[\s～~－-]+)(\d+\s*歲\s*\d*\s*個?月?)")

class PDFProcessor:
    def __init__(self):
        self.category_alias_map = {
            "知覺動作功能": ["知覺動作", "粗大動作", "精細動作", "知覺", "動作"],
            "吞嚥/口腔功能": ["吞嚥", "吞嚥功能", "吞嚥反射", "吞嚥/口腔", "吞嚥/口", "口腔", "口腔功能", "口腔動作", "腔功能"],
            "口語溝通功能": ["口語", "語言", "溝通", "通功能"],
            "認知功能": ["認知"],
            "社會情緒功能": ["社會情緒", "情緒行為", "社會適應", "情緒"]
        }

    def _bbox_intersects(self, a, b) -> bool:
        ax0, ax1, at, ab = a
        bx0, bx1, bt, bb = b
        return not (ax1 < bx0 or ax0 > bx1 or ab < bt or at > bb)

    def extract_full_text(self, pdf) -> str:
        text = ""
        for p in pdf.pages:
            t = p.extract_text()
            if t: text += t + "\n"
        return text

    def extract_document_meta(self, full_text: str) -> Dict[str, Any]:
        meta = {
            "report_title": "", "hospital": "", "patient_name": "",
            "gender": "", "birth_date": "", "age": "",
            "record_no": "", "id_no": "",
            "report_dates": {
                "doctor_visit_date": "", "first_therapy_eval_date": "",
                "report_complete_date": "", "next_review_date": ""
            },
            "page_no": 1
        }
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        if len(lines) >= 2:
            if "醫院" in lines[0] or "診所" in lines[0]: meta["hospital"] = lines[0]
            if "報告書" in lines[1] or "聯合評估" in lines[1]: meta["report_title"] = lines[1]

        def get_match(pattern):
            m = pattern.search(full_text)
            return m.group(1).strip() if m else ""

        meta["record_no"] = get_match(RE_RECORD_NO)
        meta["id_no"] = get_match(RE_ID_NO)
        meta["patient_name"] = get_match(RE_NAME)
        meta["gender"] = get_match(RE_GENDER)
        meta["birth_date"] = get_match(RE_BIRTH)
        meta["age"] = get_match(RE_AGE)
        meta["report_dates"]["doctor_visit_date"] = get_match(RE_DATE_DV)
        meta["report_dates"]["first_therapy_eval_date"] = get_match(RE_DATE_FT)
        meta["report_dates"]["report_complete_date"] = get_match(RE_DATE_RC)
        meta["report_dates"]["next_review_date"] = get_match(RE_DATE_NR)
        return meta

    def extract_summary_sections(self, full_text: str, pdf_obj=None) -> Dict[str, Any]:
        """
        Optimized version that takes a pdf_obj directly to avoid re-opening file.
        """
        doc_sec = {
            "chief_complaint": {"text": "", "page_no": 2},
            "visit_problems": {"items": [], "page_no": 2},
            "team_summary": {"text": "", "page_no": 2},
            "diagnosis": {"suspected": [], "confirmed": [], "page_no": 2},
            "evaluation_results": {"items": {}, "page_no": 3},
            "general_recommendations": {"items": {}, "page_no": 3}
        }
        if not pdf_obj: return doc_sec

        logical_lines = []
        try:
            for p_idx in [1, 2]:
                if p_idx >= len(pdf_obj.pages): break
                page = pdf_obj.pages[p_idx]
                page_checkmarks = []
                for r in page.rects:
                    w, h = r.get("width", 0), r.get("height", 0)
                    if 6 <= w <= 16 and 6 <= h <= 16:
                        expanded = (r["x0"]-1, r["x1"]+1, r["top"]-1, r["bottom"]+1)
                        hits = sum(1 for c in page.curves if self._bbox_intersects(expanded, (c["x0"], c["x1"], c["top"], c["bottom"])))
                        page_checkmarks.append({"text": "■" if hits > 1 else "□", "top": r["top"], "x0": r["x0"]})

                words = page.extract_words(x_tolerance=2, y_tolerance=3)
                all_el = sorted(words + page_checkmarks, key=lambda x: (x['top'], x['x0']))
                if not all_el: continue
                lines = [[all_el[0]]]
                for i in range(1, len(all_el)):
                    if abs(all_el[i]['top'] - all_el[i-1]['top']) <= 5: lines[-1].append(all_el[i])
                    else: lines.append([all_el[i]])
                for row in lines:
                    line_text = " ".join(el['text'] for el in sorted(row, key=lambda x: x['x0'])).strip()
                    if line_text: logical_lines.append(line_text)
        except Exception: return doc_sec

        SIDEBAR_MAP = {"主訴": "chief_complaint", "就診問題": "visit_problems", "團隊評估總結": "team_summary", "團隊評估建議": "team_summary", "疾病診斷": "diagnosis", "評估結果": "evaluation_results", "綜合建議": "general_recommendations"}
        boundaries = []
        for m_text, sec_name in SIDEBAR_MAP.items():
            ch_idx, start_idx = 0, -1
            for i, line in enumerate(logical_lines):
                c_ln = line.replace(" ", "")
                if c_ln and c_ln[0] == m_text[ch_idx]:
                    if ch_idx == 0: start_idx = i
                    ch_idx += 1
                    if ch_idx == len(m_text):
                        boundaries.append((start_idx, sec_name))
                        break
        
        boundaries.sort()
        b_map = dict(boundaries)
        current_sec, buffers, last_labels = None, {k: [] for k in ["chief_complaint", "team_summary", "recs_raw"]}, []
        for i, line in enumerate(logical_lines):
            clean_line = line.replace(" ", "")
            if i in b_map: current_sec = b_map[i]
            if re.match(r"^[0-9]+$", clean_line) or clean_line in ("類別內容", "壹、評估結果報告", "符號說明", "合", "建", "議", "就", "評"): continue

            if any(kw in clean_line for kw in ["疑似", "確定"]):
                parts = RE_DIAG_KIND.split(line)
                if len(parts) > 1:
                    if parts[0].strip() and current_sec in buffers: buffers[current_sec].append(parts[0].strip())
                    for k in range(1, len(parts), 2):
                        kind = "suspected" if parts[k] == "疑似" else "confirmed"
                        content = re.split(r'(評估結果|綜合建議|■|▲|□)', parts[k+1])[0].strip()
                        if content: doc_sec["diagnosis"][kind].append(content)
                    current_sec = "diagnosis"
                    continue
            if not current_sec: continue

            if current_sec == "chief_complaint": buffers["chief_complaint"].append(line)
            elif current_sec == "team_summary":
                if "發展遲緩" in clean_line and any(d in clean_line for d in ["知覺動作", "語言發展"]): current_sec = "evaluation_results"
                else: buffers["team_summary"].append(line)
            
            if current_sec == "visit_problems":
                items = re.findall(r"([■▲□])\s*([^■▲□\s]+)", line)
                if items:
                    for s, t in items: doc_sec["visit_problems"]["items"].append({"text": t.strip(), "checked": s != "□"})
                elif any(kw in clean_line for kw in ["生理", "認知", "人際", "活動量"]):
                    for p in line.split(): doc_sec["visit_problems"]["items"].append({"text": p, "checked": False})
            elif current_sec == "evaluation_results":
                dm = re.search(r"(知覺動作|語言發展|認知發展|社會情緒|感官功能|其他發展)\s*([^\s]*遲緩|[^\s]*異常|未進行評估|無異常)?", line)
                if dm: doc_sec["evaluation_results"]["items"][dm.group(1)] = {"status": (dm.group(2) or "").strip(), "items": []}
                elif re.match(r"^[■▲□\s]+$", clean_line):
                    symbols, domains = line.split(), list(doc_sec["evaluation_results"]["items"].keys())
                    if domains and last_labels:
                        cur_domain = domains[-1]
                        for sym, lbl in zip(symbols, last_labels):
                            doc_sec["evaluation_results"]["items"][cur_domain]["items"].append({"text": lbl, "checked": sym != "□", "qualifier": "checked" if sym == "■" else ("borderline" if sym == "▲" else "unchecked")})
                        last_labels = []
                elif len(clean_line) > 1 and clean_line not in ("□無異常", "■遲緩", "▲臨界"): last_labels = line.split()
            elif current_sec == "general_recommendations": buffers["recs_raw"].append(line)

        doc_sec["chief_complaint"]["text"] = " ".join(buffers["chief_complaint"]).strip()
        doc_sec["team_summary"]["text"] = " ".join(buffers["team_summary"]).strip()
        recs, current_sub, MAJOR_RECS_CATS = {}, "一般", ["符合證明", "追蹤評估", "相關療育", "門診追蹤"]
        for ln in buffers["recs_raw"]:
            cln = ln.replace(" ", "")
            for mc in MAJOR_RECS_CATS:
                if mc in cln and (len(cln) < 15 or ("■" not in cln and "□" not in cln)):
                    current_sub = mc if (mc != "相關療育" and "與資源" not in cln) else "相關療育與資源"
                    break
            items = re.findall(r"([■□▲☑☐])\s*([^■□▲☑☐\r\n]+)", ln)
            for s, t in items:
                for p in re.split(r'\s{2,}', t.strip()):
                    p = p.strip()
                    if not p or p in (MAJOR_RECS_CATS + ["與資源", "諮詢", "申請資格"]): continue
                    recs.setdefault(current_sub, []).append({"text": p, "checked": s in ("■", "☑", "▲"), "qualifier": "checked" if s in ("■", "☑") else ("borderline" if s == "▲" else "unchecked")})
            if not items and cln and not any(mc in cln for mc in MAJOR_RECS_CATS):
                p = ln.strip()
                if p and p not in ["與資源", "諮詢", "申請資格"]: recs.setdefault(current_sub, []).append({"text": p, "checked": False})
        doc_sec["general_recommendations"]["items"] = recs
        return doc_sec

    def extract_all_checkbox_items(self, page, page_no: int) -> List[Dict]:
        checkbox_rects = []
        for r in page.rects:
            w, h = r.get("width", 0), r.get("height", 0)
            if 6 <= w <= 20 and 6 <= h <= 20: 
                expanded = (r["x0"] - 1, r["x1"] + 1, r["top"] - 1, r["bottom"] + 1)
                hits = sum(1 for c in page.curves if self._bbox_intersects(expanded, (c["x0"], c["x1"], c["top"], c["bottom"])))
                checkbox_rects.append({"rect": r, "checked": hits > 1})
        if not checkbox_rects: return []
        words = page.extract_words(x_tolerance=1, y_tolerance=2)
        out = []
        for item in sorted(checkbox_rects, key=lambda x: x["rect"]["top"]):
            r = item["rect"]
            line_words = sorted([w for w in words if w["top"] <= r["bottom"] + 4 and w["bottom"] >= r["top"] - 4 and w["x0"] >= r["x1"] + 2], key=lambda w: w["x0"])
            text = " ".join(w["text"] for w in line_words).strip()
            if text: out.append({"text": text, "checked": item["checked"], "page_no": page_no, "x0": float(r["x0"]), "top": float(r["top"]), "source": "checkbox"})
        return out

    def extract_tables_structured(self, page, doc_id: str, page_no: int) -> List[Dict]:
        try: tables = page.extract_tables({"vertical_strategy": "lines", "horizontal_strategy": "lines", "intersection_tolerance": 5, "edge_min_length": 12})
        except Exception: return []
        out = []
        for t in tables:
            if not t or len(t) < 1: continue
            header = [re.sub(r"\s+", "", h or "") or f"Col_{i}" for i, h in enumerate(t[0])]
            is_score = any(any(kw in h for kw in ["原始分數", "標準分數", "百分等級", "發展年齡", "IQ", "CCS"]) for h in header if h)
            tid = hashlib.sha1(" | ".join(header).encode("utf-8")).hexdigest()[:16]
            
            # If it's a score table but only has header, treat header as data
            data_rows = t[1:] if len(t) > 1 else t
            for idx, r in enumerate(data_rows):
                rd = {h: (v or "").strip() for h, v in zip(header, r) if h or v}
                if rd: out.append({"doc_id": doc_id, "page_no": page_no, "table_id": tid, "row_index": idx, "header": header, "row_dict": rd, "is_score_table": is_score})
        return out

def normalize_category(raw: str, amap: Dict, last_valid: str = "") -> str:
    s = re.sub(r"\s+", "", str(raw or ""))
    if not s: return last_valid or "_unassigned"
    for std, aliases in amap.items():
        if any(a in s for a in aliases): return std
    return last_valid if s in ["內容", "評估/訓練項目", "評估項目"] else s

def merge_table_rows(table_rows: List[Dict], amap: Dict) -> List[Dict]:
    merged_tables, groups = [], defaultdict(list)
    for r in table_rows: groups[r["table_id"]].append(r)
    for tid, rows in groups.items():
        rows_sorted = sorted(rows, key=lambda x: (x["page_no"], x["row_index"]))
        is_score, merged_rows, current_rd, last_valid = rows_sorted[0].get("is_score_table"), [], None, ""
        for r in rows_sorted:
            rd = r["row_dict"]
            if is_score: merged_rows.append(r); continue
            if rd.get("類別") or rd.get("評估／訓練項目"):
                if current_rd: merged_rows.append(current_rd)
                current_rd = copy.deepcopy(r)
            elif current_rd:
                for k, v in rd.items():
                    if v: current_rd["row_dict"][k] = (current_rd["row_dict"].get(k, "") + "\n" + v).strip()
            else: current_rd = copy.deepcopy(r)
        if current_rd: merged_rows.append(current_rd)
        for idx, mr in enumerate(merged_rows):
            mr["group_id"], norm = f"{tid}-g{idx:03d}", normalize_category(mr["row_dict"].get("類別", ""), amap, last_valid)
            if norm != "_unassigned": last_valid = norm
            mr["row_dict"]["normalized_category"] = norm
        merged_tables.append({"table_id": tid, "doc_id": rows[0]["doc_id"], "is_score_table": is_score, "rows": merged_rows})
    return merged_tables

def parse_bullet_list(lines: List[str]) -> List[Dict]:
    has_bullets = any(RE_BULLET_MAJOR.match(l.strip()) for l in lines if l.strip())
    if not has_bullets:
        merged = " ".join(l.strip() for l in lines if l.strip())
        return [{"text": merged, "sub_items": []}] if merged else []
    parsed, current = [], None
    for line in (l.strip() for l in lines if l.strip()):
        m_major = RE_BULLET_MAJOR.match(line)
        if m_major: current = {"text": line, "sub_items": []}; parsed.append(current)
        elif RE_BULLET_MINOR.match(line):
            if current: current["sub_items"].append({"text": line})
            else: parsed.append({"text": line, "sub_items": []})
        elif current:
            if current["sub_items"]: current["sub_items"][-1]["text"] += " " + line
            else: current["text"] += " " + line
        else: parsed.append({"text": line, "sub_items": []})
    return parsed
def deduplicate_list(list_of_dicts: List[Dict]) -> List[Dict]:
    seen_hashes = set()
    result = []
    for d in list_of_dicts:
        h = json.dumps(d, sort_keys=True)
        if h not in seen_hashes:
            seen_hashes.add(h)
            result.append(d)
    return result

def parse_scores(lines: List[str]) -> Dict:
    raw = " ".join(lines)
    res = {"percentile": None, "development_quotient": None, "standard_score": None, "current_ability": {"from": None, "to": None}, "raw_text": raw, "subtests": []}
    
    # Extract numbers with labels
    get_vals = lambda regex: [{"val": float(m.group(1).replace("＜", "<").replace("<", "").strip()), "pos": m.start(), "qual": "<" if "<" in m.group(1) else None} for m in regex.finditer(raw) if m.group(1)]
    
    percentiles = get_vals(RE_SCORE_PERCENTILE)
    dqs = get_vals(RE_SCORE_DQ)
    raws = get_vals(RE_RAW_SCORE)
    sss = get_vals(RE_SCORE_SS)

    # Pairing Logic for Subtests
    used_sss = set()
    for rs in raws:
        # Match Raw Score to nearest subsequent Standard Score
        next_ss = next((s for s in sss if s["pos"] not in used_sss and 0 < s["pos"] - rs["pos"] < 70), None)
        if next_ss:
            used_sss.add(next_ss["pos"])
            pair = {"原始分數": rs["val"], "標準分數": next_ss["val"]}
            # Optional: Match Percentile if it's very close
            next_pct = next((p for p in percentiles if 0 < p["pos"] - next_ss["pos"] < 70), None)
            if next_pct: pair["百分等級"] = next_pct["val"]
            res["subtests"].append(pair)

    # Globals
    if percentiles: res["percentile"], res["percentile_qualifier"] = percentiles[0]["val"], percentiles[0]["qual"]
    if dqs: res["development_quotient"] = dqs[0]["val"]
    
    # Global SS: Closest to DQ or first unused
    avail_ss = [s for s in sss if s["pos"] not in used_sss]
    if dqs and avail_ss: avail_ss.sort(key=lambda s: abs(s["pos"] - dqs[0]["pos"]))
    if avail_ss: res["standard_score"] = avail_ss[0]["val"]
    elif sss and not res["subtests"]: res["standard_score"] = sss[0]["val"]

    am = RE_SCORE_ABILITY.search(raw)
    if am: res["current_ability"] = {"from": am.group(1).replace(" ", ""), "to": am.group(2).replace(" ", "")}
    return res

def format_subtest_text(s_item: Dict):
    """Formats observation item text to the user's requested 3-line style."""
    t = s_item.get("text", "")
    # Strip existing metadata if re-formatting
    t_clean = re.sub(r'[\s:：]*百分等級[:：\s]*\d+[%％\s]*', '', t)
    t_clean = re.sub(r'[\s:：]*發展年齡[:：\s]*[\d\sMY]+', '', t_clean).strip()
    
    lines = [t_clean]
    fmt = lambda x: int(x) if x == int(x) else x
    if "原始分數" in s_item and "標準分數" in s_item:
        lines.append(f"原始分數：{fmt(s_item['原始分數'])} 標準分數：{fmt(s_item['標準分數'])}")
    
    meta = []
    if "百分等級" in s_item: meta.append(f"百分等級：{fmt(s_item['百分等級'])}")
    if "發展年齡" in s_item: meta.append(f"發展年齡：{s_item['發展年齡']}")
    if meta: lines.append(" ".join(meta))
    
    s_item["text"] = "\n".join(lines)

def consolidate_subtests(p_data: Dict, p_scores: Dict):
    """Merges extracted subtest scores into the corresponding observation sub-items."""
    raw_pairs = p_scores.get("subtests", [])
    if not raw_pairs: return

    for obs_item in p_data.get("observations", []):
        sub_items = obs_item.get("sub_items", [])
        # Find indices of items that correspond to subtests
        s_idx = [i for i, si in enumerate(sub_items) if any(kw in si.get("text", "") for kw in ["百分等級", "發展年齡"])]
        
        if len(s_idx) == len(raw_pairs) and len(s_idx) > 0:
            for idx, pair in zip(s_idx, raw_pairs):
                s_item = sub_items[idx]
                t = s_item["text"]
                # Extract values already in text (don't overwrite more specific ones later)
                if (m := re.search(r'百分等級[:：\s]*(\d+)', t)): s_item["百分等級"] = float(m.group(1))
                if (m := re.search(r'發展年齡[:：\s]*([\d\sMY]+)', t)): s_item["發展年齡"] = m.group(1).strip()
                
                # Merge extracted raw/standard scores
                for k, v in pair.items():
                    if k not in s_item: s_item[k] = v
                
                format_subtest_text(s_item)
            p_scores["subtests"] = [] # Clear once matched
            break

def parse_unit_text(text: str) -> Dict[str, Any]:
    lines, parsed, curr = [L.strip() for L in text.split("\n") if L.strip()], {"evaluation_date": "", "status": "", "assessment_tools": [], "observations": [], "training_directions": [], "recommendations": [], "scores_text": []}, None
    def is_cont(line, curr_list):
        if not curr_list or not line.strip() or re.match(r'^[■▲▼●☑☒□☐○◎✓✗［］]', line.strip()): return False
        return not ('一' <= line.strip()[0] <= '鿿') or ("：" in line and "：" not in curr_list[-1].get("text", ""))

    for line in lines:
        if line.startswith("評估日期："): parsed["evaluation_date"], curr = line.split("：", 1)[-1].strip(), "eval_date"
        elif line.startswith("評估結果："): parsed["scores_text"].append(line); curr = "scores"
        elif line.startswith(("評估工具：", "評估工具、結果與訓練方向")):
            tool, curr = line.split("：", 1)[-1].strip() if "：" in line else "", "assessment_tools"
            if tool: parsed["assessment_tools"].append({"text": tool})
        elif any(kw in line for kw in ["行為觀察及綜合結果：", "行為觀察", "測驗結果", "評估時表現"]):
            obs, curr = line.split("：", 1)[-1].strip() if "：" in line else "", "observations"
            if obs: parsed["observations"].append(obs)
        elif line.startswith(("訓練方向：", "訓練方向", "訓練/建議：")): curr = "training_directions"
        elif line.startswith(("具體建議：", "居家練習")): curr = "recommendations"
        elif any(kw in line for kw in ("百分位", "發展商數", "標準分數", "T分數", "FSIQ", "IQ")) or curr == "scores": parsed["scores_text"].append(line)
        elif curr == "assessment_tools":
            if is_cont(line, parsed["assessment_tools"]): parsed["assessment_tools"][-1]["text"] += " " + line
            else: parsed["assessment_tools"].append({"text": line})
        elif curr == "observations": parsed["observations"].append(line)
        elif curr == "training_directions":
            if is_cont(line, parsed["training_directions"]): parsed["training_directions"][-1]["text"] += " " + line
            else: parsed["training_directions"].append({"text": line})
        elif curr == "recommendations": parsed["recommendations"].append(line)
        else: parsed["status"] += line + "\n"
    parsed["status"] = parsed["status"].strip()
    parsed["observations"] = parse_bullet_list(parsed["observations"])
    parsed["recommendations"] = parse_bullet_list(parsed["recommendations"])
    return parsed

def normalize_item_name(item_text: str) -> Dict[str, str]:
    res, text = {"item_name": item_text.strip(), "item_phase": "other", "status_text": ""}, item_text.replace("\n", " ").strip()
    for kw in ["需要追蹤及諮詢", "需要諮詢", "需要追蹤", "定期追蹤", "無異常", "無", "臨界發展遲緩", "發展遲緩", "疑似失調", "需要訓練", "異常", "正常"]:
        if kw in text: res["status_text"], text = kw, text.replace(kw, "").strip(); break
    res["item_phase"] = "training" if "訓練" in text and text != "訓練方向" else "evaluation"
    text = text.replace("訓練", "").strip()
    mapping = {"粗動作": "粗大動作", "理解": "口語理解", "表達": "口語表達", "認知": "認知功能", "情緒行為與社會適應機能": "情緒行為與社會適應功能"}
    res["item_name"] = mapping.get(text, text) or "綜合"
    return res

def assign_checkboxes(units: List[Dict], cbs: List[Dict]):
    lookup = {}
    for cb in cbs:
        k = re.sub(r'\s+', '', cb["text"])
        if k: lookup.setdefault(k, []).append(cb)
    def find(txt, pool, full_lookup, src_norm):
        norm = re.sub(r'\s+', '', txt)
        if not norm: return None
        for cb in pool:
            if re.sub(r'\s+', '', cb["text"]) in norm: return cb["checked"]
        for k, v in full_lookup.items():
            if k in norm and k in src_norm: return v[0]["checked"]
        if re.search(r'^[\u25a0\u25b2\u25bc\u25cf\u2611\u2612\u2713]', txt.strip()): return True
        if re.search(r'^[\u25a1\u2610\u25cb\u25ce]', txt.strip()): return False
        return None
    for u in units:
        pool = [cb for cb in cbs if u["page_start"] - 3 <= cb["page_no"] <= u["page_end"] + 3]
        src_norm = re.sub(r'\s+', '', u.get("source_text", ""))
        for td in u.get("training_directions", []): td["checked"] = find(td.get("text", ""), pool, lookup, src_norm)
        for at in u.get("assessment_tools", []): at["checked"] = find(at.get("text", ""), pool, lookup, src_norm)

def process_iep_pdf(pdf_path: str, doc_id: str = "report_1") -> Dict[str, Any]:
    processor = PDFProcessor()
    with pdfplumber.open(pdf_path) as pdf:
        doc_text = processor.extract_full_text(pdf)
        table_rows, cbs = [], []
        for pno, page in enumerate(pdf.pages, start=1):
            table_rows.extend(processor.extract_tables_structured(page, doc_id, pno))
            cbs.extend(processor.extract_all_checkbox_items(page, pno))
        
        merged = merge_table_rows(table_rows, processor.category_alias_map)
        units = []
        score_buf = []
        for tbl in merged:
            if tbl["is_score_table"]:
                score_buf.extend(r["row_dict"] for r in tbl["rows"])
                continue
            groups = defaultdict(list)
            for r in tbl["rows"]: groups[r["group_id"]].append(r)
            for gid, rs in groups.items():
                rs_s = sorted(rs, key=lambda x: (x["page_no"], x["row_index"]))
                rd0 = rs_s[0]["row_dict"]
                texts = []
                for r in rs_s:
                    txt = r["row_dict"].get("評估工具、結果與訓練方向", "")
                    extras = [v for k, v in r["row_dict"].items() if any(kw in k for kw in ["百分位", "發展商數", "標準分數", "_3", "_4", "_5", "_6"]) and v.strip()]
                    if extras: txt += " " + " ".join(extras)
                    if txt.strip(): texts.append(txt.strip())
                merged_txt = "\n\n".join(texts)
                p_data = parse_unit_text(merged_txt)
                m_tds = []
                for td in p_data["training_directions"]:
                    if m_tds and td["text"] and not re.match(r'^[\u4e00-\u9fff\uff01-\uff60]', td["text"][0]): m_tds[-1]["text"] += " " + td["text"]
                    else: m_tds.append(td)
                p_data["training_directions"] = m_tds
                # Merge scores and observations
                p_scores = parse_scores(p_data["scores_text"])
                consolidate_subtests(p_data, p_scores)
                
                u = {"category": re.sub(r"\s+", "", rd0.get("類別") or ""), "item": re.sub(r"[\r\n]+", " ", rd0.get("評估／訓練項目") or "").strip(), "normalized_category": rd0.get("normalized_category"), "page_start": min(r["page_no"] for r in rs_s), "page_end": max(r["page_no"] for r in rs_s), "source_text": merged_txt}
                if p_data.get("evaluation_date"): u["evaluation_date"] = p_data["evaluation_date"]
                if p_scores["raw_text"] or p_scores["subtests"]: u["scores"] = p_scores
                for k in ["assessment_tools", "observations", "training_directions", "recommendations"]:
                    if p_data.get(k): u[k] = p_data[k]
                units.append(u)
        assign_checkboxes(units, cbs)
        sections = {"summary_extra": defaultdict(list), "eval": defaultdict(lambda: defaultdict(lambda: {"evaluation": {}, "training": {}})), "appendix": defaultdict(list)}
        for u in units:
            cat, norm_cat, item = u["category"], u["normalized_category"] or u["category"], u["item"]
            if any(kw in cat for kw in ["病因診斷", "病因分類"]): sections["appendix"]["etiology"].append(u); continue
            if "相關疾病" in cat: sections["appendix"]["related"].append(u); continue
            if any(sc in cat for sc in ["主訴", "團隊評估", "疾病診斷", "評估結果"]): sections["summary_extra"][cat].append(u); continue
            ni = normalize_item_name(item)
            target = sections["eval"][norm_cat][ni["item_name"]]
            pdiv = target["evaluation"] if ni["item_phase"] == "evaluation" else target["training"]
            if "page_no" not in target["evaluation"]: target["evaluation"]["page_no"] = u["page_start"]
            if "page_no" not in target["training"]: target["training"]["page_no"] = u["page_start"]
            if ni["status_text"]: pdiv["status" if ni["item_phase"] == "evaluation" else "training_status"] = ni["status_text"]
            for k, v in u.items():
                if k in ["category", "item", "normalized_category", "page_start", "page_end", "source_text", "rows"]: continue
                if not v: continue
                if k == "scores":
                    existing = pdiv.get("scores", {})
                    # If existing has DQ/Percentile and new one doesn't, or new one is just subtests, keep existing.
                    # Or better: merge them, keeping the best global scores.
                    new_has_global = v.get("development_quotient") or v.get("percentile")
                    ext_has_global = existing.get("development_quotient") or existing.get("percentile")
                    if not ext_has_global and new_has_global:
                        pdiv[k] = v
                    elif ext_has_global and new_has_global:
                        # Both have global, maybe they are the same or we keep the first one found?
                        # Usually the first one in the doc is the most prominent.
                        pass
                    elif not ext_has_global and not new_has_global:
                        # Neither has global, just pick one (the first one)
                        if "standard_score" not in existing: pdiv[k] = v
                    
                    # Always merge subtests if they exist
                    if v.get("subtests"):
                        pdiv.setdefault("scores", {}).setdefault("subtests", []).extend(v["subtests"])
                else:
                    # For other fields (observations, tools, etc.), we can append or set if missing.
                    if k in ["observations", "assessment_tools", "training_directions", "recommendations"]:
                        if k not in pdiv: pdiv[k] = v
                        else:
                            # Avoid duplicates and merge
                            existing_list = pdiv[k]
                            for item in v:
                                if item not in existing_list: existing_list.append(item)
                    else:
                        if k not in pdiv: pdiv[k] = v

        eval_sec, unassigned = sections["eval"], sections["eval"].get("_unassigned", {})
        lookup_cat = {iname: cname for cname, items in eval_sec.items() if cname != "_unassigned" for iname in items}
        for iname, idata in list(unassigned.items()):
            if iname in lookup_cat:
                t = eval_sec[lookup_cat[iname]][iname]
                for k, v in idata.get("evaluation", {}).items():
                    if v and k != "page_no": t["evaluation"][k] = v
                for k, v in idata.get("training", {}).items():
                    if v and k != "page_no": t["training"][k] = v
                del unassigned[iname]
        if not unassigned: eval_sec.pop("_unassigned", None)

        # Optimized call to summary sections
        summary = processor.extract_summary_sections(doc_text, pdf_obj=pdf)
        for k, v in sections["summary_extra"].items(): summary[k] = v
        return {"document_meta": processor.extract_document_meta(doc_text), "summary_sections": summary, "evaluation_sections": eval_sec, "appendix_sections": sections["appendix"]}

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()
    if not os.path.exists(args.pdf_path): sys.exit(1)
    try:
        data = process_iep_pdf(args.pdf_path)
        with open(args.output, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Success! Output: {args.output}")
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(1)
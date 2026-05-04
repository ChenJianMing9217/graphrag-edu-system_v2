import json, os, copy
from neo4j import GraphDatabase

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_iep(self, json_data, report_id):
        with self.driver.session() as session:
            print(f"Cleaning up old data for report: {report_id}...")
            session.run("""
                MATCH (r:Report {id: $rid})
                OPTIONAL MATCH (r)-[:HAS_META|HAS_SUMMARY|HAS_DOMAIN|HAS_SUBDOMAIN|HAS_CATEGORY|HAS_CONTENT|IDENTIFIED_AS|DIAGNOSED_AS|HAS_RESULT|HAS_SUGGESTION|HAS_ASSESSMENT_TOOLS|HAS_OBSERVATIONS|HAS_RECOMMENDATIONS|HAS_TRAINING_PLAN|EVALUATED_STATUS|EVALUATED_DATE|TRAINING_STATUS|HAS_SCORES|ITEM|USED_TOOL|OBSERVED|RECOMMENDED|TRAINED_BY|HAS_VALUE|HAS_SUB_ITEM*0..10]->(node)
                DETACH DELETE r, node
            """, rid=report_id)
            session.run("MATCH (n) WHERE n.id STARTS WITH $rid DETACH DELETE n", rid=f"{report_id}_")
            session.run("MATCH (h:CategoryHub) WHERE h.id STARTS WITH $rid DETACH DELETE h", rid=report_id)
            session.execute_write(self._create_graph_batch, json_data, report_id)

    @staticmethod
    def _create_graph_batch(tx, data, report_id):
        meta = data.get("document_meta", {})
        tx.run("""
            MERGE (r:Report {id: $report_id})
            SET r.title = $title, r.hospital = $hospital
            MERGE (m:Meta {id: $report_id})
            SET m.patient_name = $patient, m.gender = $gender, m.birth_date = $birth,
                m.age = $age, m.doctor_visit_date = $dv_date, m.first_therapy_eval_date = $ft_date,
                m.report_complete_date = $rc_date, m.next_review_date = $nr_date
            MERGE (r)-[:HAS_META]->(m)
        """, report_id=report_id, title=meta.get("report_title"), hospital=meta.get("hospital"),
           patient=meta.get("patient_name"), gender=meta.get("gender"), birth=meta.get("birth_date"), age=meta.get("age"),
           dv_date=meta.get("report_dates", {}).get("doctor_visit_date"),
           ft_date=meta.get("report_dates", {}).get("first_therapy_eval_date"),
           rc_date=meta.get("report_dates", {}).get("report_complete_date"),
           nr_date=meta.get("report_dates", {}).get("next_review_date"))

        summary_data = data.get("summary_sections", {})
        tx.run("MERGE (r:Report {id: $report_id}) MERGE (s:Summary {id: $report_id}) MERGE (r)-[:HAS_SUMMARY]->(s)", report_id=report_id)

        def get_semantic_prefix(hub_name, checked, qualifier=None):
            """
            Returns a semantic prefix helper for LLM context.
            """
            if checked is None: return ""
            
            # Labels for Parent-Facing Robot
            labels = {
                "訓練方向": ("(建議加強)", "(非重點項目)", "(視情況練習)"),
                "就診問題摘要": ("(目前有此疑慮)", "(暫無此問題)", "(具發展風險)"),
                "評估結果摘要": ("(項次：遲緩)", "(項次：正常)", "(項次：臨界)"),
                "綜合建議摘要": ("(重點建議)", "(一般參考)", "(視情況建議)"),
                "評估工具": ("(已採用量表)", "(未選用量表)", ""),
                "default": ("(Checked)", "(Unchecked)", "(Partial)")
            }
            
            p_yes, p_no, p_border = labels.get(hub_name, labels["default"])
            if qualifier == "borderline": return p_border
            return p_yes if checked else p_no

        def batch_add_hub_and_items(parent_label, parent_id, hub_name, item_label, items, relationship_to_hub, relationship_to_item):
            if not items: return
            hub_id = f"{parent_id}_{hub_name}"
            tx.run(f"MATCH (p:{parent_label} {{id: $p_id}}) MERGE (h:CategoryHub {{id: $h_id}}) ON CREATE SET h.name = $h_name MERGE (p)-[:{relationship_to_hub}]->(h)", p_id=parent_id, h_id=hub_id, h_name=hub_name)
            
            item_list = []
            sub_item_list = []
            for idx, item in enumerate(items):
                txt = item.get("text") if isinstance(item, dict) else item
                if not txt: continue
                
                checked = item.get("checked") if isinstance(item, dict) else None
                qualifier = item.get("qualifier") if isinstance(item, dict) else None
                prefix = get_semantic_prefix(hub_name, checked, qualifier)
                
                # Apply semantic decoration to the node text
                decorated_text = f"{prefix} {txt}".strip()
                
                item_node_id = f"{hub_id}_i{idx}"
                props = {"id": item_node_id, "text": decorated_text, "raw_text": txt}
                if checked is not None: props["checked"] = checked
                if qualifier: props["qualifier"] = qualifier
                
                item_list.append(props)
                if isinstance(item, dict) and item.get("sub_items"):
                    for s_idx, sub in enumerate(item["sub_items"]):
                        sub_txt = sub.get("text") if isinstance(sub, dict) else sub
                        if sub_txt: sub_item_list.append({"parent_id": item_node_id, "id": f"{item_node_id}_s{s_idx}", "text": sub_txt})

            if item_list:
                tx.run(f"UNWIND $items AS item_data MATCH (h:CategoryHub {{id: $h_id}}) MERGE (i:{item_label} {{id: item_data.id}}) SET i += item_data MERGE (h)-[:{relationship_to_item}]->(i)", h_id=hub_id, items=item_list)
            if sub_item_list:
                tx.run(f"UNWIND $subs AS sub_data MATCH (i:{item_label} {{id: sub_data.parent_id}}) MERGE (c:SubItem {{id: sub_data.id}}) SET c.text = sub_data.text MERGE (i)-[:HAS_SUB_ITEM]->(c)", subs=sub_item_list)

        s_id = report_id
        cc = summary_data.get("chief_complaint", {})
        if isinstance(cc, dict) and cc.get("text"): batch_add_hub_and_items("Summary", s_id, "主訴摘要", "ChiefComplaint", [cc["text"]], "HAS_CATEGORY", "HAS_CONTENT")
        vp = summary_data.get("visit_problems", {})
        if isinstance(vp, dict) and vp.get("items"): batch_add_hub_and_items("Summary", s_id, "就診問題摘要", "VisitProblem", vp["items"], "HAS_CATEGORY", "IDENTIFIED_AS")
        ts = summary_data.get("team_summary", {})
        if isinstance(ts, dict) and ts.get("text"): batch_add_hub_and_items("Summary", s_id, "團隊評估總結", "TeamSummary", [ts["text"]], "HAS_CATEGORY", "HAS_CONTENT")
        diag = summary_data.get("diagnosis", {})
        if isinstance(diag, dict):
            diag_items = []
            for k in ["suspected", "confirmed"]:
                for d in diag.get(k, []): diag_items.append({"text": d, "type": k})
            if diag_items: batch_add_hub_and_items("Summary", s_id, "診斷摘要", "Diagnosis", diag_items, "HAS_CATEGORY", "DIAGNOSED_AS")
        er = summary_data.get("evaluation_results", {})
        if er and er.get("items"):
            er_items = [{"text": f"{d}: {v.get('status')}", "domain": d, "status": v.get('status')} for d, v in er["items"].items()]
            batch_add_hub_and_items("Summary", s_id, "評估結果摘要", "GeneralResult", er_items, "HAS_CATEGORY", "HAS_RESULT")
        gr = summary_data.get("general_recommendations", {})
        if gr and gr.get("items"):
            gr_items = []
            for cat, its in gr["items"].items():
                for it in its: 
                    it_copy = copy.deepcopy(it)
                    it_copy["category"] = cat
                    gr_items.append(it_copy)
            if gr_items: batch_add_hub_and_items("Summary", s_id, "綜合建議摘要", "GeneralRecommendation", gr_items, "HAS_CATEGORY", "HAS_SUGGESTION")

        eval_sections = data.get("evaluation_sections", {})
        for d_name, subdomains in eval_sections.items():
            d_id = f"{report_id}_{d_name}".replace(" ", "_").replace("/", "_").replace("(", "_").replace(")", "_")
            tx.run("MERGE (r:Report {id: $r_id}) MERGE (d:Domain {id: $d_id}) SET d.name = $name MERGE (r)-[:HAS_DOMAIN]->(d)", r_id=report_id, d_id=d_id, name=d_name)
            for sd_name, sd_data in subdomains.items():
                sd_id = f"{d_id}_{sd_name}".replace(" ", "_").replace("/", "_").replace("(", "_").replace(")", "_")
                tx.run("MATCH (d:Domain {id: $d_id}) MERGE (sd:Subdomain {id: $sd_id}) SET sd.name = $sd_name, sd.domain = $d_name MERGE (d)-[:HAS_SUBDOMAIN]->(sd)", d_id=d_id, sd_id=sd_id, sd_name=sd_name, d_name=d_name)
                eval_p, train_p = sd_data.get("evaluation", {}), sd_data.get("training", {})
                if eval_p.get("assessment_tools"): batch_add_hub_and_items("Subdomain", sd_id, "評估工具", "AssessmentTool", eval_p["assessment_tools"], "HAS_ASSESSMENT_TOOLS", "USED_TOOL")
                if eval_p.get("observations"): batch_add_hub_and_items("Subdomain", sd_id, "行為觀察", "Observation", eval_p["observations"], "HAS_OBSERVATIONS", "OBSERVED")
                if eval_p.get("recommendations"): batch_add_hub_and_items("Subdomain", sd_id, "具體建議", "Recommendation", eval_p["recommendations"], "HAS_RECOMMENDATIONS", "RECOMMENDED")
                if train_p.get("training_directions"): batch_add_hub_and_items("Subdomain", sd_id, "訓練方向", "TrainingDirection", train_p["training_directions"], "HAS_TRAINING_PLAN", "TRAINED_BY")
                if eval_p.get("status"): tx.run("MATCH (sd:Subdomain {id: $sd_id}) MERGE (s:StatusNode {id: $s_id}) SET s.name = '評估狀態', s.text = $stat MERGE (sd)-[:EVALUATED_STATUS]->(s)", sd_id=sd_id, s_id=f"{sd_id}_eval_status", stat=eval_p["status"])
                if eval_p.get("evaluation_date"): tx.run("MATCH (sd:Subdomain {id: $sd_id}) MERGE (s:DateNode {id: $s_id}) SET s.name = '評估日期', s.text = $dt MERGE (sd)-[:EVALUATED_DATE]->(s)", sd_id=sd_id, s_id=f"{sd_id}_eval_date", dt=eval_p["evaluation_date"])
                if train_p.get("training_status"): tx.run("MATCH (sd:Subdomain {id: $sd_id}) MERGE (s:StatusNode {id: $s_id}) SET s.name = '訓練狀態', s.text = $stat MERGE (sd)-[:TRAINING_STATUS]->(s)", sd_id=sd_id, s_id=f"{sd_id}_train_status", stat=train_p["training_status"])
                scores = eval_p.get("scores", {})
                if scores:
                    score_list = []
                    for k in ["percentile", "standard_score", "development_quotient"]:
                        if scores.get(k) is not None: score_list.append({"text": f"{k}: {scores[k]}", "type": k, "value": scores[k]})
                    if score_list: batch_add_hub_and_items("Subdomain", sd_id, "分數", "Score", score_list, "HAS_SCORES", "HAS_VALUE")

if __name__ == "__main__":
    pass

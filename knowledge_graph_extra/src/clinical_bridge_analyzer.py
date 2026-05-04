import json
import os
import sys
from universal_correspondence_engine import ClinicalCorrespondenceEngine

class ClinicalBridgeAnalyzer:
    def __init__(self, encoder=None):
        self.engine = ClinicalCorrespondenceEngine(encoder=encoder)

    def analyze_clinical_context(self, observations="", goals="", activities="", 
                                 top_n=3, deep_search=False, with_milestones=True):
        """
        全量因果分析：將所有的輸入片段 (行為、目標、建議) 
        透過圖譜對接到「核心能力 (Ability Hub)」並進行聚類。
        
        :param deep_search: 是否深挖 (增加搜尋廣度與圖譜挖掘深度)
        :param with_milestones: 是否提取相關里程碑與發展依據
        """
        # 如果深挖，增加初始搜尋的候選數量
        actual_top_n = top_n * 2 if deep_search else top_n
        
        inputs = [
            ("Observation", observations),
            ("Goal/Strategy", goals),
            ("Activity", activities)
        ]
        
        ability_clusters = {}
        gaps = []

        for category, text in inputs:
            if not text: continue
            
            segments = [s.strip() for s in re.split(r'[。\n；;]', text) if len(s.strip()) > 2]
            
            for seg in segments:
                # 執行混合搜尋
                results = self.engine.find_multi_party_correspondence(seg, limit=actual_top_n)
                
                if not results or (isinstance(results, dict) and "message" in results):
                    gaps.append({"text": seg, "category": category})
                    continue
                
                for res in results:
                    entry = res['enrty']
                    
                    # 獲取關連的能力
                    # 如果深挖，我們不僅看 1 步，還要擴展到相關能力
                    with self.engine.driver.session() as session:
                        # 動態調整路徑深度
                        path_depth = "*1..2" if deep_search else ""
                        
                        expand_query = f"""
                        MATCH (start) WHERE elementId(start) = $eid
                        OPTIONAL MATCH (start)-[:INDICATES_ABILITY|TARGETS_ABILITY|HAS_ABILITY{path_depth}]-(a:Ability)
                        WHERE NOT a.name IN $sinks
                        RETURN DISTINCT a.name AS name, a.id AS id
                        """
                        abilities = session.run(expand_query, eid=entry['elementId'], sinks=list(self.engine.sink_nodes))
                        matched_abilities = [r['name'] for r in abilities if r['name']]
                    
                    if not matched_abilities and entry['label'] == 'Ability':
                         matched_abilities = [entry['name']]
                    
                    for ab_name in set(matched_abilities):
                        if ab_name not in ability_clusters:
                            ability_clusters[ab_name] = {
                                "ability_name": ab_name,
                                "observations": set(),
                                "goals": set(),
                                "recommendations": set(),
                                "milestones": []
                            }
                        
                        if category == "Observation":
                            ability_clusters[ab_name]["observations"].add(seg)
                        elif category == "Goal/Strategy":
                            ability_clusters[ab_name]["goals"].add(seg)
                        elif category == "Activity":
                            ability_clusters[ab_name]["recommendations"].add(seg)
                            
                        # 如果需要里程碑且尚未抓取
                        if with_milestones and not ability_clusters[ab_name]["milestones"]:
                             with self.engine.driver.session() as session:
                                 # 擴展查詢路徑：僅針對當前能力或極近親屬
                                 m_query = """
                                 MATCH (a:Ability {name: $ab_name})
                                 // 僅看直接里程碑或 1 步之內的層級關聯，避免過度擴張至整個領域
                                 OPTIONAL MATCH (a)-[:HAS_MILESTONE|HAS_ABILITY|HAS_SUBDOMAIN*1..2]-(m:Milestone)
                                 OPTIONAL MATCH (m)-[:SOURCED_FROM]->(s:Source)
                                 WHERE m IS NOT NULL
                                 RETURN DISTINCT elementId(m) AS m_eid, m.expected_behavior AS behavior, 
                                        m.age_min_month AS min_age, s.name AS source
                                 ORDER BY min_age ASC, behavior ASC
                                 LIMIT 6
                                 """
                                 ms = session.run(m_query, ab_name=ab_name)
                                 ability_clusters[ab_name]["milestones"] = [
                                     {"behavior": r['behavior'], "source": r['source'], "age": r['min_age']}
                                     for r in ms if r['behavior']
                                 ]

        final_clusters = []
        for ab_name, data in ability_clusters.items():
            final_clusters.append({
                "ability": ab_name,
                "observations": list(data["observations"]),
                "goals": list(data["goals"]),
                "recommendations": list(data["recommendations"]),
                "milestones": data["milestones"],
                "is_causal_complete": len(data["observations"]) > 0 and (len(data["goals"]) > 0 or len(data["recommendations"]) > 0)
            })
        return {
            "ability_centric_clusters": final_clusters,
            "unmapped_observations": gaps
        }

    def get_developmental_timeline(self, domain=None, ability=None, age_months=None):
        """
        獲取發展常模查詢：支援 領域/能力 的時序地圖。
        1. 如果傳入個別能力，先執行『智慧定錨』與『發展鏈聚合』。
        2. 抓取目標月齡的『前、中、後』三點時序，並標註 LLM 語意狀態。
        """
        if age_months is None:
            return {"error": "Missing age_months", "target_age_months": age_months}

        with self.engine.driver.session() as session:
            # 1. 確定要查詢的能力列表 (定錨與擴展過程)
            ability_names = []
            
            if ability:
                # --- 核心升級：全標籤智慧定錨 ---
                print(f"🔍 正在針對「{ability}」執行全路徑智慧定錨...")
                matches = self.engine.find_multi_party_correspondence(ability, limit=10)
                
                ability_candidates = []
                for res in matches:
                    entry = res.get('enrty', res.get('entry', {})) # 兼容 typo
                    lbl = entry.get('label')
                    eid = entry.get('elementId')
                    
                    if lbl == 'Ability':
                        ability_candidates.append(entry['name'])
                    else:
                        # 透過圖譜向上回溯至 Ability 核心
                        with self.engine.driver.session() as sub_session:
                            back_trace = """
                            MATCH (start) WHERE elementId(start) = $eid
                            MATCH (start)-[:INDICATES_ABILITY|HAS_ABILITY|TARGETS_ABILITY*0..2]-(a:Ability)
                            RETURN DISTINCT a.name AS name LIMIT 1
                            """
                            bt_res = sub_session.run(back_trace, eid=eid).single()
                            if bt_res: ability_candidates.append(bt_res['name'])
                    
                    if ability_candidates: break # 抓到第一個關聯能力後就停止定錨
                
                if ability_candidates:
                    ab_name = ability_candidates[0]
                    # 執行目錄級檢測與分發
                    with self.engine.driver.session() as sub_session:
                        has_m = sub_session.run("MATCH (a:Ability {name: $n})-[:HAS_MILESTONE]-(m:Milestone) RETURN count(m) AS cnt", n=ab_name).single()['cnt']
                        
                        if has_m > 0:
                            ability_names = [ab_name]
                        else:
                            print(f"💡 定錨至目錄節點「{ab_name}」，正在同步聚合所屬領域發展指標...")
                            peer_query = """
                            MATCH (a:Ability {name: $n})
                            MATCH (a)-[:HAS_ABILITY|HAS_SUBDOMAIN]-(s:Subdomain)-[:HAS_ABILITY]->(peer:Ability)
                            WHERE EXISTS { (peer)-[:HAS_MILESTONE]-() }
                            RETURN DISTINCT peer.name AS name
                            """
                            peers = sub_session.run(peer_query, n=ab_name)
                            ability_names = [r['name'] for r in peers]
                    print(f"📊 最終參與時序分析的能力名單: {ability_names}")
                else:
                    ability_names = [ability]
            
            elif domain:
                # 兼容大領域與子領域查詢，僅抓取有里程碑的能力
                d_query = """
                MATCH (anchor)
                WHERE (anchor:Domain OR anchor:Subdomain) AND anchor.name = $domain
                OPTIONAL MATCH (anchor)-[:HAS_SUBDOMAIN|HAS_ABILITY*1..2]->(a:Ability)
                WHERE EXISTS { (a)-[:HAS_MILESTONE]-(:Milestone) }
                RETURN DISTINCT a.name AS name
                LIMIT 20
                """
                res = session.run(d_query, domain=domain)
                ability_names = [r['name'] for r in res if r['name']]
            
            if not ability_names:
                return {
                    "message": f"未找到「{domain or ability}」之關聯發展常模", 
                    "target_age_months": age_months,
                    "developmental_map": []
                }

            # 2. 為每個能力抓取「前、中、後」時序，並標記 LLM 語意狀態
            timeline_results = []
            for ab_name in list(set(ability_names)):
                ability_entry = {"ability": ab_name, "timeline": {}}
                
                params = {"ab_name": ab_name, "age": age_months}
                m_query = """
                MATCH (a:Ability {name: $ab_name})-[:HAS_MILESTONE]-(m:Milestone)
                RETURN m.expected_behavior AS behavior, m.age_min_month AS min, m.age_max_month AS max
                ORDER BY min
                """
                all_m = [dict(r) for r in session.run(m_query, **params)]
                if not all_m: continue

                # 分類：Achieved (已完成), Current_Target (當前目標), Next_Step (進階目標)
                # 每個分類取最近的一項，以防 Context 過載
                ability_entry["timeline"]["achieved"] = [m for m in all_m if m['max'] < age_months][-1:]
                ability_entry["timeline"]["current_target"] = [m for m in all_m if m['min'] <= age_months <= m['max']]
                ability_entry["timeline"]["next_step"] = [m for m in all_m if m['min'] > age_months][:1]
                
                # 如果這三類完全沒資料則忽略
                if not any(ability_entry["timeline"].values()): continue
                
                timeline_results.append(ability_entry)

            # 按能力名稱排序
            timeline_results.sort(key=lambda x: x['ability'])

            return {
                "target_age_months": age_months,
                "status": "success",
                "developmental_map": timeline_results
            }

    def print_markdown_report(self, analysis_result):
        """
        產出「以能力為中心」的因果判定報告
        """
        print("\n# 🩺 臨床因果路徑分析報告 (Ability-Centric)")
        
        clusters = analysis_result['ability_centric_clusters']
        # 優先顯示因果鏈完整的項目
        clusters.sort(key=lambda x: x['is_causal_complete'], reverse=True)
        
        print("## ✅ 核心能力對接路徑 (Causal Paths)")
        for chain in clusters:
            status = "🔗 [因果鏈完整]" if chain['is_causal_complete'] else "📍 [單點對接]"
            print(f"### {status} 能力核心：{chain['ability']}")
            
            if chain['observations']:
                print(f"  - **[因] 臨床觀察**：{', '.join(chain['observations'])}")
            
            if chain['goals'] or chain['recommendations']:
                print(f"  - **[果] 訓練對策**：")
                if chain['goals']: print(f"    - 方向：{', '.join(chain['goals'])}")
                if chain['recommendations']: print(f"    - 建議：{', '.join(chain['recommendations'])}")

            if chain['milestones']:
                m = chain['milestones'][0]
                print(f"  - **📚 發展依據 (Milestone)**：{m['behavior']}")
            print("")

        if analysis_result['unmapped_observations']:
            print("## ❓ 未命中的臨床片段 (Gaps)")
            for gap in analysis_result['unmapped_observations']:
                print(f"  - [{gap['category']}] {gap['text']}")
        
        print("\n---")
        print("*報告由 Clinical Bridge Analyzer (Ability Clustering 版) 生成*")

import re # 確保 re 被導入

if __name__ == "__main__":
    analyzer = ClinicalBridgeAnalyzer()
    
    # 測試多維度輸入
    obs = "個案在團體情境下難以與同儕發起有意義的口語互動。操作教具時指尖力道控制不佳。"
    goals = "增加主動表達的意願和機會。提升細小物品操作的穩定度。"
    
    print("正在執行全鏈路分析 (Qwen3 模型)...")
    result = analyzer.analyze_clinical_context(observations=obs, goals=goals)
    
    # 輸出人類報告
    analyzer.print_markdown_report(result)
    
    # 輸出 LLM Payload (可由外部程式存成檔案或傳入 API)
    # print("\n--- LLM Context Payload ---")
    # print(analyzer.format_llm_prompt(result))
    
    analyzer.engine.close()

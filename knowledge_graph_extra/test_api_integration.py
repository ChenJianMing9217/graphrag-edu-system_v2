import json
import os
from clinical_api import ClinicalBridgeService

def test_api():
    # 1. 初始化服務 (底層會自動對接 Neo4j 與 Qwen3)
    print("🚀 正在啟動 Clinical API 服務...")
    service = ClinicalBridgeService()
    
    # 2. 準備測試數據 (行為觀察 + 訓練方向)
    test_obs = "個案在團體情境下難以與同儕發起有意義的口語互動。操作教具時指尖力道控制不佳。"
    test_goal = "增加主動表達的意願和機會。提升細小物品操作的穩定度。"
    
    print("🔍 執行「深度挖掘模式 (Deep Search)」與「里程碑提取 (With Milestones)」...")
    
    # 3. 調用核心介面 (開啟深挖與里程碑)
    result = service.get_llm_payload(
        observations=test_obs,
        training_goals=test_goal,
        deep_search=True,
        with_milestones=True
    )
    
    # 4. 顯示結果 (這就是您之後要餵給 LLM 的資料)
    print("\n✅ API 成功產出 LLM 專用數據結構：")
    print("-" * 50)
    
    # 我們只打印聚群結果的前幾個，展示結構
    clusters = result.get('ability_centric_clusters', [])
    print(f"📊 總共成功識別跨領域能力群組：{len(clusters)} 個")
    
    # 轉成視覺化的 JSON 呈現
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("-" * 50)
    print("✨ 測試完成！這組 JSON 已經包含所有能力路徑、里程碑與因果關係，可直接作為 LLM 的 Context。")
    
    # 5. 關閉連線
    service.close()

if __name__ == "__main__":
    test_api()

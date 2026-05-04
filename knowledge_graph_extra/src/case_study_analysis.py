import os
import sys
import json
import re
from clinical_bridge_analyzer import ClinicalBridgeAnalyzer

def analyze_individual_case():
    analyzer = ClinicalBridgeAnalyzer()
    
    # --- 1. 個案原始報告文本 (由用戶提供) ---
    obs_text = """
    個案的溝通方式以肢體動作、聲音及部分口語為主，可以表達需求、
    拒絕、簡單社交互動和傳遞訊息等溝通功能，未做出詢問和表達情緒
    ，對他人有主動溝通意圖，溝通效度可部分被理解。
    個案目前可做出指物、揮手打招呼和搖頭表達拒絕。看到認識的動物會發出
    該動物之叫聲，媽媽表示個案於家中能說出人稱詞彙、ㄋㄟㄋㄟ、棒棒，
    表達詞彙量少且大多以"恩"加上動作表達需求。
    """
    
    goal_text = """
    增加主動表達的意願和機會
    模仿發聲或模仿語音（發聲遊戲）
    以單字加上手勢動作表達，如：我要
    加強詞彙使用，如：常見物品名稱、功能性詞彙
    """
    
    print("\n" + "="*80)
    print("   📋 臨床實例分析報告 (Case Mapping Analysis)   ")
    print("="*80)
    
    # 執行全量因果分析
    result = analyzer.analyze_clinical_context(
        observations=obs_text,
        goals=goal_text
    )
    
    # --- 2. 顯示判定結果 (自動併攏相同能力的行為與目標) ---
    analyzer.print_markdown_report(result)

    # --- 3. 輸出 LLM 可用的 Context (範例呈現) ---
    print("\n[LLM 知識定錨範例 (Ready for Grounding)]")
    # print(json.dumps(result, ensure_ascii=False, indent=2))
    print("   ✅ JSON 聚群已產出，相同能力的觀察與目標已自動合併。")
    print("="*80)

    analyzer.engine.close()

if __name__ == "__main__":
    analyze_individual_case()

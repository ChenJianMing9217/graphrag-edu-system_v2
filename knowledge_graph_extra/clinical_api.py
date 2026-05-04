import os
import sys
import json

# 自動定位 src 資料夾，確保 import 正常
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'src'))

from clinical_bridge_analyzer import ClinicalBridgeAnalyzer

class ClinicalBridgeService:
    """
    臨床對稱與證明 API 服務模組 (V4 Qwen3 版)
    
    用途：
    輸入行為、目標或建議，直接返回給 LLM 參考的結構化 JSON 資料。
    """
    def __init__(self, encoder=None):
        self.analyzer = ClinicalBridgeAnalyzer(encoder=encoder)

    def get_llm_payload(self, observations="", training_goals="", recommendations="",
                        deep_search=False, with_milestones=True):
        """
        核心介面函數：
        1. 執行語意對接
        2. 執行能力聚類與因果判定
        
        :param deep_search: 是否深挖 (找更多、更深層的關聯能力)
        :param with_milestones: 是否包含相關的臨床發展里程碑
        """
        analysis_result = self.analyzer.analyze_clinical_context(
            observations=observations,
            goals=training_goals,
            activities=recommendations,
            deep_search=deep_search,
            with_milestones=with_milestones
        )
        
        return analysis_result

    def get_developmental_map(self, domain=None, ability=None, age_months=None):
        """
        常模查詢介面：
        獲取特定 月齡/領域/能力 的發展時序地圖 (前、中、後)。
        
        :param domain: 領域名稱 (如: 粗大動作)
        :param ability: 能力名稱 (如: 詞彙表達)
        :param age_months: 目標月齡 (int)
        """
        if age_months is None:
            return {"error": "必須提供 age_months (月齡) 參數"}
            
        return self.analyzer.get_developmental_timeline(
            domain=domain,
            ability=ability,
            age_months=age_months
        )

    def close(self):
        """關閉資料庫連線"""
        self.analyzer.engine.close()

# --- 使用範例 ---
# from clinical_api import ClinicalBridgeService
# service = ClinicalBridgeService()
# data = service.get_llm_payload(observations="小朋友上樓梯不穩")
# print(json.dumps(data, ensure_ascii=False))
# service.close()

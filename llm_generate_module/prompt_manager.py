"""
LLM 提示詞和參數管理器
根據 DST 和 Task 類型動態選擇提示詞和生成參數
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class LLMGenerationConfig:
    """針對不同 DST + Task 的 LLM 生成配置（可覆蓋欄位使用 Optional，None 代表「不覆蓋 / 使用預設」）"""

    # 生成參數（可覆蓋欄位）
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # 提示詞模板（直接儲存組裝後字串，不參與可覆蓋合併）
    system_prompt_template: str = ""
    user_prompt_template: str = ""

    # 上下文處理（可覆蓋欄位）
    max_context_items: Optional[int] = None
    context_format_style: Optional[str] = None  # "detailed" | "concise" | "structured"

    # 回應風格（可覆蓋欄位）
    response_style: Optional[str] = None  # "professional" | "friendly" | "concise" | "detailed" | "step_by_step" | "explanatory" | "comprehensive" ...
    include_examples: Optional[bool] = None
    include_caution: Optional[bool] = None  # 是否包含模糊度警告

    # 模糊相關資訊（用於 build_user_prompt，不作為合併權重）
    is_ambiguous: bool = False
    active_domains: List[str] = field(default_factory=list)
    task_options: List[str] = field(default_factory=list)
    # 新設計 (Phase D)：clarify 類型（附加追問屬性，不阻塞回答）
    clarify_type: Optional[str] = None     # None | "DOMAIN_HARD" | "TASK_SOFT" | "SLOT_REGION" | "CONTEXT_MISSING"
    clarify_reason: Optional[str] = None

    @classmethod
    def default_values(cls) -> "LLMGenerationConfig":
        """取得一份帶有系統預設值的新 Config，用於回填 None 欄位"""
        return cls(
            temperature=1.0,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.3,   # 減少重複詞彙（範圍：-2.0 到 2.0，正值減少重複）
            presence_penalty=0.1,    # 減少重複主題（範圍：-2.0 到 2.0，正值減少重複）
            max_context_items=10,
            context_format_style="detailed",
            response_style="professional",
            include_examples=False,
            include_caution=False,
            is_ambiguous=False,
            active_domains=[],
            task_options=[],
        )

    def with_defaults(self) -> "LLMGenerationConfig":
        """
        回填所有可覆蓋欄位的 None 為預設值。
        - 外部使用時建議使用本方法，確保不會取得 None（行為與原本固定預設值版本相容）。
        """
        base = self.default_values()
        return LLMGenerationConfig(
            # 可覆蓋欄位：None 則回退到 base
            temperature=self.temperature if self.temperature is not None else base.temperature,
            max_tokens=self.max_tokens if self.max_tokens is not None else base.max_tokens,
            top_p=self.top_p if self.top_p is not None else base.top_p,
            frequency_penalty=(
                self.frequency_penalty
                if self.frequency_penalty is not None
                else base.frequency_penalty
            ),
            presence_penalty=(
                self.presence_penalty
                if self.presence_penalty is not None
                else base.presence_penalty
            ),
            max_context_items=(
                self.max_context_items
                if self.max_context_items is not None
                else base.max_context_items
            ),
            context_format_style=(
                self.context_format_style
                if self.context_format_style is not None
                else base.context_format_style
            ),
            response_style=(
                self.response_style
                if self.response_style is not None
                else base.response_style
            ),
            include_examples=(
                self.include_examples
                if self.include_examples is not None
                else base.include_examples
            ),
            include_caution=(
                self.include_caution
                if self.include_caution is not None
                else base.include_caution
            ),
            # 非可覆蓋欄位：維持原本行為
            system_prompt_template=self.system_prompt_template,
            user_prompt_template=self.user_prompt_template,
            is_ambiguous=self.is_ambiguous,
            active_domains=list(self.active_domains) if self.active_domains else [],
            task_options=list(self.task_options) if self.task_options else [],
        )


class LLMPromptManager:
    """管理不同 DST + Task 組合的提示詞和參數"""

    # JSON 設定檔路徑（與本檔案同目錄）
    _CONFIG_PATH = os.path.join(os.path.dirname(__file__), "prompt_config.json")

    def __init__(self):
        """初始化提示詞管理器，從 JSON 設定檔載入所有配置"""
        self._load_config()

    def _load_config(self):
        """從 prompt_config.json 載入所有設定"""
        with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Task A–M 名稱映射（中文，領域中性）
        self.TASK_NAME_ZH: Dict[str, str] = cfg["task_name_zh"]

        # Scope 名稱映射（中文）
        self.SCOPE_NAME_ZH: Dict[str, str] = cfg["scope_name_zh"]

        # semantic_flow 對應的基礎 LLM 配置表
        self._SEMANTIC_FLOW_CONFIG: Dict[str, Dict] = cfg["semantic_flow_config"]

        # retrieval_action 對應的上下文配置表
        self._RETRIEVAL_ACTION_CONFIG: Dict[str, Dict] = cfg["retrieval_action_config"]

        # Task A–M 專屬配置表（不含 scope）
        self._TASK_CONFIG: Dict[str, Dict] = cfg["task_config"]

        # Scope 專屬配置表
        self._SCOPE_CONFIG: Dict[str, Dict] = cfg["scope_config"]

        # build_user_prompt 中，各種 response_style 對應的提示模板
        self._USER_PROMPT_TEMPLATES: Dict[str, str] = cfg["user_prompt_templates"]

        # 查詢改寫提示詞
        self.QUERY_REWRITE_PROMPT: str = cfg["query_rewrite_prompt"]

        # 對話歷史濃縮提示詞（用於 Contextualization）
        self.QUERY_CONDENSE_PROMPT: str = cfg["query_condense_prompt"]

    # 使用者回應長度偏好的倍率（套在 max_tokens 上，base=1200 tokens）
    # auto: 1500 tokens 給 LLM 自由空間 / concise: 600 / standard: 1200 / detailed: 2160
    LENGTH_MULTIPLIER = {
        "auto":     1.25,
        "concise":  0.5,
        "standard": 1.0,
        "detailed": 1.8,
    }

    # 給 LLM 的長度指引（塞進 system prompt 末段）
    # 強硬措辭 + 絕對上限（中文 1 字 ≈ 1.5 token，buffer 預留 10-15% 確保不被 max_tokens 截斷）
    LENGTH_GUIDE = {
        "auto": (
            "【回答長度規範】\n"
            "- **絕對上限:1000 字**（請務必嚴守,不論問題多複雜都不可超過）\n"
            "- 自適應原則:\n"
            "  · 簡單事實 / 定義 -> 150-250 字\n"
            "  · 評估解讀 / 觀察說明 -> 300-500 字\n"
            "  · 訓練建議 / 多步驟方法 -> 500-800 字\n"
            "  · 全面解讀 / 多領域整合 -> 800-1000 字\n"
            "- 篇幅控制策略:用條列、表格代替冗長段落；避免客套、過多強調；"
            "短追問請延續上輪深度但**不重複**已答內容。"
        ),
        "concise": (
            "【回答長度規範】\n"
            "- **絕對上限:350 字**（請務必嚴守）\n"
            "- 目標:200-300 字精簡回答\n"
            "- 重點清楚即可,不展開細節,不舉例,避免客套。"
        ),
        "standard": (
            "【回答長度規範】\n"
            "- **絕對上限:700 字**（請務必嚴守）\n"
            "- 目標:400-600 字\n"
            "- 結構清楚、重點分明,不過度展開。"
        ),
        "detailed": (
            "【回答長度規範】\n"
            "- **絕對上限:1300 字**（請務必嚴守）\n"
            "- 目標:800-1200 字深入回答\n"
            "- 含完整脈絡、舉例與建議。"
        ),
    }
    def get_config(
        self,
        semantic_flow: str,
        retrieval_action: str,
        task_label: Optional[str],
        scope_label: Optional[str],
        is_ambiguous: bool,
        is_multi_domain: bool,
        top_domain: str,
        active_domains: Optional[List[str]] = None,
        domain_distribution: Optional[Dict[str, float]] = None,
        clarify_type: Optional[str] = None,
        clarify_reason: Optional[str] = None,
        response_length: str = "standard",
    ) -> LLMGenerationConfig:
        """
        根據 DST 和 Task 獲取配置

        Args:
            semantic_flow: "continue" | "shift_soft" | "shift_hard"
            retrieval_action: "NARROW_GRAPH" | "CONTEXT_FIRST" | "WIDE_IN_DOMAIN" | "DUAL_OR_CLARIFY"
            task_label: Task 類型（可選）
            scope_label: Scope 類型（可選）
            is_ambiguous: 是否模糊
            is_overview_query: 是否為整體查詢
            is_multi_domain: 是否多領域
            top_domain: 頂級領域
            active_domains: 活躍領域列表（可選，用於模糊引導）
            domain_distribution: 領域分布（可選，用於模糊引導）

        Returns:
            LLMGenerationConfig（已回填預設值）
        """
        # 基礎配置（根據 semantic_flow）
        base_config = self._get_base_config_by_flow(semantic_flow, is_ambiguous)

        # 根據 retrieval_action 調整
        retrieval_config = self._get_config_by_retrieval_action(retrieval_action)

        # 根據 task 調整
        task_config = (
            self._get_config_by_task(task_label, scope_label)
            if task_label
            else LLMGenerationConfig()
        )

        # 根據特殊情況調整
        special_config = self._get_special_config(
            is_multi_domain=is_multi_domain,
            is_ambiguous=is_ambiguous,
            top_domain=top_domain,
        )

        # 合併配置（優先級：task > special > retrieval > base）
        merged_config = self._merge_configs(
            base_config,
            retrieval_config,
            task_config,
            special_config,
        )

        # 在對外輸出前回填預設值，保留原先「欄位一定有值」的使用習慣
        final_config = merged_config.with_defaults()

        # 構建系統提示詞
        final_config.system_prompt_template = self.build_system_prompt(
            task_label,
            scope_label,
            top_domain,
            is_ambiguous,
            active_domains or [],
            domain_distribution or {},
            retrieval_action=retrieval_action,
        )

        # 將模糊相關資訊添加到 config 中（用於 build_user_prompt）
        final_config.is_ambiguous = is_ambiguous
        final_config.active_domains = active_domains or []
        # 新設計 (Phase D)：附加 clarify 屬性，生成層根據此決定追問策略
        final_config.clarify_type = clarify_type
        final_config.clarify_reason = clarify_reason

        # [NEW] 套用使用者回應長度偏好
        rl_key = response_length if response_length in self.LENGTH_MULTIPLIER else "standard"
        mult = self.LENGTH_MULTIPLIER[rl_key]
        if final_config.max_tokens:
            final_config.max_tokens = max(256, int(final_config.max_tokens * mult))
        # system prompt 末尾加上長度軟指引（讓 LLM 行為配合 max_tokens）
        guide = self.LENGTH_GUIDE[rl_key]
        if final_config.system_prompt_template:
            final_config.system_prompt_template += f"\n\n{guide}"
        else:
            final_config.system_prompt_template = guide

        return final_config

    def _get_base_config_by_flow(
        self, semantic_flow: str, is_ambiguous: bool
    ) -> LLMGenerationConfig:
        """根據 semantic_flow 取得基礎配置，主體規則改為查表"""
        config_dict = self._SEMANTIC_FLOW_CONFIG.get(
            semantic_flow, self._SEMANTIC_FLOW_CONFIG["shift_hard"]
        )
        config = LLMGenerationConfig(**config_dict)

        # 「continue + 模糊」時，略微降低溫度以提高穩定性
        if semantic_flow == "continue" and is_ambiguous:
            config.temperature = 0.15

        return config

    def _get_config_by_retrieval_action(
        self, retrieval_action: str
    ) -> LLMGenerationConfig:
        """根據 retrieval_action 取得配置，使用查表避免多層 if-elif"""
        # 預設視為 DUAL_OR_CLARIFY 行為
        config_dict = self._RETRIEVAL_ACTION_CONFIG.get(
            retrieval_action, self._RETRIEVAL_ACTION_CONFIG["DUAL_OR_CLARIFY"]
        )
        return LLMGenerationConfig(**config_dict)

    def _get_config_by_task(
        self, task_label: str, scope_label: Optional[str]
    ) -> LLMGenerationConfig:
        """根據 task A–M 和 scope 取得配置，改為由 Task / Scope 兩個表組合"""
        # 1. 先從 Task 表取得設定
        task_dict = self._TASK_CONFIG.get(
            task_label, self._TASK_CONFIG["_default"]
        )
        task_config = LLMGenerationConfig(**task_dict)

        # 2. 再根據 Scope 表進一步疊加（若有）
        if scope_label:
            scope_dict = self._SCOPE_CONFIG.get(scope_label)
            if scope_dict:
                scope_config = LLMGenerationConfig(**scope_dict)
                task_config = self._merge_configs(task_config, scope_config)

        return task_config

    def _get_special_config(
        self,
        is_multi_domain: bool,
        is_ambiguous: bool,
        top_domain: str = "",
    ) -> LLMGenerationConfig:
        """根據特殊情況（整體查詢、多領域、模糊）取得額外配置"""
        config = LLMGenerationConfig()

        # 整體查詢：偏向長輸出與結構化
        if top_domain == "整體概況":
            config.max_tokens = 3000
            config.max_context_items = 20
            config.context_format_style = "structured"
            config.response_style = "comprehensive"

        # 多領域：強制使用結構化輸出，並稍微放寬上下文數量限制 (設為 20 以避免 8k 模型溢位)
        if is_multi_domain:
            config.context_format_style = "structured"
            # 指示：原本 50 在 GPT-3.5 8k 或類似模型下會導致 input + output > 8192
            config.max_context_items = 20

        # 模糊查詢：鼓勵更穩定的輸出，並附帶警示
        if is_ambiguous:
            config.temperature = 0.15
            config.include_caution = True

        return config

    def _merge_configs(self, *configs: LLMGenerationConfig) -> LLMGenerationConfig:
        """
        合併多個配置（後面的優先級更高）

        - 使用 None 代表「不設定 / 不覆蓋」，因此：
          * 0、False、空字串 "" 都會被視為「有效的覆蓋值」並被保留。
        """
        merged = LLMGenerationConfig()

        # 定義會參與「可覆蓋」合併的欄位名稱
        overridable_fields = (
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_context_items",
            "context_format_style",
            "response_style",
            "include_examples",
            "include_caution",
        )

        for config in configs:
            if config is None:
                continue
            for field_name in overridable_fields:
                value = getattr(config, field_name)
                # 僅當 value 不是 None 時才覆蓋（允許 0 / False / ""）
                if value is not None:
                    setattr(merged, field_name, value)

        return merged

    # === System Prompt 組裝 ===

    def build_system_prompt(
        self,
        task_label: Optional[str],
        scope_label: Optional[str],
        top_domain: str,
        is_ambiguous: bool,
        active_domains: Optional[List[str]] = None,
        domain_distribution: Optional[Dict[str, float]] = None,
        retrieval_action: Optional[str] = None,
    ) -> str:
        """
        構建系統提示詞

        - 改為以多個常數片段與條件片段組裝，而不是大量 base_prompt +=
        """
        # 核心固定片段
        intro = (
            "你是一位專業的早療系統助手，能夠根據評估報告和檢索到的相關資訊，"
            "為家長和治療師提供專業的建議和回答。\n\n"
        )

        language_requirement = (
            "【語言規範】務必繁體中文 + 台灣在地用語（醫護/早療專業語境）。"
            "常見替換：信息→訊息、視頻→影片、質量→品質、康復→復健/療育、水平→程度/能力。"
            "嚴禁簡體字與大陸用語。\n\n"
        )

        style_guidance = (
            "【回應格式要求】\n"
            "1. 使用 Markdown 格式：根據內容邏輯使用合適的分段標題（例如 `### 建議與作法`）、`**粗體**` 強調關鍵重點、`- ` 條列細節。\n"
            "2. **先回答問題，再提供建議**：優先處理用戶的核心疑問，行有餘力再進行延伸補充，確保重點不被掩蓋。\n"
            "3. 使用友善、專業且溫暖的語氣，讓家長感到被引導與支持。務必呈現台灣醫護、教育體系的專業與溫度。\n"
            "4. **請自然地融入背景資訊，絕不要使用「根據資料一...」或「依照提供的紀錄...」等生硬的起手式來回答。**\n\n"
        )

        content_guidance = (
            "【回答內容建議】\n"
            "1. 摘要並整合重點：請先理解檢索到的內容，再用自己的話串接成連貫、易懂的建議，避免散碎的資訊堆疊。\n"
            "2. 語氣自然流暢：避免過於生硬、機械化的開場（如「以下為回答：」），直接切入重點或用自然的過場語句引導。\n"
            "3. 結構分明：利用標題（###）讓家長一眼看到目前在討論哪個面向（例如：評估結果、居家練習）。\n"
            "4. 延伸補充放置於結尾：如果有相關但非核心的提醒，請放在回答的最末段，並賦予一個自然、具體的標題。\n\n"
        )

        parts: List[str] = [
            intro,
            language_requirement,
            style_guidance,
            content_guidance,
        ]

        # Task A–M 角色片段
        if task_label:
            task_name = self.TASK_NAME_ZH.get(task_label, task_label)
            parts.append(
                f"\n\n你的專長是「{task_name}」，請根據使用者需求提供相應的專業建議與說明。"
            )

        # Scope 範圍片段
        if scope_label:
            scope_name = self.SCOPE_NAME_ZH.get(scope_label, scope_label)
            if scope_label == "S_overview":
                parts.append(
                    f"\n\n本次查詢是「{scope_name}」，請整合多個領域的資訊，提供全面但結構化的回答。"
                )
            elif scope_label == "S_domain":
                parts.append(
                    f"\n\n本次查詢聚焦「{scope_name}」（{top_domain}），請針對該領域深入回答。"
                )
            elif scope_label == "S_multi_domain":
                parts.append(
                    f"\n\n本次查詢是「{scope_name}」，請分別針對各相關領域回答，必要時說明彼此關聯。"
                )

        # 模糊查詢處理片段
        active_domains = active_domains or []
        if is_ambiguous:
            ambiguity_parts = [
                "\n\n【模糊查詢處理】本次查詢可能較為模糊，請按照以下方式處理：\n",
                "1. 首先用親切、自然的語氣說明您理解查詢可能涉及多個面向，不要用制式化的開場白。\n",
                "2. **使用親切自然的反問**，避免制式化的問法。例如：\n",
                "   - 好的：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想了解哪個部分呢？」\n",
                "   - 好的：「您還想了解什麼呢？」或「還有其他想知道的嗎？」\n",
                "   - 避免：「請問您是問 [領域名稱] 相關的 [任務類型] 嗎？」（太制式化）\n",
                "3. 如果有多個可能的領域，用自然的方式逐一詢問，讓家長選擇。例如：\n",
                "   - 「您是想了解粗大動作的評估結果嗎？」\n",
                "   - 「還是想問精細動作的訓練建議？」\n",
                "4. 提供可能的任務類型選項時，用親切的語氣，例如：「您是想了解評估結果，還是想獲得訓練建議呢？」\n",
                "5. 鼓勵使用 Markdown 語法（如表格、加粗、條列符號 - 或 *）讓回答更生動、結構更清楚。\n",
                "6. 語氣要親切、自然、溫暖，就像朋友在聊天一樣，不要讓家長感到被質疑或困惑。\n",
            ]

            if active_domains and len(active_domains) > 1:
                domains_text = "、".join(active_domains[:5])  # 最多顯示 5 個
                ambiguity_parts.append(f"\n可能的相關領域包括：{domains_text}。\n")
                ambiguity_parts.append(
                    "請用親切自然的語氣反問這些領域，例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」讓家長選擇。\n"
                )

            parts.extend(ambiguity_parts)

        # 在地資源反問處理
        if retrieval_action == "LOCAL_RESOURCE_CLARIFY":
            parts.append(
                "\n\n【在地資源反問】目前的查詢涉及在地醫療或早療資源，但缺乏地區資訊。\n"
                "請用親切、溫暖且自然的語氣詢問用戶所在的縣市（例如：台北市、台中市等）。\n"
                "不要進行醫療建議或解釋報告，只需單純詢問地區即可。\n"
            )

        # 整體查詢補充說明
        if top_domain == "整體概況":
            parts.append(
                "\n\n本次查詢是整體性查詢，請整合多個領域的資訊，提供全面但結構化的回答。"
            )

        # 結尾語氣提醒
        parts.append(
            "\n\n請用友善、專業、溫暖的語氣回答問題，讓家長感到被理解和支持。"
        )

        return "".join(parts)

    # === User Prompt 組裝 ===

    def build_user_prompt(
        self,
        user_query: str,
        retrieved_context: List[Dict],
        config: LLMGenerationConfig,
        is_ambiguous: bool = False,
        active_domains: Optional[List[str]] = None,
        task_options: Optional[List[str]] = None,
        prev_context: Optional[List[Dict]] = None,
    ) -> str:
        """
        構建用戶提示詞（包含上下文）

        - 先依 score 排序 retrieved_context，再截取 max_context_items
        - response_style 對應的提示改由模板字典管理
        - 模糊查詢引導文字抽成共用 helper
        - prev_context: 上一輪檢索結果（輔助背景），以較低權重呈現
        """
        # 確保 config 欄位已回填預設值，外部行為維持原本「一定有值」的假設
        resolved_config = config.with_defaults()

        # 預先產生模糊查詢引導文字（可能為空字串）
        ambiguity_guidance = self._build_ambiguity_guidance(
            is_ambiguous=is_ambiguous,
            active_domains=active_domains or [],
            task_options=task_options or [],
        )

        # 沒有檢索上下文的情況
        if not retrieved_context:
            if is_ambiguous:
                return user_query + ambiguity_guidance
            return user_query

        # 先依 score 由高到低排序，再取前 max_context_items 筆
        def get_score(x):
            if isinstance(x, dict):
                return x.get("score", 0.0)
            return getattr(x, "score", 0.0)

        sorted_context = sorted(
            retrieved_context,
            key=get_score,
            reverse=True,
        )
        max_items = resolved_config.max_context_items or len(sorted_context)
        top_context = sorted_context[:max_items]

        # 根據格式風格格式化上下文（使用 formatter strategy）
        context_text = self._format_context_by_style(
            top_context,
            resolved_config.context_format_style or "structured",
        )

        # 附加上一輪參考 context（若有）
        if prev_context:
            prev_text = self._format_context_by_style(
                prev_context,
                resolved_config.context_format_style or "structured",
            )
            context_text += (
                "\n\n---\n"
                "【前次對話參考】（以下為上一輪檢索的相關內容，僅供輔助參照，請以上方本次檢索結果為主要依據）\n"
                + prev_text
            )

        # 根據 response_style 從模板表取得對應模板
        style_key = resolved_config.response_style or "_default"
        template = self._USER_PROMPT_TEMPLATES.get(
            style_key,
            self._USER_PROMPT_TEMPLATES["_default"],
        )

        user_prompt = template.format(context=context_text, query=user_query)

        # 若為模糊查詢，附加引導提示
        if is_ambiguous and ambiguity_guidance:
            user_prompt += ambiguity_guidance

        # 新設計 (Phase D)：clarify_type 驅動的追問引導（附加在回答末尾）
        clarify_guidance = self._build_clarify_guidance(
            clarify_type=getattr(resolved_config, "clarify_type", None),
            active_domains=active_domains or [],
            task_options=task_options or [],
        )
        if clarify_guidance:
            user_prompt += clarify_guidance

        return user_prompt

    def _build_clarify_guidance(
        self,
        clarify_type: Optional[str],
        active_domains: List[str],
        task_options: List[str],
    ) -> str:
        """
        根據 clarify_type 附加對應的追問引導（新設計：clarify 屬性）。
        目的：讓 LLM 在回答正文後，自然加入一句引導，而非阻塞回答。
        """
        if not clarify_type:
            return ""

        if clarify_type == "DOMAIN_HARD":
            options = "、".join(active_domains) if active_domains else "粗大動作、認知、情緒、轉介資源等"
            return (
                "\n\n【系統提示：請使用者釐清】\n"
                f"使用者的問題較為開放，請在回答中先誠實說明無法精確判斷意圖，並主動詢問使用者想了解的方向（如：{options}）。"
            )

        if clarify_type == "CONTEXT_MISSING":
            return (
                "\n\n【系統提示：補問前文指涉】\n"
                "使用者在首輪使用了「對了／剛才／那個」等接續詞，但沒有實際的前次對話。請先就 query 的字面意涵給出初步回答，"
                "並在結尾自然地問：「請問您剛才提到的是指報告中的哪個部分呢？」"
            )

        if clarify_type == "SLOT_REGION":
            return (
                "\n\n【系統提示：補問所在地區】\n"
                "請先給出全台共通的資源/補助類型介紹，然後在回答末尾自然地加入一句："
                "「如果您方便告訴我所在縣市，我可以推薦更具體的當地資源與申請窗口。」"
            )

        if clarify_type == "TASK_SOFT":
            opt_str = "、".join(task_options) if task_options else "（多個可能方向）"
            return (
                "\n\n【系統提示：綜合回答 + 引導深入】\n"
                f"此 query 可能對應多個方向（{opt_str}）。請給出涵蓋主要候選方向的綜合回答，並在結尾自然地問："
                "「您比較想深入了解哪個部分呢？」"
            )

        if clarify_type == "OUT_OF_DOMAIN":
            return (
                "\n\n【系統提示：使用者偏離早療話題】\n"
                "使用者目前的問題似乎與兒童早療、發展評估、報告解讀等本系統涵蓋的範疇無關。"
                "請以友善的語氣：\n"
                "(1) 簡短、溫和地回應使用者的問題（若適合的話），不要冷漠拒絕；\n"
                "(2) 說明本系統專注於早療諮詢，無法深入回答此類問題；\n"
                "(3) 邀請使用者回到早療相關話題，例如：\n"
                "「如果您有關於孩子發展評估、報告解讀、居家訓練或轉介資源的問題，"
                "我可以更深入地幫您。」\n"
                "請保持耐心和體貼，避免讓使用者感到被拒絕。"
            )

        return ""

    def _build_ambiguity_guidance(
        self,
        is_ambiguous: bool,
        active_domains: List[str],
        task_options: List[str],
    ) -> str:
        """
        建立「模糊查詢」時附加在 user prompt 後方的引導說明文字。

        - 抽成共用 helper，供有無上下文兩支流程共用。
        """
        if not is_ambiguous:
            return ""

        lines: List[str] = []
        lines.append("\n\n【引導提示】由於查詢較為模糊，請在回答中：\n")
        lines.append(
            "1. **使用親切自然的反問**，避免制式化問法。例如：「您是想了解 [領域] 的 [內容] 嗎？」或「關於 [領域]，您想問什麼呢？」\n"
        )

        if active_domains and len(active_domains) > 1:
            lines.append(
                f"2. 可能的領域包括：{', '.join(active_domains[:5])}，請用親切的語氣逐一詢問這些領域。\n"
            )
            lines.append(
                "   例如：「您是想了解 [領域1] 的評估結果嗎？」或「還是想問 [領域2] 的訓練建議？」\n"
            )

        # 若有 task_options，可鼓勵模型自然地拋出幾個任務型選項（不強制使用）
        if task_options:
            lines.append(
                "3. 可以自然地提供幾種可能的需求方向，協助家長聚焦，例如：\n"
            )
            preview = "、".join(task_options[:5])
            lines.append(f"   「目前看起來可能與：{preview} 有關，您比較想先了解哪一部分呢？」\n")
            extra_index_base = 4
        else:
            extra_index_base = 3

        lines.append(
            f"{extra_index_base}. 提供任務類型選項時，用自然親切的語氣引導家長更清楚地表達需求。\n"
        )
        lines.append(
            f"{extra_index_base + 1}. 請用自己的話摘要和重新組織內容，不要直接引用原始文字。\n"
        )
        lines.append(
            f"{extra_index_base + 2}. 用白話文詳細解釋，篇幅以家長容易閱讀為主，不必刻意拉長。\n"
        )
        lines.append(
            f"{extra_index_base + 3}. 請用親切、自然、溫暖的語氣，就像朋友在聊天一樣，引導家長更清楚地表達需求。\n"
        )

        return "".join(lines)

    # === Context Formatter（Strategy Mapping） ===

    def _format_context_by_style(
        self, retrieved_context: List[Dict], style: str
    ) -> str:
        """
        根據格式風格格式化上下文
        """
        formatter_mapping = {
            "detailed": self._format_detailed_context,
            "concise": self._format_concise_context,
            "structured": self._format_structured_context,
        }
        formatter = formatter_mapping.get(style, self._format_structured_context)
        
        # 針對 ClinicalNorm 節點進行預先處理 (如果它是 JSON，則先轉為好讀 Markdown)
        for item in retrieved_context:
            try:
                if self._get_val(item, "label") == "ClinicalNorm":
                    raw_text = self._get_val(item, "text", "")
                    data = json.loads(raw_text)
                    if isinstance(data, dict):
                        # 判定是哪種 Clinical 資料
                        if "ability_centric_clusters" in data:
                            pretty_text = self._format_clinical_cluster(data)
                        elif "developmental_map" in data:
                            pretty_text = self._format_developmental_map(data)
                        else:
                            pretty_text = "\n".join([f"    ▪ {k}: {v}" for k, v in data.items() if v])
                        
                        # 寫回 item (若是 dict 則直接寫，若是 CandidateNode 則 setattr)
                        if isinstance(item, dict):
                            item["text"] = pretty_text
                        else:
                            setattr(item, "text", pretty_text)
            except:
                pass

        base_context = formatter(retrieved_context)

        # 檢查是否有圖譜上下文 (PCST)
        graph_info = ""
        for item in retrieved_context:
            metadata = self._get_val(item, "metadata", {})
            if isinstance(metadata, dict) and "subgraph_context" in metadata:
                graph_info = self._format_subgraph_context(metadata["subgraph_context"])
                break

        if graph_info:
            return base_context + "\n\n【知識圖譜關聯脈絡 (Connected Context)】\n" + graph_info

        return base_context

    def _format_subgraph_context(self, subgraph_edges: List[Dict[str, Any]]) -> str:
        """格式化 PCST 產生的子圖關聯"""
        lines = []
        for edge in subgraph_edges:
            src = edge["source"]
            target = edge["target"]
            rel = edge["relation"]
            lines.append(f"  - ({src['label']}: {src['name']}) --[{rel}]--> ({target['label']}: {target['name']})")
        return "\n".join(lines)

    def _format_clinical_cluster(self, data: Dict) -> str:
        """格式化「臨床能力聚類」資料"""
        lines = ["【專家分析：個案行為與臨床能力對接】"]
        clusters = data.get("ability_centric_clusters", [])
        for c in clusters:
            lines.append(f"### 📍 能力核心：{c.get('ability')}")
            if c.get("observations"):
                lines.append(f"  - **現狀觀察**：{', '.join(c['observations'])}")
            if c.get("milestones"):
                m = c["milestones"][0]
                lines.append(f"  - **發展基準**：{m.get('behavior')} ({m.get('age')}個月)")
            if c.get("goals") or c.get("recommendations"):
                lines.append(f"  - **建議對策**：{', '.join(c.get('goals', []) + c.get('recommendations', []))}")
            lines.append("")
            
        gaps = data.get("unmapped_observations", [])
        if gaps:
            lines.append(f"  *(註：另有 {len(gaps)} 項觀察暫無直接常模對照)*")
        return "\n".join(lines)

    def _format_developmental_map(self, data: Dict) -> str:
        """格式化「發展里程碑地圖」資料"""
        age = data.get("target_age_months")
        lines = [f"【專家知識：{age} 個月發展常模參考】"]
        maps = data.get("developmental_map", [])
        for item in maps:
            lines.append(f"### 📊 領域：{item.get('ability')}")
            tm = item.get("timeline", {})
            if tm.get("achieved"):
                lines.append(f"  - **前置能力 (已完成)**：{tm['achieved'][0].get('behavior')}")
            if tm.get("current_target"):
                lines.append(f"  - **當前目標 (發展中)**：{tm['current_target'][0].get('behavior')}")
            if tm.get("next_step"):
                lines.append(f"  - **進階目標 (未來方向)**：{tm['next_step'][0].get('behavior')}")
            lines.append("")
        return "\n".join(lines)

    def _get_val(self, item, key, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def _format_detailed_context(self, retrieved_context: List[Any]) -> str:
        """詳細格式：以自然語句串連完整資訊"""
        formatted_parts = []
        for item in retrieved_context:
            # 支援 dict 或 CandidateNode
            path = self._get_val(item, "path", {})
            raw_text = self._get_val(item, "text", "")
            
            # 若為 ClinicalNorm json，嘗試解析為易讀格式
            try:
                if self._get_val(item, "label", "") == "ClinicalNorm":
                    data = json.loads(raw_text)
                    if isinstance(data, dict):
                        raw_text = "\n".join([f"    ▪ {k}: {v}" for k, v in data.items() if v])
            except:
                pass

            subdomain = path.get("subdomain", "N/A") if isinstance(path, dict) else getattr(path, "subdomain", "N/A")
            section_type = path.get("section_type", "N/A") if isinstance(path, dict) else getattr(path, "section_type", "N/A")
            section_name = path.get("section_name", "N/A") if isinstance(path, dict) else getattr(path, "section_name", "N/A")

            # 針對額外來源自訂前綴
            if subdomain == "臨床常模":
                prefix = f"針對「{section_name}」常模資訊提到：\n"
            elif subdomain == "外部知識":
                prefix = f"GPT 外部知識補充：\n"
            else:
                prefix = f"在「{subdomain}」的「{section_type}」（{section_name}）中提到：\n"

            formatted_parts.append(
                f"{prefix}{raw_text[:600]}{'...' if len(raw_text) > 600 else ''}\n"
            )
        return "\n".join(formatted_parts)

    def _format_concise_context(self, retrieved_context: List[Any]) -> str:
        """簡潔格式：以自然語句串連關鍵資訊"""
        formatted_parts = []
        for item in retrieved_context:
            path = self._get_val(item, "path", {})
            text = self._get_val(item, "text", "")

            subdomain = path.get("subdomain", "N/A") if isinstance(path, dict) else getattr(path, "subdomain", "N/A")
            section_type = path.get("section_type", "N/A") if isinstance(path, dict) else getattr(path, "section_type", "N/A")

            formatted_parts.append(
                f"關於「{subdomain}」的「{section_type}」指出：{text[:300]}{'...' if len(text) > 300 else ''}"
            )
        return "\n".join(formatted_parts)

    def _format_structured_context(self, retrieved_context: List[Any]) -> str:
        """結構化格式：按領域分組，並形成自然語句的脈絡"""
        # 按領域分組
        by_domain: Dict[str, List[Any]] = {}
        for item in retrieved_context:
            path = self._get_val(item, "path", {})
            subdomain = self._get_val(path, "subdomain", "N/A")
            by_domain.setdefault(subdomain, []).append(item)

        formatted_parts: List[str] = []
        for domain, items in by_domain.items():
            if domain == "臨床常模":
                domain_text = "【臨床發展常模與里程碑】：\n"
            elif domain == "外部知識":
                domain_text = "【外部通識與補充知識】：\n"
            else:
                domain_text = f"關於「{domain}」的相關報告紀錄：\n"
                
            for item in items:
                path = self._get_val(item, "path", {})
                raw_text = self._get_val(item, "text", "")
                section_type = self._get_val(path, "section_type", "N/A")
                section_name = self._get_val(path, "section_name", "N/A")

                try:
                    if self._get_val(item, "label", "") == "ClinicalNorm":
                        data = json.loads(raw_text)
                        if isinstance(data, dict):
                            raw_text = " ".join([f"{k}: {v}" for k, v in data.items() if v])
                except:
                    pass

                # 將類型資訊嵌入段落，形成自然語句
                if domain in ["臨床常模", "外部知識"]:
                    domain_text += f"- [{section_name}] {raw_text[:400].replace(chr(10), ' ')}{'...' if len(raw_text) > 400 else ''}\n"
                else:
                    domain_text += (
                        f"- 在「{section_type}」（{section_name}）方面提到："
                        f"{raw_text[:400].replace(chr(10), ' ')}{'...' if len(raw_text) > 400 else ''}\n"
                    )
            formatted_parts.append(domain_text)

        return "\n".join(formatted_parts)
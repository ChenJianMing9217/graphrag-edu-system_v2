"""
LLM 生成模組
使用 LM Studio 的 OpenAI 兼容 API 生成回應
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI
from .prompt_manager import LLMPromptManager, LLMGenerationConfig


@dataclass
class LLMConfig:
    """LLM 基礎配置（API 相關）"""
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    def __post_init__(self):
        # 從主設定檔載入
        from config import LLM_CONFIG
        self.base_url = self.base_url or LLM_CONFIG.get('base_url', "")
        self.api_key = self.api_key or LLM_CONFIG.get('api_key', "vllm-key")
        self.model = self.model or LLM_CONFIG.get('model', "google/gemma-3-4b-it")


class LLMGenerator:
    """LLM 生成器"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 生成器
        
        Args:
            config: LLM 基礎配置，如果為 None 則使用默認配置
        """
        self.config = config or LLMConfig()
        # 確保 openai 庫正確調用 vLLM
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key
        )
        self.prompt_manager = LLMPromptManager()
    
    def generate_response(
        self,
        user_query: str,
        retrieved_context: List[Dict] = None,
        conversation_history: List[Dict] = None,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
        prev_context: Optional[List[Dict]] = None,
        on_delta = None,  # 若提供，啟用 streaming 並 callback 每個 token delta
    ) -> str:
        """
        生成 LLM 回應
        
        Args:
            user_query: 使用者查詢
            retrieved_context: 檢索到的上下文資料（可選）
            conversation_history: 對話歷史（可選）
            system_prompt: 系統提示詞（可選，如果提供則覆蓋 generation_config 中的提示詞）
            generation_config: 生成配置（可選，如果提供則使用此配置的參數和提示詞）
        
        Returns:
            LLM 生成的回應
        """
        # 使用 generation_config 或默認配置
        if generation_config is None:
            generation_config = LLMGenerationConfig()
        
        # 構建消息列表
        messages = []
        
        # 添加系統提示詞（優先使用傳入的 system_prompt，否則使用 generation_config 中的）
        final_system_prompt = system_prompt or generation_config.system_prompt_template
        if not final_system_prompt:
            final_system_prompt = "你是一位專業的早療系統助手，能夠根據評估報告和檢索到的相關資訊，為家長和治療師提供專業的建議和回答。請用友善、專業的語氣回答問題。"
        
        messages.append({"role": "system", "content": final_system_prompt})
        
        # 添加對話歷史
        if conversation_history:
            messages.extend(conversation_history)
        
        # 構建用戶查詢（包含檢索到的上下文）
        if generation_config.user_prompt_template:
            # 使用配置中的模板
            user_content = generation_config.user_prompt_template.format(
                query=user_query,
                context=self._format_retrieved_context(retrieved_context or [], generation_config) if retrieved_context else ""
            )
        else:
            # 使用 prompt_manager 構建
            # 從 generation_config 中提取模糊相關資訊
            is_ambiguous = generation_config.is_ambiguous if hasattr(generation_config, 'is_ambiguous') else False
            active_domains = generation_config.active_domains if hasattr(generation_config, 'active_domains') else None
            task_options = generation_config.task_options if hasattr(generation_config, 'task_options') else None
            
            user_content = self.prompt_manager.build_user_prompt(
                user_query,
                retrieved_context or [],
                generation_config,
                is_ambiguous=is_ambiguous,
                active_domains=active_domains or [],
                task_options=task_options or [],
                prev_context=prev_context,
            )
        
        messages.append({"role": "user", "content": user_content})
        
        # 打印生成的 Prompt（偵錯用）
        print("\n" + "="*50)
        print("[LLM Prompt] 即將發送給 LLM 的 User Content:")
        print(user_content)
        print("="*50 + "\n")
        
        # 正規化訊息列表，確保符合 vLLM 的交替順序要求
        messages = self._normalize_messages(messages)
        
        try:
            # Streaming 模式（提供 on_delta callback 時啟用）
            if on_delta is not None:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=generation_config.temperature,
                    max_completion_tokens=generation_config.max_tokens,
                    top_p=generation_config.top_p,
                    frequency_penalty=generation_config.frequency_penalty,
                    presence_penalty=generation_config.presence_penalty,
                    stream=True,
                )
                full_text = ""
                for chunk in response:
                    try:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            full_text += delta
                            try:
                                on_delta(delta)
                            except Exception as cb_e:
                                print(f"[LLM stream] on_delta callback error: {cb_e}")
                    except Exception:
                        continue
                return full_text.strip()

            # 非 streaming（原有行為）
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=generation_config.temperature,
                max_completion_tokens=generation_config.max_tokens,
                top_p=generation_config.top_p,
                frequency_penalty=generation_config.frequency_penalty,
                presence_penalty=generation_config.presence_penalty
            )
            generated_text = response.choices[0].message.content.strip()
            return generated_text

        except Exception as e:
            print(f"[LLM 生成錯誤] {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，生成回應時發生錯誤：{str(e)}"
    
    def _normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        正規化訊息列表，確保符合 vLLM/OpenAI 的交替順序要求：
        1. 合併連續的相同角色內容。
        2. 確保第一個非 system 訊息是 user。
        3. 確保最後一個訊息是 user。
        4. 確保中間角色嚴格交替。
        """
        if not messages:
            return []

        # 1. 提取 System Prompt
        system_msg = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]

        if not other_msgs:
            return system_msg

        # 2. 合併連續的相同角色
        merged = []
        for m in other_msgs:
            if not merged or merged[-1]["role"] != m["role"]:
                merged.append({"role": m["role"], "content": m["content"]})
            else:
                merged[-1]["content"] += "\n\n" + m["content"]

        # 3. 確保第一個訊息是 user (如果開頭是 assistant 則移除，直到遇到 user)
        while merged and merged[0]["role"] == "assistant":
            merged.pop(0)

        if not merged:
            return system_msg

        # 4. 確保最後一個訊息是 user (如果結尾是 assistant 則移除)
        while merged and merged[-1]["role"] == "assistant":
            merged.pop()

        if not merged:
            return system_msg

        # 5. 確保中間嚴格交替（理論上合併後除了開頭結尾處理，中間已經交替了）
        # 如果因為移除操作導致有連續角色（極端情況），可在此再次運行合併
        final_history = []
        for m in merged:
            if not final_history or final_history[-1]["role"] != m["role"]:
                final_history.append(m)
            else:
                final_history[-1]["content"] += "\n\n" + m["content"]

        return system_msg + final_history

    def generate_chitchat(self, user_input: str) -> str:
        """
        處理問候語與閒聊（不走 RAG，直接由 LLM 回應）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": (
                        "你是一位親切的早療系統助手。使用者正在跟你打招呼或進行簡單的問候。"
                        "請用溫暖、簡短的方式回應，並輕輕引導他們提出早療相關的問題。"
                        "不要回答超過兩句話。"
                    )},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_completion_tokens=80,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMGenerator] generate_chitchat 失敗: {e}")
            return "您好！有什麼關於孩子早療的問題需要我協助嗎？😊"

    def _format_retrieved_context(self, context: List[Dict], config: LLMGenerationConfig) -> str:
        """
        格式化檢索到的上下文（使用 prompt_manager）
        
        Args:
            retrieved_context: 檢索結果列表
            config: 生成配置
        
        Returns:
            格式化後的上下文文字
        """
        return self.prompt_manager._format_context_by_style(
            retrieved_context[:config.max_context_items],
            config.context_format_style
        )
    


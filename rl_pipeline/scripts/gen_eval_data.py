import json
import random

def generate_dataset(output_path, num_samples=100):
    dataset = []
    
    memory_actions = ["STAY", "REFRESH", "CLARIFY"]
    sections_list = ["assessment", "observation", "training", "suggestion", "community_resources", "external_gpt"]
    difficulties = ["Easy", "Medium", "Hard"]
    
    # 建立不同的對話情境範本庫
    # 情境 1: STAY - 延續話題 (追問分數、追問細節)
    stay_templates = [
        {
            "ctx": [
                {"role": "user", "content": "我想看小明的魏氏智力測驗分數。"},
                {"role": "assistant", "content": "沒問題，小明的魏氏智力測驗總智商分數為 85，落在中下區間。"}
            ],
            "query": "那他的語言發展表現呢？",
            "gold_memory": "STAY",
            "gold_sections": ["assessment"],
            "gold_retrieval": ["語文理解指標(VCI)", "語言測驗分數"],
            "gold_ans": "在語言發展表現上，他的語文理解指標落在正常範圍，相較於非語文能力表現較佳。",
            "style": "客觀報告分數並進行簡單比較",
            "diff": "Easy",
            "notes": "對話直接延續同一份報告的下一個項目，Memory 必須 STAY。"
        },
        {
            "ctx": [
                {"role": "user", "content": "老師說他最近在班上很容易分心。"},
                {"role": "assistant", "content": "了解，根據學習觀察紀錄，他在團體活動時確實有注意力不集中的情形。"}
            ],
            "query": "這個狀況有建議家長回家怎麼引導嗎？",
            "gold_memory": "STAY",
            "gold_sections": ["suggestion"],
            "gold_retrieval": ["注意力缺失的居家建議", "親職教養策略"],
            "gold_ans": "建議家長在回家後，可以將任務切分成小步驟，並透過計時器幫助他建立時間觀念，逐步拉長專注力。",
            "style": "具體且可執行的教養建議",
            "diff": "Medium",
            "notes": "使用者以「這個狀況」代指前文的「分心」，需要精準的 STAY 來保留 Context。"
        },
        {
            "ctx": [
                {"role": "user", "content": "診所安排了下週的職能治療計畫。"},
                {"role": "assistant", "content": "好的，職能治療計畫主要是針對精細動作與感覺統合。"}
            ],
            "query": "那這項治療的具體目標是什麼？",
            "gold_memory": "STAY",
            "gold_sections": ["training"],
            "gold_retrieval": ["職能治療長期目標", "精細動作訓練項目"],
            "gold_ans": "這項職能治療的目標是提升手指精細操作能力，並改善握筆姿勢，以幫助未來的書寫發展。",
            "style": "醫療與訓練計畫說明的專業語氣",
            "diff": "Hard",
            "notes": "提問中的「這項治療」需依賴歷史判定「職能治療」。"
        }
    ]

    # 情境 2: REFRESH - 轉換話題 (突然問完全無關的問題)
    refresh_templates = [
        {
            "ctx": [
                {"role": "user", "content": "評估報告建議我們做每週兩次的物理治療。"},
                {"role": "assistant", "content": "是的，這有助於改善下肢肌張力不足的問題。"}
            ],
            "query": "對了，請問如果要申請身心障礙手冊，流程是什麼？",
            "gold_memory": "REFRESH",
            "gold_sections": ["community_resources"],
            "gold_retrieval": ["身心障礙手冊申請資格", "社會局申請流程"],
            "gold_ans": "申請身心障礙手冊需要先向戶籍地區公所領取鑑定表，接著前往指定醫療機構進行門診鑑定。完成後醫院會將資料送交社會局核定。",
            "style": "步驟清晰的行政手續指南",
            "diff": "Easy",
            "notes": "使用者使用「對了」強行轉換話題至行政資源申請，必須清空物理治療的 Context。"
        },
        {
            "ctx": [
                {"role": "user", "content": "他的理解能力好像比同年齡孩子慢。"},
                {"role": "assistant", "content": "從評估報告來看，確實落後約半年，建議可以多安排生活實境互動。"}
            ],
            "query": "一般來說，自閉症的初期徵兆會有哪些？",
            "gold_memory": "REFRESH",
            "gold_sections": ["external_gpt"],
            "gold_retrieval": ["自閉症光譜(ASD)核心症狀", "早期徵兆"],
            "gold_ans": "一般而言，自閉症的初期徵兆包含：社交互動困難（如不看人、不理人）、語言發展遲緩，以及重複性或固著的行為模式。",
            "style": "醫學與衛教知識的客觀說明",
            "diff": "Medium",
            "notes": "問題轉向通用的衛教知識，完全跳脫個人報告範疇，需觸發 REFRESH 搭配 external_gpt。"
        }
    ]

    # 情境 3: CLARIFY - 語意不清或多重意圖 (需要反問)
    clarify_templates = [
        {
            "ctx": [
                {"role": "user", "content": "我們之前做了物理治療和語言治療的評估。"},
                {"role": "assistant", "content": "了解。物理治療側重於大動作發展，語言治療則關注發音與理解。請問您想先看哪一份報告？"}
            ],
            "query": "那這樣會不會很嚴重？",
            "gold_memory": "CLARIFY",
            "gold_sections": [],
            "gold_retrieval": [],
            "gold_ans": "請問您是指物理治療的大動作遲緩，還是語言發展的遲緩呢？因為兩份報告的嚴重程度評估不同喔。",
            "style": "友善且精確地反問",
            "diff": "Hard",
            "notes": "前文有多焦點，User 的「這樣」極度模糊，系統應該觸發 CLARIFY 不盲目檢索。"
        },
        {
            "ctx": [
                {"role": "user", "content": "今天剛拿到最新報告。"},
                {"role": "assistant", "content": "好的，我已經準備好協助您解讀。"}
            ],
            "query": "怎麼辦？",
            "gold_memory": "CLARIFY",
            "gold_sections": [],
            "gold_retrieval": [],
            "gold_ans": "家長不要著急，請問您是看到報告中哪一項分數覺得比較擔心呢？我可以幫您詳細解說。",
            "style": "安撫情緒並引導提問",
            "diff": "Medium",
            "notes": "完全缺乏實體關鍵詞，必須要反問澄清。"
        }
    ]
    
    # 組合與隨機變化
    all_templates = stay_templates * 15 + refresh_templates * 15 + clarify_templates * 15
    random.shuffle(all_templates)
    
    names = ["小新", "小華", "安安", "佑佑", "樂樂", "孩子"]
    
    for i in range(num_samples):
        tmpl = all_templates[i % len(all_templates)].copy()
        
        # 微小的字串替代增加多樣性
        name = random.choice(names)
        if "小明" in tmpl["ctx"][0]["content"]:
            tmpl["ctx"][0]["content"] = tmpl["ctx"][0]["content"].replace("小明", name)
            tmpl["gold_ans"] = tmpl["gold_ans"].replace("小明", name)
            
        record = {
            "sample_id": f"eval_{i+1:04d}",
            "conversation_context": tmpl["ctx"],
            "user_query": tmpl["query"],
            "gold_memory_action": tmpl["gold_memory"],
            "gold_sections": tmpl["gold_sections"],
            "gold_retrieval_targets": tmpl["gold_retrieval"],
            "gold_answer": tmpl["gold_ans"],
            "answer_style_notes": tmpl["style"],
            "difficulty": tmpl["diff"],
            "notes_for_evaluation": tmpl["notes"]
        }
        dataset.append(record)
        
    with open(output_path, "w", encoding="utf-8") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {num_samples} samples to: {output_path}")

if __name__ == "__main__":
    generate_dataset("c:/Users/88696/Desktop/edu_sys/app_v7/offline_eval_dataset_100.jsonl", 100)

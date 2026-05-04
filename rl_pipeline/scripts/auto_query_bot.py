"""
auto_query_bot.py — 自動化測試 & 訓練資料收集器 (v3)

功能：
1. 自動登入系統
2. 逐一發問（涵蓋全領域與多樣化意圖）
3. 對話紀錄與 flow_state 自動寫入 SQL（由 Flask 後端處理）
4. Scenario 結束後送出該情境的 final_feedback（+1/-1）

注意：本腳本「不」進行 LLM 評分。
      所有 LLM Judge 評分由 unified_train_db.py 在離線訓練時統一處理。
      feedback_value 欄位保留給：(a) 真實使用者按讚/踩，(b) Scenario 層級最終評分。

用法：
    1. 先啟動 Flask 伺服器: python app.py
    2. 執行本腳本: python rl_pipeline/scripts/auto_query_bot.py
    3. 執行訓練: python rl_pipeline/scripts/unified_train_db.py
"""
import requests
import json
import time
import sys
import os
from dotenv import load_dotenv
from generated_scenarios import GENERATED_SCENARIOS

load_dotenv() # 讀取 .env 檔案中的 OPENAI_API_KEY

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- 設定區 ---
BASE_URL = "http://127.0.0.1:5001"
ACCESS_CODE = "N0NBW2IT"      # 使用者提供的存取碼
CHILD_ID = 1
DELAY_BETWEEN_QUESTIONS = 1.5   # 每題間隔秒數




# SCENARIOS 測試集 (超過 100 題的多輪對話，涵蓋各類意圖與滿意度)
SCENARIOS = [
    # --- 動作與體能 ---
    {"name": "粗大動作初步", "steps": ["他現在走路穩嗎？", "單腳站立的表現怎麼樣？", "那我要做啥可以幫助他？"], "final_feedback": 1},
    {"name": "戶外與球類", "steps": ["跑步急停會有困難嗎？", "有什麼推薦的球類練習？", "踢球的準確度如何？", "有建議的增加肌肉張力的運動嗎？"], "final_feedback": 1},
    {"name": "輔具與安全", "steps": ["需要準備什麼輔具嗎？", "下樓梯需要扶手嗎？", "他在戶外活動時的安全意識夠嗎？"], "final_feedback": 1},
    {"name": "精細動作操作", "steps": ["他現在會用剪刀了嗎？", "畫圓圈或橫線的表現如何？", "扣鈕扣對他來說會不會太難？", "堆積木能堆到幾層？"], "final_feedback": 1},
    {"name": "自理與精細動作", "steps": ["拿湯匙餵食的時候會漏嗎？", "他抓握筆的姿勢正確嗎？", "如何訓練他的手眼協調？", "對於精細動作有什麼居家練習？"], "final_feedback": 1},

    # --- 語言與認知 ---
    {"name": "口語表達", "steps": ["他的口語表達有進步嗎？", "他會用完整的句子說話嗎？", "為什麼他還是常用手指代替說話？"], "final_feedback": 1},
    {"name": "聽覺與理解", "steps": ["他聽得懂兩步驟的指令嗎？", "聽覺理解的部分有落後嗎？", "他能理解「上下、前後」的空間詞嗎？", "如何提升他的詞彙量？"], "final_feedback": 1},
    {"name": "認知發展", "steps": ["他的認知發展指標有達到嗎？", "他能分辨紅色、黃色、藍色嗎？", "數數可以數到多少？", "他對新事物的學習速度快嗎？"], "final_feedback": 1},
    {"name": "邏輯與記憶", "steps": ["他能完成簡單的拼圖嗎？", "記憶力表現如何？", "邏輯推理能力有評估到嗎？", "如何加強他的專注力？"], "final_feedback": 1},
    {"name": "語言進階", "steps": ["他會模仿大人說話嗎？", "他能正確叫出常見物品的名稱嗎？", "他的發音清晰度需要調整嗎？", "有什麼推薦的語言訓練遊戲？"], "final_feedback": 1},

    # --- 社會與情緒 ---
    {"name": "團體參與", "steps": ["他在團體活動中的參與度？", "他會跟小朋友分享玩具嗎？", "他會輪流排隊嗎？", "他在同儕間的互動型態？"], "final_feedback": 1},
    {"name": "情緒控制", "steps": ["老師反應他會推人，這跟情緒有關嗎？", "情緒失控時該如何安撫？", "如何建立他的自信心？", "如何訓練他的挫折容忍度？"], "final_feedback": -1}, # 刻意給一次負向回饋測試減分效果
    {"name": "適應與社交技巧", "steps": ["他離開父母時會有分離焦慮嗎？", "他在陌生環境的適應力？", "是否有社交退縮的現象？", "如何教他正確的社交技巧？"], "final_feedback": 1},
    {"name": "規則與同理心", "steps": ["他能遵守簡單的遊戲規則嗎？", "他能理解別人的表情代表什麼心情嗎？", "他有眼神對視嗎？"], "final_feedback": 1},

    # --- 生活自理 ---
    {"name": "穿脫與如廁", "steps": ["他會自己穿襪子和鞋子嗎？", "穿脫衣服需要幫忙嗎？", "戒尿布的進度理想嗎？"], "final_feedback": 1},
    {"name": "進食與衛生", "steps": ["他能獨自用杯子喝水嗎？", "他有挑食的狀況嗎？", "他在進食時的專注度？", "洗手的時候能照步驟做嗎？", "刷牙習慣建立得如何？"], "final_feedback": 1},
    {"name": "獨立性養成", "steps": ["他能自己收玩具嗎？", "如何提升他的生活獨立性？", "有建議的居家小幫手任務嗎？"], "final_feedback": 1},

    # --- 報告數據與未來規劃 ---
    {"name": "看懂數據", "steps": ["報告裡的分數怎麼看？", "百分等級 8 是什麼意思？", "這份報告的有效期是多久？"], "final_feedback": 1},
    {"name": "強弱項分析", "steps": ["標準分數 20 代表落後嗎？", "他的強項在哪裡？", "他最需要先加強哪一項？", "跟半年前比有明顯進步嗎？"], "final_feedback": 1},
    {"name": "就學與機構", "steps": ["這代表他需要去上特教班嗎？", "目前的評估結果有顯示任何異狀嗎？", "如果我想申請補助，這份報告可以嗎？", "有沒有推薦的早療機構？"], "final_feedback": 1},
    {"name": "治療師與家長配合", "steps": ["發展商數 (DQ) 79 算正常嗎？", "家長在家可以配合做什麼？", "專業治療師的建議有哪些？", "這些分數會跟著他一輩子嗎？"], "final_feedback": 1},
    {"name": "後續發展目標", "steps": ["他在同年齡層中的 PR 值是多少？", "目前的發展目標是什麼？", "如果三個月後再測一次會比較準嗎？", "我該如何跟老師溝通這份結果？", "我該預約下一次評估嗎？"], "final_feedback": 1},

    # --- 記憶與話題切換 (Memory Agent 特殊測項) ---
    {"name": "同領域模糊追問 (STAY)", "steps": ["精細動作的評估分數是幾分？", "那標準分數代表什麼意思？", "跟同齡小孩相比算落後嗎？"], "final_feedback": 1},
    {"name": "硬切換後接續 (REFRESH)", "steps": ["他在樓梯上的移位表現？", "口語表達有什麼訓練建議？", "他會主動表達需求嗎？"], "final_feedback": 1},
    {"name": "從數據切建議 (STAY)", "steps": ["粗大動作的百分等級是多少？", "那我在家可以怎麼幫助他練習？", "需要買什麼器材嗎？"], "final_feedback": 1},
    {"name": "大幅跳躍 (REFRESH)", "steps": ["雙腳跳遠的距離合格嗎？", "他的情緒管理有問題嗎？", "在學校會打人嗎？", "老師有反映什麼嗎？"], "final_feedback": 1},
    {"name": "完全模糊代名詞 (STAY)", "steps": ["感覺統合方面有做什麼測驗嗎？", "那結果怎麼樣？", "這樣算好還是不好？"], "final_feedback": -1},
    {"name": "回馬槍提問 (REFRESH)", "steps": ["對了剛剛說的精細動作分數幾分？", "那我再問一下口語理解的部分"], "final_feedback": 1},
    {"name": "跨領域比較 (REFRESH/STAY)", "steps": ["他粗大動作跟精細動作哪個比較好？", "那語言跟認知呢？"], "final_feedback": 1},
    {"name": "總結與結束 (REFRESH)", "steps": ["整體來看他最弱的是哪一項？", "那我最應該先加強什麼？", "好的謝謝，我會好好配合"], "final_feedback": 1},
    {"name": "PR值數據精確查詢", "steps": ["請問他的百分等級 PR 是多少？", "那粗大動作跟精細動作的 PR 分別是多少？", "這在同年齡層算什麼等級？"], "final_feedback": 1},
    {"name": "行為與情緒深度建議", "steps": ["他在家很不聽話怎麼辦？", "關於搶玩具的問題，有具體的練習方法嗎？", "老師說他在學校坐不住，這跟感覺統合有關嗎？"], "final_feedback": 1},
    {"name": "外部資源與補助諮詢", "steps": ["請問這份報告可以申請什麼補助嗎？", "附近有推薦的兒童復健診所嗎？", "早療補助要去哪裡申請？"], "final_feedback": 1},
    {"name": "行政與身分核對", "steps": ["這份報告是哪一天做的？", "測驗老師的名字是什麼？", "小孩現在幾個月大？"], "final_feedback": 1},
    {"name": "臨床常模與定義", "steps": ["什麼是 DSM-5？", "自閉症的診斷標準有哪些？", "這份報告有提到過動嗎？"], "final_feedback": 1},
    {"name": "日常生活應用", "steps": ["洗澡怕水是感統問題嗎？", "吃飯慢是因為口腔動作不好嗎？", "他會主動跟小朋友玩嗎？"], "final_feedback": 1},
    {"name": "精緻動作細分", "steps": ["他的抓握能力怎麼樣？", "使用剪刀有困難嗎？", "疊積木可以疊幾個？"], "final_feedback": 1},
    {"name": "閒聊與純感謝 (測試雜訊)", "steps": ["謝謝妳的回答", "今天辛苦了", "妳好嗎？"], "final_feedback": 1},
    {"name": "MySQL區域機構查詢", "steps": ["台北市有哪些推薦的復健診所？", "新北市附近有早療據點嗎？", "這兩區的機構有什麼差別？"], "final_feedback": 1},
    {"name": "MySQL補助與申請", "steps": ["請問早療補助要怎麼申請？", "桃園市有提供交通費補助嗎？", "申請這些補助需要什麼證明？"], "final_feedback": 1},
    {"name": "MySQL特定單位搜尋", "steps": ["幫我查一下新北的社福單位", "有沒有聯評醫院的名單？", "這些單位的電話是多少？"], "final_feedback": 1},
    {"name": "跨報告縱向追蹤 (Intent I)", "steps": ["跟上個月的報告比起來，他有進步嗎？", "他的 PR 值有提高嗎？", "目前進步最多的領域是哪一個？"], "final_feedback": 1},
    {"name": "IEP訓練目標進度 (Intent J)", "steps": ["他這個階段的訓練目標是什麼？", "目前的進度有達到預期嗎？", "下一階段要加強哪一部分？"], "final_feedback": 1},
    {"name": "家長心理情緒支持 (Intent M)", "steps": ["我最近教到好累，不知道該怎麼辦", "我覺得他好像都沒有進步，壓力好大", "妳可以給我一點正能量嗎？"], "final_feedback": 1},
    {"name": "肌肉張力與PR對比 (必選2)", "steps": ["他肌肉張力好像很低，這會影響他的百分等級嗎？", "這份報告中有提到具體的 PR 值是多少嗎？", "那這種張力問題，在家裡有什麼建議的練習？"], "final_feedback": 1},
    {"name": "動作落後與社福諮詢 (必選2)", "steps": ["他現在走路還是很不穩，請問有推薦的早期療育機構嗎？", "申請這些機構需要看報告中的哪些分數？", "有相關受測日期的限制嗎？"], "final_feedback": 1},
    {"name": "發展遲緩與家長衛教 (必選2)", "steps": ["報告說他臨界發展遲緩，這跟自閉症有關係嗎？", "我可以去哪裡找更權威的醫療診斷規範？", "妳有推薦的網路資源或 GPT 衛教資訊嗎？"], "final_feedback": 1},
    {"name": "精細動作與訓練計畫 (必選2)", "steps": ["他拿筆不穩，PR 值表現如何？", "這跟他的粗大動作發展有關連嗎？", "針對拿筆這件事，有什麼具體的訓練建議？"], "final_feedback": 1},
    {"name": "綜合追蹤建議 (必選2)", "steps": ["請問他上次跟這次比有進步嗎？", "針對目前最差的項目，哪裡有物理治療資源？", "家長每天要花多少時間陪伴練習比較好？"], "final_feedback": 1},
    {"name": "純行為細節觀察 (Forced Observation)", "steps": ["老師在學校觀察到他有什麼具體的社交動作？", "他在教室裡會主動找老師嗎？", "這些觀察內容在報告中的哪一頁？"]},
    {"name": "純居家介入建議 (Forced Suggestion)", "steps": ["回家之後，家長一定要注意的禁忌有哪些？", "對於家人的陪伴，報告提供了什麼最關鍵的建議？", "日常生活的作息要怎麼調整比較好？"]},
    {"name": "臨床表現描述 (Forced Observation)", "steps": ["他拿小積木的動作順暢嗎？", "他的精細動作表現，在臨床上有什麼具體描述？", "這跟他在家裡拿湯匙的狀況吻合嗎？"]},
    {"name": "專家介入建議 (Forced Suggestion)", "steps": ["關於未來三個月的介入方向，專家的首要建議是什麼？", "這份建議是針對居家還是學校環境？", "建議中是否有提到需要配合其他專業治療？"]},
    {"name": "測驗當下反應 (Forced Observation)", "steps": ["他在做測驗的時候，配合度高嗎？", "有沒有出現害怕或是哭鬧的行為？", "治療師對他當下的狀態有什麼觀察評語？"]},
    {"name": "生活運用能力 (Forced Observation)", "steps": ["他自己穿鞋子會不會有困難？", "這項能力在報告中是被記錄在哪邊？", "他單腳站立能撐幾秒鐘？"]},
    {"name": "教具與環境調整 (Forced Suggestion)", "steps": ["為了讓他進步，家裡有哪些環境需要調整？", "有推薦購買什麼特定的教具或玩具給他玩嗎？", "在使用這些教具時家長該怎麼引導？"]},
    {"name": "情緒與行為對策 (Forced Suggestion)", "steps": ["他如果不願意配合練習生氣了，有什麼建議的好方法？", "遇到挫折時，家長應該用什麼態度回應？", "報告裡有提到他固執的狀況該怎麼處理嗎？"]},
    {"name": "認知與指令觀察 (Forced Observation)", "steps": ["他聽得懂『把球拿給我』這種指令嗎？", "在測驗中，他有表現出理解因果關係的能力嗎？", "他的專注力大概可以維持幾分鐘？"]},
    {"name": "玩具共享反應 (Forced Observation)", "steps": ["別的小孩拿他的玩具，他會有什麼反應？", "他會用眼神尋求大人的幫助嗎？", "他會不會有出手推人的動作？"]},
    {"name": "飲食與口腔建議 (Forced Suggestion)", "steps": ["他挑食的問題，報告有給什麼改善建議嗎？", "需要給他咬比較硬的食物來練習咀嚼嗎？", "吃飯時間拉太長該怎麼縮短？"]},
    {"name": "肢體張力表現 (Forced Observation)", "steps": ["他在靜止的時候，身體的肌肉摸起來是緊繃的嗎？", "這在臨床上被稱為高張力還是低張力？", "他的關節活動角度有受限嗎？"]},
    {"name": "戶外活動建議 (Forced Suggestion)", "steps": ["帶他去公園的時候，最推薦玩什麼設施？", "溜滑梯對他的前庭覺有幫助嗎？", "需要避免讓他玩盪鞦韆嗎？"]},
    {"name": "發音與口語觀察 (Forced Observation)", "steps": ["他現在會發出哪些單音？", "他會模仿別人說話的聲音嗎？", "在想要東西時，他是用指的還是用叫的？"]},
    {"name": "溝通互動策略 (Forced Suggestion)", "steps": ["當他一直尖叫不說話的時候，大人應該怎麼回應？", "需要要求他看著我的眼睛才給東西嗎？", "有推薦使用圖卡來輔助溝通嗎？"]},
    {"name": "平衡感動作觀察 (Forced Observation)", "steps": ["他走在直線上的時候會不會左右搖晃？", "他跳躍的時候雙腳能同時離地嗎？", "有沒有容易跌倒的紀錄？"]},
    {"name": "睡眠障礙對策 (Forced Suggestion)", "steps": ["他晚上很難入睡，有什麼睡前的放鬆建議嗎？", "這跟白天活動量不足有關係嗎？", "需要把房間的燈光調到多暗比較好？"]},
    {"name": "大肌肉力量觀察 (Forced Observation)", "steps": ["他把球丟出去的距離大概有多遠？", "丟球的時候是一隻手還是兩隻手？", "他拿重物時手會發抖嗎？"]},
    {"name": "精細動作訓練建議 (Forced Suggestion)", "steps": ["在家裡有什麼生活瑣事可以讓他練習手指協調？", "用夾子夾豆子的遊戲適合他現在的程度嗎？", "握筆姿勢錯誤需要立刻糾正嗎？"]},
    {"name": "社交遊戲觀察 (Forced Observation)", "steps": ["他會主動加入其他小孩的遊戲嗎？", "他在旁邊看的時間多，還是實際下去玩的時間多？", "如果遊戲規則改變，他能接受嗎？"]},
    {"name": "轉銜焦慮建議 (Forced Suggestion)", "steps": ["從遊戲切換到吃飯時間，他總是崩潰，該怎麼辦？", "使用預告計時器會有效嗎？", "還有什麼方法可以降低他的轉換焦慮？"]},
    {"name": "手眼協調觀察 (Forced Observation)", "steps": ["他串珠珠的時候，可以順利穿過去嗎？", "他在畫畫時，眼睛有看著紙嗎？", "他能接到滾過來的球嗎？"]},
    {"name": "感統刺激建議 (Forced Suggestion)", "steps": ["為了增加本體覺刺激，可以在家玩什麼遊戲？", "讓他推重物或是棉被包壽司的遊戲適合嗎？", "這些活動一天要做幾次？"]},
    {"name": "生活自理觀察 (Forced Observation)", "steps": ["他會自己用湯匙把飯送進嘴巴嗎？", "吃飯時食物會掉出碗外嗎？", "他會自己上廁所了嗎？"]},
    {"name": "如廁訓練建議 (Forced Suggestion)", "steps": ["現在這個階段適合開始戒尿布了嗎？", "如果他上廁所失敗，家長應該怎麼安慰？", "有推薦的兒童馬桶座嗎？"]},
    {"name": "眼神交流觀察 (Forced Observation)", "steps": ["叫他的名字時，他會轉頭看人嗎？", "他看人的眼神會閃躲嗎？", "在玩喜歡的玩具時，他會拿起來給大人看嗎？"]},
    {"name": "親子共讀建議 (Forced Suggestion)", "steps": ["唸繪本給他聽的時候，他都不專心，有甚麼技巧嗎？", "適合選有很多文字還是很多圖片的書？", "需要要求他跟著唸嗎？"], "final_feedback": 1},

    # --- Task L: 後續追蹤 (Follow-up Tracking) ---
    {"name": "後續追蹤-進步幅度 (Task L)", "steps": ["跟上次報告相比，他整體有進步嗎？", "哪個領域進步最明顯？", "有哪個領域退步或停滯了嗎？", "這樣的進步速度算正常嗎？"], "final_feedback": 1},
    {"name": "後續追蹤-目標達成 (Task L)", "steps": ["這次報告達到上次設定的訓練目標了嗎？", "哪個短期目標還沒達成？", "下一個階段的追蹤重點應該放在哪裡？"], "final_feedback": 1},
    {"name": "後續追蹤-再評估規劃 (Task L)", "steps": ["我應該什麼時候安排下一次完整評估？", "三個月後重測比較好還是半年？", "這次結果有沒有需要轉介給其他治療師的地方？"], "final_feedback": 1},

    # --- 多任務 H+K: 轉介資源 + 補助申請 ---
    {"name": "多任務-機構查詢+區域補助 (H+K)", "steps": ["台北市有哪些早療相關的社福補助？", "我們在新北市，這邊有適合的療育機構嗎？", "申請補助跟申請機構的流程有什麼不一樣？", "補助金額大概是多少？"], "final_feedback": 1},
    {"name": "多任務-區域資源+機構申請 (K+H)", "steps": ["桃園市有早療聯評中心嗎？", "聯評之後如果需要療育，有推薦的機構嗎？", "這些機構有提供交通補助嗎？", "如果我想換機構，流程複雜嗎？"], "final_feedback": 1},

    # --- 多任務 E+J: 在家訓練 + 學校合作 ---
    {"name": "多任務-發展報告+IEP目標 (E+J)", "steps": ["這份報告的整體評估結果是什麼？", "根據這個結果，他的 IEP 目標應該怎麼設定？", "這個發展狀況符合 DSM-5 的哪些描述？", "訓練計畫要從哪個領域開始優先處理？"], "final_feedback": 1},
    {"name": "多任務-DSM診斷+訓練進度 (J+E)", "steps": ["DSM-5 對發展遲緩的判斷標準是什麼？", "他的報告數據對比這個標準大概在哪個程度？", "下一階段的訓練目標怎麼訂比較合適？", "這些目標跟他目前的觀察紀錄吻合嗎？"], "final_feedback": 1},

    # --- 多任務 A+B: 報告總覽 + 分數解讀 ---
    {"name": "多任務-報告總覽+分數解讀 (A+B)", "steps": ["這份報告我第一次看，要從哪裡開始讀？", "那各個分數區塊分別代表什麼意思？", "百分等級跟標準分數有什麼不一樣？", "最重要的數字是哪幾個？"], "final_feedback": 1},

    # --- 多任務 B+N: 分數解讀 + 進步查詢 ---
    {"name": "多任務-分數比較+進步確認 (B+N)", "steps": ["他這次的標準分數是多少？", "跟上次相比，這個分數有進步嗎？", "進步的幅度在臨床上算有意義嗎？", "哪個分測驗進步最多？"], "final_feedback": 1},

    # --- 多任務 C+D: 臨床觀察 + 能力剖面 ---
    {"name": "多任務-觀察表現+優弱勢分析 (C+D)", "steps": ["報告裡臨床觀察記錄了什麼具體表現？", "從這些觀察來看，他的優勢和弱項分別是什麼？", "最需要優先處理的需求是哪一項？", "這個能力剖面跟測驗分數吻合嗎？"], "final_feedback": 1},

    # --- 多任務 D+E: 能力剖面 + 在家訓練 ---
    {"name": "多任務-弱點確認+居家介入 (D+E)", "steps": ["從報告看他最弱的能力是什麼？", "針對這個弱點，在家可以怎麼練習？", "每天要花多少時間訓練比較合適？", "有沒有融入遊戲的方式讓他不排斥？"], "final_feedback": 1},

    # --- 多任務 F+M: 日常作息練習 + 家長情緒支持 ---
    {"name": "多任務-日常練習+家長壓力 (F+M)", "steps": ["我很累，有沒有不需要特別安排時間的練習方法？", "把練習融入日常作息的具體做法有哪些？", "每天這樣做，家長心理上怎麼調適？", "有沒有辦法讓練習變成親子互動而不是壓力？"], "final_feedback": 1},

    # --- 多任務 G+H: 是否需要早療 + 轉介資源 ---
    {"name": "多任務-早療評估+機構轉介 (G+H)", "steps": ["根據這份報告，他現在需要去做早療嗎？", "如果需要，要去哪裡轉介？", "轉介需要準備這份報告嗎？有時間限制嗎？", "台北市有哪些推薦的早療機構？"], "final_feedback": 1},

    # --- 多任務 G+L: 是否需要早療 + 後續追蹤 ---
    {"name": "多任務-成效追蹤+再評估 (G+L)", "steps": ["他目前的療育成效算理想嗎？", "什麼時候應該安排下一次評估來確認成效？", "如果進步不如預期，要怎麼調整介入方向？", "追蹤的重點指標應該放在哪些領域？"], "final_feedback": 1},

    # --- 多任務 I+H: 報告分享/隱私 + 轉介資源 ---
    {"name": "多任務-報告分享+機構轉介 (I+H)", "steps": ["我想把這份報告提供給早療機構，有什麼要注意的嗎？", "報告裡的個人資料需要遮蔽嗎？", "機構會怎麼使用這份資料？", "我要去哪裡找有資格接受這份報告的機構？"], "final_feedback": 1},

    # --- 多任務 J+K: 學校合作 + 補助申請 ---
    {"name": "多任務-學校合作+補助申請 (J+K)", "steps": ["學校老師說要幫他申請特教資源，我要準備什麼？", "這份報告可以用來申請學校的資源嗎？", "除了學校，政府還有哪些相關補助可以申請？", "申請補助跟申請學校資源可以同時進行嗎？"], "final_feedback": 1},

    # --- 多任務 L+N: 後續追蹤 + 進步查詢 ---
    {"name": "多任務-追蹤進度+進步確認 (L+N)", "steps": ["這次的追蹤結果，跟上次評估比，他整體有進步嗎？", "哪個領域的進步幅度最大？", "有沒有達到上次設定的追蹤目標？", "下一個追蹤時間點要設在什麼時候？"], "final_feedback": 1},
]
SCENARIOS = GENERATED_SCENARIOS

def run_auto_bot():
    session = requests.Session()

    print(f"{'=' * 60}")
    print(f" 自動化測試 & RL 訓練資料收集器 v2 (Scenarios)")
    print(f" 目標伺服器: {BASE_URL}")
    print(f" 劇本數: {len(SCENARIOS)}")
    print(f"{'=' * 60}")

    # 1. 登入
    print(f"\n正在使用存取碼登入: {ACCESS_CODE}...")
    try:
        login_res = session.post(f"{BASE_URL}/api/login_with_code", json={
            "code": ACCESS_CODE
        })

        if login_res.status_code != 200:
            print(f"登入失敗 ({login_res.status_code}): {login_res.text}")
            return
        print(f"登入成功。兒童姓名: {login_res.json().get('child_name')}\n")
    except Exception as e:
        print(f"登入連線失敗: {e}")
        return


    # 2. 逐一執行情境
    success_count = 0
    total_q = sum(len(s["steps"]) for s in SCENARIOS)
    
    for s_idx, scenario in enumerate(SCENARIOS):
        print(f"\n>>> 啟動情境 [{s_idx+1}/{len(SCENARIOS)}]: {scenario['name']}")
        
        # 開啟新對話 Session
        try:
            res = session.post(f"{BASE_URL}/api/new_chat")
            if res.status_code != 200:
                print("開啟新對話 (reset) 失敗")
        except Exception as e:
            print(f"新對話連線失敗: {e}")
            break
            
        last_msg_id = None
        conversation_history = []
        
        for q_idx, q in enumerate(scenario["steps"]):
            print(f"  [步驟 {q_idx+1}/{len(scenario['steps'])}] 發問: {q}")
            try:
                res = session.post(f"{BASE_URL}/api/chat", json={
                    "message": q
                })
                
                if res.status_code == 200:
                    try:
                        data = res.json()
                        reply = data.get('message', '')
                        msg_id = data.get('message_id', '')
                        last_msg_id = msg_id
                        print(f"  → 回覆: {reply[:80]}...")
                        success_count += 1
                        conversation_history.append(reply)
                        
                    except Exception as e:
                        print(f"  解析 JSON 失敗: {e}")
                        print(f"  原始回應內容：\n{res.text[:500]}")
                        break
                    
                    # 模擬人類思考時間
                    time.sleep(DELAY_BETWEEN_QUESTIONS)
                else:
                    print(f"  請求失敗: {res.status_code}")
                    print(f"  原始回應內容：\n{res.text[:500]}")
                    break
                    
            except Exception as e:
                print(f"  連線出錯: {e}")
                break
                
        # 5. 情境最終回饋（Scenario 設計者標注的 final_feedback，非 LLM 生成）
        final_feedback_val = scenario.get("final_feedback")
        if last_msg_id and final_feedback_val is not None:
            try:
                fb_res = session.post(f"{BASE_URL}/api/feedback", json={
                    "message_id": last_msg_id,
                    "feedback": final_feedback_val
                })
                if fb_res.status_code == 200:
                    print(f"  → 情境最終回饋已送出: {final_feedback_val}")
                else:
                    print(f"  → 回饋送出失敗: {fb_res.status_code}")
            except Exception as e:
                print(f"  → 回饋送出出錯: {e}")

    print(f"\n{'=' * 60}")
    print(f" 測試結束")
    print(f"{'=' * 60}")
    print(f"  成功發問: {success_count}/{total_q} 題")
    print(f"\n下一步：執行離線訓練（LLM Judge 將在訓練時統一處理無回饋的資料）")
    print(f"  python rl_pipeline/scripts/unified_train_db.py")


if __name__ == "__main__":
    run_auto_bot()

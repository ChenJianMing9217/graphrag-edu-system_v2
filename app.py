from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import threading
import queue as _queue
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os, re, uuid, random, string, json, time
from datetime import datetime

from config import get_mysql_uri, SECRET_KEY, get_neo4j_uri, get_neo4j_auth, UPLOAD_FOLDER

# 導入對話大腦（實際 instance 在 db 建立之後創建，見下方 dialogue_mgr = DialogueManager(sql_db=db)）
from dialogue_manager import DialogueManager
dialogue_mgr = None  # 延遲初始化

# ============================================================================
# Turn Log（每輪對話寫一行 JSON 到 logs/turns_YYYYMMDD.jsonl）
# 提供測試期回顧、debug、跟 Claude 討論時用。
# env 控制：TURN_LOG_DISABLE=1 關閉
# ============================================================================
_TURN_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_TURN_LOG_DIR, exist_ok=True)


def _write_turn_log(*, query, response, turn_state, retrieved, session_id, child_id,
                    duration_ms=None, msg_uuid=None, error=None):
    """每輪對話寫入 JSONL 日誌"""
    if os.environ.get("TURN_LOG_DISABLE") == "1":
        return
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        path = os.path.join(_TURN_LOG_DIR, f"turns_{date_str}.jsonl")
        # 縮減 retrieved_context（留 top 8 + 必要欄位，避免單行太大）
        compact_retrieved = []
        if isinstance(retrieved, list):
            for r in retrieved[:8]:
                if isinstance(r, dict):
                    compact_retrieved.append({
                        "score": r.get("score"),
                        "label": r.get("label"),
                        "category": r.get("category") or (r.get("properties") or {}).get("category"),
                        "subdomain": r.get("subdomain") or (r.get("properties") or {}).get("subdomain"),
                        "text_preview": (r.get("text") or "")[:160],
                    })
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "child_id": child_id,
            "msg_uuid": msg_uuid,
            "duration_ms": duration_ms,
            "query": query,
            "response": (response or "")[:1500],   # 截斷避免單行爆炸
            "response_full_len": len(response or ""),
            "flow": {
                "task": (turn_state or {}).get("task_pred"),
                "secondary_tasks": (turn_state or {}).get("secondary_tasks"),
                "task_dist": (turn_state or {}).get("task_dist"),
                "task_top_score": (turn_state or {}).get("task_top_score"),
                "memory_action": (turn_state or {}).get("memory_action"),
                "retrieval_action": (turn_state or {}).get("retrieval_action"),
                "clarify_type": (turn_state or {}).get("clarify_type"),
                "scope_pred": (turn_state or {}).get("scope_pred"),
                "top_domain": (turn_state or {}).get("top_domain"),
                "active_domains": (turn_state or {}).get("active_domains"),
                "detected_region": (turn_state or {}).get("detected_region"),
                "planning_active": ((turn_state or {}).get("planning_info") or {}).get("active"),
                "planning_probs": ((turn_state or {}).get("planning_info") or {}).get("probs"),
                # Memory Agent 訓練/分析訊號（log instrument，不影響行為）
                "memory_features": (turn_state or {}).get("memory_features"),
                "memory_probs": (turn_state or {}).get("memory_probs"),
                # Shadow log: 8d 新特徵 + Agent 內部狀態 + override 觸發 + 上一輪資訊（Phase 1+ 重訓用）
                "memory_features_v2": (turn_state or {}).get("memory_features_v2"),
                "agent_used": (turn_state or {}).get("agent_used"),
                "agent_decision_raw": (turn_state or {}).get("agent_decision_raw"),
                "fallback_reason": (turn_state or {}).get("fallback_reason"),
                "overrides_fired": (turn_state or {}).get("overrides_fired"),
                "prev_query": (turn_state or {}).get("prev_query"),
                "prev_task": (turn_state or {}).get("prev_task"),
                "prev_task_dist": (turn_state or {}).get("prev_task_dist"),
                "topic_overlap_raw": (turn_state or {}).get("topic_overlap"),
                "tv_distance": (turn_state or {}).get("tv_distance"),
                "context_sim": (turn_state or {}).get("context_sim"),
                "normalized_entropy": (turn_state or {}).get("normalized_entropy"),
            },
            "retrieved_top8": compact_retrieved,
            "n_candidates": (turn_state or {}).get("num_candidates"),
            "error": error,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as _e:
        # 不影響主流程
        print(f"[TurnLog] write failed: {_e}")


# 導入 pdf_parser 處理流程
from pdf_parser.pdf_processor_main import IEPPipeline
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = get_mysql_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# [FIX 2026-05-07] db 建好後才初始化 DialogueManager，把 sql_db 注入給 RetrievalModuleV2，
# 才能讓 MySQL 在地資源/補助查詢正常運作。
dialogue_mgr = DialogueManager(sql_db=db)

# ============================================================================
# 資料庫模型 (v2: 以兒童為中心)
# ============================================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False) # 擴展長度
    password_hash = db.Column(db.String(255), nullable=False) # 擴展長度
    role = db.Column(db.String(20), nullable=False) # 'therapist', 'caregiver', 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Child(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    birth_date = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    age = db.Column(db.String(50), nullable=True) # 從 PDF 提取的年齡描述
    
    # 核心：存取碼 (Access Code)
    access_code = db.Column(db.String(20), unique=True, nullable=False, index=True)
    
    # 關聯
    creator_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True) # 建立者 (通常是治療師)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    reports = db.relationship('Report', backref='child', lazy=True)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    child_id = db.Column(db.Integer, db.ForeignKey('child.id'), nullable=False)
    assessment_date = db.Column(db.String(50), nullable=True) # 從 PDF 提取的日期
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=True) # [新增] Session ID 用於串接多輪對話
    msg_uuid = db.Column(db.String(36), unique=True, nullable=True) # 前端對應 ID
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    child_id = db.Column(db.Integer, db.ForeignKey('child.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    is_user_message = db.Column(db.Boolean, nullable=False, default=True)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    flow_state = db.Column(db.Text, nullable=True) # JSON 格式
    retrieval_info = db.Column(db.Text, nullable=True) # JSON 格式
    feedback_value = db.Column(db.Integer, default=0) # 1: Helpful, -1: Not Helpful, 0: 未回饋

class SubsidyProgram(db.Model):
    """各縣市早療補助方案（從官方 PDF 匯入）"""
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column(db.String(20), nullable=False, index=True)        # 臺北市、新北市...
    eligibility = db.Column(db.Text, nullable=True)                    # 補助對象/資格條件
    subsidy_items = db.Column(db.Text, nullable=True)                  # 補助項目描述
    transport_fee = db.Column(db.String(200), nullable=True)           # 交通補助金額
    training_cap = db.Column(db.String(200), nullable=True)            # 療育訓練費上限（一般戶）
    low_income_cap = db.Column(db.String(200), nullable=True)          # 低收入戶上限
    excluded_items = db.Column(db.Text, nullable=True)                 # 不補助項目
    required_docs = db.Column(db.Text, nullable=True)                  # 申請應備文件
    apply_deadline = db.Column(db.String(300), nullable=True)          # 申請期限
    apply_where = db.Column(db.Text, nullable=True)                    # 申請方式/窗口
    notes = db.Column(db.Text, nullable=True)                          # 其他注意事項
    full_text = db.Column(db.Text, nullable=True)                      # 完整原文
    source_file = db.Column(db.String(100), nullable=True)             # 來源檔名
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# ============================================================================
# 工具函式
# ============================================================================
def minguo_to_iso(text):
    """將 民國113年03月30日 轉換為 2024-03-30"""
    if not text: return None
    try:
        # 使用正規表達式抓取年、月、日
        match = re.search(r"民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日", text)
        if match:
            year = int(match.group(1)) + 1911
            month = int(match.group(2))
            day = int(match.group(3))
            return f"{year:04d}-{month:02d}-{day:02d}"
    except Exception:
        pass
    return text # 若失敗則回傳原樣

def generate_access_code(length=8):
    """生成唯一的存取碼"""
    characters = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(characters, k=length))
        if not Child.query.filter_by(access_code=code).first():
            return code

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ----------------------------------------------------------------------------
# 解析與圖譜串接函數 (仿 v6)
# ----------------------------------------------------------------------------

def process_report_task(report_id, file_path, child_id_str):
    """處理 PDF 並建置圖譜 (應改為非同步，此處先以同步示範)"""
    try:
        neo4j_config = {
            'uri': get_neo4j_uri(),
            'user': get_neo4j_auth()[0],
            'password': get_neo4j_auth()[1]
        }
        
        doc_id = f"v7_report_{report_id}_{child_id_str}"
        archive_dir = os.path.join(app.config['UPLOAD_FOLDER'], "json_archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        pipeline = IEPPipeline(neo4j_config=neo4j_config, archive_dir=archive_dir)
        # 使用固定的 v7 前綴 ID 格式，確保解析與檢索一致
        success, result_data = pipeline.run(file_path, child_id=doc_id)
        
        return success, result_data
    except Exception as e:
        print(f"[ERROR] 圖譜建置失敗: {e}")
        return False, str(e)

# ----------------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "沒有選擇檔案"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "沒有選擇檔案"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_name = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(file_path)
        
        # 1. 處理身份：若未登入，建立新兒童
        if 'active_child_id' not in session:
            new_code = generate_access_code()
            new_child = Child(
                name="新評估兒童",
                access_code=new_code
            )
            db.session.add(new_child)
            db.session.commit()
            session['active_child_id'] = new_child.id
            session['access_code'] = new_code
            is_new = True
        else:
            new_child = db.session.get(Child, session['active_child_id'])
            is_new = False
            
        # 2. 建立報告記錄
        new_report = Report(
            filename=save_name,
            file_path=file_path,
            child_id=new_child.id
        )
        db.session.add(new_report)
        db.session.commit()
        
        # 設定為當前活動報告
        session['active_report_id'] = new_report.id
        
        # 3. 啟動圖譜建置 (此處同步，正式版建議 Thread)
        # 重要：使用 new_report.id 作為 doc_id，確保多份報告在 Neo4j 中獨立存在不會覆寫
        success, result_data = process_report_task(new_report.id, file_path, str(new_child.id))
        
        if success:
            # --- 4. 自動從 PDF Meta 補完成員資訊 ---
            meta = result_data.get("document_meta", {})
            patient_name = meta.get("patient_name")
            gender = meta.get("gender")
            age = meta.get("age")
            birth_date_str = meta.get("birth_date")
            report_dates = meta.get("report_dates", {})
            complete_date = report_dates.get("report_complete_date")
            
            # 若為新建立的臨時兒童，將姓名更新為 PDF 上的真實姓名
            if is_new and patient_name:
                new_child.name = patient_name
            
            # 更新兒童基本資訊 (性別, 歲數, 生日)
            if gender:
                new_child.gender = gender
            if age:
                new_child.age = age
            if birth_date_str:
                iso_birth = minguo_to_iso(birth_date_str)
                if iso_birth:
                    try:
                        # 使用 minguo_to_iso 轉換後的 YYYY-MM-DD 格式（datetime 已在頂層 import）
                        new_child.birth_date = datetime.strptime(iso_birth, "%Y-%m-%d").date()
                    except Exception as e:
                        print(f"[ERROR] 生日轉換失敗: {e}")
            
            # 存入報告日期 (標準化為 ISO 格式以利排序)
            if complete_date:
                new_report.assessment_date = minguo_to_iso(complete_date)
            
            db.session.commit() # 再次提交更新後的資訊
            
            msg = f"報告上傳成功！您的存取碼為：{new_child.access_code}。您現在可以針對這份報告詢問任何問題，例如：「這份報告的初步評估結論是什麼？」\n\n📌 溫馨提醒：請務必妥善保存您的存取碼，以便日後登入繼續諮詢。"
            return jsonify({
                "status": "success",
                "message": msg,
                "access_code": new_child.access_code,
                "child_name": new_child.name,
                "is_new": is_new
            })
        else:
            return jsonify({"status": "error", "message": f"報告已上傳，但圖譜建置失敗: {result_data}"}), 500
            
    return jsonify({"status": "error", "message": "請上傳 PDF 檔案"}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    response_length = data.get('response_length', 'standard')  # concise / standard / detailed

    # 判斷當前兒童 ID (若未登入則為 None)
    child_id = session.get('active_child_id')
    _t_start = time.time()

    # 調用對話大腦
    try:
        # 獲取該兒童的所有報告列表，供進步查詢任務對比使用
        all_reports = []
        age_months = None
        if child_id:
            all_reports = Report.query.filter_by(child_id=child_id).order_by(Report.assessment_date.desc()).all()
            
            # 計算 age_months 供 ClinicalBridgeService (常模API) 使用
            active_child = db.session.get(Child, child_id)
            if active_child:
                # 優先用 age 文字欄位（報告上的實際年齡，如 "3 歲 1 月"）
                if active_child.age:
                    import re
                    m = re.search(r'(\d+)\s*歲\s*(\d+)?', active_child.age)
                    if m:
                        years = int(m.group(1))
                        months = int(m.group(2)) if m.group(2) else 0
                        age_months = max(1, years * 12 + months)

                # Fallback：若 age 欄位為空或解析失敗，用 birth_date 計算
                if age_months is None and active_child.birth_date:
                    from datetime import date
                    if isinstance(active_child.birth_date, str):
                        try:
                            b_date = datetime.strptime(active_child.birth_date, "%Y-%m-%d").date()
                        except:
                            b_date = None
                    else:
                        b_date = active_child.birth_date

                    if b_date:
                        delta = date.today() - b_date
                        age_months = max(1, int(delta.days / 30.436875))
            
        response_text, msg_uuid, turn_state, retrieved_context = dialogue_mgr.get_response(
            user_input,
            child_id,
            session,
            all_reports=all_reports,
            age_months=age_months,
            response_length=response_length,
        )

        # [NEW] 儲存提問與回覆至 SQL (供未來 RL 訓練用)
        try:
            # 取得當前真實登入使用者的 ID (如果有登入的話)
            current_user_id = current_user.id if current_user and current_user.is_authenticated else None
            
            # [FIX] 如果 current_user_id 為空 (Caregiver 模式)，則嘗試使用該兒童的建立者 ID，或預設為 1
            if not current_user_id and child_id:
                active_child = db.session.get(Child, child_id)
                if active_child and active_child.creator_id:
                    current_user_id = active_child.creator_id
                else:
                    current_user_id = 1 # Fallback 預設管理員 ID
            elif not current_user_id:
                current_user_id = 1 # 無兒童資訊時也預設為 1

            # 確保有 chat_session_id
            if 'chat_session_id' not in session:
                session['chat_session_id'] = uuid.uuid4().hex
            current_session_id = session['chat_session_id']
            
            # 使用者的訊息
            user_msg = ChatMessage(
                session_id=current_session_id,
                user_id=current_user_id,
                child_id=child_id,
                message=user_input,
                is_user_message=True
            )
            db.session.add(user_msg)
            
            # 助理的回覆
            bot_msg = ChatMessage(
                session_id=current_session_id,
                msg_uuid=msg_uuid,
                user_id=current_user_id,
                child_id=child_id,
                message=response_text,
                is_user_message=False,
                flow_state=json.dumps(turn_state, ensure_ascii=False) if turn_state else None,
                retrieval_info=json.dumps(retrieved_context, ensure_ascii=False) if retrieved_context else None
            )
            db.session.add(bot_msg)
            db.session.commit()
        except Exception as db_e:
            db.session.rollback()
            print(f"[SQL] 儲存對話紀錄失敗: {db_e}")
            
        # 寫 turn log（測試期回顧用）
        try:
            _write_turn_log(
                query=user_input,
                response=response_text,
                turn_state=turn_state,
                retrieved=retrieved_context,
                session_id=session.get('chat_session_id'),
                child_id=child_id,
                duration_ms=int((time.time() - _t_start) * 1000),
                msg_uuid=msg_uuid,
            )
        except Exception:
            pass

        return jsonify({
            "status": "success",
            "message": response_text,
            "message_id": msg_uuid,
            "flow_state": turn_state,
            "retrieval_info": retrieved_context,
            "raw_retrieval": retrieved_context # 使用同一個來源，或視情況細分
        })
    except Exception as e:
        traceback.print_exc()
        # 失敗也記錄
        try:
            _write_turn_log(
                query=user_input,
                response="",
                turn_state=None,
                retrieved=None,
                session_id=session.get('chat_session_id'),
                child_id=child_id,
                duration_ms=int((time.time() - _t_start) * 1000),
                error=str(e),
            )
        except Exception:
            pass
        return jsonify({"status": "error", "message": f"對話處理出錯: {str(e)}"}), 500


# ============================================================================
# Streaming Chat（SSE）
# 與 /api/chat 同樣 pipeline，但 LLM 生成階段以 token 流的形式逐筆送回前端。
# 使用 threading + queue 解耦：worker 跑 dialogue_mgr，SSE generator 從 queue 拿 deltas。
# ============================================================================
@app.route('/api/chat_stream', methods=['POST'])
def chat_stream():
    data = request.get_json() or {}
    user_input = data.get('message', '')
    response_length = data.get('response_length', 'standard')
    child_id = session.get('active_child_id')
    chat_session_id = session.get('chat_session_id') or uuid.uuid4().hex
    session['chat_session_id'] = chat_session_id

    # 在進 thread 前抓 session/auth 必要欄位
    captured_session = dict(session)
    cur_user_id = current_user.id if current_user and current_user.is_authenticated else None
    _t_start = time.time()

    q = _queue.Queue()
    result_holder = {}

    def worker():
        try:
            with app.app_context():
                # 取得 reports + 計算 age_months（與 /api/chat 同步路徑一致）
                all_reports = []
                age_months = None
                if child_id:
                    all_reports = Report.query.filter_by(child_id=child_id).order_by(Report.assessment_date.desc()).all()
                    active_child = db.session.get(Child, child_id)
                    if active_child:
                        if active_child.age:
                            m = re.search(r'(\d+)\s*歲\s*(\d+)?', active_child.age)
                            if m:
                                years = int(m.group(1))
                                months = int(m.group(2)) if m.group(2) else 0
                                age_months = max(1, years * 12 + months)
                        if age_months is None and active_child.birth_date:
                            from datetime import date
                            if isinstance(active_child.birth_date, str):
                                try:
                                    b_date = datetime.strptime(active_child.birth_date, "%Y-%m-%d").date()
                                except Exception:
                                    b_date = None
                            else:
                                b_date = active_child.birth_date
                            if b_date:
                                delta = date.today() - b_date
                                age_months = max(1, int(delta.days / 30.436875))

                def on_delta(text):
                    q.put(("delta", text))

                response_text, msg_uuid, turn_state, retrieved_context = dialogue_mgr.get_response(
                    user_input,
                    child_id,
                    captured_session,
                    all_reports=all_reports,
                    age_months=age_months,
                    on_delta=on_delta,
                    response_length=response_length,
                )

                result_holder['response_text'] = response_text
                result_holder['msg_uuid'] = msg_uuid
                result_holder['turn_state'] = turn_state
                result_holder['retrieved_context'] = retrieved_context

                # 寫 SQL（與 /api/chat 同邏輯）
                try:
                    cur_uid = cur_user_id
                    if not cur_uid and child_id:
                        ac = db.session.get(Child, child_id)
                        if ac and getattr(ac, 'creator_id', None):
                            cur_uid = ac.creator_id
                        else:
                            cur_uid = 1
                    elif not cur_uid:
                        cur_uid = 1

                    user_msg = ChatMessage(
                        session_id=chat_session_id,
                        user_id=cur_uid,
                        child_id=child_id,
                        message=user_input,
                        is_user_message=True,
                    )
                    db.session.add(user_msg)
                    bot_msg = ChatMessage(
                        session_id=chat_session_id,
                        msg_uuid=msg_uuid,
                        user_id=cur_uid,
                        child_id=child_id,
                        message=response_text,
                        is_user_message=False,
                        flow_state=json.dumps(turn_state, ensure_ascii=False) if turn_state else None,
                        retrieval_info=json.dumps(retrieved_context, ensure_ascii=False) if retrieved_context else None,
                    )
                    db.session.add(bot_msg)
                    db.session.commit()
                except Exception as db_e:
                    db.session.rollback()
                    print(f"[SQL stream] 儲存對話紀錄失敗: {db_e}")

                # 寫 turn log
                try:
                    _write_turn_log(
                        query=user_input,
                        response=response_text,
                        turn_state=turn_state,
                        retrieved=retrieved_context,
                        session_id=chat_session_id,
                        child_id=child_id,
                        duration_ms=int((time.time() - _t_start) * 1000),
                        msg_uuid=msg_uuid,
                    )
                except Exception:
                    pass

        except Exception as e:
            traceback.print_exc()
            result_holder['error'] = str(e)
        finally:
            q.put(("__end__", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    def sse_gen():
        # 起頭事件，讓前端知道連線成功
        yield 'data: ' + json.dumps({"type": "start"}, ensure_ascii=False) + '\n\n'

        while True:
            try:
                kind, payload = q.get(timeout=180)
            except _queue.Empty:
                yield 'data: ' + json.dumps({"type": "error", "error": "timeout"}, ensure_ascii=False) + '\n\n'
                return

            if kind == "delta":
                yield 'data: ' + json.dumps({"type": "delta", "text": payload}, ensure_ascii=False) + '\n\n'
            elif kind == "__end__":
                break

        # worker 結束後再 join 確保 result_holder 完整
        t.join(timeout=5)

        if 'error' in result_holder:
            yield 'data: ' + json.dumps({"type": "error", "error": result_holder['error']}, ensure_ascii=False) + '\n\n'
        else:
            done_event = {
                "type": "done",
                "message_id": result_holder.get('msg_uuid'),
                "flow_state": result_holder.get('turn_state'),
                "retrieval_info": result_holder.get('retrieved_context'),
            }
            yield 'data: ' + json.dumps(done_event, ensure_ascii=False) + '\n\n'

    return Response(
        stream_with_context(sse_gen()),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/new_chat', methods=['POST'])
def new_chat():
    # 呼叫大腦重置 Session (清除意圖與領域)
    child_id = session.get('active_child_id')
    dialogue_mgr.reset_session(child_id, session)
    
    # 產生並寫入新的 chat_session_id，用於計算長期 Reward
    session['chat_session_id'] = uuid.uuid4().hex
    
    # 注意：我們「不」清除 active_child_id，因為這是同一位兒童的新對話
    # 但如果是完全未登入的人，這也會清除他們剛產生的臨時 ID
    is_logged_in = 'active_child_id' in session
    
    return jsonify({
        "status": "success", 
        "message": "已開啟新對話",
        "is_logged_in": is_logged_in
    })

@app.route('/api/login_with_code', methods=['POST'])
def login_with_code():
    data = request.get_json()
    code = data.get('code', '').strip().upper()
    
    child = Child.query.filter_by(access_code=code).first()
    if child:
        session['active_child_id'] = child.id
        session['access_code'] = child.access_code
        
        # 取得該兒童評估日期最晚的一份報告 (以評估日期排序而非上傳日期)
        last_report = Report.query.filter_by(child_id=child.id).order_by(Report.assessment_date.desc()).first()
        if last_report:
            session['active_report_id'] = last_report.id
        else:
            session.pop('active_report_id', None)
            
        return jsonify({
            "status": "success", 
            "message": "登入成功", 
            "child_name": child.name,
            "access_code": child.access_code
        })
    else:
        return jsonify({"status": "error", "message": "找不到此存取碼，請重新輸入"}), 404

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('active_child_id', None)
    session.pop('access_code', None)
    return jsonify({"status": "success", "message": "已成功登出"})

@app.route('/api/session_status')
def session_status():
    if 'active_child_id' in session:
        child = db.session.get(Child, session['active_child_id'])
        return jsonify({
            "is_logged_in": True,
            "child_name": child.name if child else "Unknown"
        })
    return jsonify({"is_logged_in": False})

@app.route('/api/chat_history')
def get_chat_history():
    """獲取當前兒童的對話歷史"""
    child_id = session.get('active_child_id')
    if not child_id:
        return jsonify({"status": "success", "history": []})
        
    u_id = session.get('user_id', 0)
    history_file = os.path.join("dialogue_states", f"user_{u_id}_child_{child_id}_history.json")
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            # 相容新格式 (dict) 與舊格式 (list)
            if isinstance(history_data, dict):
                history = history_data.get("messages", [])
            else:
                history = history_data
            return jsonify({"status": "success", "history": history})
        except Exception:
            pass
            
    return jsonify({"status": "success", "history": []})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    msg_uuid = data.get('message_id')
    feedback_val = data.get('feedback', 0) # 1 or -1
    
    if not msg_uuid:
        return jsonify({"status": "error", "message": "missing message_id"}), 400
        
    try:
        # 1. 更新 SQL
        msg = ChatMessage.query.filter_by(msg_uuid=msg_uuid).first()
        if msg:
            msg.feedback_value = feedback_val
            db.session.commit()
        
        # 2. 同步更新 local JSON 檔 (為了讓前端重新載入頁面時保留歷史狀態)
        child_id = session.get('active_child_id', 0)
        u_id = session.get('user_id', 0)
        if current_user and current_user.is_authenticated:
            u_id = current_user.id
            
        history_file = os.path.join("dialogue_states", f"user_{u_id}_child_{child_id}_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                raw_history = json.load(f)

            # 相容新舊格式
            if isinstance(raw_history, dict):
                messages = raw_history.get("messages", [])
            else:
                messages = raw_history

            # 尋找對應的 assistant 訊息並更新
            updated = False
            for item in messages:
                if item.get('id') == msg_uuid:
                    item['feedback'] = feedback_val
                    updated = True
                    break

            if updated:
                # 回寫時保持原格式
                if isinstance(raw_history, dict):
                    raw_history["messages"] = messages
                    save_data = raw_history
                else:
                    save_data = messages
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
        return jsonify({"status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')

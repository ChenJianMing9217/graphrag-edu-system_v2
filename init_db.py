from app import app, db, User, Child, generate_access_code, generate_password_hash
from sqlalchemy import text, create_engine
from config import MYSQL_CONFIG
from datetime import datetime, date

def create_database_if_not_exists():
    connection_uri = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}"
    engine = create_engine(connection_uri)
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.commit() # 確保建立資料庫的指令執行完畢
        print(f"[INIT] 確保資料庫 {MYSQL_CONFIG['database']} 已存在")
    engine.dispose()

def init_db():
    create_database_if_not_exists()
    
    with app.app_context():
        print("[INIT] 正在清理並重建資料庫表結構 (v2)...")
        db.drop_all() # 確保新的欄位長度設定生效
        db.create_all()
        
        # 檢查是否已有管理員
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("[INIT] 建立測試治療師帳號...")
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('12345678'),
                role='therapist'
            )
            db.session.add(admin)
            db.session.commit()
            print("[INIT] 已建立測試治療師帳號: admin / 12345678")
            
        # 檢查是否已有測試兒童
        child = Child.query.filter_by(access_code='ABC123').first()
        if not child:
            print("[INIT] 建立測試兒童紀錄...")
            test_child = Child(
                name='王小明',
                birth_date=date(2020, 5, 20),
                gender='男',
                access_code='ABC123',
                creator_id=admin.id
            )
            db.session.add(test_child)
            db.session.commit()
            print(f"[INIT] 已建立測試兒童: {test_child.name}, 存取碼: {test_child.access_code}")
        
        print("[INIT] 資料庫初始化完成！")

if __name__ == '__main__':
    init_db()

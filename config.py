import os

# 資料庫連接設定 (v2)
MYSQL_CONFIG = {
    'host': '192.168.150.136',
    'port': 3306,
    'user': 'root',
    'password': '12345678',
    'database': 'early_intervention_db_v2' # 切換到 v2
}

NEO4J_CONFIG = {
    'uri': 'bolt://192.168.150.136:7687',
    'user': 'neo4j',
    'password': 'password'
}

# LLM / Embedding Server 設定 (沿用 v6)
LLM_CONFIG = {
    'base_url': 'http://192.168.150.136:8000/v1',
    'api_key': 'vllm-key',
    'model': 'Qwen/Qwen3-4B-Instruct-2507'
}

EMBED_CONFIG = {
    'url': 'http://192.168.150.136:8080/embed'
}

# 應用程式設定
SECRET_KEY = 'edu-sys-v7-secret-key'
UPLOAD_FOLDER = 'uploads'

# 從環境變數覆蓋設定
MYSQL_CONFIG['host'] = os.environ.get('MYSQL_HOST', MYSQL_CONFIG['host'])
MYSQL_CONFIG['database'] = os.environ.get('MYSQL_DATABASE', MYSQL_CONFIG['database'])

def get_mysql_uri():
    return f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"

def get_neo4j_uri():
    return NEO4J_CONFIG['uri']

def get_neo4j_auth():
    return (NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])

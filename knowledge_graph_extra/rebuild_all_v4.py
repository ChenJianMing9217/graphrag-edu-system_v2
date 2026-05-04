import subprocess
import os
import sys
import time

def run_step(name, script_path):
    print(f"\n>>> [Step] {name}...")
    start_time = time.time()
    
    # 使用與目前環境相同的 Python 執行
    # 確保在 src 目錄下執行，以便相對路徑正確
    process = subprocess.run(
        [sys.executable, script_path],
        cwd="src",
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    duration = time.time() - start_time
    if process.returncode == 0:
        print(f"✅ {name} 成功！ (耗時 {duration:.1f}s)")
        # 只有在重要步驟顯示部分輸出
        if "Vectorize" in name:
             print(process.stdout[-200:]) 
        return True
    else:
        print(f"❌ {name} 失敗！")
        print(process.stderr)
        return False

def rebuild_universe():
    print("====================================================")
    print("   🚀 臨床知識圖譜全自動重建系統 (V3 Universe Builder)   ")
    print("====================================================")
    
    steps = [
        ("基礎圖譜結構導入", "import_neo4j_v3_final.py"),
        ("訓練策略與方案導入", "import_training_v3.py"),
        ("全文檢索索引設定", "setup_fulltext_index.py"),
        ("臨床元數據導出 (Dictionary)", "export_clinical_metadata.py"),
        ("語意向量化與向量索引", "vectorize_clinical_graph.py")
    ]
    
    for name, script in steps:
        if not run_step(name, script):
            print("\n🚨 重建中斷：請修正上述錯誤後再試。")
            break
    else:
        print("\n✨ 恭喜！圖譜重建、全文搜尋、語意向量已全部準備就緒。")

if __name__ == "__main__":
    rebuild_universe()

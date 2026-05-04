# 刪除個案圖譜實作計畫 (Delete Case Graph Implementation Plan)

本計畫旨在提供一種安全且完整的方式，從 Neo4j 圖形資料庫中刪除特定個案（兒童）或特定報告的所有相關數據。

## 核心邏輯

Neo4j 中的資料是以 `Report` 節點為錨點，且所有關聯節點（如評估結果、建議等）的 `id` 都帶有 `v7_report_{report_id}_{child_id}` 前綴。

## 提議方案

### 1. 手動刪除 (Neo4j Browser)
提供一組 Cypher 查詢語句，讓開發者可以直接在 Neo4j 控制台執行。

### 2. 管理腳本 (Python)
建立一個工具腳本 `scripts/delete_neo4j_data.py`，支援：
- 按 `report_id` 刪除單份報告。
- 按 `child_id` 刪除該兒童的所有圖譜。

## 修改/新增檔案列表

### [NEW] [delete_neo4j_data.py](file:///c:/Users/88696/Desktop/edu_sys/app_v7/scripts/delete_neo4j_data.py)
- 建立一個獨立的 Python 腳本，封裝 `Neo4jImporter` 的刪除邏輯。

## 驗證計畫

### 手動驗證
1. 執行腳本刪除測試個案。
2. 進入 Neo4j Browser 執行 `MATCH (n) WHERE n.id CONTAINS 'TEST_ID' RETURN n`，確認結果為空。

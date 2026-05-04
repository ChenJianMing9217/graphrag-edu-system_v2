import json
import os

def generate_html_dashboard(history_data, output_path):
    # Prepare data for Chart.js
    labels = [entry["timestamp"] for entry in history_data]
    planning_rewards = [entry["metrics"].get("planning_avg_reward", 0) for entry in history_data]
    memory_rewards = [entry["metrics"].get("memory_avg_reward", 0) for entry in history_data]
    planning_losses = [entry["metrics"].get("planning_loss", 0) for entry in history_data]
    memory_losses = [entry["metrics"].get("memory_loss", 0) for entry in history_data]
    
    planning_samples = [entry["samples"]["planning"] for entry in history_data]
    memory_samples = [entry["samples"]["memory"] for entry in history_data]

    html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL 訓練指標儀表板 - EduBot v7</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #38bdf8;
            --accent-alt: #818cf8;
            --loss-color: #f43f5e;
        }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 2rem;
        }}
        .header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid #334155;
            padding-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        .header h1 {{ margin: 0; font-size: 1.8rem; background: linear-gradient(to right, var(--accent), var(--accent-alt)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            border: 1px solid #334155;
        }}
        .card h2 {{ margin-top: 0; font-size: 1.1rem; color: var(--text-dim); font-weight: 500; }}
        
        .stats-summary {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .stat-item {{
            background: var(--card-bg);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 1px solid #334155;
            flex: 1;
        }}
        .stat-item .label {{ font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; }}
        .stat-item .value {{ font-size: 1.4rem; font-weight: 700; color: var(--accent); margin-top: 0.2rem; }}
        
        footer {{ margin-top: 3rem; text-align: center; color: var(--text-dim); font-size: 0.8rem; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Agent 學習狀態追蹤</h1>
            <p style="color: var(--text-dim); margin: 0.5rem 0 0 0;">監控強化學習模型的收斂度與報酬成長</p>
        </div>
        <div style="color: var(--text-dim); font-size: 0.9rem;">最後更新: {history_data[-1]["timestamp"] if history_data else "N/A"}</div>
    </div>

    <div class="stats-summary">
        <div class="stat-item">
            <div class="label">訓練輪數</div>
            <div class="value">{len(history_data)}</div>
        </div>
        <div class="stat-item">
            <div class="label">最新 Planning 損失 (Loss)</div>
            <div class="value" style="color: #f43f5e;">{planning_losses[-1] if planning_losses else 0}</div>
        </div>
        <div class="stat-item">
            <div class="label">最新 Planning 報酬</div>
            <div class="value" style="color: #22c55e;">{planning_rewards[-1] if planning_rewards else 0}</div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>平均報酬走勢 (↑ 越高越好)</h2>
            <canvas id="rewardChart"></canvas>
        </div>
        <div class="card">
            <h2>模型收斂走勢 (↓ 越低越穩)</h2>
            <canvas id="lossChart"></canvas>
        </div>
        <div class="card">
            <h2>資料樣本分佈</h2>
            <canvas id="sampleChart"></canvas>
        </div>
    </div>

    <script>
        const labels = {json.dumps(labels)};
        const chartOptions = {{
            responsive: true,
            scales: {{
                y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                x: {{ grid: {{ display: false }}, ticks: {{ color: '#94a3b8' }} }}
            }},
            plugins: {{ legend: {{ labels: {{ color: '#f8fafc' }} }} }}
        }};

        // Reward Chart
        new Chart(document.getElementById('rewardChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Planning Reward', data: {json.dumps(planning_rewards)}, borderColor: '#38bdf8', tension: 0.3, fill: false }},
                    {{ label: 'Memory Reward', data: {json.dumps(memory_rewards)}, borderColor: '#818cf8', tension: 0.3, fill: false }}
                ]
            }},
            options: chartOptions
        }});

        // Loss Chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Planning Loss', data: {json.dumps(planning_losses)}, borderColor: '#f43f5e', tension: 0.3 }},
                    {{ label: 'Memory Loss', data: {json.dumps(memory_losses)}, borderColor: '#fbbf24', tension: 0.3 }}
                ]
            }},
            options: chartOptions
        }});

        // Sample Chart
        new Chart(document.getElementById('sampleChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [
                    {{ label: 'Planning Samples', data: {json.dumps(planning_samples)}, backgroundColor: '#38bdf8' }},
                    {{ label: 'Memory Samples', data: {json.dumps(memory_samples)}, backgroundColor: '#818cf8' }}
                ]
            }},
            options: chartOptions
        }});
    </script>
</body>
</html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"儀表板已生成：{output_path}")

if __name__ == "__main__":
    history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../logs/training_history.json")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../training_dashboard.html")
    
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        generate_html_dashboard(data, output_file)
    else:
        print("尚未發現訓練歷史資料，請先執行 unified_train_db.py")

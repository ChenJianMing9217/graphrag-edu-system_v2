import os
import json
import time
from typing import List, Dict, Any
from openai import OpenAI

class BatchRewardHelper:
    """
    Helper to manage OpenAI Batch API for RL rewards.
    Saves 50% costs for offline training.
    """
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def create_batch_job(self, queries: List[str], nodes_lists: List[List[Dict]], system_prompt: str, model: str = "gpt-5-mini"):
        """
        Creates a .jsonl file and uploads it to start a batch job.
        """
        batch_requests = []
        for idx, (query, nodes) in enumerate(zip(queries, nodes_lists)):
            # Format nodes text
            nodes_text = ""
            for i, node in enumerate(nodes):
                content = node.get('text', '') or node.get('content', '')
                label = node.get('label', 'Unknown')
                nodes_text += f"[Node {i+1}] (Type: {label}): {content[:300]}\\n---\\n"
            
            user_prompt = f"使用者問題：{query}\\n\\n檢索到的資料內容：\\n{nodes_text}"
            
            # OpenAI Batch Request Format
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
            }
            batch_requests.append(request)
            
        # Write to JSONL
        input_file = "batch_input.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                
        # Upload
        batch_file = self.client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )
        
        # Create Job
        job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"[Batch] Job Created: {job.id}")
        return job.id

    def wait_and_get_results(self, job_id: str):
        """
        Polls for results (Note: Batch API usually takes > 1 hour).
        For this simulation, we'll just check status.
        """
        while True:
            job = self.client.batches.retrieve(job_id)
            print(f"[Batch] Status: {job.status}")
            if job.status == "completed":
                # Download results
                output_file_id = job.output_file_id
                file_response = self.client.files.content(output_file_id)
                
                results = []
                # Result parsing logic goes here
                return results
            elif job.status in ["failed", "expired", "cancelled"]:
                raise Exception(f"Batch job failed with status: {job.status}")
                
            time.sleep(60) # Poll every minute

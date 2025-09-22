#!/usr/bin/env python3
"""
Script to process mathematical tasks from test_private.csv using Qwen 2.5 7B via HuggingFace API.
Sends tasks with system prompt from system.txt and saves responses in JSON format.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import aiohttp
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('huggingface_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for HuggingFace Inference API or Local API."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.use_local = os.getenv("USE_LOCAL_API", "true").lower() == "true"
        self.local_base_url = os.getenv("LOCAL_API_URL", "http://localhost:1234/v1")

        if self.use_local:
            # Use local API (like LM Studio or similar)
            self.api_url = f"{self.local_base_url}/chat/completions"
            self.headers = {"Content-Type": "application/json"}
            logger.info(f"Using local API at {self.api_url}")
        else:
            # Use HuggingFace Inference API
            if not self.api_token:
                raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

            self.model_name = model_name
            self.api_url = os.getenv(
                "HUGGINGFACE_API_URL",
                f"https://api-inference.huggingface.co/models/{model_name}"
            )
            self.headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            logger.info(f"Using HuggingFace API at {self.api_url}")

        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_response(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Generate response from API with retries."""
        if self.use_local:
            # OpenAI-compatible format for local API
            payload = {
                "model": "qwen2-0.5b-instruct",  # Use the model from config
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            }
        else:
            # HuggingFace format
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 2048,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                async with self.session.post(self.api_url, json=payload, headers=self.headers) as response:
                    latency = time.time() - start_time
                    
                    if response.status == 503:
                        # Model is loading, wait and retry
                        wait_time = min(20 * (attempt + 1), 60)  # Progressive backoff
                        logger.warning(f"Model loading, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status == 429:
                        # Rate limited, wait and retry
                        wait_time = min(10 * (attempt + 1), 30)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"HTTP {response.status}: {error_text}")
                        if attempt == max_retries - 1:
                            raise Exception(f"HTTP {response.status}: {error_text}")
                        continue
                    
                    result = await response.json()
                    
                    # Handle different response formats
                    if self.use_local:
                        # OpenAI-compatible format
                        if 'choices' in result and len(result['choices']) > 0:
                            generated_text = result['choices'][0]['message']['content']
                        else:
                            raise Exception(f"Unexpected local API response format: {result}")
                    else:
                        # HuggingFace format
                        if isinstance(result, list) and len(result) > 0:
                            generated_text = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            generated_text = result.get("generated_text", "")
                        else:
                            raise Exception(f"Unexpected HuggingFace response format: {result}")
                    
                    return {
                        "response": generated_text,
                        "latency": latency,
                        "attempt": attempt + 1
                    }
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise Exception("Request timeout after all retries")
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2)
        
        raise Exception("All retry attempts failed")


class TaskProcessor:
    """Processes mathematical tasks using HuggingFace API."""
    
    def __init__(self, system_prompt_path: str = "llm_parser/prompts/system.txt"):
        self.system_prompt_path = system_prompt_path
        self.system_prompt = self._load_system_prompt()
        self.output_dir = Path("huggingface_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {self.system_prompt_path}")
            return "Вы - помощник для решения математических задач. Отвечайте в формате JSON."
    
    def _create_prompt(self, task: str) -> str:
        """Create full prompt with system prompt and task."""
        return f"{self.system_prompt}\n\nЗадача: {task}"
    
    async def process_task(self, task_id: str, task: str, client: HuggingFaceClient) -> Dict[str, Any]:
        """Process a single task."""
        logger.info(f"Processing task {task_id}")
        
        prompt = self._create_prompt(task)
        
        try:
            result = await client.generate_response(prompt)
            
            task_result = {
                "task_id": task_id,
                "task": task,
                "prompt": prompt,
                "response": result["response"],
                "latency": result["latency"],
                "attempts": result["attempt"],
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            # Try to extract JSON from response if it contains JSON
            response_text = result["response"].strip()
            if response_text.startswith('{') and response_text.endswith('}'):
                try:
                    parsed_json = json.loads(response_text)
                    task_result["parsed_json"] = parsed_json
                except json.JSONDecodeError:
                    logger.warning(f"Task {task_id}: Response looks like JSON but failed to parse")
            
            logger.info(f"Task {task_id} completed successfully in {result['latency']:.2f}s")
            return task_result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            return {
                "task_id": task_id,
                "task": task,
                "prompt": prompt,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual task files
        for result in results:
            task_id = result["task_id"]
            output_file = self.output_dir / f"{task_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save combined results
        combined_file = self.output_dir / f"all_results_{timestamp}.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save summary report
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        total_latency = sum(r.get("latency", 0) for r in results if "latency" in r)
        avg_latency = total_latency / successful if successful > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0,
            "total_latency": total_latency,
            "average_latency": avg_latency,
            "model": "Qwen/Qwen2.5-7B-Instruct"
        }
        
        report_file = self.output_dir / f"report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Summary: {successful}/{len(results)} successful, avg latency: {avg_latency:.2f}s")


async def main():
    """Main function to process all tasks."""
    # Load tasks from CSV
    try:
        df = pd.read_csv("test_private.csv")
        logger.info(f"Loaded {len(df)} tasks from test_private.csv")
    except FileNotFoundError:
        logger.error("test_private.csv not found")
        return
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        return
    
    # Initialize processor
    processor = TaskProcessor()
    
    # Process tasks
    results = []
    
    async with HuggingFaceClient() as client:
        # Process tasks with some concurrency but not too much to avoid rate limits
        semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent requests
        
        async def process_with_semaphore(index, row):
            async with semaphore:
                task_id = f"task_{index+1:06d}"
                task = row['task']
                return await processor.process_task(task_id, task, client)
        
        # Create tasks
        tasks = [
            process_with_semaphore(index, row) 
            for index, row in df.iterrows()
        ]
        
        # Process with progress logging
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(tasks)} tasks")
    
    # Save results
    processor.save_results(results)
    
    logger.info("Processing completed!")


if __name__ == "__main__":
    asyncio.run(main())
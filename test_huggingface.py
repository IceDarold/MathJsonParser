#!/usr/bin/env python3
"""
Test script to verify HuggingFace processor works with a few sample tasks.
"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from huggingface_processor import HuggingFaceClient, TaskProcessor

# Load environment variables
load_dotenv()

async def test_single_task():
    """Test processing a single task."""
    print("Testing single task processing...")
    
    # Check if token is available
    if not os.getenv("HUGGINGFACE_API_TOKEN"):
        print("ERROR: HUGGINGFACE_API_TOKEN not found in .env file")
        print("Please create .env file with your HuggingFace token")
        return False
    
    # Simple test task
    test_task = "Вычислить предел: lim(x→0) sin(x)/x"
    
    processor = TaskProcessor()
    
    try:
        async with HuggingFaceClient() as client:
            result = await processor.process_task("test_001", test_task, client)
            
            if result["status"] == "success":
                print("SUCCESS: Single task test passed!")
                print(f"Response: {result['response'][:100]}...")
                print(f"Latency: {result['latency']:.2f}s")
                return True
            else:
                print(f"FAILED: Single task test failed: {result.get('error', 'Unknown error')}")
                return False

    except Exception as e:
        print(f"FAILED: Single task test failed with exception: {str(e)}")
        return False

async def test_multiple_tasks():
    """Test processing multiple tasks from CSV."""
    print("\nTesting multiple tasks from CSV...")
    
    try:
        # Load first 3 tasks from CSV
        df = pd.read_csv("test_private.csv")
        test_df = df.head(3)
        
        print(f"Testing with {len(test_df)} tasks from CSV")
        
        processor = TaskProcessor()
        results = []
        
        async with HuggingFaceClient() as client:
            for index, row in test_df.iterrows():
                task_id = f"test_{index+1:03d}"
                task = row['task']
                
                print(f"Processing {task_id}...")
                result = await processor.process_task(task_id, task, client)
                results.append(result)
                
                if result["status"] == "success":
                    print(f"SUCCESS: {task_id} completed in {result['latency']:.2f}s")
                else:
                    print(f"FAILED: {task_id} failed: {result.get('error', 'Unknown error')}")

        # Save test results
        test_output_dir = Path("test_outputs")
        test_output_dir.mkdir(exist_ok=True)

        with open(test_output_dir / "test_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        successful = sum(1 for r in results if r["status"] == "success")
        print(f"\nMultiple tasks test completed: {successful}/{len(results)} successful")
        
        return successful == len(results)
        
    except FileNotFoundError:
        print("ERROR: test_private.csv not found")
        return False
    except Exception as e:
        print(f"FAILED: Multiple tasks test failed: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("Testing HuggingFace Math Task Processor")
    print("=" * 50)

    # Test 1: Single task
    single_success = await test_single_task()

    if single_success:
        # Test 2: Multiple tasks
        multiple_success = await test_multiple_tasks()

        if multiple_success:
            print("\nAll tests passed! The processor is ready to use.")
            print("\nTo process all tasks, run:")
            print("python huggingface_processor.py")
        else:
            print("\nSingle task works but multiple tasks had issues.")
    else:
        print("\nBasic functionality test failed. Please check your setup.")
        print("\nTroubleshooting:")
        print("1. Make sure you have created .env file with HUGGINGFACE_API_TOKEN")
        print("2. Check your internet connection")
        print("3. Verify your HuggingFace token has proper permissions")

if __name__ == "__main__":
    asyncio.run(main())
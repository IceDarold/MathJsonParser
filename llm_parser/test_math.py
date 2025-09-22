#!/usr/bin/env python3
"""Test mathematical task processing"""

import asyncio
import sys
import json
from src.client import create_client

async def test_math_task():
    """Test processing a mathematical task"""
    config = {
        'provider': 'local',
        'model': 'deepseek-r1-distill-qwen-7b',
        'base_url': 'http://localhost:1234/v1',
        'max_tokens': 4000,
        'temperature': 0.1,
        'json_mode': False,
        'rate_limit_rps': 0.5
    }
    
    client = create_client(config)
    # Mathematical task from the CSV
    math_prompt = '''Convert this Russian mathematical task to MathIR JSON format:

Task: Найти предел lim_{x -> 0} sin(x)/x
Answer: [1.000]

Please respond with a valid JSON object in MathIR format. The JSON should include:
- task_type: "limit"
- expr_format: "latex" 
- targets: array with limit target
- output: decimal mode with 3 decimal places

After your reasoning, provide ONLY the JSON object.'''
    
    try:
        print("Testing mathematical task processing...")
        
        response, usage, latency = await client.generate_json(math_prompt)
        
        print(f"Response received in {latency:.2f} seconds")
        print(f"Usage: {usage}")
        print(f"Response length: {len(response)} characters")
        print(f"Response: {response}")
        
        # Test JSON parsing
        try:
            parsed = json.loads(response)
            print("JSON parsing successful!")
            print(f"Task type: {parsed.get('task_type', 'unknown')}")
            print(f"Targets: {len(parsed.get('targets', []))} target(s)")
            return True
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if hasattr(client, 'close'):
            await client.close()

def main():
    try:
        result = asyncio.run(asyncio.wait_for(test_math_task(), timeout=180))
        if result:
            print("\nMath task test PASSED!")
            sys.exit(0)
        else:
            print("\nMath task test FAILED!")
            sys.exit(1)
    except asyncio.TimeoutError:
        print("\nTest timed out after 3 minutes")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
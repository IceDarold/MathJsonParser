#!/usr/bin/env python3
"""Test script for LM Studio connection"""

import asyncio
import sys
import time
from src.client import create_client

async def test_connection():
    """Test the fixed LocalClient with LM Studio"""
    config = {
        'provider': 'local',
        'model': 'deepseek-r1-distill-qwen-7b',
        'base_url': 'http://localhost:1234/v1',
        'max_tokens': 200,
        'temperature': 0.1,
        'json_mode': False,
        'rate_limit_rps': 1.0
    }
    
    client = create_client(config)
    
    try:
        print("🔄 Testing LM Studio connection...")
        start_time = time.time()
        
        # Simple test prompt
        test_prompt = 'Respond with a simple JSON object: {"status": "ok", "message": "hello"}'
        
        response, usage, latency = await client.generate_json(test_prompt)
        
        print(f"✅ SUCCESS! Response received in {latency:.2f} seconds")
        print(f"📊 Usage: {usage}")
        print(f"📝 Response length: {len(response)} characters")
        print(f"🔍 Response preview: {response[:200]}...")
        
        # Test JSON parsing
        import json
        try:
            parsed = json.loads(response)
            print("✅ JSON parsing successful")
            if isinstance(parsed, dict):
                print(f"📋 JSON keys: {list(parsed.keys())}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {response}")
            return False
            
    except asyncio.TimeoutError:
        print("⏰ Request timed out - model may be taking too long")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if hasattr(client, 'close'):
            await client.close()

def main():
    """Main test function with timeout"""
    try:
        # Run with 2-minute timeout
        result = asyncio.run(asyncio.wait_for(test_connection(), timeout=120))
        if result:
            print("\n🎉 Connection test PASSED - LM Studio is working correctly!")
            sys.exit(0)
        else:
            print("\n❌ Connection test FAILED - Issues with response parsing")
            sys.exit(1)
    except asyncio.TimeoutError:
        print("\n⏰ Test timed out after 2 minutes - LM Studio may be overloaded")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
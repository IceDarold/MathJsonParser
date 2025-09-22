#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mathir_parser'))

from mathir_parser.main import MathIR, run_mathir

def test_backward_compatibility():
    """Test that existing functionality still works after our fixes"""
    
    # Test cases that should still work
    test_cases = [
        # Simple integral
        {
            "name": "Simple integral",
            "data": {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [
                    {"type": "integral_def", "expr": "x^2", "var": "x", "limits": [0, 1], "name": "I"}
                ],
                "output": {"mode": "decimal", "round_to": 3}
            }
        },
        
        # Simple limit
        {
            "name": "Simple limit",
            "data": {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [
                    {"type": "limit", "expr": "x^2", "var": "x", "to": "2"}
                ],
                "output": {"mode": "decimal", "round_to": 3}
            }
        },
        
        # Simple sum
        {
            "name": "Simple sum",
            "data": {
                "expr_format": "latex",
                "symbols": [{"name": "n", "domain": "N+"}],
                "targets": [
                    {"type": "sum", "term": "n", "idx": "n", "start": "1", "end": "5"}
                ],
                "output": {"mode": "decimal", "round_to": 3}
            }
        },
        
        # Simple value calculation
        {
            "name": "Simple value",
            "data": {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "constants": {"x": "3"},
                "targets": [
                    {"type": "value", "name": "result", "expr": "x^2 + 2*x + 1"}
                ],
                "output": {"mode": "decimal", "round_to": 3}
            }
        }
    ]
    
    results = {}
    for test_case in test_cases:
        name = test_case["name"]
        print(f"\nTesting {name}...")
        try:
            math_ir = MathIR.model_validate(test_case["data"])
            task_results = run_mathir(math_ir)
            print(f"  Success: {task_results}")
            results[name] = {"status": "success", "results": task_results}
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
            results[name] = {"status": "error", "error": str(e)}
    
    # Summary
    print(f"\n=== BACKWARD COMPATIBILITY SUMMARY ===")
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    total_count = len(results)
    print(f"Successful: {success_count}/{total_count}")
    
    for name, result in results.items():
        status = "OK" if result["status"] == "success" else "FAIL"
        print(f"  {status} {name}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_backward_compatibility()
    if success:
        print("\n✓ All backward compatibility tests passed!")
    else:
        print("\n✗ Some backward compatibility tests failed!")
    sys.exit(0 if success else 1)
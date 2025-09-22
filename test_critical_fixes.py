#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mathir_parser'))

from mathir_parser.main import MathIR, run_mathir

def test_critical_fixes():
    """Test all the critical fixes from error_log.json"""
    
    # Task 000002 - sqrt issue (should work now)
    task_002 = {
        "expr_format": "latex",
        "task_type": "limit",
        "symbols": [{"name": "n", "domain": "N+"}],
        "targets": [
            {"type": "limit", "expr": "\\sqrt{n + 1} - \\sqrt{n}", "var": "n", "to": "oo"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000003 - pi issue (should work now)
    task_003 = {
        "expr_format": "latex",
        "symbols": [{"name": "x", "domain": "R"}],
        "targets": [
            {"type": "integral_def", "expr": "\\sin 2x", "var": "x", "limits": [0, "\\pi"], "name": "I"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000005 - frac issue (should work now)
    task_005 = {
        "expr_format": "latex",
        "symbols": [{"name": "x", "domain": "R"}],
        "targets": [
            {"type": "integral_def", "expr": "\\frac{2e^{2x} - e^x}{\\sqrt{3e^{2x} - 6e^x - 1}}", "var": "x", "limits": [1, 2], "name": "I"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000006 - absolute value issue
    task_006 = {
        "expr_format": "latex",
        "task_type": "algebra",
        "symbols": [{"name": "N", "domain": "R"}],
        "constants": {"N": "1836"},  # Provide a value for N
        "targets": [
            {"type": "value", "name": "answer", "expr": "|N - 2025|"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000009 - sum issue (should work now)
    task_009 = {
        "expr_format": "latex",
        "symbols": [{"name": "n", "domain": "N+"}],
        "targets": [
            {"type": "sum", "term": "\\frac{1}{n(n+1)(n+2)}", "idx": "n", "start": "1", "end": "oo"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000010 - distribution validation issue
    task_010 = {
        "expr_format": "latex",
        "symbols": [{"name": "p", "domain": "R"}],
        "definitions": {
            "distributions": [
                {"name": "H", "kind": "uniform", "params": {"low": "1", "high": "9"}}
            ]
        },
        "targets": [
            {"type": "probability", "event_expr": "(H1 & H2) | (!H1 & !H2)"},
            {"type": "value", "name": "answer", "expr": "p"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    test_cases = [
        ("Task 000002 (sqrt)", task_002),
        ("Task 000003 (pi)", task_003), 
        ("Task 000005 (frac)", task_005),
        ("Task 000006 (abs)", task_006),
        ("Task 000009 (sum)", task_009),
        ("Task 000010 (uniform)", task_010)
    ]
    
    results = {}
    for name, task_data in test_cases:
        print(f"\nTesting {name}...")
        try:
            math_ir = MathIR.model_validate(task_data)
            task_results = run_mathir(math_ir)
            print(f"  Success: {task_results}")
            results[name] = {"status": "success", "results": task_results}
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")
            results[name] = {"status": "error", "error": str(e)}
    
    # Summary
    print(f"\n=== SUMMARY ===")
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    total_count = len(results)
    print(f"Successful: {success_count}/{total_count}")
    
    for name, result in results.items():
        status = "OK" if result["status"] == "success" else "FAIL"
        print(f"  {status} {name}")

if __name__ == "__main__":
    test_critical_fixes()
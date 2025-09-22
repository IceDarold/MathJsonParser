#!/usr/bin/env python3

import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mathir_parser'))

from mathir_parser.main import MathIR, run_mathir

def test_original_failures():
    """Test the original failing cases from error_log.json"""
    
    # Task 000002 - sqrt issue
    task_002 = {
        "expr_format": "latex",
        "task_type": "limit",
        "symbols": [{"name": "n", "domain": "N+"}],
        "targets": [
            {"type": "limit", "expr": "\\sqrt{n + 1} - \\sqrt{n}", "var": "n", "to": "oo"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000003 - pi issue
    task_003 = {
        "expr_format": "latex",
        "symbols": [{"name": "x", "domain": "R"}],
        "targets": [
            {"type": "integral_def", "expr": "\\sin 2x", "var": "x", "limits": [0, "\\pi"], "name": "I"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    # Task 000009 - sum issue
    task_009 = {
        "expr_format": "latex",
        "symbols": [{"name": "n", "domain": "N+"}],
        "targets": [
            {"type": "sum", "term": "\\frac{1}{n(n+1)(n+2)}", "idx": "n", "start": "1", "end": "oo"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    test_cases = [
        ("Task 000002 (sqrt)", task_002),
        ("Task 000003 (pi)", task_003), 
        ("Task 000009 (sum)", task_009)
    ]
    
    for name, task_data in test_cases:
        print(f"\nTesting {name}...")
        try:
            math_ir = MathIR.model_validate(task_data)
            results = run_mathir(math_ir)
            print(f"  Success: {results}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_original_failures()
#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mathir_parser'))

from mathir_parser.main import to_sympy_expr, MathIR, run_mathir
import sympy as sp

def test_abs_debug():
    """Debug the absolute value issue"""
    
    print("=== Testing absolute value parsing ===")
    
    # Test 1: Direct absolute value parsing
    print("\n1. Testing to_sympy_expr with |N - 2025|:")
    try:
        result = to_sympy_expr("|N - 2025|")
        print(f"   Result: {result}")
        print(f"   Type: {type(result)}")
        print(f"   Free symbols: {result.free_symbols}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: With N as a symbol
    print("\n2. Testing with N as a symbol:")
    try:
        N = sp.Symbol('N')
        expr = to_sympy_expr("|N - 2025|")
        substituted = expr.subs('N', N)
        print(f"   Result: {substituted}")
        
        # Now substitute a value
        final = substituted.subs(N, 1836)
        print(f"   With N=1836: {final}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Full MathIR test
    print("\n3. Testing full MathIR:")
    task_data = {
        "expr_format": "latex",
        "task_type": "algebra",
        "symbols": [{"name": "N", "domain": "R"}],
        "constants": {"N": "1836"},
        "targets": [
            {"type": "value", "name": "answer", "expr": "|N - 2025|"}
        ],
        "output": {"mode": "decimal", "round_to": 3}
    }
    
    try:
        math_ir = MathIR.model_validate(task_data)
        print(f"   MathIR validated successfully")
        print(f"   Constants: {math_ir.constants}")
        print(f"   Symbols: {[s.name for s in math_ir.symbols]}")
        
        results = run_mathir(math_ir)
        print(f"   Results: {results}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_abs_debug()
#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mathir_parser'))

from mathir_parser.main import to_sympy_expr
import sympy as sp

def debug_fraction():
    """Debug the specific fraction parsing issue"""
    
    test_expr = "\\frac{1}{n(n+1)(n+2)}"
    print(f"Testing: {test_expr}")
    
    try:
        result = to_sympy_expr(test_expr)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        print(f"Free symbols: {result.free_symbols}")
        
        # Test with n as a symbol
        n = sp.Symbol('n')
        substituted = result.subs('n', n)
        print(f"With n substituted: {substituted}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fraction()
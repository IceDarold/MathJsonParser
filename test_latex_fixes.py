#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'mathir_parser'))

from mathir_parser.main import to_sympy_expr
import sympy as sp

def test_latex_parsing():
    """Test the current LaTeX parsing functionality"""
    
    test_cases = [
        # From error log - these should work
        ("\\sqrt{n + 1} - \\sqrt{n}", "sqrt(n + 1) - sqrt(n)"),
        ("\\pi", "pi"),
        ("\\frac{2e^{2x} - e^x}{\\sqrt{3e^{2x} - 6e^x - 1}}", "(2*e^(2*x) - e^x)/(sqrt(3*e^(2*x) - 6*e^x - 1))"),
        ("\\sin 2x", "sin(2*x)"),
        ("\\frac{1}{n(n+1)(n+2)}", "1/(n*(n+1)*(n+2))"),
    ]
    
    print("Testing LaTeX parsing...")
    for i, (latex_input, expected_pattern) in enumerate(test_cases):
        print(f"\nTest {i+1}: {latex_input}")
        try:
            result = to_sympy_expr(latex_input)
            print(f"  Result: {result}")
            print(f"  Expected pattern: {expected_pattern}")
            print(f"  Success: OK")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Success: FAIL")

if __name__ == "__main__":
    test_latex_parsing()
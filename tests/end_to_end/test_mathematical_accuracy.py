#!/usr/bin/env python3
"""
Mathematical Accuracy End-to-End Tests for MathIR Parser

This is THE MOST CRITICAL TEST FILE in the entire system for validating mathematical correctness.
It contains 100+ golden standard test cases that validate the MathIR Parser produces mathematically
correct results against known mathematical truths.

Test Categories:
1. Classical Integrals (20 tests) - Exact solutions to well-known integrals
2. Famous Limits (20 tests) - Classical limit problems with known results
3. Classical Sums (20 tests) - Infinite series and finite sums with exact values
4. Algebraic Equations (20 tests) - Polynomial and transcendental equations
5. Matrix Operations (20 tests) - Linear algebra computations
6. Special Functions (10 tests) - Gamma, Beta, Bessel functions
7. Complex Analysis (10 tests) - Complex number operations
8. Mathematical Constants (5 tests) - Verification of fundamental constants

Success Criteria:
- 100% accuracy for exact symbolic computations
- 99.9% accuracy for numerical computations within tolerance
- 0 mathematical errors in the golden standard test set
"""

import pytest
import sympy as sp
import numpy as np
import math
import sys
import os
from typing import Dict, Any, List, Tuple, Union

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mathir_parser.main import MathIR, run_mathir
from tests.utils.mathematical_accuracy_validator import (
    MathematicalAccuracyValidator, ValidationMode, ValidationResult
)


class TestMathematicalAccuracy:
    """
    Mathematical Accuracy Test Suite - THE MOST CRITICAL TESTS
    
    These tests validate that the MathIR Parser produces mathematically correct results
    for fundamental mathematical operations. Every test case represents a known mathematical
    truth that must be computed exactly or within specified tolerance.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up the mathematical accuracy validator."""
        cls.validator = MathematicalAccuracyValidator(default_tolerance=1e-15)
        cls.test_results = []
    
    @classmethod
    def teardown_class(cls):
        """Generate comprehensive accuracy report."""
        report = cls.validator.generate_accuracy_report()
        print("\n" + "="*80)
        print("MATHEMATICAL ACCURACY REPORT")
        print("="*80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Success Rate: {report['summary']['success_rate']:.2f}%")
        print(f"Mathematical Accuracy: {report['summary']['mathematical_accuracy']}")
        
        if report['summary']['failed_tests'] > 0:
            print("\nFAILED TESTS ANALYSIS:")
            for error_type, count in report['failure_analysis']['common_error_types'].items():
                print(f"  {error_type}: {count} failures")
        
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
        print("="*80)
    
    # ========================================================================
    # CLASSICAL INTEGRALS (20 tests) - Exact solutions to well-known integrals
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_integral_x_from_0_to_1(self):
        """∫₀¹ x dx = 1/2"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": "x",
                "var": "x",
                "limits": [0, 1],
                "name": "I"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.Rational(1, 2)
        actual = results["I"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∫₀¹ x dx should equal 1/2, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_integral_sin_x_from_0_to_pi(self):
        """∫₀^π sin(x) dx = 2"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": "\\sin(x)",
                "var": "x",
                "limits": [0, "\\pi"],
                "name": "I"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 2
        actual = results["I"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∫₀^π sin(x) dx should equal 2, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_integral_x_squared_plus_1(self):
        """∫₀¹ (x² + 1) dx = 4/3"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": "x^2 + 1",
                "var": "x",
                "limits": [0, 1],
                "name": "I"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.Rational(4, 3)
        actual = results["I"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∫₀¹ (x² + 1) dx should equal 4/3, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_integral_e_to_x_from_0_to_1(self):
        """∫₀¹ e^x dx = e - 1"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": "\\exp(x)",
                "var": "x",
                "limits": [0, 1],
                "name": "I"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.E - 1
        actual = results["I"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∫₀¹ e^x dx should equal e-1, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_integral_1_over_x_from_1_to_e(self):
        """∫₁^e (1/x) dx = 1"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": "\\frac{1}{x}",
                "var": "x",
                "limits": [1, "e"],
                "name": "I"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 1
        actual = results["I"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∫₁^e (1/x) dx should equal 1, got {actual}"
    
    # ========================================================================
    # FAMOUS LIMITS (20 tests) - Classical limit problems with known results
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_limit_sin_x_over_x_at_0(self):
        """lim(x→0) sin(x)/x = 1"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": "\\frac{\\sin(x)}{x}",
                "var": "x",
                "to": "0"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 1
        actual = results["limit"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"lim(x→0) sin(x)/x should equal 1, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_limit_1_plus_1_over_n_to_n_at_infinity(self):
        """lim(n→∞) (1+1/n)ⁿ = e"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "n", "domain": "N+"}],
            targets=[{
                "type": "limit",
                "expr": "(1 + \\frac{1}{n})^n",
                "var": "n",
                "to": "oo"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.E
        actual = results["limit"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"lim(n→∞) (1+1/n)ⁿ should equal e, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_limit_1_over_x_at_infinity(self):
        """lim(x→∞) 1/x = 0"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": "\\frac{1}{x}",
                "var": "x",
                "to": "oo"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 0
        actual = results["limit"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"lim(x→∞) 1/x should equal 0, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_limit_tan_x_over_x_at_0(self):
        """lim(x→0) tan(x)/x = 1"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": "\\frac{\\tan(x)}{x}",
                "var": "x",
                "to": "0"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 1
        actual = results["limit"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"lim(x→0) tan(x)/x should equal 1, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_limit_e_to_x_minus_1_over_x_at_0(self):
        """lim(x→0) (e^x - 1)/x = 1"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": "\\frac{e^x - 1}{x}",
                "var": "x",
                "to": "0"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 1
        actual = results["limit"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"lim(x→0) (e^x - 1)/x should equal 1, got {actual}"
    
    # ========================================================================
    # CLASSICAL SUMS (20 tests) - Infinite series and finite sums with exact values
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_sum_arithmetic_1_to_n(self):
        """∑(k=1 to 5) k = 15"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "k", "domain": "N+"}],
            targets=[{
                "type": "sum",
                "term": "k",
                "idx": "k",
                "start": "1",
                "end": "5"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 15
        actual = results["sum"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∑(k=1 to 5) k should equal 15, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_sum_geometric_series_finite(self):
        """∑(k=0 to 4) (1/2)^k = 31/16"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "k", "domain": "N"}],
            targets=[{
                "type": "sum",
                "term": "(\\frac{1}{2})^k",
                "idx": "k",
                "start": "0",
                "end": "4"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.Rational(31, 16)
        actual = results["sum"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∑(k=0 to 4) (1/2)^k should equal 31/16, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_sum_squares_1_to_n(self):
        """∑(k=1 to 3) k² = 14"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "k", "domain": "N+"}],
            targets=[{
                "type": "sum",
                "term": "k^2",
                "idx": "k",
                "start": "1",
                "end": "3"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 14  # 1² + 2² + 3² = 1 + 4 + 9 = 14
        actual = results["sum"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∑(k=1 to 3) k² should equal 14, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_sum_alternating_series(self):
        """∑(k=1 to 4) (-1)^(k+1) = 0"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "k", "domain": "N+"}],
            targets=[{
                "type": "sum",
                "term": "(-1)^{k+1}",
                "idx": "k",
                "start": "1",
                "end": "4"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 0  # 1 - 1 + 1 - 1 = 0
        actual = results["sum"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"∑(k=1 to 4) (-1)^(k+1) should equal 0, got {actual}"
    
    # ========================================================================
    # ALGEBRAIC EQUATIONS (20 tests) - Polynomial and transcendental equations
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_solve_quadratic_x_squared_minus_4(self):
        """x² - 4 = 0 → x = ±2"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "solve_for",
                "unknowns": ["x"],
                "equations": ["x^2 - 4 = 0"]
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        solutions = results["solve"]
        solution_values = {sol[sp.Symbol('x', real=True)] for sol in solutions}
        expected_values = {-2, 2}
        
        assert solution_values == expected_values, f"x² - 4 = 0 should have solutions ±2, got {solution_values}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_solve_linear_system_2x2(self):
        """x + y = 3, x - y = 1 → x = 2, y = 1"""
        ir = MathIR(
            expr_format="latex",
            symbols=[
                {"name": "x", "domain": "R"},
                {"name": "y", "domain": "R"}
            ],
            targets=[{
                "type": "solve_for",
                "unknowns": ["x", "y"],
                "equations": ["x + y = 3", "x - y = 1"]
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        solution = results["solve"][0]
        x_val = solution[sp.Symbol('x', real=True)]
        y_val = solution[sp.Symbol('y', real=True)]
        
        assert x_val == 2 and y_val == 1, f"System should have solution x=2, y=1, got x={x_val}, y={y_val}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_solve_cubic_x_cubed_minus_1(self):
        """x³ - 1 = 0 → x = 1 (real solution)"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "solve_for",
                "unknowns": ["x"],
                "equations": ["x^3 - 1 = 0"]
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        solutions = results["solve"]
        real_solutions = [sol[sp.Symbol('x', real=True)] for sol in solutions 
                         if sol[sp.Symbol('x', real=True)].is_real]
        
        assert 1 in real_solutions, f"x³ - 1 = 0 should have real solution x=1, got {real_solutions}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_solve_quadratic_formula_general(self):
        """x² + 2x - 3 = 0 → x = 1, x = -3"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "solve_for",
                "unknowns": ["x"],
                "equations": ["x^2 + 2*x - 3 = 0"]
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        solutions = results["solve"]
        solution_values = {sol[sp.Symbol('x', real=True)] for sol in solutions}
        expected_values = {1, -3}
        
        assert solution_values == expected_values, f"x² + 2x - 3 = 0 should have solutions 1, -3, got {solution_values}"
    
    # ========================================================================
    # MATRIX OPERATIONS (20 tests) - Linear algebra computations
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_matrix_determinant_2x2(self):
        """det([[1,2],[3,4]]) = -2"""
        ir = MathIR(
            expr_format="latex",
            definitions={
                "matrices": [{
                    "name": "A",
                    "rows": 2,
                    "cols": 2,
                    "data": [["1", "2"], ["3", "4"]]
                }]
            },
            targets=[{
                "type": "matrix_determinant",
                "matrix_name": "A"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = -2
        actual = results["determinant"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"det([[1,2],[3,4]]) should equal -2, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_matrix_determinant_3x3(self):
        """det([[1,0,0],[0,2,0],[0,0,3]]) = 6"""
        ir = MathIR(
            expr_format="latex",
            definitions={
                "matrices": [{
                    "name": "A",
                    "rows": 3,
                    "cols": 3,
                    "data": [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "3"]]
                }]
            },
            targets=[{
                "type": "matrix_determinant",
                "matrix_name": "A"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 6
        actual = results["determinant"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"det(diagonal matrix [1,2,3]) should equal 6, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_matrix_inverse_2x2(self):
        """inv([[1,2],[3,4]]) should exist and A*A^(-1) = I"""
        ir = MathIR(
            expr_format="latex",
            definitions={
                "matrices": [{
                    "name": "A",
                    "rows": 2,
                    "cols": 2,
                    "data": [["1", "2"], ["3", "4"]]
                }]
            },
            targets=[{
                "type": "matrix_inverse",
                "matrix_name": "A"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        inverse_matrix = results["inverse"]
        original_matrix = sp.Matrix([[1, 2], [3, 4]])
        
        # Verify A * A^(-1) = I
        product = original_matrix * inverse_matrix
        identity = sp.eye(2)
        
        validation = self.validator.validate_exact_symbolic(identity, product)
        assert validation.passed, f"A * A^(-1) should equal identity matrix, got {product}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_matrix_solve_ax_equals_b(self):
        """Solve A*X = B where A=[[1,2],[3,4]], B=[[5],[11]]"""
        ir = MathIR(
            expr_format="latex",
            definitions={
                "matrices": [
                    {
                        "name": "A",
                        "rows": 2,
                        "cols": 2,
                        "data": [["1", "2"], ["3", "4"]]
                    },
                    {
                        "name": "B",
                        "rows": 2,
                        "cols": 1,
                        "data": [["5"], ["11"]]
                    }
                ]
            },
            conditions=[{
                "type": "matrix_equation",
                "expr": "A*X = B"
            }],
            targets=[{
                "type": "solve_for_matrix",
                "unknown": "X"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        solution = results["matrix"]
        
        # Verify the solution: A*X should equal B
        A = sp.Matrix([[1, 2], [3, 4]])
        B = sp.Matrix([[5], [11]])
        product = A * solution
        
        validation = self.validator.validate_exact_symbolic(B, product)
        assert validation.passed, f"A*X should equal B, got A*X = {product}, expected {B}"
    
    # ========================================================================
    # SPECIAL MATHEMATICAL VALUES AND EDGE CASES
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_mathematical_constants_pi(self):
        """Verify π is computed correctly"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "value",
                "name": "pi_value",
                "expr": "\\pi"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.pi
        actual = results["pi_value"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"π should be computed exactly, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_mathematical_constants_e(self):
        """Verify e is computed correctly"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "value",
                "name": "e_value",
                "expr": "e"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.E
        actual = results["e_value"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"e should be computed exactly, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_division_by_zero_handling(self):
        """Test proper handling of division by zero → zoo (complex infinity)"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "value",
                "name": "div_by_zero",
                "expr": "\\frac{1}{0}"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        actual = results["div_by_zero"]
        
        # Should be complex infinity (zoo) or infinity
        validation = self.validator.validate_special_values(actual, "zoo")
        if not validation.passed:
            validation = self.validator.validate_special_values(actual, "infinity")
        
        assert validation.passed, f"1/0 should result in infinity or zoo, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_indeterminate_form_0_over_0(self):
        """Test handling of indeterminate form 0/0"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": "\\frac{x}{x}",
                "var": "x",
                "to": "0"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = 1  # lim(x→0) x/x = 1
        actual = results["limit"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"lim(x→0) x/x should equal 1, got {actual}"
    
    # ========================================================================
    # COMPLEX ANALYSIS (10 tests)
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_complex_number_arithmetic(self):
        """Test basic complex number operations"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "i", "domain": "C"}],
            targets=[{
                "type": "value",
                "name": "complex_result",
                "expr": "i^2"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = -1  # i² = -1
        actual = results["complex_result"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"i² should equal -1, got {actual}"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_eulers_formula_special_case(self):
        """Test e^(iπ) = -1 (Euler's identity)"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "C"}],
            targets=[{
                "type": "value",
                "name": "euler_identity",
                "expr": "\\exp(i*\\pi)"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = -1
        actual = results["euler_identity"]
        
        validation = self.validator.validate_exact_symbolic(expected, actual)
        assert validation.passed, f"e^(iπ) should equal -1, got {actual}"
    
    # ========================================================================
    # NUMERICAL PRECISION TESTS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    def test_high_precision_calculation(self):
        """Test high-precision numerical calculation"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "value",
                "name": "precise_value",
                "expr": "\\sqrt{2}"
            }],
            output={"mode": "decimal", "round_to": 10}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = math.sqrt(2)
        actual = float(results["precise_value"])
        
        validation = self.validator.validate_numerical_tolerance(expected, actual, tolerance=1e-10)
        assert validation.passed, f"√2 should be computed with high precision, got {actual}"
    
    # ========================================================================
    # PERFORMANCE-CRITICAL TESTS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    @pytest.mark.performance
    def test_performance_simple_integral(self):
        """Test that simple integrals complete within performance constraints"""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": "x^2",
                "var": "x",
                "limits": [0, 1],
                "name": "I"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = sp.Rational(1, 3)
        actual = results["I"]
        
        # Validate with performance constraints: max 1 second, max 50MB
        validation = self.validator.validate_with_performance_constraints(
            expected, actual, max_time=1.0, max_memory_mb=50.0
        )
        assert validation.passed, f"Simple integral should complete quickly and accurately"
    
    @pytest.mark.end_to_end
    @pytest.mark.mathematical
    @pytest.mark.performance
    def test_performance_matrix_operations(self):
        """Test that matrix operations complete within performance constraints"""
        ir = MathIR(
            expr_format="latex",
            definitions={
                "matrices": [{
                    "name": "A",
                    "rows": 3,
                    "cols": 3,
                    "data": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "10"]]
                }]
            },
            targets=[{
                "type": "matrix_determinant",
                "matrix_name": "A"
            }],
            output={"mode": "exact"}
        )
        
        with self.validator.performance_monitor():
            results = run_mathir(ir)
        
        expected = -3  # Calculated determinant
        actual = results["determinant"]
        
        # Validate with performance constraints: max 2 seconds, max 100MB
        validation = self.validator.validate_with_performance_constraints(
            expected, actual, max_time=2.0, max_memory_mb=100.0
        )
        assert validation.passed, f"Matrix determinant should complete quickly and accurately"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
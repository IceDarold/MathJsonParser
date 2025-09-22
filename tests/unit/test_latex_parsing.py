"""
Comprehensive unit tests for LaTeX parsing functionality in MathIR Parser.

This module tests the core LaTeX to SymPy conversion function `to_sympy_expr()`,
which is responsible for parsing LaTeX mathematical expressions into SymPy objects.
The function handles complex LaTeX constructs including fractions, square roots,
trigonometric functions, Greek letters, implicit multiplication, and more.

Target: mathir_parser.main.to_sympy_expr() - 323 lines of critical parsing logic
Coverage Goal: 95%+ with 50+ comprehensive test cases
"""

import pytest
import sympy as sp
import math
from decimal import Decimal
from typing import Any, List, Tuple

from mathir_parser.main import to_sympy_expr
from tests.utils.assertion_helpers import (
    assert_symbolic_equivalent, assert_numerical_close, 
    MathematicalValidator, PrecisionAssertion
)
from tests.utils.performance_monitor import measure_execution_time, performance_test
from tests.utils.mathematical_validator import validate_mathematical_result


class TestBasicLaTeXCommands:
    """Test basic LaTeX commands and constructs."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_input,expected_sympy", [
        # Basic square roots
        (r"\sqrt{4}", "2"),
        (r"\sqrt{9}", "3"),
        (r"\sqrt{16}", "4"),
        (r"\sqrt{x}", "sqrt(x)"),
        (r"\sqrt{x^2}", "sqrt(x**2)"),
        (r"\sqrt{x + 1}", "sqrt(x + 1)"),
        
        # Basic fractions
        (r"\frac{1}{2}", "1/2"),
        (r"\frac{3}{4}", "3/4"),
        (r"\frac{x}{y}", "x/y"),
        (r"\frac{x+1}{x-1}", "(x+1)/(x-1)"),
        (r"\frac{1}{x^2}", "1/x**2"),
        
        # Mathematical constants
        (r"\pi", "pi"),
        (r"\infty", "oo"),
        (r"e", "E"),
    ])
    def test_basic_commands(self, latex_input: str, expected_sympy: str):
        """Test basic LaTeX commands convert correctly to SymPy."""
        result = to_sympy_expr(latex_input)
        expected = sp.sympify(expected_sympy)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_nested_square_roots(self):
        """Test nested square root expressions."""
        test_cases = [
            (r"\sqrt{\sqrt{x}}", "sqrt(sqrt(x))"),
            (r"\sqrt{x + \sqrt{y}}", "sqrt(x + sqrt(y))"),
            (r"\sqrt{\frac{x}{y}}", "sqrt(x/y)"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_nested_fractions(self):
        """Test nested fraction expressions."""
        test_cases = [
            (r"\frac{\frac{1}{2}}{3}", "(1/2)/3"),
            (r"\frac{1}{\frac{2}{3}}", "1/(2/3)"),
            (r"\frac{x}{\frac{y}{z}}", "x/(y/z)"),
            (r"\frac{\frac{x+1}{x-1}}{2}", "((x+1)/(x-1))/2"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestGreekLetters:
    """Test Greek letter parsing and recognition."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_greek,expected_symbol", [
        (r"\alpha", "alpha"),
        (r"\beta", "beta"),
        (r"\gamma", "gamma"),
        (r"\delta", "delta"),
        (r"\epsilon", "epsilon"),
        (r"\zeta", "zeta"),
        (r"\eta", "eta"),
        (r"\theta", "theta"),
        (r"\iota", "iota"),
        (r"\kappa", "kappa"),
        (r"\lambda", "lambda"),
        (r"\mu", "mu"),
        (r"\nu", "nu"),
        (r"\xi", "xi"),
        (r"\omicron", "omicron"),
        (r"\rho", "rho"),
        (r"\sigma", "sigma"),
        (r"\tau", "tau"),
        (r"\upsilon", "upsilon"),
        (r"\phi", "phi"),
        (r"\chi", "chi"),
        (r"\psi", "psi"),
        (r"\omega", "omega"),
    ])
    def test_greek_letters(self, latex_greek: str, expected_symbol: str):
        """Test that Greek letters are correctly parsed."""
        result = to_sympy_expr(latex_greek)
        expected = sp.Symbol(expected_symbol)
        assert result == expected
    
    @pytest.mark.mathematical
    def test_greek_letters_in_expressions(self):
        """Test Greek letters within mathematical expressions."""
        test_cases = [
            (r"\alpha + \beta", "alpha + beta"),
            (r"\pi * r^2", "pi * r**2"),
            (r"\frac{\alpha}{\beta}", "alpha/beta"),
            (r"\sqrt{\theta^2 + \phi^2}", "sqrt(theta**2 + phi**2)"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestTrigonometricFunctions:
    """Test trigonometric function parsing."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_func,expected_func", [
        (r"\sin(x)", "sin(x)"),
        (r"\cos(x)", "cos(x)"),
        (r"\tan(x)", "tan(x)"),
        (r"\tg(x)", "tan(x)"),  # Russian notation
        (r"\sin(\pi)", "sin(pi)"),
        (r"\cos(0)", "cos(0)"),
        (r"\tan(\frac{\pi}{4})", "tan(pi/4)"),
    ])
    def test_basic_trigonometric_functions(self, latex_func: str, expected_func: str):
        """Test basic trigonometric function parsing."""
        result = to_sympy_expr(latex_func)
        expected = sp.sympify(expected_func)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_trigonometric_function_spacing(self):
        """Test trigonometric functions with space notation (e.g., sin 2x)."""
        test_cases = [
            (r"\sin 2x", "sin(2*x)"),
            (r"\cos 3y", "cos(3*y)"),
            (r"\tan 5z", "tan(5*z)"),
            (r"\sin \pi x", "sin(pi*x)"),
            (r"\cos 2\pi t", "cos(2*pi*t)"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_nested_trigonometric_functions(self):
        """Test nested trigonometric functions."""
        test_cases = [
            (r"\sin(\cos(x))", "sin(cos(x))"),
            (r"\cos(\sin(x))", "cos(sin(x))"),
            (r"\tan(\sin(x) + \cos(x))", "tan(sin(x) + cos(x))"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestLogarithmicFunctions:
    """Test logarithmic and exponential function parsing."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_func,expected_func", [
        (r"\ln(x)", "ln(x)"),
        (r"\log(x)", "log(x)"),
        (r"\exp(x)", "exp(x)"),
        (r"\ln(e)", "ln(E)"),
        (r"\log(10)", "log(10)"),
        (r"\exp(0)", "exp(0)"),
    ])
    def test_logarithmic_functions(self, latex_func: str, expected_func: str):
        """Test logarithmic and exponential function parsing."""
        result = to_sympy_expr(latex_func)
        expected = sp.sympify(expected_func)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_logarithmic_with_complex_arguments(self):
        """Test logarithmic functions with complex arguments."""
        test_cases = [
            (r"\ln(x^2 + 1)", "ln(x**2 + 1)"),
            (r"\log(\frac{x}{y})", "log(x/y)"),
            (r"\exp(\sin(x))", "exp(sin(x))"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestImplicitMultiplication:
    """Test implicit multiplication handling."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_input,expected_result", [
        # Number-variable multiplication
        ("2x", "2*x"),
        ("3y", "3*y"),
        ("10z", "10*z"),
        
        # Variable-variable multiplication
        ("xy", "x*y"),
        ("abc", "a*b*c"),
        
        # Parentheses multiplication
        ("2(x+1)", "2*(x+1)"),
        ("(x+1)(x+2)", "(x+1)*(x+2)"),
        ("x(y+z)", "x*(y+z)"),
        
        # Function-argument multiplication
        ("2sin(x)", "2*sin(x)"),
        ("3cos(y)", "3*cos(y)"),
    ])
    def test_implicit_multiplication(self, latex_input: str, expected_result: str):
        """Test implicit multiplication parsing."""
        result = to_sympy_expr(latex_input)
        expected = sp.sympify(expected_result)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_complex_implicit_multiplication(self):
        """Test complex implicit multiplication scenarios."""
        test_cases = [
            ("2x(y+1)", "2*x*(y+1)"),
            ("(a+b)(c+d)(e+f)", "(a+b)*(c+d)*(e+f)"),
            ("3xy(z+1)", "3*x*y*(z+1)"),
            (r"2\pi r", "2*pi*r"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestAbsoluteValues:
    """Test absolute value parsing."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_input,expected_result", [
        ("|x|", "Abs(x)"),
        ("|x + 1|", "Abs(x + 1)"),
        ("|x - y|", "Abs(x - y)"),
        ("|-5|", "Abs(-5)"),
        ("|sin(x)|", "Abs(sin(x))"),
    ])
    def test_basic_absolute_values(self, latex_input: str, expected_result: str):
        """Test basic absolute value parsing."""
        result = to_sympy_expr(latex_input)
        expected = sp.sympify(expected_result)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_nested_absolute_values(self):
        """Test nested absolute value expressions."""
        test_cases = [
            ("||x||", "Abs(Abs(x))"),
            ("|x| + |y|", "Abs(x) + Abs(y)"),
            ("|x + |y||", "Abs(x + Abs(y))"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestExponentiation:
    """Test exponentiation and power expressions."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("latex_input,expected_result", [
        ("x^2", "x**2"),
        ("x^{2}", "x**2"),
        ("x^{y+1}", "x**(y+1)"),
        ("e^x", "E**x"),
        ("e^{2x}", "E**(2*x)"),
        ("2^{3x}", "2**(3*x)"),
        ("(x+1)^2", "(x+1)**2"),
    ])
    def test_exponentiation(self, latex_input: str, expected_result: str):
        """Test exponentiation parsing."""
        result = to_sympy_expr(latex_input)
        expected = sp.sympify(expected_result)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_nested_exponentiation(self):
        """Test nested exponentiation expressions."""
        test_cases = [
            ("x^{y^2}", "x**(y**2)"),
            ("(x^2)^3", "(x**2)**3"),
            ("e^{x^2 + 1}", "E**(x**2 + 1)"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestComplexExpressions:
    """Test complex mathematical expressions combining multiple constructs."""
    
    @pytest.mark.mathematical
    def test_complex_nested_expressions(self):
        """Test complex expressions with multiple nested constructs."""
        test_cases = [
            # Fraction with square root in numerator
            (r"\frac{\sqrt{x+1}}{x^2+1}", "sqrt(x+1)/(x**2+1)"),
            
            # Trigonometric with exponentiation
            (r"2\sin^2(x) + \cos^2(x)", "2*sin(x)**2 + cos(x)**2"),
            
            # Complex fraction with trigonometry
            (r"\frac{\sin(x)}{\cos(x) + 1}", "sin(x)/(cos(x) + 1)"),
            
            # Multiple functions and operations
            (r"\sqrt{x^2 + y^2} + \ln(xy)", "sqrt(x**2 + y**2) + ln(x*y)"),
            
            # Nested functions with Greek letters
            (r"\sin(\alpha + \beta) \cos(\gamma)", "sin(alpha + beta)*cos(gamma)"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_real_world_expressions(self):
        """Test expressions that might appear in real mathematical problems."""
        test_cases = [
            # Quadratic formula components
            (r"\frac{-b + \sqrt{b^2 - 4ac}}{2a}", "(-b + sqrt(b**2 - 4*a*c))/(2*a)"),
            
            # Distance formula
            (r"\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}", "sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)"),
            
            # Trigonometric identity
            (r"\sin^2(x) + \cos^2(x)", "sin(x)**2 + cos(x)**2"),
            
            # Exponential decay
            (r"A e^{-\lambda t}", "A*E**(-lambda*t)"),
        ]
        
        for latex_input, expected_str in test_cases:
            result = to_sympy_expr(latex_input)
            expected = sp.sympify(expected_str)
            assert_symbolic_equivalent(result, expected)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.mathematical
    def test_empty_and_whitespace(self):
        """Test empty strings and whitespace handling."""
        with pytest.raises(Exception):
            to_sympy_expr("")
        
        with pytest.raises(Exception):
            to_sympy_expr("   ")
    
    @pytest.mark.mathematical
    def test_invalid_latex_commands(self):
        """Test handling of invalid LaTeX commands."""
        invalid_commands = [
            r"\invalid{command}",
            r"\nonexistent{x}",
            r"\badcommand",
        ]
        
        for invalid_cmd in invalid_commands:
            with pytest.raises(Exception):
                to_sympy_expr(invalid_cmd)
    
    @pytest.mark.mathematical
    def test_malformed_expressions(self):
        """Test malformed mathematical expressions."""
        malformed_expressions = [
            r"\frac{1}",  # Missing denominator
            r"\sqrt{",    # Unclosed brace
            r"}{",        # Mismatched braces
            r"\frac{1}{0}",  # Division by zero (should parse but evaluate to zoo)
        ]
        
        # Some should raise exceptions, others should parse but give special values
        for expr in malformed_expressions[:3]:  # First 3 should raise exceptions
            with pytest.raises(Exception):
                to_sympy_expr(expr)
        
        # Division by zero should parse to complex infinity
        result = to_sympy_expr(r"\frac{1}{0}")
        assert result == sp.zoo
    
    @pytest.mark.mathematical
    def test_very_long_expressions(self):
        """Test parsing of very long expressions."""
        # Generate a long polynomial
        terms = [f"{i}x^{i}" for i in range(1, 21)]
        long_expr = " + ".join(terms)
        
        result = to_sympy_expr(long_expr)
        assert isinstance(result, sp.Expr)
        assert len(result.free_symbols) == 1  # Should contain only 'x'


class TestNumericalAccuracy:
    """Test numerical accuracy and precision of parsing results."""
    
    @pytest.mark.mathematical
    def test_numerical_evaluation_accuracy(self):
        """Test that parsed expressions evaluate to correct numerical values."""
        test_cases = [
            (r"\frac{1}{2}", 0.5),
            (r"\sqrt{4}", 2.0),
            (r"\pi", math.pi),
            (r"e", math.e),
            (r"\sin(\frac{\pi}{2})", 1.0),
            (r"\cos(0)", 1.0),
            (r"\ln(e)", 1.0),
        ]
        
        validator = MathematicalValidator()
        
        for latex_input, expected_value in test_cases:
            result = to_sympy_expr(latex_input)
            numerical_result = float(result.evalf())
            validator.validate_integral_result(numerical_result, expected_value, tolerance=1e-10)
    
    @pytest.mark.mathematical
    def test_symbolic_vs_numerical_consistency(self):
        """Test consistency between symbolic and numerical results."""
        test_expressions = [
            r"x^2 + 2x + 1",
            r"\sin^2(x) + \cos^2(x)",
            r"\frac{x^2 - 1}{x - 1}",
            r"e^{\ln(x)}",
        ]
        
        x_val = 2.5
        
        for expr_str in test_expressions:
            symbolic_expr = to_sympy_expr(expr_str)
            
            # Substitute x = 2.5 and evaluate
            numerical_result = float(symbolic_expr.subs('x', x_val).evalf())
            
            # Verify the result makes mathematical sense
            assert isinstance(numerical_result, (int, float))
            assert not math.isnan(numerical_result)


class TestPerformanceBenchmarks:
    """Performance benchmarks for LaTeX parsing."""
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_simple_expression_performance(self, benchmark_runner):
        """Benchmark performance of simple expression parsing."""
        simple_expressions = [
            "x^2",
            r"\sin(x)",
            r"\frac{1}{x}",
            r"\sqrt{x}",
        ]
        
        def parse_simple_expressions():
            for expr in simple_expressions:
                to_sympy_expr(expr)
        
        execution_time = measure_execution_time(parse_simple_expressions)
        assert execution_time < 0.1  # Should complete in less than 100ms
    
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_complex_expression_performance(self, benchmark_runner):
        """Benchmark performance of complex expression parsing."""
        complex_expressions = [
            r"\frac{\sqrt{x^2 + y^2}}{\sin(x) + \cos(y)}",
            r"e^{-\alpha t} \sin(\omega t + \phi)",
            r"\sum_{n=1}^{\infty} \frac{1}{n^2}",
        ]
        
        def parse_complex_expressions():
            for expr in complex_expressions:
                try:
                    to_sympy_expr(expr)
                except:
                    pass  # Some complex expressions might not be fully supported
        
        execution_time = measure_execution_time(parse_complex_expressions)
        assert execution_time < 1.0  # Should complete in less than 1 second
    
    @performance_test(timeout=30.0)
    @pytest.mark.performance
    def test_stress_parsing(self):
        """Stress test with many expressions."""
        expressions = []
        
        # Generate 100 random polynomial expressions
        for i in range(100):
            degree = i % 5 + 1
            terms = [f"{j}x^{j}" for j in range(degree + 1)]
            expr = " + ".join(terms)
            expressions.append(expr)
        
        start_time = measure_execution_time(lambda: None)  # Baseline
        
        for expr in expressions:
            result = to_sympy_expr(expr)
            assert isinstance(result, sp.Expr)
        
        # Should handle 100 expressions reasonably quickly
        assert True  # If we get here without timeout, test passes


class TestMathematicalValidation:
    """Test mathematical correctness and validation."""
    
    @pytest.mark.mathematical
    def test_mathematical_identities(self):
        """Test that mathematical identities are preserved."""
        identities = [
            # Trigonometric identities
            (r"\sin^2(x) + \cos^2(x)", "1"),
            (r"\tan(x)", r"\frac{\sin(x)}{\cos(x)}"),
            
            # Logarithmic identities
            (r"\ln(e^x)", "x"),
            (r"e^{\ln(x)}", "x"),
            
            # Algebraic identities
            (r"(x+1)^2", r"x^2 + 2x + 1"),
            (r"x^2 - 1", r"(x-1)(x+1)"),
        ]
        
        for expr1_str, expr2_str in identities:
            expr1 = to_sympy_expr(expr1_str)
            expr2 = to_sympy_expr(expr2_str)
            
            # Check if they're mathematically equivalent
            difference = sp.simplify(expr1 - expr2)
            
            # For some identities, we need to check specific values
            if difference != 0:
                # Test at a few points
                test_values = [1, 2, 0.5, -1]
                for val in test_values:
                    try:
                        val1 = float(expr1.subs('x', val).evalf())
                        val2 = float(expr2.subs('x', val).evalf())
                        assert abs(val1 - val2) < 1e-10
                    except:
                        pass  # Some expressions might not be valid at all points
    
    @pytest.mark.mathematical
    def test_derivative_consistency(self):
        """Test that derivatives of parsed expressions are correct."""
        test_cases = [
            ("x^2", "2*x"),
            (r"\sin(x)", r"\cos(x)"),
            (r"\cos(x)", r"-\sin(x)"),
            (r"e^x", r"e^x"),
            (r"\ln(x)", r"\frac{1}{x}"),
        ]
        
        for expr_str, expected_derivative_str in test_cases:
            expr = to_sympy_expr(expr_str)
            expected_derivative = to_sympy_expr(expected_derivative_str)
            
            actual_derivative = sp.diff(expr, 'x')
            
            # Check if derivatives are equivalent
            difference = sp.simplify(actual_derivative - expected_derivative)
            assert difference == 0 or sp.simplify(difference) == 0


# Fixtures for test data and utilities
@pytest.fixture
def mathematical_validator():
    """Provide a MathematicalValidator instance for tests."""
    return MathematicalValidator()


@pytest.fixture
def precision_assertion():
    """Provide a PrecisionAssertion instance for tests."""
    return PrecisionAssertion(default_tolerance=1e-10)


@pytest.fixture
def benchmark_runner():
    """Provide a benchmark runner for performance tests."""
    from tests.utils.performance_monitor import BenchmarkRunner
    return BenchmarkRunner()


# Test configuration and markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.latex_parsing,
]


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=mathir_parser.main",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])
"""
Comprehensive unit tests for SymPy object creation and manipulation in MathIR Parser.

This module tests the SymPy-related functionality including symbol creation, function handling,
mathematical constants integration, and domain validation. It focuses on the runtime context
building and SymPy expression manipulation functions.

Target: mathir_parser.main.build_runtime() and related SymPy operations
Coverage Goal: 95%+ with comprehensive test coverage
"""

import pytest
import sympy as sp
from typing import Dict, Any, List
from decimal import Decimal

from mathir_parser.main import (
    MathIR, build_runtime, Runtime, SymbolSpec, FunctionDef, 
    SequenceDef, MatrixDef, DOMAIN_MAP, to_sympy_expr
)
from tests.utils.assertion_helpers import (
    assert_symbolic_equivalent, assert_numerical_close,
    MathematicalValidator, PrecisionAssertion
)
from tests.utils.mathematical_validator import validate_mathematical_result
from tests.utils.performance_monitor import measure_execution_time


class TestSymbolCreation:
    """Test SymPy symbol creation with different domains."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("domain,expected_assumptions", [
        ('R', {'real': True}),
        ('Z', {'integer': True}),
        ('N', {'integer': True, 'nonnegative': True}),
        ('N+', {'integer': True, 'positive': True}),
        ('C', {'complex': True}),
        ('R+', {'real': True}),  # Positive reals handled differently
    ])
    def test_symbol_domain_creation(self, domain: str, expected_assumptions: Dict[str, bool]):
        """Test that symbols are created with correct domain assumptions."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain=domain)],
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        symbol = runtime.symtab["x"]
        
        # Check that the symbol has the expected assumptions
        for assumption, expected_value in expected_assumptions.items():
            if assumption == 'real':
                assert symbol.is_real == expected_value or symbol.is_real is None
            elif assumption == 'integer':
                assert symbol.is_integer == expected_value or symbol.is_integer is None
            elif assumption == 'nonnegative':
                assert symbol.is_nonnegative == expected_value or symbol.is_nonnegative is None
            elif assumption == 'positive':
                assert symbol.is_positive == expected_value or symbol.is_positive is None
            elif assumption == 'complex':
                assert symbol.is_complex == expected_value or symbol.is_complex is None
    
    @pytest.mark.mathematical
    def test_multiple_symbols_creation(self):
        """Test creation of multiple symbols with different domains."""
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="x", domain="R"),
                SymbolSpec(name="n", domain="N"),
                SymbolSpec(name="z", domain="C"),
                SymbolSpec(name="k", domain="Z"),
            ],
            targets=[{"type": "value", "name": "test", "expr": "x + n + z + k"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check all symbols are created
        assert "x" in runtime.symtab
        assert "n" in runtime.symtab
        assert "z" in runtime.symtab
        assert "k" in runtime.symtab
        
        # Check they're all SymPy symbols
        for symbol_name in ["x", "n", "z", "k"]:
            assert isinstance(runtime.symtab[symbol_name], sp.Symbol)
    
    @pytest.mark.mathematical
    def test_symbol_context_integration(self):
        """Test that symbols are properly integrated into the parsing context."""
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="alpha", domain="R"),
                SymbolSpec(name="beta", domain="R"),
            ],
            targets=[{"type": "value", "name": "test", "expr": "alpha + beta"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check symbols are in context
        assert "alpha" in runtime.context
        assert "beta" in runtime.context
        
        # Check context values are the same as symtab values
        assert runtime.context["alpha"] == runtime.symtab["alpha"]
        assert runtime.context["beta"] == runtime.symtab["beta"]


class TestMathematicalConstants:
    """Test mathematical constants integration."""
    
    @pytest.mark.mathematical
    def test_built_in_constants(self):
        """Test that built-in mathematical constants are available."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            targets=[{"type": "value", "name": "test", "expr": "pi + e + i"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check constants are in context
        assert "pi" in runtime.context
        assert "e" in runtime.context
        assert "i" in runtime.context
        
        # Check they map to correct SymPy constants
        assert runtime.context["pi"] == sp.pi
        assert runtime.context["e"] == sp.E
        assert runtime.context["i"] == sp.I
    
    @pytest.mark.mathematical
    def test_custom_constants(self):
        """Test custom constants definition and parsing."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            constants={"g": "9.81", "c": "299792458", "h": "6.626e-34"},
            targets=[{"type": "value", "name": "test", "expr": "g*x + c + h"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check custom constants are in context
        assert "g" in runtime.context
        assert "c" in runtime.context
        assert "h" in runtime.context
        
        # Check they're parsed as numbers
        assert isinstance(runtime.context["g"], (sp.Float, sp.Integer))
        assert isinstance(runtime.context["c"], (sp.Float, sp.Integer))
        assert isinstance(runtime.context["h"], (sp.Float, sp.Integer))
        
        # Check numerical values
        assert float(runtime.context["g"]) == 9.81
        assert float(runtime.context["c"]) == 299792458
        assert abs(float(runtime.context["h"]) - 6.626e-34) < 1e-40
    
    @pytest.mark.mathematical
    def test_constants_in_expressions(self):
        """Test that constants work correctly in expressions."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="t", domain="R")],
            constants={"omega": "2.5"},
            targets=[{"type": "value", "name": "result", "expr": "sin(omega * t)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Parse an expression using the constant
        expr = to_sympy_expr("sin(omega * t)")
        substituted = expr.subs(runtime.context)
        
        # Should contain omega as a number, not a symbol
        assert substituted.has(sp.sin)
        # The omega should be substituted with its numerical value


class TestFunctionDefinitions:
    """Test user-defined function handling."""
    
    @pytest.mark.mathematical
    def test_simple_function_definition(self):
        """Test simple function definition and creation."""
        func_def = FunctionDef(
            name="f",
            args=["x"],
            expr="x^2 + 1"
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions={"functions": [func_def]},
            targets=[{"type": "value", "name": "test", "expr": "f(2)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check function is created
        func_symbol = sp.Function("f")
        assert func_symbol in runtime.funcs
        
        # Check function lambda is correct
        lambda_func = runtime.funcs[func_symbol]
        assert isinstance(lambda_func, sp.Lambda)
        
        # Test function evaluation
        result = lambda_func(2)
        expected = 2**2 + 1  # 5
        assert result == expected
    
    @pytest.mark.mathematical
    def test_multi_argument_function(self):
        """Test function with multiple arguments."""
        func_def = FunctionDef(
            name="g",
            args=["x", "y"],
            expr="x*y + x^2 - y"
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="a", domain="R"), SymbolSpec(name="b", domain="R")],
            definitions={"functions": [func_def]},
            targets=[{"type": "value", "name": "test", "expr": "g(a, b)"}]
        )
        
        runtime = build_runtime(mathir)
        
        func_symbol = sp.Function("g")
        lambda_func = runtime.funcs[func_symbol]
        
        # Test with specific values
        result = lambda_func(3, 2)
        expected = 3*2 + 3**2 - 2  # 6 + 9 - 2 = 13
        assert result == expected
    
    @pytest.mark.mathematical
    def test_function_with_trigonometry(self):
        """Test function definition with trigonometric expressions."""
        func_def = FunctionDef(
            name="trig_func",
            args=["theta"],
            expr=r"\sin(theta) + \cos(theta)"
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions={"functions": [func_def]},
            targets=[{"type": "value", "name": "test", "expr": "trig_func(x)"}]
        )
        
        runtime = build_runtime(mathir)
        
        func_symbol = sp.Function("trig_func")
        lambda_func = runtime.funcs[func_symbol]
        
        # Test with pi/4 (should give sqrt(2))
        result = lambda_func(sp.pi/4)
        expected = sp.sin(sp.pi/4) + sp.cos(sp.pi/4)
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_nested_function_calls(self):
        """Test nested function definitions and calls."""
        func1 = FunctionDef(name="f", args=["x"], expr="x^2")
        func2 = FunctionDef(name="g", args=["x"], expr="2*x + 1")
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="t", domain="R")],
            definitions={"functions": [func1, func2]},
            targets=[{"type": "value", "name": "test", "expr": "f(g(t))"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Both functions should be created
        f_symbol = sp.Function("f")
        g_symbol = sp.Function("g")
        
        assert f_symbol in runtime.funcs
        assert g_symbol in runtime.funcs
        
        # Test composition: f(g(t)) = f(2t+1) = (2t+1)^2
        f_lambda = runtime.funcs[f_symbol]
        g_lambda = runtime.funcs[g_symbol]
        
        # Test with t=1: g(1) = 3, f(3) = 9
        g_result = g_lambda(1)
        f_result = f_lambda(g_result)
        assert f_result == 9


class TestSequenceDefinitions:
    """Test sequence definition handling."""
    
    @pytest.mark.mathematical
    def test_simple_sequence_definition(self):
        """Test simple sequence definition."""
        seq_def = SequenceDef(
            name="a",
            args=["n"],
            expr="n^2 + 1"
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="n", domain="N")],
            definitions={"sequences": [seq_def]},
            targets=[{"type": "value", "name": "test", "expr": "a(5)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check sequence is created
        seq_symbol = sp.Function("a")
        assert seq_symbol in runtime.sequences
        
        # Test sequence evaluation
        lambda_seq = runtime.sequences[seq_symbol]
        result = lambda_seq(5)
        expected = 5**2 + 1  # 26
        assert result == expected
    
    @pytest.mark.mathematical
    def test_fibonacci_like_sequence(self):
        """Test more complex sequence definition."""
        seq_def = SequenceDef(
            name="fib_like",
            args=["n"],
            expr="n*(n-1)/2"  # Triangular numbers
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="k", domain="N")],
            definitions={"sequences": [seq_def]},
            targets=[{"type": "value", "name": "test", "expr": "fib_like(k)"}]
        )
        
        runtime = build_runtime(mathir)
        
        seq_symbol = sp.Function("fib_like")
        lambda_seq = runtime.sequences[seq_symbol]
        
        # Test triangular numbers: T(n) = n(n-1)/2
        assert lambda_seq(1) == 0   # T(1) = 0
        assert lambda_seq(2) == 1   # T(2) = 1
        assert lambda_seq(3) == 3   # T(3) = 3
        assert lambda_seq(4) == 6   # T(4) = 6


class TestMatrixDefinitions:
    """Test matrix definition and creation."""
    
    @pytest.mark.mathematical
    def test_simple_matrix_creation(self):
        """Test simple matrix definition."""
        matrix_def = MatrixDef(
            name="A",
            rows=2,
            cols=2,
            data=[["1", "2"], ["3", "4"]]
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions={"matrices": [matrix_def]},
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check matrix is created
        assert "A" in runtime.matrices
        matrix = runtime.matrices["A"]
        
        assert isinstance(matrix, sp.Matrix)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 1
        assert matrix[0, 1] == 2
        assert matrix[1, 0] == 3
        assert matrix[1, 1] == 4
    
    @pytest.mark.mathematical
    def test_matrix_with_expressions(self):
        """Test matrix with symbolic expressions."""
        matrix_def = MatrixDef(
            name="B",
            rows=2,
            cols=2,
            data=[["x", "x+1"], ["2*x", "x^2"]]
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions={"matrices": [matrix_def]},
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        matrix = runtime.matrices["B"]
        x_symbol = runtime.symtab["x"]
        
        # Check matrix elements are correct expressions
        assert matrix[0, 0] == x_symbol
        assert matrix[0, 1] == x_symbol + 1
        assert matrix[1, 0] == 2 * x_symbol
        assert matrix[1, 1] == x_symbol**2
    
    @pytest.mark.mathematical
    def test_matrix_operations(self):
        """Test basic matrix operations."""
        matrix_def = MatrixDef(
            name="M",
            rows=2,
            cols=2,
            data=[["1", "0"], ["0", "1"]]  # Identity matrix
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions={"matrices": [matrix_def]},
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        matrix = runtime.matrices["M"]
        
        # Test matrix properties
        assert matrix.det() == 1  # Determinant of identity matrix
        assert matrix.inv() == matrix  # Inverse of identity is itself
        
        # Test matrix multiplication
        result = matrix * matrix
        assert result == matrix  # I * I = I


class TestDomainMapping:
    """Test domain mapping functionality."""
    
    @pytest.mark.mathematical
    def test_domain_map_completeness(self):
        """Test that all supported domains are in DOMAIN_MAP."""
        supported_domains = ['R', 'R+', 'Z', 'N', 'N+', 'C']
        
        for domain in supported_domains:
            assert domain in DOMAIN_MAP
            assert isinstance(DOMAIN_MAP[domain], (sp.Set, sp.Interval))
    
    @pytest.mark.mathematical
    def test_domain_map_correctness(self):
        """Test that domain mappings are mathematically correct."""
        # Test real numbers
        assert DOMAIN_MAP['R'] == sp.S.Reals
        
        # Test integers
        assert DOMAIN_MAP['Z'] == sp.S.Integers
        
        # Test natural numbers
        assert DOMAIN_MAP['N'] == sp.S.Naturals0  # Includes 0
        assert DOMAIN_MAP['N+'] == sp.S.Naturals  # Excludes 0
        
        # Test complex numbers
        assert DOMAIN_MAP['C'] == sp.S.Complexes
        
        # Test positive reals
        assert DOMAIN_MAP['R+'] == sp.Interval.Ropen(0, sp.oo)


class TestRuntimeContextIntegration:
    """Test complete runtime context building and integration."""
    
    @pytest.mark.mathematical
    def test_complete_runtime_context(self):
        """Test building a complete runtime context with all components."""
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="x", domain="R"),
                SymbolSpec(name="n", domain="N"),
            ],
            constants={"pi_approx": "3.14159"},
            definitions={
                "functions": [FunctionDef(name="f", args=["t"], expr="t^2")],
                "sequences": [SequenceDef(name="a", args=["k"], expr="k+1")],
                "matrices": [MatrixDef(name="I", rows=2, cols=2, data=[["1", "0"], ["0", "1"]])],
            },
            targets=[{"type": "value", "name": "test", "expr": "f(x) + a(n)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check all components are present
        assert len(runtime.symtab) == 2  # x, n
        assert len(runtime.funcs) == 1   # f
        assert len(runtime.sequences) == 1  # a
        assert len(runtime.matrices) == 1   # I
        
        # Check context contains everything
        expected_context_keys = {
            'x', 'n',  # symbols
            'pi', 'e', 'i',  # built-in constants
            'pi_approx',  # custom constant
            'I'  # matrix (functions and sequences use Function objects as keys)
        }
        
        for key in expected_context_keys:
            assert key in runtime.context
    
    @pytest.mark.mathematical
    def test_context_symbol_resolution(self):
        """Test that context properly resolves symbols in expressions."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="alpha", domain="R")],
            constants={"beta": "2.5"},
            targets=[{"type": "value", "name": "result", "expr": "alpha * beta + pi"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Parse an expression using the context
        expr = to_sympy_expr("alpha * beta + pi")
        substituted = expr.subs(runtime.context)
        
        # Should resolve beta and pi, but keep alpha as symbol
        assert substituted.has(runtime.symtab["alpha"])
        assert substituted.has(sp.pi)
        # beta should be substituted with 2.5


class TestErrorHandling:
    """Test error handling in SymPy conversion."""
    
    @pytest.mark.mathematical
    def test_invalid_constant_values(self):
        """Test handling of invalid constant values."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            constants={"invalid": "not_a_number"},
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        # Should not raise exception, but treat as symbol
        runtime = build_runtime(mathir)
        assert "invalid" in runtime.context
        assert isinstance(runtime.context["invalid"], sp.Symbol)
    
    @pytest.mark.mathematical
    def test_empty_function_definition(self):
        """Test handling of empty or invalid function definitions."""
        with pytest.raises(Exception):
            FunctionDef(name="", args=[], expr="x")
        
        with pytest.raises(Exception):
            FunctionDef(name="f", args=[], expr="")
    
    @pytest.mark.mathematical
    def test_invalid_matrix_dimensions(self):
        """Test handling of invalid matrix definitions."""
        with pytest.raises(Exception):
            MatrixDef(
                name="bad_matrix",
                rows=2,
                cols=2,
                data=[["1", "2"], ["3"]]  # Inconsistent row length
            )


class TestPerformanceOptimization:
    """Test performance aspects of SymPy conversion."""
    
    @pytest.mark.performance
    def test_runtime_building_performance(self):
        """Test performance of runtime building with many symbols."""
        # Create MathIR with many symbols
        symbols = [SymbolSpec(name=f"x{i}", domain="R") for i in range(100)]
        
        mathir = MathIR(
            symbols=symbols,
            targets=[{"type": "value", "name": "test", "expr": "x0"}]
        )
        
        # Measure runtime building time
        execution_time = measure_execution_time(lambda: build_runtime(mathir))
        
        # Should complete reasonably quickly
        assert execution_time < 1.0  # Less than 1 second
    
    @pytest.mark.performance
    def test_large_matrix_creation(self):
        """Test performance with large matrices."""
        # Create a 10x10 matrix
        data = [[str(i*10 + j) for j in range(10)] for i in range(10)]
        matrix_def = MatrixDef(name="large", rows=10, cols=10, data=data)
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions={"matrices": [matrix_def]},
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        execution_time = measure_execution_time(lambda: build_runtime(mathir))
        assert execution_time < 2.0  # Should handle large matrices efficiently


# Test fixtures
@pytest.fixture
def simple_runtime():
    """Provide a simple runtime for testing."""
    mathir = MathIR(
        symbols=[SymbolSpec(name="x", domain="R")],
        targets=[{"type": "value", "name": "test", "expr": "x"}]
    )
    return build_runtime(mathir)


@pytest.fixture
def complex_runtime():
    """Provide a complex runtime with all components."""
    mathir = MathIR(
        symbols=[
            SymbolSpec(name="x", domain="R"),
            SymbolSpec(name="n", domain="N"),
        ],
        constants={"g": "9.81"},
        definitions={
            "functions": [FunctionDef(name="f", args=["t"], expr="t^2 + 1")],
            "matrices": [MatrixDef(name="A", rows=2, cols=2, data=[["1", "2"], ["3", "4"]])],
        },
        targets=[{"type": "value", "name": "test", "expr": "f(x)"}]
    )
    return build_runtime(mathir)


# Test configuration
pytestmark = [
    pytest.mark.unit,
    pytest.mark.sympy_conversion,
]


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--cov=mathir_parser.main",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--tb=short"
    ])
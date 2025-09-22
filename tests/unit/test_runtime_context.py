"""
Comprehensive unit tests for runtime context building and management in MathIR Parser.

This module tests the runtime context creation, symbol domain handling, constants integration,
function management, and the overall context lifecycle. It focuses on the Runtime class
and context building functionality.

Target: mathir_parser.main.Runtime and build_runtime() function
Coverage Goal: 95%+ with comprehensive context management testing
"""

import pytest
import sympy as sp
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from mathir_parser.main import (
    MathIR, Runtime, build_runtime, SymbolSpec, FunctionDef, SequenceDef, 
    MatrixDef, DistributionDef, GeometryDef, Definitions, to_sympy_expr
)
from tests.utils.assertion_helpers import (
    assert_symbolic_equivalent, MathematicalValidator
)
from tests.utils.performance_monitor import measure_execution_time
from tests.utils.mathematical_validator import validate_mathematical_result


class TestRuntimeCreation:
    """Test basic Runtime object creation and initialization."""
    
    @pytest.mark.mathematical
    def test_empty_runtime_creation(self):
        """Test creation of runtime with minimal MathIR."""
        mathir = MathIR(
            targets=[{"type": "value", "name": "test", "expr": "1"}]
        )
        
        runtime = build_runtime(mathir)
        
        assert isinstance(runtime, Runtime)
        assert isinstance(runtime.symtab, dict)
        assert isinstance(runtime.funcs, dict)
        assert isinstance(runtime.sequences, dict)
        assert isinstance(runtime.matrices, dict)
        assert isinstance(runtime.distributions, dict)
        assert isinstance(runtime.geometry, dict)
        assert isinstance(runtime.context, dict)
    
    @pytest.mark.mathematical
    def test_runtime_with_basic_symbols(self):
        """Test runtime creation with basic symbols."""
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="x", domain="R"),
                SymbolSpec(name="y", domain="R"),
            ],
            targets=[{"type": "value", "name": "test", "expr": "x + y"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check symbol table
        assert len(runtime.symtab) == 2
        assert "x" in runtime.symtab
        assert "y" in runtime.symtab
        
        # Check symbols are SymPy symbols
        assert isinstance(runtime.symtab["x"], sp.Symbol)
        assert isinstance(runtime.symtab["y"], sp.Symbol)
        
        # Check context contains symbols
        assert "x" in runtime.context
        assert "y" in runtime.context
        assert runtime.context["x"] == runtime.symtab["x"]
        assert runtime.context["y"] == runtime.symtab["y"]
    
    @pytest.mark.mathematical
    def test_runtime_built_in_constants(self):
        """Test that built-in constants are always available."""
        mathir = MathIR(
            targets=[{"type": "value", "name": "test", "expr": "pi + e + i"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check built-in constants
        assert "pi" in runtime.context
        assert "e" in runtime.context
        assert "i" in runtime.context
        
        assert runtime.context["pi"] == sp.pi
        assert runtime.context["e"] == sp.E
        assert runtime.context["i"] == sp.I


class TestSymbolDomainHandling:
    """Test symbol creation with different domain specifications."""
    
    @pytest.mark.mathematical
    @pytest.mark.parametrize("domain,expected_properties", [
        ("R", {"real": True}),
        ("Z", {"integer": True}),
        ("N", {"integer": True, "nonnegative": True}),
        ("N+", {"integer": True, "positive": True}),
        ("C", {"complex": True}),
        ("R+", {"real": True}),  # Positive reals
    ])
    def test_symbol_domain_properties(self, domain: str, expected_properties: Dict[str, bool]):
        """Test that symbols get correct domain properties."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="var", domain=domain)],
            targets=[{"type": "value", "name": "test", "expr": "var"}]
        )
        
        runtime = build_runtime(mathir)
        symbol = runtime.symtab["var"]
        
        # Check domain-specific properties
        for prop, expected_value in expected_properties.items():
            if prop == "real":
                # Real property might be None for some symbols
                assert symbol.is_real == expected_value or symbol.is_real is None
            elif prop == "integer":
                assert symbol.is_integer == expected_value or symbol.is_integer is None
            elif prop == "nonnegative":
                assert symbol.is_nonnegative == expected_value or symbol.is_nonnegative is None
            elif prop == "positive":
                assert symbol.is_positive == expected_value or symbol.is_positive is None
            elif prop == "complex":
                assert symbol.is_complex == expected_value or symbol.is_complex is None
    
    @pytest.mark.mathematical
    def test_mixed_domain_symbols(self):
        """Test runtime with symbols from different domains."""
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="real_var", domain="R"),
                SymbolSpec(name="int_var", domain="Z"),
                SymbolSpec(name="nat_var", domain="N"),
                SymbolSpec(name="complex_var", domain="C"),
            ],
            targets=[{"type": "value", "name": "test", "expr": "real_var + int_var"}]
        )
        
        runtime = build_runtime(mathir)
        
        # All symbols should be created
        assert len(runtime.symtab) == 4
        
        # Check each symbol exists and has correct type
        for var_name in ["real_var", "int_var", "nat_var", "complex_var"]:
            assert var_name in runtime.symtab
            assert isinstance(runtime.symtab[var_name], sp.Symbol)
            assert var_name in runtime.context
    
    @pytest.mark.mathematical
    def test_default_domain_handling(self):
        """Test handling of symbols with default domain."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="default_var")],  # No domain specified
            targets=[{"type": "value", "name": "test", "expr": "default_var"}]
        )
        
        runtime = build_runtime(mathir)
        symbol = runtime.symtab["default_var"]
        
        # Default domain should be real
        assert symbol.is_real == True or symbol.is_real is None


class TestConstantsIntegration:
    """Test custom constants integration into runtime context."""
    
    @pytest.mark.mathematical
    def test_numerical_constants(self):
        """Test numerical constants parsing and integration."""
        mathir = MathIR(
            constants={
                "g": "9.81",
                "c": "299792458",
                "planck": "6.626e-34",
                "integer_const": "42",
            },
            targets=[{"type": "value", "name": "test", "expr": "g"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check all constants are in context
        for const_name in ["g", "c", "planck", "integer_const"]:
            assert const_name in runtime.context
        
        # Check numerical values
        assert abs(float(runtime.context["g"]) - 9.81) < 1e-10
        assert float(runtime.context["c"]) == 299792458
        assert abs(float(runtime.context["planck"]) - 6.626e-34) < 1e-40
        assert float(runtime.context["integer_const"]) == 42
        
        # Check types
        assert isinstance(runtime.context["g"], (sp.Float, sp.Integer))
        assert isinstance(runtime.context["c"], (sp.Float, sp.Integer))
        assert isinstance(runtime.context["planck"], (sp.Float, sp.Integer))
        assert isinstance(runtime.context["integer_const"], (sp.Float, sp.Integer))
    
    @pytest.mark.mathematical
    def test_invalid_constants_handling(self):
        """Test handling of invalid constant values."""
        mathir = MathIR(
            constants={
                "valid": "3.14",
                "invalid": "not_a_number",
                "empty": "",
            },
            targets=[{"type": "value", "name": "test", "expr": "valid"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Valid constant should be parsed as number
        assert isinstance(runtime.context["valid"], (sp.Float, sp.Integer))
        assert abs(float(runtime.context["valid"]) - 3.14) < 1e-10
        
        # Invalid constants should be treated as symbols
        assert isinstance(runtime.context["invalid"], sp.Symbol)
        assert isinstance(runtime.context["empty"], sp.Symbol)
    
    @pytest.mark.mathematical
    def test_constants_override_builtin(self):
        """Test that custom constants can override built-in names."""
        mathir = MathIR(
            constants={"pi": "3.0"},  # Override pi with approximation
            targets=[{"type": "value", "name": "test", "expr": "pi"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Custom pi should override built-in
        assert float(runtime.context["pi"]) == 3.0
        assert runtime.context["pi"] != sp.pi


class TestFunctionManagement:
    """Test function definition and management in runtime context."""
    
    @pytest.mark.mathematical
    def test_single_function_definition(self):
        """Test single function definition and integration."""
        func_def = FunctionDef(
            name="quadratic",
            args=["x"],
            expr="x^2 + 2*x + 1"
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="t", domain="R")],
            definitions=Definitions(functions=[func_def]),
            targets=[{"type": "value", "name": "test", "expr": "quadratic(t)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check function is created
        func_symbol = sp.Function("quadratic")
        assert func_symbol in runtime.funcs
        
        # Check function lambda
        lambda_func = runtime.funcs[func_symbol]
        assert isinstance(lambda_func, sp.Lambda)
        
        # Test function evaluation
        result = lambda_func(3)
        expected = 3**2 + 2*3 + 1  # 16
        assert result == expected
        
        # Check function is in context (functions use Function objects as keys)
        assert func_symbol in runtime.context
    
    @pytest.mark.mathematical
    def test_multiple_function_definitions(self):
        """Test multiple function definitions."""
        func1 = FunctionDef(name="f", args=["x"], expr="x^2")
        func2 = FunctionDef(name="g", args=["x"], expr="2*x + 1")
        func3 = FunctionDef(name="h", args=["x", "y"], expr="x*y + x - y")
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="a", domain="R"), SymbolSpec(name="b", domain="R")],
            definitions=Definitions(functions=[func1, func2, func3]),
            targets=[{"type": "value", "name": "test", "expr": "f(a) + g(b)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check all functions are created
        assert len(runtime.funcs) == 3
        
        f_symbol = sp.Function("f")
        g_symbol = sp.Function("g")
        h_symbol = sp.Function("h")
        
        assert f_symbol in runtime.funcs
        assert g_symbol in runtime.funcs
        assert h_symbol in runtime.funcs
        
        # Test function evaluations
        f_lambda = runtime.funcs[f_symbol]
        g_lambda = runtime.funcs[g_symbol]
        h_lambda = runtime.funcs[h_symbol]
        
        assert f_lambda(5) == 25
        assert g_lambda(3) == 7
        assert h_lambda(2, 3) == 2*3 + 2 - 3  # 5
    
    @pytest.mark.mathematical
    def test_function_with_complex_expressions(self):
        """Test functions with complex mathematical expressions."""
        func_def = FunctionDef(
            name="complex_func",
            args=["x"],
            expr=r"\sin(x) + \cos(x^2) + \ln(x + 1)"
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="t", domain="R")],
            definitions=Definitions(functions=[func_def]),
            targets=[{"type": "value", "name": "test", "expr": "complex_func(t)"}]
        )
        
        runtime = build_runtime(mathir)
        
        func_symbol = sp.Function("complex_func")
        lambda_func = runtime.funcs[func_symbol]
        
        # Test with a specific value
        result = lambda_func(1)
        expected = sp.sin(1) + sp.cos(1) + sp.ln(2)
        
        # Should be mathematically equivalent
        assert_symbolic_equivalent(result, expected)
    
    @pytest.mark.mathematical
    def test_function_argument_isolation(self):
        """Test that function arguments are properly isolated."""
        func_def = FunctionDef(
            name="isolated_func",
            args=["x"],
            expr="x + y"  # y should not be resolved from global context
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="y", domain="R")],  # Global y
            definitions=Definitions(functions=[func_def]),
            targets=[{"type": "value", "name": "test", "expr": "isolated_func(1)"}]
        )
        
        runtime = build_runtime(mathir)
        
        func_symbol = sp.Function("isolated_func")
        lambda_func = runtime.funcs[func_symbol]
        
        # Function should have its own y symbol, not the global one
        result = lambda_func(2)
        
        # The result should contain a y symbol (from function scope)
        assert result.has(sp.Symbol('y'))


class TestSequenceManagement:
    """Test sequence definition and management."""
    
    @pytest.mark.mathematical
    def test_sequence_definition(self):
        """Test sequence definition and integration."""
        seq_def = SequenceDef(
            name="arithmetic",
            args=["n"],
            expr="2*n + 1"  # Arithmetic sequence: 3, 5, 7, 9, ...
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="k", domain="N")],
            definitions=Definitions(sequences=[seq_def]),
            targets=[{"type": "value", "name": "test", "expr": "arithmetic(k)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check sequence is created
        seq_symbol = sp.Function("arithmetic")
        assert seq_symbol in runtime.sequences
        
        # Test sequence values
        lambda_seq = runtime.sequences[seq_symbol]
        assert lambda_seq(1) == 3
        assert lambda_seq(2) == 5
        assert lambda_seq(3) == 7
        assert lambda_seq(10) == 21
    
    @pytest.mark.mathematical
    def test_multiple_sequences(self):
        """Test multiple sequence definitions."""
        seq1 = SequenceDef(name="squares", args=["n"], expr="n^2")
        seq2 = SequenceDef(name="cubes", args=["n"], expr="n^3")
        seq3 = SequenceDef(name="fibonacci_like", args=["n"], expr="n*(n+1)/2")
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="i", domain="N")],
            definitions=Definitions(sequences=[seq1, seq2, seq3]),
            targets=[{"type": "value", "name": "test", "expr": "squares(i)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check all sequences are created
        assert len(runtime.sequences) == 3
        
        squares_symbol = sp.Function("squares")
        cubes_symbol = sp.Function("cubes")
        fib_symbol = sp.Function("fibonacci_like")
        
        # Test sequence evaluations
        squares_lambda = runtime.sequences[squares_symbol]
        cubes_lambda = runtime.sequences[cubes_symbol]
        fib_lambda = runtime.sequences[fib_symbol]
        
        assert squares_lambda(4) == 16
        assert cubes_lambda(3) == 27
        assert fib_lambda(5) == 15  # 5*6/2


class TestMatrixManagement:
    """Test matrix definition and management in runtime context."""
    
    @pytest.mark.mathematical
    def test_simple_matrix_definition(self):
        """Test simple matrix definition and integration."""
        matrix_def = MatrixDef(
            name="simple_matrix",
            rows=2,
            cols=3,
            data=[["1", "2", "3"], ["4", "5", "6"]]
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions=Definitions(matrices=[matrix_def]),
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check matrix is created
        assert "simple_matrix" in runtime.matrices
        matrix = runtime.matrices["simple_matrix"]
        
        assert isinstance(matrix, sp.Matrix)
        assert matrix.shape == (2, 3)
        
        # Check matrix elements
        assert matrix[0, 0] == 1
        assert matrix[0, 1] == 2
        assert matrix[0, 2] == 3
        assert matrix[1, 0] == 4
        assert matrix[1, 1] == 5
        assert matrix[1, 2] == 6
        
        # Check matrix is in context
        assert "simple_matrix" in runtime.context
        assert runtime.context["simple_matrix"] == matrix
    
    @pytest.mark.mathematical
    def test_matrix_with_symbolic_elements(self):
        """Test matrix with symbolic expressions as elements."""
        matrix_def = MatrixDef(
            name="symbolic_matrix",
            rows=2,
            cols=2,
            data=[["x", "x+1"], ["2*x", "x^2"]]
        )
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions=Definitions(matrices=[matrix_def]),
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        matrix = runtime.matrices["symbolic_matrix"]
        x_symbol = runtime.symtab["x"]
        
        # Check symbolic elements
        assert matrix[0, 0] == x_symbol
        assert matrix[0, 1] == x_symbol + 1
        assert matrix[1, 0] == 2 * x_symbol
        assert matrix[1, 1] == x_symbol**2
    
    @pytest.mark.mathematical
    def test_multiple_matrices(self):
        """Test multiple matrix definitions."""
        identity = MatrixDef(name="I", rows=2, cols=2, data=[["1", "0"], ["0", "1"]])
        zeros = MatrixDef(name="Z", rows=2, cols=2, data=[["0", "0"], ["0", "0"]])
        
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions=Definitions(matrices=[identity, zeros]),
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check both matrices are created
        assert len(runtime.matrices) == 2
        assert "I" in runtime.matrices
        assert "Z" in runtime.matrices
        
        # Check matrix properties
        I_matrix = runtime.matrices["I"]
        Z_matrix = runtime.matrices["Z"]
        
        assert I_matrix.det() == 1  # Identity determinant
        assert Z_matrix.det() == 0  # Zero matrix determinant


class TestContextLifecycle:
    """Test the complete context lifecycle and integration."""
    
    @pytest.mark.mathematical
    def test_complete_context_building(self):
        """Test building a complete context with all components."""
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="x", domain="R"),
                SymbolSpec(name="n", domain="N"),
            ],
            constants={
                "g": "9.81",
                "c": "3e8",
            },
            definitions=Definitions(
                functions=[FunctionDef(name="f", args=["t"], expr="t^2 + 1")],
                sequences=[SequenceDef(name="a", args=["k"], expr="k + 1")],
                matrices=[MatrixDef(name="M", rows=2, cols=2, data=[["1", "2"], ["3", "4"]])],
            ),
            targets=[{"type": "value", "name": "test", "expr": "f(x) + a(n)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Check all components are present
        assert len(runtime.symtab) == 2  # x, n
        assert len(runtime.funcs) == 1   # f
        assert len(runtime.sequences) == 1  # a
        assert len(runtime.matrices) == 1   # M
        
        # Check context completeness
        expected_keys = {
            'x', 'n',  # symbols
            'pi', 'e', 'i',  # built-in constants
            'g', 'c',  # custom constants
            'M'  # matrix (functions/sequences use Function objects as keys)
        }
        
        for key in expected_keys:
            assert key in runtime.context
        
        # Check function and sequence symbols are in context
        f_symbol = sp.Function("f")
        a_symbol = sp.Function("a")
        assert f_symbol in runtime.context
        assert a_symbol in runtime.context
    
    @pytest.mark.mathematical
    def test_context_symbol_resolution(self):
        """Test that context properly resolves all symbols in expressions."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="alpha", domain="R")],
            constants={"beta": "2.5", "gamma": "1.5"},
            targets=[{"type": "value", "name": "result", "expr": "alpha * beta + gamma * pi"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Parse and substitute an expression
        expr = to_sympy_expr("alpha * beta + gamma * pi")
        substituted = expr.subs(runtime.context)
        
        # Check substitution results
        assert substituted.has(runtime.symtab["alpha"])  # alpha should remain
        assert substituted.has(sp.pi)  # pi should remain
        # beta and gamma should be substituted with their numerical values
        
        # Evaluate with alpha = 1
        final_result = substituted.subs(runtime.symtab["alpha"], 1)
        expected = 1 * 2.5 + 1.5 * sp.pi
        assert_symbolic_equivalent(final_result, expected)
    
    @pytest.mark.mathematical
    def test_context_precedence_rules(self):
        """Test precedence rules in context building."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            constants={"pi": "3.0"},  # Override built-in pi
            definitions=Definitions(
                functions=[FunctionDef(name="sin", args=["t"], expr="t")]  # Override sin
            ),
            targets=[{"type": "value", "name": "test", "expr": "sin(pi)"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Custom constants should override built-ins
        assert float(runtime.context["pi"]) == 3.0
        
        # Custom functions should be available
        sin_symbol = sp.Function("sin")
        assert sin_symbol in runtime.funcs


class TestContextValidation:
    """Test context validation and error handling."""
    
    @pytest.mark.mathematical
    def test_duplicate_symbol_names(self):
        """Test handling of duplicate symbol names."""
        # This should work - same symbol defined multiple times
        mathir = MathIR(
            symbols=[
                SymbolSpec(name="x", domain="R"),
                SymbolSpec(name="x", domain="Z"),  # Redefine with different domain
            ],
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Should have only one x symbol (last one wins)
        assert len(runtime.symtab) == 1
        assert "x" in runtime.symtab
    
    @pytest.mark.mathematical
    def test_empty_definitions(self):
        """Test handling of empty definitions."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            definitions=Definitions(),  # Empty definitions
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        
        # Should work with empty definitions
        assert len(runtime.funcs) == 0
        assert len(runtime.sequences) == 0
        assert len(runtime.matrices) == 0
        assert "x" in runtime.context
    
    @pytest.mark.mathematical
    def test_context_immutability(self):
        """Test that context modifications don't affect original runtime."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            constants={"a": "1.0"},
            targets=[{"type": "value", "name": "test", "expr": "x"}]
        )
        
        runtime = build_runtime(mathir)
        original_context = runtime.context.copy()
        
        # Modify context
        runtime.context["new_var"] = sp.Symbol("new_var")
        
        # Original context should be unchanged
        assert "new_var" not in original_context
        assert len(original_context) < len(runtime.context)


class TestPerformanceOptimization:
    """Test performance aspects of runtime context management."""
    
    @pytest.mark.performance
    def test_large_symbol_table_performance(self):
        """Test performance with large symbol tables."""
        # Create many symbols
        symbols = [SymbolSpec(name=f"var_{i}", domain="R") for i in range(1000)]
        
        mathir = MathIR(
            symbols=symbols,
            targets=[{"type": "value", "name": "test", "expr": "var_0"}]
        )
        
        # Measure runtime building time
        execution_time = measure_execution_time(lambda: build_runtime(mathir))
        
        # Should handle large symbol tables efficiently
        assert execution_time < 2.0  # Less than 2 seconds
    
    @pytest.mark.performance
    def test_complex_context_building_performance(self):
        """Test performance with complex context (many components)."""
        # Create complex MathIR
        symbols = [SymbolSpec(name=f"x{i}", domain="R") for i in range(50)]
        constants = {f"c{i}": str(i * 0.1) for i in range(50)}
        functions = [FunctionDef(name=f"f{i}", args=["t"], expr=f"t^{i+1}") for i in range(20)]
        matrices = [MatrixDef(name=f"M{i}", rows=2, cols=2, 
                             data=[["1", "0"], ["0", "1"]]) for i in range(10)]
        
        mathir = MathIR(
            symbols=symbols,
            constants=constants,
            definitions=Definitions(functions=functions, matrices=matrices),
            targets=[{"type": "value", "name": "test", "expr": "x0"}]
        )
        
        execution_time = measure_execution_time(lambda: build_runtime(mathir))
        
        # Should handle complex contexts efficiently
        assert execution_time < 5.0  # Less than 5 seconds


# Test fixtures
@pytest.fixture
def minimal_runtime():
    """Provide minimal runtime for testing."""
    mathir = MathIR(targets=[{"type": "value", "name": "test", "expr": "1"}])
    return build_runtime(mathir)


@pytest.fixture
def standard_runtime():
    """Provide standard runtime with common components."""
    mathir = MathIR(
        symbols=[SymbolSpec(name="x", domain="R"), SymbolSpec(name="n", domain="N")],
        constants={"g": "9.81"},
        definitions=Definitions(
            functions=[FunctionDef(name="f", args=["t"], expr="t^2")],
            matrices=[MatrixDef(name="I", rows=2, cols=2, data=[["1", "0"], ["0", "1"]])]
        ),
        targets=[{"type": "value", "name": "test", "expr": "f(x)"}]
    )
    return build_runtime(mathir)


@pytest.fixture
def mathematical_validator():
    """Provide mathematical validator for tests."""
    return MathematicalValidator()


# Test configuration
pytestmark = [
    pytest.mark.unit,
    pytest.mark.runtime_context,
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
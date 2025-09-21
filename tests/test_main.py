import pytest
import json
import os
from mathir_parser.main import MathIR, run_mathir
import sympy as sp

# Helper to load JSON and create MathIR
def load_mathir_from_file(filepath: str) -> MathIR:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return MathIR.model_validate(data)

class TestMathIRParser:
    """Comprehensive tests for MathIR parser covering all target types."""

    def test_integral_definite_simple(self):
        """Test definite integral computation."""
        ir = MathIR.model_validate({
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [
                {
                    "type": "integral_def",
                    "expr": "x^2 + 1",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I"
                }
            ],
            "output": {"mode": "decimal", "round_to": 3}
        })
        results = run_mathir(ir)
        assert "I" in results
        # Integral of x^2 + 1 from 0 to 1 is 1/3 + 1 = 4/3 ≈ 1.333
        assert abs(float(results["I"]) - 1.333) < 0.001

    def test_integral_with_value_reference(self):
        """Test integral followed by value evaluation."""
        ir = MathIR.model_validate({
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [
                {
                    "type": "integral_def",
                    "expr": "x",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I"
                },
                {
                    "type": "value",
                    "name": "result",
                    "expr": "I"
                }
            ],
            "output": {"mode": "exact"}
        })
        results = run_mathir(ir)
        assert "result" in results
        # Integral of x from 0 to 1 is 1/2
        assert results["result"] == sp.Rational(1, 2)

    def test_limit_basic(self):
        """Test limit computation."""
        ir = MathIR.model_validate({
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [
                {
                    "type": "limit",
                    "expr": "\\frac{\\sin(x)}{x}",
                    "var": "x",
                    "to": "0"
                }
            ],
            "output": {"mode": "exact"}
        })
        results = run_mathir(ir)
        assert "limit" in results
        assert results["limit"] == 1

    def test_limit_to_infinity(self):
        """Test limit to infinity."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[
                {
                    "type": "limit",
                    "expr": "\\frac{1}{x}",
                    "var": "x",
                    "to": "oo"
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "limit" in results
        assert results["limit"] == 0

    def test_sum_finite(self):
        """Test finite summation."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "n", "domain": "N"}],
            targets=[
                {
                    "type": "sum",
                    "term": "n",
                    "idx": "n",
                    "start": "1",
                    "end": "5"
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "sum" in results
        assert results["sum"] == 15  # 1+2+3+4+5=15

    def test_sum_geometric(self):
        """Test geometric series summation."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "n", "domain": "N"}],
            targets=[
                {
                    "type": "sum",
                    "term": "(\\frac{1}{2})^n",
                    "idx": "n",
                    "start": "0",
                    "end": "10"
                }
            ],
            output={"mode": "decimal", "round_to": 5}
        )
        results = run_mathir(ir)
        assert "sum" in results
        # Geometric series sum ≈ 2 - (1/2)^11 ≈ 1.999023
        assert abs(float(results["sum"]) - 1.999023) < 0.0001

    def test_solve_linear_system(self):
        """Test solving system of linear equations."""
        ir = MathIR.model_validate({
            "expr_format": "latex",
            "symbols": [
                {"name": "x", "domain": "R"},
                {"name": "y", "domain": "R"}
            ],
            "targets": [
                {
                    "type": "solve_for",
                    "unknowns": ["x", "y"],
                    "equations": [
                        "x + y = 6",
                        "x - y = 2"
                    ]
                }
            ],
            "output": {"mode": "exact"}
        })
        results = run_mathir(ir)
        assert "solve" in results
        # Should solve to x=4, y=2
        solution = results["solve"]
        assert len(solution) == 1
        sol_dict = solution[0]
        assert sol_dict[sp.Symbol('x', real=True)] == 4
        assert sol_dict[sp.Symbol('y', real=True)] == 2

    def test_solve_quadratic(self):
        """Test solving quadratic equation."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[
                {
                    "type": "solve_for",
                    "unknowns": ["x"],
                    "equations": ["x^2 - 4 = 0"]
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "solve" in results
        solution = results["solve"]
        roots = set(sol[sp.Symbol('x', real=True)] for sol in solution)
        assert roots == {-2, 2}

    def test_inequalities_simple(self):
        """Test solving simple inequality."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[
                {
                    "type": "inequalities",
                    "inequalities": ["x^2 > 4"]
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "inequalities" in results
        # Should be (-oo, -2) U (2, oo)
        ineq_result = results["inequalities"]
        assert isinstance(ineq_result, sp.Or)
        x = sp.Symbol('x', real=True)
        expected_ineqs = {x < -2, x > 2}
        actual_ineqs = set(sp.simplify(ineq) for ineq in ineq_result.args)
        assert actual_ineqs == expected_ineqs

    def test_matrix_solve_basic(self):
        """Test solving matrix equation A*X = B."""
        ir = MathIR(
            expr_format="latex",
            symbols=[
                {"name": "x", "domain": "R"},
                {"name": "y", "domain": "R"}
            ],
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
            conditions=[
                {
                    "type": "matrix_equation",
                    "expr": "A*X = B"
                }
            ],
            targets=[
                {
                    "type": "solve_for_matrix",
                    "unknown": "X"
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "matrix" in results
        # A = [[1,2],[3,4]], B = [[5],[11]]
        # X should be [[1], [2]] since A*X = [1+4, 3+8] = [5,11]
        matrix_result = results["matrix"]
        assert matrix_result[0, 0] == 1
        assert matrix_result[1, 0] == 2

    def test_function_definition_and_use(self):
        """Test defining and using custom functions."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            definitions={
                "functions": [
                    {
                        "name": "f",
                        "args": ["x"],
                        "expr": "x^2 + 1"
                    }
                ]
            },
            targets=[
                {
                    "type": "value",
                    "name": "result",
                    "expr": "f(3)"
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "result" in results
        assert results["result"] == 10  # 3^2 + 1 = 10

    def test_sequence_definition_and_sum(self):
        """Test defining and summing sequences."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "n", "domain": "N"}],
            definitions={
                "sequences": [
                    {
                        "name": "a",
                        "args": ["n"],
                        "expr": "2*n + 1"
                    }
                ]
            },
            targets=[
                {
                    "type": "sum",
                    "term": "a(n)",
                    "idx": "n",
                    "start": "0",
                    "end": "2"
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "sum" in results
        # a(0)=1, a(1)=3, a(2)=5, sum=9
        assert results["sum"] == 9

    def test_complex_expression_with_constants(self):
        """Test expressions with mathematical constants."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[
                {
                    "type": "value",
                    "name": "result",
                    "expr": "\\sin(\\pi) + e^2"
                }
            ],
            output={"mode": "decimal", "round_to": 3}
        )
        results = run_mathir(ir)
        assert "result" in results
        # sin(pi) + e^2 ≈ 0 + 7.389 ≈ 7.389
        assert abs(float(results["result"]) - 7.389) < 0.001


    def test_error_handling_invalid_latex(self):
        """Test handling of invalid LaTeX expressions."""
        ir = MathIR(
            targets=[
                {
                    "type": "value",
                    "name": "result",
                    "expr": "\\invalidcommand{x}"
                }
            ]
        )
        # Should not raise exception, but may return error or unexpected result
        results = run_mathir(ir)
        assert "result" in results

    def test_matrix_solve_non_invertible(self):
        """Test matrix solve with non-invertible matrix."""
        ir = MathIR(
            definitions={
                "matrices": [
                    {
                        "name": "A",
                        "rows": 2,
                        "cols": 2,
                        "data": [["1", "2"], ["2", "4"]]  # Not invertible
                    },
                    {
                        "name": "B",
                        "rows": 2,
                        "cols": 1,
                        "data": [["1"], ["2"]]
                    }
                ]
            },
            conditions=[
                {
                    "type": "matrix_equation",
                    "expr": "A*X = B"
                }
            ],
            targets=[
                {
                    "type": "solve_for_matrix",
                    "unknown": "X"
                }
            ]
        )
        results = run_mathir(ir)
        assert "matrix" in results
        assert "error" in results["matrix"]

    def test_empty_targets(self):
        """Test with no targets."""
        ir = MathIR(targets=[])
        results = run_mathir(ir)
        assert results == {}

    def test_multiple_targets(self):
        """Test multiple targets in sequence."""
        ir = MathIR(
            expr_format="latex",
            symbols=[{"name": "x", "domain": "R"}],
            targets=[
                {
                    "type": "integral_def",
                    "expr": "x",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "half"
                },
                {
                    "type": "value",
                    "name": "double",
                    "expr": "2*half"
                },
                {
                    "type": "limit",
                    "expr": "\\frac{1}{x}",
                    "var": "x",
                    "to": "0"
                }
            ],
            output={"mode": "exact"}
        )
        results = run_mathir(ir)
        assert "half" in results
        assert "double" in results
        assert "limit" in results
        assert results["half"] == sp.Rational(1,2)
        assert results["double"] == 1
        assert results["limit"] == sp.oo

    # Integration tests with existing JSON files
    @pytest.mark.parametrize("json_file", [
        "test_simple_integral.json",
        "test_limit.json",
        "test_sum.json",
        "test_solve.json",
        "test_inequalities.json",
        "test_matrix_solve.json"
    ])
    def test_existing_json_files(self, json_file):
        """Test that existing JSON test files can be processed without errors."""
        filepath = os.path.join("tests", json_file)
        ir = load_mathir_from_file(filepath)
        results = run_mathir(ir)
        assert isinstance(results, dict)
        assert len(results) > 0

if __name__ == "__main__":
    pytest.main([__file__])
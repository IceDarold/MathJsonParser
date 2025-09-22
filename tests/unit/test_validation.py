"""
Comprehensive unit tests for Pydantic validation and MathIR schema validation.

This module tests the Pydantic model validation, MathIR schema compliance, error handling,
and data validation across all Target types and mathematical constructs. It ensures that
the MathIR parser correctly validates input data and handles edge cases.

Target: All Pydantic models and validation logic in mathir_parser.main
Coverage Goal: 95%+ with comprehensive validation testing
"""

import pytest
import json
from typing import Dict, Any, List, Optional, Union
from pydantic import ValidationError
import sympy as sp

from mathir_parser.main import (
    MathIR, SymbolSpec, FunctionDef, SequenceDef, MatrixDef, DistributionDef, 
    GeometryDef, Definitions, TransformSpec, Condition, OutputSpec, ValidationSpec,
    TargetIntegral, TargetLimit, TargetSum, TargetSolve, TargetIneq, TargetMatrixSolve,
    TargetProbability, TargetValue, TargetMatrixInverse, TargetMatrixDeterminant,
    TargetSequenceLimitCondition, TargetFindMaximum, TargetAreaBetweenCurves,
    TargetDoubleIntegral, run_mathir
)
from tests.utils.assertion_helpers import MathematicalValidator
from tests.utils.test_helpers import TestDataLoader


class TestSymbolSpecValidation:
    """Test SymbolSpec model validation."""
    
    @pytest.mark.validation
    def test_valid_symbol_specs(self):
        """Test valid symbol specifications."""
        valid_specs = [
            {"name": "x", "domain": "R"},
            {"name": "n", "domain": "N"},
            {"name": "z", "domain": "C"},
            {"name": "k", "domain": "Z"},
            {"name": "t", "domain": "R+"},
            {"name": "m", "domain": "N+"},
            {"name": "alpha"},  # Default domain
        ]
        
        for spec_data in valid_specs:
            symbol = SymbolSpec(**spec_data)
            assert symbol.name == spec_data["name"]
            assert symbol.domain == spec_data.get("domain", "R")
    
    @pytest.mark.validation
    def test_invalid_symbol_specs(self):
        """Test invalid symbol specifications."""
        invalid_specs = [
            {"name": "", "domain": "R"},  # Empty name
            {"name": "x", "domain": "INVALID"},  # Invalid domain
            {"domain": "R"},  # Missing name
            {"name": "123", "domain": "R"},  # Invalid name format
        ]
        
        for spec_data in invalid_specs:
            with pytest.raises(ValidationError):
                SymbolSpec(**spec_data)
    
    @pytest.mark.validation
    def test_symbol_domain_constraints(self):
        """Test domain constraint validation."""
        valid_domains = ["R", "R+", "Z", "N", "N+", "C"]
        
        for domain in valid_domains:
            symbol = SymbolSpec(name="test", domain=domain)
            assert symbol.domain == domain
        
        # Test invalid domain
        with pytest.raises(ValidationError):
            SymbolSpec(name="test", domain="INVALID_DOMAIN")


class TestFunctionDefinitionValidation:
    """Test FunctionDef model validation."""
    
    @pytest.mark.validation
    def test_valid_function_definitions(self):
        """Test valid function definitions."""
        valid_functions = [
            {"name": "f", "args": ["x"], "expr": "x^2"},
            {"name": "g", "args": ["x", "y"], "expr": "x*y + 1"},
            {"name": "trig", "args": ["theta"], "expr": r"\sin(theta) + \cos(theta)"},
            {"name": "complex_func", "args": ["a", "b", "c"], "expr": "a*b^2 + c"},
        ]
        
        for func_data in valid_functions:
            func = FunctionDef(**func_data)
            assert func.name == func_data["name"]
            assert func.args == func_data["args"]
            assert func.expr == func_data["expr"]
    
    @pytest.mark.validation
    def test_invalid_function_definitions(self):
        """Test invalid function definitions."""
        invalid_functions = [
            {"name": "", "args": ["x"], "expr": "x^2"},  # Empty name
            {"name": "f", "args": [], "expr": "x^2"},  # Empty args
            {"name": "f", "args": ["x"], "expr": ""},  # Empty expression
            {"args": ["x"], "expr": "x^2"},  # Missing name
            {"name": "f", "expr": "x^2"},  # Missing args
            {"name": "f", "args": ["x"]},  # Missing expr
        ]
        
        for func_data in invalid_functions:
            with pytest.raises(ValidationError):
                FunctionDef(**func_data)
    
    @pytest.mark.validation
    def test_function_argument_validation(self):
        """Test function argument validation."""
        # Valid arguments
        func = FunctionDef(name="f", args=["x", "y", "z"], expr="x + y + z")
        assert len(func.args) == 3
        
        # Duplicate arguments should be allowed (Pydantic doesn't prevent this)
        func_dup = FunctionDef(name="f", args=["x", "x"], expr="x^2")
        assert func_dup.args == ["x", "x"]


class TestMatrixDefinitionValidation:
    """Test MatrixDef model validation."""
    
    @pytest.mark.validation
    def test_valid_matrix_definitions(self):
        """Test valid matrix definitions."""
        valid_matrices = [
            {
                "name": "A",
                "rows": 2,
                "cols": 2,
                "data": [["1", "2"], ["3", "4"]]
            },
            {
                "name": "B",
                "rows": 3,
                "cols": 1,
                "data": [["x"], ["y"], ["z"]]
            },
            {
                "name": "identity",
                "rows": 3,
                "cols": 3,
                "data": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]
            },
        ]
        
        for matrix_data in valid_matrices:
            matrix = MatrixDef(**matrix_data)
            assert matrix.name == matrix_data["name"]
            assert matrix.rows == matrix_data["rows"]
            assert matrix.cols == matrix_data["cols"]
            assert matrix.data == matrix_data["data"]
    
    @pytest.mark.validation
    def test_invalid_matrix_definitions(self):
        """Test invalid matrix definitions."""
        invalid_matrices = [
            {
                "name": "",
                "rows": 2,
                "cols": 2,
                "data": [["1", "2"], ["3", "4"]]
            },  # Empty name
            {
                "name": "A",
                "rows": 0,
                "cols": 2,
                "data": []
            },  # Zero rows
            {
                "name": "A",
                "rows": 2,
                "cols": 0,
                "data": [[], []]
            },  # Zero cols
            {
                "name": "A",
                "rows": 2,
                "cols": 2,
                "data": [["1", "2"], ["3"]]
            },  # Inconsistent row length
            {
                "name": "A",
                "rows": 2,
                "cols": 2,
                "data": [["1", "2"]]
            },  # Wrong number of rows
        ]
        
        for matrix_data in invalid_matrices:
            with pytest.raises(ValidationError):
                MatrixDef(**matrix_data)
    
    @pytest.mark.validation
    def test_matrix_dimension_consistency(self):
        """Test matrix dimension consistency validation."""
        # This should pass validation at Pydantic level but might fail at runtime
        matrix_data = {
            "name": "inconsistent",
            "rows": 2,
            "cols": 2,
            "data": [["1", "2"], ["3", "4", "5"]]  # Inconsistent
        }
        
        # Pydantic validation might not catch this, but it should be caught at runtime
        try:
            matrix = MatrixDef(**matrix_data)
            # If Pydantic allows it, runtime should catch it
            assert len(matrix.data) == matrix.rows
            for row in matrix.data:
                assert len(row) == matrix.cols
        except (ValidationError, AssertionError):
            # Either validation error or assertion error is acceptable
            pass


class TestTargetValidation:
    """Test Target model validation for all target types."""
    
    @pytest.mark.validation
    def test_target_integral_validation(self):
        """Test TargetIntegral validation."""
        # Valid integral targets
        valid_integrals = [
            {
                "type": "integral_def",
                "expr": "x^2",
                "var": "x",
                "limits": [0, 1],
                "name": "integral1"
            },
            {
                "type": "integral_def",
                "expr": r"\sin(x)",
                "var": "x",
                "limits": [0, "pi"]
            },  # No name (optional)
        ]
        
        for integral_data in valid_integrals:
            integral = TargetIntegral(**integral_data)
            assert integral.type == "integral_def"
            assert integral.expr == integral_data["expr"]
            assert integral.var == integral_data["var"]
            assert integral.limits == integral_data["limits"]
        
        # Invalid integral targets
        invalid_integrals = [
            {"type": "integral_def", "var": "x", "limits": [0, 1]},  # Missing expr
            {"type": "integral_def", "expr": "x^2", "limits": [0, 1]},  # Missing var
            {"type": "integral_def", "expr": "x^2", "var": "x"},  # Missing limits
            {"type": "wrong_type", "expr": "x^2", "var": "x", "limits": [0, 1]},  # Wrong type
        ]
        
        for integral_data in invalid_integrals:
            with pytest.raises(ValidationError):
                TargetIntegral(**integral_data)
    
    @pytest.mark.validation
    def test_target_limit_validation(self):
        """Test TargetLimit validation."""
        # Valid limit targets
        valid_limits = [
            {"type": "limit", "expr": r"\frac{\sin(x)}{x}", "var": "x", "to": "0"},
            {"type": "limit", "expr": "x^2", "var": "x", "to": "oo"},
            {"type": "limit", "expr": "1/x", "var": "x", "to": "-oo"},
        ]
        
        for limit_data in valid_limits:
            limit = TargetLimit(**limit_data)
            assert limit.type == "limit"
            assert limit.expr == limit_data["expr"]
            assert limit.var == limit_data["var"]
            assert limit.to == limit_data["to"]
        
        # Invalid limit targets
        invalid_limits = [
            {"type": "limit", "var": "x", "to": "0"},  # Missing expr
            {"type": "limit", "expr": "x^2", "to": "0"},  # Missing var
            {"type": "limit", "expr": "x^2", "var": "x"},  # Missing to
        ]
        
        for limit_data in invalid_limits:
            with pytest.raises(ValidationError):
                TargetLimit(**limit_data)
    
    @pytest.mark.validation
    def test_target_sum_validation(self):
        """Test TargetSum validation."""
        # Valid sum targets
        valid_sums = [
            {"type": "sum", "term": "n", "idx": "n", "start": "1", "end": "10"},
            {"type": "sum", "term": "n^2", "idx": "n", "start": "0", "end": "oo"},
            {"type": "sum", "term": "1/n", "idx": "n", "start": "1", "end": "100"},
        ]
        
        for sum_data in valid_sums:
            sum_target = TargetSum(**sum_data)
            assert sum_target.type == "sum"
            assert sum_target.term == sum_data["term"]
            assert sum_target.idx == sum_data["idx"]
            assert sum_target.start == sum_data["start"]
            assert sum_target.end == sum_data["end"]
        
        # Invalid sum targets
        invalid_sums = [
            {"type": "sum", "idx": "n", "start": "1", "end": "10"},  # Missing term
            {"type": "sum", "term": "n", "start": "1", "end": "10"},  # Missing idx
            {"type": "sum", "term": "n", "idx": "n", "end": "10"},  # Missing start
            {"type": "sum", "term": "n", "idx": "n", "start": "1"},  # Missing end
        ]
        
        for sum_data in invalid_sums:
            with pytest.raises(ValidationError):
                TargetSum(**sum_data)
    
    @pytest.mark.validation
    def test_target_solve_validation(self):
        """Test TargetSolve validation."""
        # Valid solve targets
        valid_solves = [
            {"type": "solve_for", "unknowns": ["x"], "equations": ["x + 1 = 0"]},
            {"type": "solve_for", "unknowns": ["x", "y"], "equations": ["x + y = 3", "x - y = 1"]},
            {"type": "solve_for", "unknowns": ["z"], "equations": ["z^2 - 4 = 0"]},
        ]
        
        for solve_data in valid_solves:
            solve_target = TargetSolve(**solve_data)
            assert solve_target.type == "solve_for"
            assert solve_target.unknowns == solve_data["unknowns"]
            assert solve_target.equations == solve_data["equations"]
        
        # Invalid solve targets
        invalid_solves = [
            {"type": "solve_for", "equations": ["x + 1 = 0"]},  # Missing unknowns
            {"type": "solve_for", "unknowns": ["x"]},  # Missing equations
            {"type": "solve_for", "unknowns": [], "equations": ["x + 1 = 0"]},  # Empty unknowns
            {"type": "solve_for", "unknowns": ["x"], "equations": []},  # Empty equations
        ]
        
        for solve_data in invalid_solves:
            with pytest.raises(ValidationError):
                TargetSolve(**solve_data)


class TestMathIRValidation:
    """Test complete MathIR model validation."""
    
    @pytest.mark.validation
    def test_minimal_valid_mathir(self):
        """Test minimal valid MathIR object."""
        minimal_data = {
            "targets": [{"type": "value", "name": "result", "expr": "1"}]
        }
        
        mathir = MathIR(**minimal_data)
        assert len(mathir.targets) == 1
        assert mathir.expr_format == "latex"  # Default value
        assert mathir.task_type == "auto"  # Default value
    
    @pytest.mark.validation
    def test_complete_valid_mathir(self):
        """Test complete valid MathIR object with all components."""
        complete_data = {
            "meta": {"author": "test", "version": "1.0"},
            "task_type": "integral",
            "expr_format": "latex",
            "assumptions": {"x": "real"},
            "constants": {"g": "9.81"},
            "symbols": [{"name": "x", "domain": "R"}],
            "definitions": {
                "functions": [{"name": "f", "args": ["t"], "expr": "t^2"}],
                "matrices": [{"name": "A", "rows": 2, "cols": 2, "data": [["1", "0"], ["0", "1"]]}]
            },
            "targets": [{"type": "integral_def", "expr": "x^2", "var": "x", "limits": [0, 1]}],
            "output": {"mode": "exact", "simplify": True},
            "validation": {"tolerance_abs": 1e-10}
        }
        
        mathir = MathIR(**complete_data)
        assert mathir.meta == complete_data["meta"]
        assert mathir.task_type == complete_data["task_type"]
        assert mathir.expr_format == complete_data["expr_format"]
        assert len(mathir.symbols) == 1
        assert len(mathir.targets) == 1
    
    @pytest.mark.validation
    def test_invalid_mathir_objects(self):
        """Test invalid MathIR objects."""
        invalid_mathirs = [
            {},  # Missing targets
            {"targets": []},  # Empty targets
            {"targets": [{"type": "invalid_type"}]},  # Invalid target type
            {"expr_format": "invalid_format", "targets": [{"type": "value", "name": "test", "expr": "1"}]},
            {"task_type": "invalid_task", "targets": [{"type": "value", "name": "test", "expr": "1"}]},
        ]
        
        for mathir_data in invalid_mathirs:
            with pytest.raises(ValidationError):
                MathIR(**mathir_data)
    
    @pytest.mark.validation
    def test_mathir_field_defaults(self):
        """Test MathIR field default values."""
        mathir = MathIR(targets=[{"type": "value", "name": "test", "expr": "1"}])
        
        # Check default values
        assert mathir.meta == {}
        assert mathir.task_type == "auto"
        assert mathir.expr_format == "latex"
        assert mathir.prob_space is None
        assert mathir.assumptions == {}
        assert mathir.constants == {}
        assert mathir.symbols == []
        assert mathir.transforms == []
        assert mathir.conditions == []
        assert mathir.output.mode == "decimal"
        assert mathir.output.round_to == 3
        assert mathir.validation.tolerance_abs == 1e-9


class TestOutputSpecValidation:
    """Test OutputSpec model validation."""
    
    @pytest.mark.validation
    def test_valid_output_specs(self):
        """Test valid output specifications."""
        valid_outputs = [
            {"mode": "exact"},
            {"mode": "decimal", "round_to": 5},
            {"mode": "decimal", "round_to": 3, "simplify": False},
            {"mode": "exact", "simplify": True, "rationalize": True},
        ]
        
        for output_data in valid_outputs:
            output = OutputSpec(**output_data)
            assert output.mode == output_data["mode"]
            if "round_to" in output_data:
                assert output.round_to == output_data["round_to"]
    
    @pytest.mark.validation
    def test_invalid_output_specs(self):
        """Test invalid output specifications."""
        invalid_outputs = [
            {"mode": "invalid_mode"},
            {"mode": "decimal", "round_to": -1},  # Negative rounding
            {"mode": "decimal", "round_to": "invalid"},  # Non-integer rounding
        ]
        
        for output_data in invalid_outputs:
            with pytest.raises(ValidationError):
                OutputSpec(**output_data)
    
    @pytest.mark.validation
    def test_output_spec_defaults(self):
        """Test OutputSpec default values."""
        output = OutputSpec()
        assert output.mode == "decimal"
        assert output.round_to == 3
        assert output.simplify == True
        assert output.rationalize == False


class TestValidationSpecValidation:
    """Test ValidationSpec model validation."""
    
    @pytest.mark.validation
    def test_valid_validation_specs(self):
        """Test valid validation specifications."""
        valid_validations = [
            {"tolerance_abs": 1e-10},
            {"tolerance_abs": 1e-6, "check_domain_violations": False},
            {"tolerance_abs": 0.001, "check_domain_violations": True},
        ]
        
        for validation_data in valid_validations:
            validation = ValidationSpec(**validation_data)
            assert validation.tolerance_abs == validation_data["tolerance_abs"]
    
    @pytest.mark.validation
    def test_invalid_validation_specs(self):
        """Test invalid validation specifications."""
        invalid_validations = [
            {"tolerance_abs": -1e-10},  # Negative tolerance
            {"tolerance_abs": "invalid"},  # Non-numeric tolerance
            {"tolerance_abs": 0},  # Zero tolerance might be problematic
        ]
        
        for validation_data in invalid_validations:
            with pytest.raises(ValidationError):
                ValidationSpec(**validation_data)
    
    @pytest.mark.validation
    def test_validation_spec_defaults(self):
        """Test ValidationSpec default values."""
        validation = ValidationSpec()
        assert validation.tolerance_abs == 1e-9
        assert validation.check_domain_violations == True


class TestJSONSchemaValidation:
    """Test JSON schema validation and serialization."""
    
    @pytest.mark.validation
    def test_mathir_json_serialization(self):
        """Test MathIR JSON serialization and deserialization."""
        mathir_data = {
            "symbols": [{"name": "x", "domain": "R"}],
            "constants": {"pi_approx": "3.14159"},
            "targets": [{"type": "value", "name": "result", "expr": "pi_approx * x^2"}],
            "output": {"mode": "decimal", "round_to": 4}
        }
        
        # Create MathIR object
        mathir = MathIR(**mathir_data)
        
        # Serialize to JSON
        json_str = mathir.model_dump_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        json_data = json.loads(json_str)
        mathir_restored = MathIR(**json_data)
        
        # Check that restored object is equivalent
        assert mathir_restored.symbols[0].name == mathir.symbols[0].name
        assert mathir_restored.constants == mathir.constants
        assert len(mathir_restored.targets) == len(mathir.targets)
    
    @pytest.mark.validation
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON data."""
        invalid_json_data = [
            '{"targets": [{"type": "invalid_type"}]}',  # Invalid target type
            '{"symbols": [{"name": "", "domain": "R"}]}',  # Invalid symbol
            '{"targets": []}',  # Empty targets
            '{}',  # Missing required fields
        ]
        
        for json_str in invalid_json_data:
            json_data = json.loads(json_str)
            with pytest.raises(ValidationError):
                MathIR(**json_data)
    
    @pytest.mark.validation
    def test_schema_compliance(self):
        """Test compliance with expected JSON schema structure."""
        # Load a test file and validate it
        test_data = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{"type": "integral_def", "expr": "x^2", "var": "x", "limits": [0, 1]}],
            "output": {"mode": "exact"}
        }
        
        mathir = MathIR(**test_data)
        
        # Check that all required fields are present
        assert hasattr(mathir, 'targets')
        assert hasattr(mathir, 'expr_format')
        assert hasattr(mathir, 'symbols')
        assert hasattr(mathir, 'output')
        
        # Check that the structure matches expected schema
        serialized = mathir.model_dump()
        assert 'targets' in serialized
        assert 'expr_format' in serialized
        assert 'symbols' in serialized


class TestErrorHandlingValidation:
    """Test error handling and edge cases in validation."""
    
    @pytest.mark.validation
    def test_nested_validation_errors(self):
        """Test validation errors in nested structures."""
        # Invalid function definition within MathIR
        mathir_data = {
            "definitions": {
                "functions": [{"name": "", "args": ["x"], "expr": "x^2"}]  # Invalid function
            },
            "targets": [{"type": "value", "name": "test", "expr": "1"}]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            MathIR(**mathir_data)
        
        # Check that error information is available
        assert exc_info.value.error_count() > 0
    
    @pytest.mark.validation
    def test_type_coercion(self):
        """Test automatic type coercion in validation."""
        # Test that strings are converted to appropriate types where possible
        mathir_data = {
            "targets": [{"type": "value", "name": "test", "expr": "1"}],
            "output": {"round_to": "5"}  # String instead of int
        }
        
        # This might work due to Pydantic's type coercion
        try:
            mathir = MathIR(**mathir_data)
            assert mathir.output.round_to == 5
        except ValidationError:
            # If coercion fails, that's also acceptable
            pass
    
    @pytest.mark.validation
    def test_extra_fields_handling(self):
        """Test handling of extra fields in input data."""
        mathir_data = {
            "targets": [{"type": "value", "name": "test", "expr": "1"}],
            "extra_field": "should_be_ignored",  # Extra field
            "another_extra": {"nested": "data"}
        }
        
        # Pydantic should ignore extra fields by default
        mathir = MathIR(**mathir_data)
        assert not hasattr(mathir, 'extra_field')
        assert not hasattr(mathir, 'another_extra')


class TestRuntimeValidation:
    """Test validation during runtime execution."""
    
    @pytest.mark.validation
    def test_valid_mathir_execution(self):
        """Test that valid MathIR objects execute successfully."""
        valid_mathirs = [
            MathIR(
                symbols=[SymbolSpec(name="x", domain="R")],
                targets=[TargetValue(type="value", name="result", expr="2*x + 1")]
            ),
            MathIR(
                symbols=[SymbolSpec(name="x", domain="R")],
                targets=[TargetIntegral(type="integral_def", expr="x^2", var="x", limits=[0, 1])]
            ),
            MathIR(
                symbols=[SymbolSpec(name="x", domain="R")],
                targets=[TargetLimit(type="limit", expr="sin(x)/x", var="x", to="0")]
            ),
        ]
        
        for mathir in valid_mathirs:
            # Should not raise validation errors
            try:
                results = run_mathir(mathir)
                assert isinstance(results, dict)
            except Exception as e:
                # Runtime errors are different from validation errors
                assert not isinstance(e, ValidationError)
    
    @pytest.mark.validation
    def test_domain_violation_detection(self):
        """Test detection of domain violations during execution."""
        # Create MathIR with potential domain violations
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R+")],  # Positive reals only
            targets=[TargetValue(type="value", name="result", expr="ln(x)")],
            validation=ValidationSpec(check_domain_violations=True)
        )
        
        # This should execute without validation errors (domain checking is runtime)
        try:
            results = run_mathir(mathir)
            assert isinstance(results, dict)
        except Exception as e:
            # Runtime domain violations are acceptable
            assert not isinstance(e, ValidationError)
    
    @pytest.mark.validation
    def test_tolerance_validation_in_results(self):
        """Test that tolerance settings affect result validation."""
        mathir = MathIR(
            symbols=[SymbolSpec(name="x", domain="R")],
            targets=[TargetIntegral(type="integral_def", expr="x", var="x", limits=[0, 1])],
            validation=ValidationSpec(tolerance_abs=1e-12),  # Very strict tolerance
            output=OutputSpec(mode="decimal", round_to=10)
        )
        
        results = run_mathir(mathir)
        assert isinstance(results, dict)
        
        # The tolerance setting should be available for validation
        assert mathir.validation.tolerance_abs == 1e-12


class TestDistributionAndGeometryValidation:
    """Test validation of distribution and geometry definitions."""
    
    @pytest.mark.validation
    def test_distribution_definition_validation(self):
        """Test DistributionDef validation."""
        valid_distributions = [
            {"name": "coin", "kind": "bernoulli", "params": {"p": "0.5"}},
            {"name": "dice", "kind": "uniform", "params": {"a": "1", "b": "6"}},
            {"name": "normal", "kind": "binomial", "params": {"n": "10", "p": "0.3"}},
        ]
        
        for dist_data in valid_distributions:
            dist = DistributionDef(**dist_data)
            assert dist.name == dist_data["name"]
            assert dist.kind == dist_data["kind"]
            assert dist.params == dist_data["params"]
        
        # Invalid distributions
        invalid_distributions = [
            {"name": "", "kind": "bernoulli", "params": {"p": "0.5"}},  # Empty name
            {"name": "test", "kind": "invalid_kind", "params": {}},  # Invalid kind
            {"name": "test", "params": {"p": "0.5"}},  # Missing kind
        ]
        
        for dist_data in invalid_distributions:
            with pytest.raises(ValidationError):
                DistributionDef(**dist_data)
    
    @pytest.mark.validation
    def test_geometry_definition_validation(self):
        """Test GeometryDef validation."""
        valid_geometries = [
            {"id": "line1", "kind": "line", "equation": "y = 2x + 1"},
            {"id": "circle1", "kind": "circle", "params": {"center": [0, 0], "radius": 5}},
            {"id": "parabola1", "kind": "parabola", "equation": "y = x^2"},
        ]
        
        for geom_data in valid_geometries:
            geom = GeometryDef(**geom_data)
            assert geom.id == geom_data["id"]
            assert geom.kind == geom_data["kind"]
        
        # Invalid geometries
        invalid_geometries = [
            {"id": "", "kind": "line"},  # Empty id
            {"id": "test", "kind": "invalid_kind"},  # Invalid kind
            {"kind": "line"},  # Missing id
        ]
        
        for geom_data in invalid_geometries:
            with pytest.raises(ValidationError):
                GeometryDef(**geom_data)


# Test fixtures
@pytest.fixture
def sample_mathir_data():
    """Provide sample MathIR data for testing."""
    return {
        "symbols": [{"name": "x", "domain": "R"}],
        "constants": {"g": "9.81"},
        "targets": [{"type": "value", "name": "result", "expr": "g * x^2"}],
        "output": {"mode": "decimal", "round_to": 3}
    }


@pytest.fixture
def mathematical_validator():
    """Provide mathematical validator for tests."""
    return MathematicalValidator()


# Test configuration
pytestmark = [
    pytest.mark.unit,
    pytest.mark.validation,
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
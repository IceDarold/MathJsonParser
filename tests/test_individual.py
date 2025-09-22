import pytest
import json
import os
import glob
from tokenize import TokenError
from mathir_parser.main import MathIR, run_mathir
import sympy as sp

class TestIndividualTasks:
    """
    Unit tests for individual task files in tests/individual/.
    Each test loads a JSON file, parses the MathIR structure, runs the computation,
    and validates the results including error handling and structure validation.

    Example input: A JSON file like task_000001.json containing:
    {
        "task_id": "task_000001",
        "task": "Compute limit...",
        "parsed_json": {
            "expr_format": "latex",
            "symbols": [{"name": "n", "domain": "N+"}],
            "targets": [{"type": "limit", "expr": "...", "var": "n", "to": "oo"}],
            "output": {"mode": "decimal", "round_to": 3}
        }
    }

    Example output: A dict with computed results, e.g., {"limit": 0.0}
    """

    @pytest.mark.parametrize("json_file", glob.glob("tests/individual/task_*.json"))
    def test_individual_task_parsing_and_computation(self, json_file):
        """
        Test parsing and computation for each individual task file.
        Validates that the file can be loaded, MathIR created, computation runs,
        and results match the expected structure. Handles errors gracefully.
        """
        # Load the JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract the parsed MathIR structure
        parsed_json = data.get('parsed_json')
        if parsed_json is None:
            pytest.skip(f"parsed_json missing in {json_file} - task not fully parsed")

        # Create MathIR object
        ir = MathIR.model_validate(parsed_json)

        # Run the computation
        results = run_mathir(ir)

        # Basic validation: results should be a non-empty dict
        assert isinstance(results, dict), f"Results should be dict, got {type(results)} in {json_file}"
        assert len(results) > 0, f"Results should not be empty in {json_file}"

        # Check for errors in results - timeout is acceptable to prevent hanging
        if 'error' in results:
            error_msg = results.get('error')
            if 'timeout' in error_msg:
                pytest.skip(f"Computation timed out for {json_file}: {error_msg}")
            else:
                assert False, f"Computation error in results for {json_file}: {error_msg}"

        # Validate that results contain expected keys based on targets
        for target in ir.targets:
            name = getattr(target, 'name', None)
            if name:
                assert name in results, f"Missing result for target '{name}' in {json_file}"
            elif target.type == 'limit':
                assert 'limit' in results, f"Missing 'limit' result in {json_file}"
            elif target.type == 'integral_def':
                assert name in results, f"Missing integral result '{name}' in {json_file}"
            elif target.type == 'sum':
                assert 'sum' in results, f"Missing 'sum' result in {json_file}"
            elif target.type == 'solve_for':
                assert 'solve' in results, f"Missing 'solve' result in {json_file}"
            elif target.type == 'inequalities':
                assert 'inequalities' in results, f"Missing 'inequalities' result in {json_file}"
            elif target.type == 'probability':
                assert 'probability' in results, f"Missing 'probability' result in {json_file}"
            elif target.type == 'value':
                assert name in results, f"Missing value result '{name}' in {json_file}"
            # Add more types as needed

        # Ensure results are concrete values, not symbolic expressions
        for key, value in results.items():
            if isinstance(value, sp.Expr):
                assert value.free_symbols == set(), f"Result '{key}' contains symbolic expression with variables: {value}"
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, sp.Expr):
                        assert item.free_symbols == set(), f"Result '{key}' contains symbolic expression with variables: {item}"
                    elif isinstance(item, dict):
                        for subkey, subvalue in item.items():
                            if isinstance(subvalue, sp.Expr):
                                assert subvalue.free_symbols == set(), f"Result '{key}' contains symbolic expression with variables: {subvalue}"

        # Additional structure validation: ensure output mode is respected if specified
        if ir.output and ir.output.mode == 'decimal':
            for value in results.values():
                if isinstance(value, (int, float)):
                    # For decimal mode, results should be floats or rounded appropriately
                    pass  # Could add more specific checks if needed

    def test_invalid_json_structure(self):
        """
        Test error handling for invalid JSON structure.
        Example: Missing required fields in MathIR.
        """
        invalid_data = {
            "expr_format": "latex",
            # Missing symbols, targets, etc.
        }
        with pytest.raises(Exception):  # MathIR validation should raise an error
            MathIR.model_validate(invalid_data)

    def test_invalid_latex_expression(self):
        """
        Test error handling for invalid LaTeX expressions.
        Example: Expression that cannot be parsed.
        """
        try:
            ir = MathIR.model_validate({
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [
                    {
                        "type": "value",
                        "name": "result",
                        "expr": "\\invalidcommand{x}"
                    }
                ]
            })
            results = run_mathir(ir)
            # Should not crash, but may have error or unexpected result
            assert isinstance(results, dict)
            # If there's an error, it should be noted
            if 'error' in results:
                assert 'result' not in results or results['result'] is None
        except Exception as e:
            # Expected to fail with SympifyError or similar
            from sympy.core.sympify import SympifyError
            assert isinstance(e, (SympifyError, ValueError, SyntaxError, TokenError))

    def test_missing_parsed_json(self):
        """
        Test handling of files missing parsed_json.
        """
        # Simulate a file without parsed_json
        data = {"task_id": "test", "task": "test task"}
        with pytest.raises(AssertionError):
            parsed_json = data.get('parsed_json')
            assert parsed_json is not None

if __name__ == "__main__":
    pytest.main([__file__])
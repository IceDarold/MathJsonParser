"""
Common test helper functions for MathIR Parser testing.

This module provides utility functions for loading test data, creating MathIR objects,
and performing common test operations across different test suites.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from mathir_parser.main import MathIR, run_mathir
import sympy as sp


class TestDataLoader:
    """Utility class for loading test data from various sources."""
    
    @staticmethod
    def load_json_file(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON data from a file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Dictionary containing the loaded JSON data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Test data file not found: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_mathir_from_file(filepath: Union[str, Path]) -> MathIR:
        """
        Load a MathIR object from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing MathIR data
            
        Returns:
            MathIR object created from the file data
        """
        data = TestDataLoader.load_json_file(filepath)
        
        # Handle files with nested parsed_json structure
        if 'parsed_json' in data:
            data = data['parsed_json']
            
        return MathIR.model_validate(data)
    
    @staticmethod
    def load_test_cases_from_directory(directory: Union[str, Path], 
                                     pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Load all test cases from a directory matching a pattern.
        
        Args:
            directory: Directory containing test files
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            List of dictionaries containing test case data
        """
        directory = Path(directory)
        test_cases = []
        
        for filepath in directory.glob(pattern):
            try:
                test_case = TestDataLoader.load_json_file(filepath)
                test_case['_source_file'] = str(filepath)
                test_cases.append(test_case)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Failed to load test case from {filepath}: {e}")
                
        return test_cases


class MathIRTestHelper:
    """Helper class for creating and manipulating MathIR objects in tests."""
    
    @staticmethod
    def create_simple_mathir(expr: str, 
                           target_type: str = "value",
                           target_name: str = "result",
                           output_mode: str = "exact",
                           symbols: Optional[List[Dict[str, str]]] = None) -> MathIR:
        """
        Create a simple MathIR object for testing.
        
        Args:
            expr: Mathematical expression
            target_type: Type of target (default: "value")
            target_name: Name of the target (default: "result")
            output_mode: Output mode (default: "exact")
            symbols: List of symbol definitions
            
        Returns:
            MathIR object
        """
        if symbols is None:
            symbols = [{"name": "x", "domain": "R"}]
            
        target = {
            "type": target_type,
            "name": target_name,
            "expr": expr
        }
        
        return MathIR(
            expr_format="latex",
            symbols=symbols,
            targets=[target],
            output={"mode": output_mode}
        )
    
    @staticmethod
    def create_integral_mathir(expr: str,
                             var: str,
                             limits: List[Union[str, int, float]],
                             name: str = "integral",
                             output_mode: str = "exact") -> MathIR:
        """
        Create a MathIR object for integral computation.
        
        Args:
            expr: Expression to integrate
            var: Variable of integration
            limits: Integration limits [lower, upper]
            name: Name for the result
            output_mode: Output mode
            
        Returns:
            MathIR object for integral computation
        """
        return MathIR(
            expr_format="latex",
            symbols=[{"name": var, "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": expr,
                "var": var,
                "limits": limits,
                "name": name
            }],
            output={"mode": output_mode}
        )
    
    @staticmethod
    def create_limit_mathir(expr: str,
                          var: str,
                          to: str,
                          output_mode: str = "exact") -> MathIR:
        """
        Create a MathIR object for limit computation.
        
        Args:
            expr: Expression for limit
            var: Variable approaching the limit
            to: Value the variable approaches
            output_mode: Output mode
            
        Returns:
            MathIR object for limit computation
        """
        return MathIR(
            expr_format="latex",
            symbols=[{"name": var, "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": expr,
                "var": var,
                "to": to
            }],
            output={"mode": output_mode}
        )


class TestResultValidator:
    """Utility class for validating test results."""
    
    @staticmethod
    def validate_result_structure(results: Dict[str, Any], 
                                expected_keys: List[str]) -> bool:
        """
        Validate that results contain expected keys.
        
        Args:
            results: Results dictionary to validate
            expected_keys: List of keys that should be present
            
        Returns:
            True if all expected keys are present
        """
        return all(key in results for key in expected_keys)
    
    @staticmethod
    def validate_no_errors(results: Dict[str, Any]) -> bool:
        """
        Validate that results don't contain errors.
        
        Args:
            results: Results dictionary to validate
            
        Returns:
            True if no errors are present
        """
        return 'error' not in results
    
    @staticmethod
    def validate_symbolic_results(results: Dict[str, Any]) -> bool:
        """
        Validate that symbolic results don't contain free variables.
        
        Args:
            results: Results dictionary to validate
            
        Returns:
            True if all symbolic expressions are fully evaluated
        """
        for key, value in results.items():
            if isinstance(value, sp.Expr):
                if value.free_symbols:
                    return False
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, sp.Expr) and item.free_symbols:
                        return False
                    elif isinstance(item, dict):
                        for subvalue in item.values():
                            if isinstance(subvalue, sp.Expr) and subvalue.free_symbols:
                                return False
        return True


def run_mathir_test(mathir: MathIR) -> Dict[str, Any]:
    """
    Run a MathIR computation and return results with error handling.
    
    Args:
        mathir: MathIR object to execute
        
    Returns:
        Dictionary containing results or error information
    """
    try:
        return run_mathir(mathir)
    except Exception as e:
        return {"error": str(e), "exception_type": type(e).__name__}


def get_test_data_path(relative_path: str) -> Path:
    """
    Get the absolute path to test data files.
    
    Args:
        relative_path: Relative path from the tests directory
        
    Returns:
        Absolute path to the test data file
    """
    tests_dir = Path(__file__).parent.parent
    return tests_dir / relative_path


def compare_test_results(actual: Dict[str, Any], 
                        expected: Dict[str, Any],
                        tolerance: float = 1e-10) -> bool:
    """
    Compare actual and expected test results with tolerance for numerical values.
    
    Args:
        actual: Actual results from computation
        expected: Expected results
        tolerance: Numerical tolerance for floating-point comparisons
        
    Returns:
        True if results match within tolerance
    """
    if set(actual.keys()) != set(expected.keys()):
        return False
        
    for key in actual.keys():
        actual_val = actual[key]
        expected_val = expected[key]
        
        # Handle numerical comparisons
        if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)):
            if abs(actual_val - expected_val) > tolerance:
                return False
        # Handle SymPy expressions
        elif isinstance(actual_val, sp.Expr) and isinstance(expected_val, sp.Expr):
            if not sp.simplify(actual_val - expected_val) == 0:
                return False
        # Handle exact equality for other types
        elif actual_val != expected_val:
            return False
            
    return True
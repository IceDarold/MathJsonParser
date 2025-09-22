"""
Mathematical assertion helpers with precision handling for MathIR Parser testing.

This module provides specialized assertion functions for mathematical computations,
handling both exact symbolic results and numerical approximations with appropriate
tolerance levels.
"""

import math
from typing import Any, Union, Optional, Dict, List
import sympy as sp
from decimal import Decimal, getcontext


class PrecisionAssertion:
    """Class for precision-aware mathematical assertions."""
    
    def __init__(self, default_tolerance: float = 1e-10):
        """
        Initialize precision assertion helper.
        
        Args:
            default_tolerance: Default tolerance for numerical comparisons
        """
        self.default_tolerance = default_tolerance
        
    def assert_numerical_equal(self, 
                             actual: Union[int, float, complex, Decimal], 
                             expected: Union[int, float, complex, Decimal],
                             tolerance: Optional[float] = None,
                             message: Optional[str] = None) -> None:
        """
        Assert that two numerical values are equal within tolerance.
        
        Args:
            actual: Actual numerical value
            expected: Expected numerical value
            tolerance: Tolerance for comparison (uses default if None)
            message: Custom error message
            
        Raises:
            AssertionError: If values are not equal within tolerance
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        if isinstance(actual, complex) or isinstance(expected, complex):
            diff = abs(complex(actual) - complex(expected))
        else:
            diff = abs(float(actual) - float(expected))
            
        if diff > tolerance:
            error_msg = f"Values not equal within tolerance {tolerance}: {actual} != {expected} (diff: {diff})"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    def assert_symbolic_equal(self,
                            actual: sp.Expr,
                            expected: sp.Expr,
                            simplify: bool = True,
                            message: Optional[str] = None) -> None:
        """
        Assert that two SymPy expressions are mathematically equal.
        
        Args:
            actual: Actual SymPy expression
            expected: Expected SymPy expression
            simplify: Whether to simplify the difference before checking
            message: Custom error message
            
        Raises:
            AssertionError: If expressions are not mathematically equal
        """
        try:
            # Handle special cases first
            if actual == expected:
                return
            
            # Handle infinity cases
            if actual == sp.oo and expected == sp.oo:
                return
            if actual == -sp.oo and expected == -sp.oo:
                return
            if str(actual) == 'zoo' and str(expected) == 'zoo':
                return
            if actual == sp.nan and expected == sp.nan:
                return
            
            # Try symbolic comparison
            diff = actual - expected
            if simplify:
                diff = sp.simplify(diff)
                
            if diff == 0:
                return
                
            # If difference is not zero, check if they're equivalent
            if sp.simplify(actual.equals(expected)) is True:
                return
                
            error_msg = f"Symbolic expressions not equal: {actual} != {expected}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
            
        except Exception as e:
            # For special values that can't be subtracted, try direct comparison
            if str(actual) == str(expected):
                return
            error_msg = f"Error comparing symbolic expressions: {actual} != {expected} ({e})"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
    
    def assert_matrix_equal(self, 
                          actual: sp.Matrix, 
                          expected: sp.Matrix,
                          tolerance: Optional[float] = None,
                          message: Optional[str] = None) -> None:
        """
        Assert that two matrices are equal within tolerance.
        
        Args:
            actual: Actual matrix
            expected: Expected matrix
            tolerance: Tolerance for numerical comparisons
            message: Custom error message
            
        Raises:
            AssertionError: If matrices are not equal
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        if actual.shape != expected.shape:
            error_msg = f"Matrix shapes don't match: {actual.shape} != {expected.shape}"
            if message:
                error_msg = f"{message}: {error_msg}"
            raise AssertionError(error_msg)
        
        for i in range(actual.rows):
            for j in range(actual.cols):
                actual_elem = actual[i, j]
                expected_elem = expected[i, j]
                
                try:
                    if isinstance(actual_elem, (int, float)) and isinstance(expected_elem, (int, float)):
                        self.assert_numerical_equal(actual_elem, expected_elem, tolerance)
                    else:
                        self.assert_symbolic_equal(actual_elem, expected_elem)
                except AssertionError as e:
                    error_msg = f"Matrix element [{i},{j}] mismatch: {e}"
                    if message:
                        error_msg = f"{message}: {error_msg}"
                    raise AssertionError(error_msg)


class MathematicalValidator:
    """Validator class for different types of mathematical operations."""
    
    def __init__(self, precision_assertion: Optional[PrecisionAssertion] = None):
        """
        Initialize mathematical validator.
        
        Args:
            precision_assertion: PrecisionAssertion instance to use
        """
        self.precision = precision_assertion or PrecisionAssertion()
    
    def validate_integral_result(self, 
                               result: Any, 
                               expected: Any,
                               tolerance: Optional[float] = None) -> None:
        """
        Validate integral computation result.
        
        Args:
            result: Computed integral result
            expected: Expected integral result
            tolerance: Tolerance for numerical comparison
        """
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            self.precision.assert_numerical_equal(result, expected, tolerance, 
                                                "Integral result mismatch")
        elif isinstance(result, sp.Expr) and isinstance(expected, sp.Expr):
            self.precision.assert_symbolic_equal(result, expected, 
                                               message="Integral result mismatch")
        else:
            # Try to convert and compare numerically
            try:
                result_float = float(result)
                expected_float = float(expected)
                self.precision.assert_numerical_equal(result_float, expected_float, tolerance,
                                                    "Integral result mismatch")
            except (ValueError, TypeError):
                if result != expected:
                    raise AssertionError(f"Integral result mismatch: {result} != {expected}")
    
    def validate_limit_result(self, 
                            result: Any, 
                            expected: Any,
                            tolerance: Optional[float] = None) -> None:
        """
        Validate limit computation result.
        
        Args:
            result: Computed limit result
            expected: Expected limit result
            tolerance: Tolerance for numerical comparison
        """
        # Handle special limit values
        if result == sp.oo and expected == sp.oo:
            return
        if result == -sp.oo and expected == -sp.oo:
            return
        if result == sp.nan and expected == sp.nan:
            return
        if str(result) == 'zoo' and str(expected) == 'zoo':  # Complex infinity
            return
            
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            self.precision.assert_numerical_equal(result, expected, tolerance,
                                                "Limit result mismatch")
        elif isinstance(result, sp.Expr) and isinstance(expected, sp.Expr):
            self.precision.assert_symbolic_equal(result, expected,
                                               message="Limit result mismatch")
        else:
            if result != expected:
                raise AssertionError(f"Limit result mismatch: {result} != {expected}")
    
    def validate_sum_result(self, 
                          result: Any, 
                          expected: Any,
                          tolerance: Optional[float] = None) -> None:
        """
        Validate summation result.
        
        Args:
            result: Computed sum result
            expected: Expected sum result
            tolerance: Tolerance for numerical comparison
        """
        if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
            self.precision.assert_numerical_equal(result, expected, tolerance,
                                                "Sum result mismatch")
        elif isinstance(result, sp.Expr) and isinstance(expected, sp.Expr):
            self.precision.assert_symbolic_equal(result, expected,
                                               message="Sum result mismatch")
        else:
            try:
                result_float = float(result)
                expected_float = float(expected)
                self.precision.assert_numerical_equal(result_float, expected_float, tolerance,
                                                    "Sum result mismatch")
            except (ValueError, TypeError):
                if result != expected:
                    raise AssertionError(f"Sum result mismatch: {result} != {expected}")
    
    def validate_solve_result(self, 
                            result: List[Dict[sp.Symbol, Any]], 
                            expected: List[Dict[sp.Symbol, Any]],
                            tolerance: Optional[float] = None) -> None:
        """
        Validate equation solving result.
        
        Args:
            result: Computed solution set
            expected: Expected solution set
            tolerance: Tolerance for numerical comparison
        """
        if len(result) != len(expected):
            raise AssertionError(f"Solution count mismatch: {len(result)} != {len(expected)}")
        
        # Sort solutions for comparison (if possible)
        try:
            result_sorted = sorted(result, key=lambda x: str(sorted(x.items())))
            expected_sorted = sorted(expected, key=lambda x: str(sorted(x.items())))
        except:
            result_sorted = result
            expected_sorted = expected
        
        for i, (actual_sol, expected_sol) in enumerate(zip(result_sorted, expected_sorted)):
            if set(actual_sol.keys()) != set(expected_sol.keys()):
                raise AssertionError(f"Solution {i} variables mismatch: "
                                   f"{set(actual_sol.keys())} != {set(expected_sol.keys())}")
            
            for var in actual_sol.keys():
                actual_val = actual_sol[var]
                expected_val = expected_sol[var]
                
                if isinstance(actual_val, (int, float)) and isinstance(expected_val, (int, float)):
                    self.precision.assert_numerical_equal(actual_val, expected_val, tolerance,
                                                        f"Solution {i} variable {var} mismatch")
                elif isinstance(actual_val, sp.Expr) and isinstance(expected_val, sp.Expr):
                    self.precision.assert_symbolic_equal(actual_val, expected_val,
                                                       message=f"Solution {i} variable {var} mismatch")
                else:
                    if actual_val != expected_val:
                        raise AssertionError(f"Solution {i} variable {var} mismatch: "
                                           f"{actual_val} != {expected_val}")
    
    def validate_inequality_result(self, 
                                 result: Any, 
                                 expected: Any) -> None:
        """
        Validate inequality solving result.
        
        Args:
            result: Computed inequality result
            expected: Expected inequality result
        """
        # Inequality results can be complex, so we use symbolic comparison
        if isinstance(result, sp.Expr) and isinstance(expected, sp.Expr):
            # Try to simplify and compare
            try:
                simplified_diff = sp.simplify(result.equals(expected))
                if simplified_diff is not True:
                    # Try alternative comparison methods
                    if not self._compare_inequalities(result, expected):
                        raise AssertionError(f"Inequality result mismatch: {result} != {expected}")
            except:
                if str(result) != str(expected):
                    raise AssertionError(f"Inequality result mismatch: {result} != {expected}")
        else:
            if result != expected:
                raise AssertionError(f"Inequality result mismatch: {result} != {expected}")
    
    def _compare_inequalities(self, result: sp.Expr, expected: sp.Expr) -> bool:
        """
        Helper method to compare inequality expressions.
        
        Args:
            result: Computed inequality
            expected: Expected inequality
            
        Returns:
            True if inequalities are equivalent
        """
        try:
            # Convert to sets and compare
            if hasattr(result, 'as_set') and hasattr(expected, 'as_set'):
                return result.as_set() == expected.as_set()
            
            # Try string comparison as fallback
            return str(sp.simplify(result)) == str(sp.simplify(expected))
        except:
            return False


# Convenience functions for common assertions
def assert_integral_equal(actual: Any, expected: Any, tolerance: float = 1e-10) -> None:
    """Assert that integral results are equal."""
    validator = MathematicalValidator()
    validator.validate_integral_result(actual, expected, tolerance)


def assert_limit_equal(actual: Any, expected: Any, tolerance: float = 1e-10) -> None:
    """Assert that limit results are equal."""
    validator = MathematicalValidator()
    validator.validate_limit_result(actual, expected, tolerance)


def assert_sum_equal(actual: Any, expected: Any, tolerance: float = 1e-10) -> None:
    """Assert that sum results are equal."""
    validator = MathematicalValidator()
    validator.validate_sum_result(actual, expected, tolerance)


def assert_solve_equal(actual: List[Dict[sp.Symbol, Any]], 
                      expected: List[Dict[sp.Symbol, Any]], 
                      tolerance: float = 1e-10) -> None:
    """Assert that solve results are equal."""
    validator = MathematicalValidator()
    validator.validate_solve_result(actual, expected, tolerance)


def assert_inequality_equal(actual: Any, expected: Any) -> None:
    """Assert that inequality results are equal."""
    validator = MathematicalValidator()
    validator.validate_inequality_result(actual, expected)


def assert_numerical_close(actual: Union[int, float, complex], 
                          expected: Union[int, float, complex],
                          tolerance: float = 1e-10) -> None:
    """Assert that numerical values are close within tolerance."""
    precision = PrecisionAssertion()
    precision.assert_numerical_equal(actual, expected, tolerance)


def assert_symbolic_equivalent(actual: sp.Expr, expected: sp.Expr) -> None:
    """Assert that symbolic expressions are mathematically equivalent."""
    precision = PrecisionAssertion()
    precision.assert_symbolic_equal(actual, expected)
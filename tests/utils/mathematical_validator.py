"""
Comprehensive mathematical validation framework for MathIR Parser testing.

This module provides a complete validation system for mathematical computations,
supporting exact symbolic results, numerical approximations, and specialized
validation for different mathematical operation types.
"""

import math
from typing import Any, Union, Optional, Dict, List, Tuple, Callable
from decimal import Decimal, getcontext
from enum import Enum
import sympy as sp
from sympy import oo, zoo, nan, I, E, pi
import numpy as np


class ValidationMode(Enum):
    """Enumeration of validation modes."""
    EXACT = "exact"
    NUMERICAL = "numerical"
    SYMBOLIC = "symbolic"
    MIXED = "mixed"


class ToleranceLevel(Enum):
    """Enumeration of tolerance levels for numerical validation."""
    STRICT = 1e-12
    NORMAL = 1e-10
    RELAXED = 1e-8
    LOOSE = 1e-6


class MathematicalAccuracyValidator:
    """Validator for mathematical accuracy across different operation types."""
    
    def __init__(self, 
                 default_tolerance: float = ToleranceLevel.NORMAL.value,
                 validation_mode: ValidationMode = ValidationMode.MIXED):
        """
        Initialize mathematical accuracy validator.
        
        Args:
            default_tolerance: Default numerical tolerance
            validation_mode: Default validation mode
        """
        self.default_tolerance = default_tolerance
        self.validation_mode = validation_mode
        self.special_values = {
            'infinity': [oo, float('inf'), np.inf],
            'negative_infinity': [-oo, float('-inf'), -np.inf],
            'nan': [nan, float('nan'), np.nan],
            'complex_infinity': [zoo],
            'imaginary_unit': [I, 1j, complex(0, 1)],
            'euler': [E, math.e, np.e],
            'pi': [pi, math.pi, np.pi]
        }
    
    def validate_integral_result(self, 
                               actual: Any, 
                               expected: Any,
                               tolerance: Optional[float] = None,
                               integration_bounds: Optional[Tuple[Any, Any]] = None) -> bool:
        """
        Validate integral computation results with specialized handling.
        
        Args:
            actual: Computed integral result
            expected: Expected integral result
            tolerance: Numerical tolerance (uses default if None)
            integration_bounds: Integration bounds for context
            
        Returns:
            True if validation passes
            
        Raises:
            AssertionError: If validation fails
        """
        tolerance = tolerance or self.default_tolerance
        
        try:
            # Handle definite integrals
            if integration_bounds is not None:
                return self._validate_definite_integral(actual, expected, tolerance, integration_bounds)
            
            # Handle indefinite integrals (symbolic expressions)
            if isinstance(actual, sp.Expr) and isinstance(expected, sp.Expr):
                return self._validate_symbolic_integral(actual, expected)
            
            # Handle numerical results
            if self._is_numerical(actual) and self._is_numerical(expected):
                return self._validate_numerical_equality(actual, expected, tolerance)
            
            # Handle special cases
            if self._is_special_value(actual) or self._is_special_value(expected):
                return self._validate_special_values(actual, expected)
            
            # Fallback to exact comparison
            return self._validate_exact_equality(actual, expected)
            
        except Exception as e:
            raise AssertionError(f"Integral validation failed: {e}")
    
    def validate_limit_result(self, 
                            actual: Any, 
                            expected: Any,
                            limit_point: Any = None,
                            tolerance: Optional[float] = None) -> bool:
        """
        Validate limit computation results.
        
        Args:
            actual: Computed limit result
            expected: Expected limit result
            limit_point: Point where limit is taken
            tolerance: Numerical tolerance
            
        Returns:
            True if validation passes
        """
        tolerance = tolerance or self.default_tolerance
        
        try:
            # Handle limits to infinity
            if self._represents_infinity(limit_point):
                return self._validate_limit_to_infinity(actual, expected, tolerance)
            
            # Handle limits to finite points
            if self._is_numerical(limit_point):
                return self._validate_limit_to_point(actual, expected, limit_point, tolerance)
            
            # Handle special limit cases (0/0, ∞/∞, etc.)
            if self._is_indeterminate_form(actual, expected):
                return self._validate_indeterminate_limit(actual, expected, tolerance)
            
            # General validation
            return self._validate_general_result(actual, expected, tolerance)
            
        except Exception as e:
            raise AssertionError(f"Limit validation failed: {e}")
    
    def validate_sum_result(self, 
                          actual: Any, 
                          expected: Any,
                          sum_bounds: Optional[Tuple[Any, Any]] = None,
                          tolerance: Optional[float] = None) -> bool:
        """
        Validate summation results.
        
        Args:
            actual: Computed sum result
            expected: Expected sum result
            sum_bounds: Summation bounds (start, end)
            tolerance: Numerical tolerance
            
        Returns:
            True if validation passes
        """
        tolerance = tolerance or self.default_tolerance
        
        try:
            # Handle finite sums
            if sum_bounds and self._is_finite_sum(sum_bounds):
                return self._validate_finite_sum(actual, expected, tolerance)
            
            # Handle infinite series
            if sum_bounds and self._is_infinite_sum(sum_bounds):
                return self._validate_infinite_series(actual, expected, tolerance)
            
            # Handle convergence tests
            if self._requires_convergence_test(actual, expected):
                return self._validate_series_convergence(actual, expected, tolerance)
            
            # General numerical validation
            return self._validate_numerical_equality(actual, expected, tolerance)
            
        except Exception as e:
            raise AssertionError(f"Sum validation failed: {e}")
    
    def validate_solve_result(self, 
                            actual: List[Dict[sp.Symbol, Any]], 
                            expected: List[Dict[sp.Symbol, Any]],
                            equation_type: str = "algebraic",
                            tolerance: Optional[float] = None) -> bool:
        """
        Validate equation solving results.
        
        Args:
            actual: Computed solution set
            expected: Expected solution set
            equation_type: Type of equation (algebraic, transcendental, etc.)
            tolerance: Numerical tolerance
            
        Returns:
            True if validation passes
        """
        tolerance = tolerance or self.default_tolerance
        
        try:
            # Validate solution count
            if len(actual) != len(expected):
                raise AssertionError(f"Solution count mismatch: {len(actual)} != {len(expected)}")
            
            # Handle different equation types
            if equation_type == "algebraic":
                return self._validate_algebraic_solutions(actual, expected, tolerance)
            elif equation_type == "transcendental":
                return self._validate_transcendental_solutions(actual, expected, tolerance)
            elif equation_type == "differential":
                return self._validate_differential_solutions(actual, expected, tolerance)
            else:
                return self._validate_general_solutions(actual, expected, tolerance)
                
        except Exception as e:
            raise AssertionError(f"Solve validation failed: {e}")
    
    def validate_matrix_result(self, 
                             actual: sp.Matrix, 
                             expected: sp.Matrix,
                             operation_type: str = "general",
                             tolerance: Optional[float] = None) -> bool:
        """
        Validate matrix computation results.
        
        Args:
            actual: Computed matrix result
            expected: Expected matrix result
            operation_type: Type of matrix operation
            tolerance: Numerical tolerance
            
        Returns:
            True if validation passes
        """
        tolerance = tolerance or self.default_tolerance
        
        try:
            # Validate matrix dimensions
            if actual.shape != expected.shape:
                raise AssertionError(f"Matrix shape mismatch: {actual.shape} != {expected.shape}")
            
            # Element-wise validation
            for i in range(actual.rows):
                for j in range(actual.cols):
                    actual_elem = actual[i, j]
                    expected_elem = expected[i, j]
                    
                    if not self._validate_general_result(actual_elem, expected_elem, tolerance):
                        raise AssertionError(f"Matrix element [{i},{j}] mismatch: {actual_elem} != {expected_elem}")
            
            # Special validations for specific operations
            if operation_type == "eigenvalues":
                return self._validate_eigenvalues(actual, expected, tolerance)
            elif operation_type == "determinant":
                return self._validate_determinant(actual, expected, tolerance)
            elif operation_type == "inverse":
                return self._validate_matrix_inverse(actual, expected, tolerance)
            
            return True
            
        except Exception as e:
            raise AssertionError(f"Matrix validation failed: {e}")
    
    # Helper methods for validation
    def _validate_definite_integral(self, actual: Any, expected: Any, 
                                  tolerance: float, bounds: Tuple[Any, Any]) -> bool:
        """Validate definite integral results."""
        # Check for improper integrals
        if self._is_improper_integral(bounds):
            return self._validate_improper_integral(actual, expected, tolerance)
        
        # Standard definite integral validation
        return self._validate_numerical_equality(actual, expected, tolerance)
    
    def _validate_symbolic_integral(self, actual: sp.Expr, expected: sp.Expr) -> bool:
        """Validate symbolic integral results."""
        # Check if derivatives are equal (fundamental theorem of calculus)
        try:
            actual_derivative = sp.diff(actual, sp.Symbol('x'))
            expected_derivative = sp.diff(expected, sp.Symbol('x'))
            return sp.simplify(actual_derivative - expected_derivative) == 0
        except:
            # Fallback to direct symbolic comparison
            return sp.simplify(actual - expected) == 0
    
    def _validate_numerical_equality(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate numerical equality with tolerance."""
        try:
            actual_float = complex(actual) if isinstance(actual, complex) else float(actual)
            expected_float = complex(expected) if isinstance(expected, complex) else float(expected)
            
            if isinstance(actual_float, complex) or isinstance(expected_float, complex):
                return abs(actual_float - expected_float) <= tolerance
            else:
                return abs(actual_float - expected_float) <= tolerance
        except (ValueError, TypeError, OverflowError):
            return False
    
    def _validate_special_values(self, actual: Any, expected: Any) -> bool:
        """Validate special mathematical values."""
        # Check if both are the same type of special value
        for value_type, values in self.special_values.items():
            if self._is_value_type(actual, values) and self._is_value_type(expected, values):
                return True
        
        return False
    
    def _validate_exact_equality(self, actual: Any, expected: Any) -> bool:
        """Validate exact equality."""
        return actual == expected
    
    def _validate_general_result(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """General result validation with multiple strategies."""
        # Try numerical validation first
        if self._is_numerical(actual) and self._is_numerical(expected):
            return self._validate_numerical_equality(actual, expected, tolerance)
        
        # Try symbolic validation
        if isinstance(actual, sp.Expr) and isinstance(expected, sp.Expr):
            try:
                return sp.simplify(actual - expected) == 0
            except:
                pass
        
        # Try special values
        if self._is_special_value(actual) or self._is_special_value(expected):
            return self._validate_special_values(actual, expected)
        
        # Fallback to exact equality
        return self._validate_exact_equality(actual, expected)
    
    # Utility methods
    def _is_numerical(self, value: Any) -> bool:
        """Check if value is numerical."""
        return isinstance(value, (int, float, complex, Decimal, np.number))
    
    def _is_special_value(self, value: Any) -> bool:
        """Check if value is a special mathematical value."""
        for values in self.special_values.values():
            if self._is_value_type(value, values):
                return True
        return False
    
    def _is_value_type(self, value: Any, value_list: List[Any]) -> bool:
        """Check if value matches any in the value list."""
        for v in value_list:
            try:
                if value == v or str(value) == str(v):
                    return True
                # Handle NaN specially
                if str(v) == 'nan' and (math.isnan(float(value)) if self._is_numerical(value) else False):
                    return True
            except:
                continue
        return False
    
    def _represents_infinity(self, value: Any) -> bool:
        """Check if value represents infinity."""
        return self._is_value_type(value, self.special_values['infinity'] + 
                                 self.special_values['negative_infinity'])
    
    def _is_finite_sum(self, bounds: Tuple[Any, Any]) -> bool:
        """Check if summation bounds are finite."""
        start, end = bounds
        return (self._is_numerical(start) and self._is_numerical(end) and 
                not self._represents_infinity(start) and not self._represents_infinity(end))
    
    def _is_infinite_sum(self, bounds: Tuple[Any, Any]) -> bool:
        """Check if summation involves infinity."""
        start, end = bounds
        return self._represents_infinity(start) or self._represents_infinity(end)
    
    def _is_improper_integral(self, bounds: Tuple[Any, Any]) -> bool:
        """Check if integral is improper."""
        start, end = bounds
        return self._represents_infinity(start) or self._represents_infinity(end)
    
    def _is_indeterminate_form(self, actual: Any, expected: Any) -> bool:
        """Check if result involves indeterminate forms."""
        indeterminate_patterns = ['0/0', 'inf/inf', '0*inf', 'inf-inf', '1^inf', '0^0', 'inf^0']
        actual_str = str(actual).replace(' ', '')
        expected_str = str(expected).replace(' ', '')
        
        return any(pattern in actual_str or pattern in expected_str 
                  for pattern in indeterminate_patterns)
    
    def _requires_convergence_test(self, actual: Any, expected: Any) -> bool:
        """Check if series convergence testing is required."""
        # This would be implemented based on specific series patterns
        return False  # Placeholder
    
    # Specialized validation methods (placeholders for full implementation)
    def _validate_limit_to_infinity(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate limits approaching infinity."""
        return self._validate_general_result(actual, expected, tolerance)
    
    def _validate_limit_to_point(self, actual: Any, expected: Any, point: Any, tolerance: float) -> bool:
        """Validate limits approaching a finite point."""
        return self._validate_general_result(actual, expected, tolerance)
    
    def _validate_indeterminate_limit(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate indeterminate form limits."""
        return self._validate_general_result(actual, expected, tolerance)
    
    def _validate_finite_sum(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate finite summation results."""
        return self._validate_numerical_equality(actual, expected, tolerance)
    
    def _validate_infinite_series(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate infinite series results."""
        return self._validate_general_result(actual, expected, tolerance)
    
    def _validate_series_convergence(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate series convergence."""
        return self._validate_general_result(actual, expected, tolerance)
    
    def _validate_algebraic_solutions(self, actual: List[Dict], expected: List[Dict], tolerance: float) -> bool:
        """Validate algebraic equation solutions."""
        return self._validate_solution_sets(actual, expected, tolerance)
    
    def _validate_transcendental_solutions(self, actual: List[Dict], expected: List[Dict], tolerance: float) -> bool:
        """Validate transcendental equation solutions."""
        return self._validate_solution_sets(actual, expected, tolerance)
    
    def _validate_differential_solutions(self, actual: List[Dict], expected: List[Dict], tolerance: float) -> bool:
        """Validate differential equation solutions."""
        return self._validate_solution_sets(actual, expected, tolerance)
    
    def _validate_general_solutions(self, actual: List[Dict], expected: List[Dict], tolerance: float) -> bool:
        """Validate general equation solutions."""
        return self._validate_solution_sets(actual, expected, tolerance)
    
    def _validate_solution_sets(self, actual: List[Dict], expected: List[Dict], tolerance: float) -> bool:
        """Validate solution sets with tolerance."""
        if len(actual) != len(expected):
            return False
        
        # Sort solutions for comparison
        try:
            actual_sorted = sorted(actual, key=lambda x: str(sorted(x.items())))
            expected_sorted = sorted(expected, key=lambda x: str(sorted(x.items())))
        except:
            actual_sorted = actual
            expected_sorted = expected
        
        for actual_sol, expected_sol in zip(actual_sorted, expected_sorted):
            if set(actual_sol.keys()) != set(expected_sol.keys()):
                return False
            
            for var in actual_sol.keys():
                if not self._validate_general_result(actual_sol[var], expected_sol[var], tolerance):
                    return False
        
        return True
    
    def _validate_improper_integral(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate improper integral results."""
        return self._validate_general_result(actual, expected, tolerance)
    
    def _validate_eigenvalues(self, actual: sp.Matrix, expected: sp.Matrix, tolerance: float) -> bool:
        """Validate matrix eigenvalues."""
        # This would implement eigenvalue-specific validation
        return True  # Placeholder
    
    def _validate_determinant(self, actual: Any, expected: Any, tolerance: float) -> bool:
        """Validate matrix determinant."""
        return self._validate_numerical_equality(actual, expected, tolerance)
    
    def _validate_matrix_inverse(self, actual: sp.Matrix, expected: sp.Matrix, tolerance: float) -> bool:
        """Validate matrix inverse."""
        # Check if A * A^(-1) = I
        try:
            identity_check = actual * expected
            identity_matrix = sp.eye(actual.rows)
            
            for i in range(identity_check.rows):
                for j in range(identity_check.cols):
                    expected_val = 1 if i == j else 0
                    if not self._validate_numerical_equality(identity_check[i, j], expected_val, tolerance):
                        return False
            return True
        except:
            return False


# Convenience functions for common validations
def validate_mathematical_result(actual: Any, 
                               expected: Any, 
                               operation_type: str = "general",
                               tolerance: float = ToleranceLevel.NORMAL.value,
                               **kwargs) -> bool:
    """
    Validate mathematical computation results.
    
    Args:
        actual: Computed result
        expected: Expected result
        operation_type: Type of mathematical operation
        tolerance: Numerical tolerance
        **kwargs: Additional validation parameters
        
    Returns:
        True if validation passes
    """
    validator = MathematicalAccuracyValidator(default_tolerance=tolerance)
    
    if operation_type == "integral":
        return validator.validate_integral_result(actual, expected, tolerance, 
                                                kwargs.get('integration_bounds'))
    elif operation_type == "limit":
        return validator.validate_limit_result(actual, expected, 
                                             kwargs.get('limit_point'), tolerance)
    elif operation_type == "sum":
        return validator.validate_sum_result(actual, expected, 
                                           kwargs.get('sum_bounds'), tolerance)
    elif operation_type == "solve":
        return validator.validate_solve_result(actual, expected, 
                                             kwargs.get('equation_type', 'algebraic'), tolerance)
    elif operation_type == "matrix":
        return validator.validate_matrix_result(actual, expected, 
                                              kwargs.get('matrix_operation', 'general'), tolerance)
    else:
        return validator._validate_general_result(actual, expected, tolerance)
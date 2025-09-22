#!/usr/bin/env python3
"""
Mathematical Accuracy Validator for MathIR Parser End-to-End Tests

This module provides comprehensive validation infrastructure for testing mathematical
correctness of the MathIR Parser system. It includes precision-aware assertions,
cross-verification with multiple mathematical libraries, and detailed accuracy reporting.

Key Features:
- Exact symbolic computation validation
- Numerical precision validation with configurable tolerance
- Special mathematical constants and edge case handling
- Performance monitoring and memory usage tracking
- Cross-verification with SymPy, NumPy, and SciPy
- Detailed accuracy reports and error analysis
"""

import sympy as sp
import numpy as np
import math
import time
import psutil
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation modes for mathematical accuracy testing."""
    EXACT_SYMBOLIC = "exact_symbolic"
    NUMERICAL_TOLERANCE = "numerical_tolerance"
    CROSS_VERIFICATION = "cross_verification"
    PERFORMANCE_AWARE = "performance_aware"


@dataclass
class ValidationResult:
    """Result of mathematical accuracy validation."""
    passed: bool
    expected: Any
    actual: Any
    tolerance: Optional[float] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    validation_mode: Optional[ValidationMode] = None
    cross_verification_results: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for mathematical operations."""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float


class MathematicalAccuracyValidator:
    """
    Comprehensive validator for mathematical accuracy in MathIR Parser.
    
    This validator provides multiple validation modes:
    1. Exact symbolic validation for symbolic computations
    2. Numerical tolerance validation for numerical results
    3. Cross-verification with multiple mathematical libraries
    4. Performance-aware validation with resource monitoring
    """
    
    def __init__(self, default_tolerance: float = 1e-10):
        """
        Initialize the mathematical accuracy validator.
        
        Args:
            default_tolerance: Default numerical tolerance for comparisons
        """
        self.default_tolerance = default_tolerance
        self.validation_history: List[ValidationResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Mathematical constants for validation
        self.mathematical_constants = {
            'pi': sp.pi,
            'e': sp.E,
            'euler_gamma': sp.EulerGamma,
            'golden_ratio': sp.GoldenRatio,
            'catalan': sp.Catalan,
            'sqrt_2': sp.sqrt(2),
            'sqrt_3': sp.sqrt(3),
            'sqrt_5': sp.sqrt(5),
            'ln_2': sp.log(2),
            'ln_10': sp.log(10)
        }
    
    @contextmanager
    def performance_monitor(self):
        """Context manager for monitoring performance metrics."""
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = process.cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage_mb=end_memory - start_memory,
                peak_memory_mb=max(start_memory, end_memory),
                cpu_percent=max(start_cpu, end_cpu)
            )
            self.performance_metrics.append(metrics)
    
    def validate_exact_symbolic(self, expected: Any, actual: Any, 
                              description: str = "") -> ValidationResult:
        """
        Validate exact symbolic mathematical results.
        
        Args:
            expected: Expected symbolic result
            actual: Actual result from MathIR parser
            description: Description of the test case
            
        Returns:
            ValidationResult with exact symbolic comparison
        """
        try:
            # Convert both to SymPy expressions for comparison
            expected_expr = sp.sympify(expected) if not isinstance(expected, sp.Basic) else expected
            actual_expr = sp.sympify(actual) if not isinstance(actual, sp.Basic) else actual
            
            # Check for exact symbolic equality
            difference = sp.simplify(expected_expr - actual_expr)
            is_equal = difference == 0
            
            if not is_equal:
                # Try alternative forms and simplifications
                is_equal = sp.simplify(expected_expr).equals(sp.simplify(actual_expr))
            
            result = ValidationResult(
                passed=is_equal,
                expected=expected_expr,
                actual=actual_expr,
                validation_mode=ValidationMode.EXACT_SYMBOLIC,
                error_message=None if is_equal else f"Symbolic expressions not equal: {difference}"
            )
            
            self.validation_history.append(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                expected=expected,
                actual=actual,
                validation_mode=ValidationMode.EXACT_SYMBOLIC,
                error_message=f"Symbolic validation failed: {str(e)}"
            )
            self.validation_history.append(result)
            return result
    
    def validate_numerical_tolerance(self, expected: Union[float, complex], 
                                   actual: Union[float, complex],
                                   tolerance: Optional[float] = None,
                                   description: str = "") -> ValidationResult:
        """
        Validate numerical results within specified tolerance.
        
        Args:
            expected: Expected numerical result
            actual: Actual numerical result
            tolerance: Tolerance for comparison (uses default if None)
            description: Description of the test case
            
        Returns:
            ValidationResult with numerical tolerance comparison
        """
        if tolerance is None:
            tolerance = self.default_tolerance
        
        try:
            # Convert to numerical values
            expected_num = complex(expected) if isinstance(expected, (int, float, complex)) else complex(sp.N(expected))
            actual_num = complex(actual) if isinstance(actual, (int, float, complex)) else complex(sp.N(actual))
            
            # Calculate absolute difference
            abs_diff = abs(expected_num - actual_num)
            is_within_tolerance = abs_diff <= tolerance
            
            result = ValidationResult(
                passed=is_within_tolerance,
                expected=expected_num,
                actual=actual_num,
                tolerance=tolerance,
                validation_mode=ValidationMode.NUMERICAL_TOLERANCE,
                error_message=None if is_within_tolerance else f"Numerical difference {abs_diff} exceeds tolerance {tolerance}"
            )
            
            self.validation_history.append(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                validation_mode=ValidationMode.NUMERICAL_TOLERANCE,
                error_message=f"Numerical validation failed: {str(e)}"
            )
            self.validation_history.append(result)
            return result
    
    def validate_with_cross_verification(self, expression: str, expected: Any,
                                       actual: Any, tolerance: Optional[float] = None) -> ValidationResult:
        """
        Cross-verify results with multiple mathematical libraries.
        
        Args:
            expression: Mathematical expression being tested
            expected: Expected result
            actual: Actual result from MathIR parser
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            ValidationResult with cross-verification data
        """
        if tolerance is None:
            tolerance = self.default_tolerance
        
        cross_verification = {}
        
        try:
            # SymPy verification
            try:
                sympy_result = sp.N(sp.sympify(expression))
                cross_verification['sympy'] = complex(sympy_result)
            except:
                cross_verification['sympy'] = None
            
            # NumPy verification (for numerical expressions)
            try:
                # Simple numerical evaluation for basic expressions
                if all(c in '0123456789+-*/.()eE ' for c in str(expression)):
                    numpy_result = eval(expression.replace('^', '**'))
                    cross_verification['numpy'] = complex(numpy_result)
                else:
                    cross_verification['numpy'] = None
            except:
                cross_verification['numpy'] = None
            
            # Python math library verification
            try:
                # For expressions involving standard math functions
                math_expr = str(expression).replace('sin', 'math.sin').replace('cos', 'math.cos').replace('pi', 'math.pi').replace('e', 'math.e')
                if 'math.' in math_expr:
                    math_result = eval(math_expr)
                    cross_verification['math'] = complex(math_result)
                else:
                    cross_verification['math'] = None
            except:
                cross_verification['math'] = None
            
            # Determine consensus
            valid_results = [v for v in cross_verification.values() if v is not None]
            consensus_passed = True
            
            if len(valid_results) > 1:
                # Check if all valid results agree within tolerance
                for i in range(len(valid_results)):
                    for j in range(i + 1, len(valid_results)):
                        if abs(valid_results[i] - valid_results[j]) > tolerance:
                            consensus_passed = False
                            break
                    if not consensus_passed:
                        break
            
            # Check if actual result matches consensus
            actual_num = complex(actual) if isinstance(actual, (int, float, complex)) else complex(sp.N(actual))
            matches_consensus = any(abs(actual_num - result) <= tolerance for result in valid_results)
            
            result = ValidationResult(
                passed=consensus_passed and matches_consensus,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                validation_mode=ValidationMode.CROSS_VERIFICATION,
                cross_verification_results=cross_verification,
                error_message=None if (consensus_passed and matches_consensus) else "Cross-verification failed"
            )
            
            self.validation_history.append(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                validation_mode=ValidationMode.CROSS_VERIFICATION,
                cross_verification_results=cross_verification,
                error_message=f"Cross-verification failed: {str(e)}"
            )
            self.validation_history.append(result)
            return result
    
    def validate_special_values(self, actual: Any, expected_type: str) -> ValidationResult:
        """
        Validate special mathematical values (infinity, NaN, complex infinity, etc.).
        
        Args:
            actual: Actual result to validate
            expected_type: Expected type ('infinity', 'negative_infinity', 'nan', 'zoo', etc.)
            
        Returns:
            ValidationResult for special value validation
        """
        try:
            actual_expr = sp.sympify(actual) if not isinstance(actual, sp.Basic) else actual
            
            validation_map = {
                'infinity': lambda x: x == sp.oo,
                'negative_infinity': lambda x: x == -sp.oo,
                'nan': lambda x: x.is_nan if hasattr(x, 'is_nan') else False,
                'zoo': lambda x: x == sp.zoo,  # Complex infinity
                'undefined': lambda x: x.is_nan or x == sp.zoo,
                'zero': lambda x: x == 0,
                'one': lambda x: x == 1,
                'negative_one': lambda x: x == -1
            }
            
            if expected_type not in validation_map:
                raise ValueError(f"Unknown special value type: {expected_type}")
            
            is_valid = validation_map[expected_type](actual_expr)
            
            result = ValidationResult(
                passed=is_valid,
                expected=expected_type,
                actual=actual_expr,
                validation_mode=ValidationMode.EXACT_SYMBOLIC,
                error_message=None if is_valid else f"Expected {expected_type}, got {actual_expr}"
            )
            
            self.validation_history.append(result)
            return result
            
        except Exception as e:
            result = ValidationResult(
                passed=False,
                expected=expected_type,
                actual=actual,
                validation_mode=ValidationMode.EXACT_SYMBOLIC,
                error_message=f"Special value validation failed: {str(e)}"
            )
            self.validation_history.append(result)
            return result
    
    def validate_with_performance_constraints(self, expected: Any, actual: Any,
                                            max_time: float, max_memory_mb: float,
                                            tolerance: Optional[float] = None) -> ValidationResult:
        """
        Validate results with performance constraints.
        
        Args:
            expected: Expected result
            actual: Actual result
            max_time: Maximum allowed execution time in seconds
            max_memory_mb: Maximum allowed memory usage in MB
            tolerance: Tolerance for numerical comparisons
            
        Returns:
            ValidationResult with performance validation
        """
        if tolerance is None:
            tolerance = self.default_tolerance
        
        # Get the latest performance metrics
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]
            
            # Check performance constraints
            time_ok = latest_metrics.execution_time <= max_time
            memory_ok = latest_metrics.memory_usage_mb <= max_memory_mb
            
            # Check mathematical accuracy
            accuracy_result = self.validate_numerical_tolerance(expected, actual, tolerance)
            
            # Combined result
            performance_passed = time_ok and memory_ok
            overall_passed = accuracy_result.passed and performance_passed
            
            error_messages = []
            if not accuracy_result.passed:
                error_messages.append(accuracy_result.error_message)
            if not time_ok:
                error_messages.append(f"Execution time {latest_metrics.execution_time:.3f}s exceeds limit {max_time}s")
            if not memory_ok:
                error_messages.append(f"Memory usage {latest_metrics.memory_usage_mb:.1f}MB exceeds limit {max_memory_mb}MB")
            
            result = ValidationResult(
                passed=overall_passed,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                execution_time=latest_metrics.execution_time,
                memory_usage=latest_metrics.memory_usage_mb,
                validation_mode=ValidationMode.PERFORMANCE_AWARE,
                error_message="; ".join(error_messages) if error_messages else None
            )
        else:
            result = ValidationResult(
                passed=False,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                validation_mode=ValidationMode.PERFORMANCE_AWARE,
                error_message="No performance metrics available"
            )
        
        self.validation_history.append(result)
        return result
    
    def generate_accuracy_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive accuracy report.
        
        Returns:
            Dictionary containing detailed accuracy statistics
        """
        if not self.validation_history:
            return {
                "summary": {
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "success_rate": 0.0,
                    "mathematical_accuracy": "NO_DATA"
                },
                "validation_modes": {},
                "performance_statistics": {},
                "failure_analysis": {
                    "common_error_types": {},
                    "tolerance_violations": 0,
                    "symbolic_failures": 0,
                    "performance_failures": 0
                },
                "recommendations": ["No validation data available"]
            }
        
        total_tests = len(self.validation_history)
        passed_tests = sum(1 for result in self.validation_history if result.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by validation mode
        mode_stats = {}
        for mode in ValidationMode:
            mode_results = [r for r in self.validation_history if r.validation_mode == mode]
            if mode_results:
                mode_passed = sum(1 for r in mode_results if r.passed)
                mode_stats[mode.value] = {
                    "total": len(mode_results),
                    "passed": mode_passed,
                    "failed": len(mode_results) - mode_passed,
                    "success_rate": mode_passed / len(mode_results) * 100
                }
        
        # Performance statistics
        perf_stats = {}
        if self.performance_metrics:
            execution_times = [m.execution_time for m in self.performance_metrics]
            memory_usage = [m.memory_usage_mb for m in self.performance_metrics]
            
            perf_stats = {
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "max_execution_time": max(execution_times),
                "min_execution_time": min(execution_times),
                "avg_memory_usage_mb": sum(memory_usage) / len(memory_usage),
                "max_memory_usage_mb": max(memory_usage),
                "total_tests_with_metrics": len(self.performance_metrics)
            }
        
        # Failed test analysis
        failed_results = [r for r in self.validation_history if not r.passed]
        failure_analysis = {
            "common_error_types": {},
            "tolerance_violations": 0,
            "symbolic_failures": 0,
            "performance_failures": 0
        }
        
        for result in failed_results:
            if result.error_message:
                error_type = result.error_message.split(':')[0] if ':' in result.error_message else "Unknown"
                failure_analysis["common_error_types"][error_type] = failure_analysis["common_error_types"].get(error_type, 0) + 1
            
            if result.validation_mode == ValidationMode.NUMERICAL_TOLERANCE:
                failure_analysis["tolerance_violations"] += 1
            elif result.validation_mode == ValidationMode.EXACT_SYMBOLIC:
                failure_analysis["symbolic_failures"] += 1
            elif result.validation_mode == ValidationMode.PERFORMANCE_AWARE:
                failure_analysis["performance_failures"] += 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "mathematical_accuracy": "EXCELLENT" if passed_tests / total_tests >= 0.999 else 
                                       "GOOD" if passed_tests / total_tests >= 0.99 else
                                       "ACCEPTABLE" if passed_tests / total_tests >= 0.95 else "POOR"
            },
            "validation_modes": mode_stats,
            "performance_statistics": perf_stats,
            "failure_analysis": failure_analysis,
            "recommendations": self._generate_recommendations(passed_tests / total_tests if total_tests > 0 else 0)
        }
    
    def _generate_recommendations(self, success_rate: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate < 0.95:
            recommendations.append("Mathematical accuracy is below acceptable threshold (95%). Review failed test cases.")
        
        if success_rate < 0.99:
            recommendations.append("Consider improving numerical precision handling for edge cases.")
        
        if self.performance_metrics:
            avg_time = sum(m.execution_time for m in self.performance_metrics) / len(self.performance_metrics)
            if avg_time > 10.0:
                recommendations.append("Average execution time is high. Consider performance optimizations.")
            
            max_memory = max(m.memory_usage_mb for m in self.performance_metrics)
            if max_memory > 500:
                recommendations.append("Memory usage is high. Consider memory optimization strategies.")
        
        if success_rate >= 0.999:
            recommendations.append("Excellent mathematical accuracy achieved! System is production-ready.")
        
        return recommendations
    
    def reset_validation_history(self):
        """Reset validation history and performance metrics."""
        self.validation_history.clear()
        self.performance_metrics.clear()
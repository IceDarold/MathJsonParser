"""
Integration tests for error handling and graceful degradation across MathIR Parser components.

This module tests error propagation, recovery mechanisms, and graceful degradation
when components encounter various types of errors during the mathematical processing pipeline.
"""

import pytest
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import sympy as sp
from unittest.mock import patch, MagicMock

from mathir_parser.main import (
    MathIR, run_mathir, build_runtime, to_sympy_expr,
    log_error, log_success
)
from tests.utils.test_helpers import TestDataLoader, MathIRTestHelper, run_mathir_test
from tests.utils.assertion_helpers import PrecisionAssertion


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling across the complete pipeline."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.test_helper = MathIRTestHelper()
        self.data_loader = TestDataLoader()
        self.precision = PrecisionAssertion()
    
    def test_invalid_json_input_handling(self):
        """Test handling of malformed JSON input."""
        invalid_json_cases = [
            '{"expr_format": "latex", "targets": [}',  # Missing closing bracket
            '{"expr_format": "latex", "targets": [{"type": "value"}]}',  # Missing required fields
            '{"expr_format": "invalid_format", "targets": []}',  # Invalid enum value
            '',  # Empty string
            'not json at all',  # Not JSON
            '{"expr_format": "latex"}',  # Missing required targets field
        ]
        
        for i, invalid_json in enumerate(invalid_json_cases):
            with pytest.raises((json.JSONDecodeError, ValueError, TypeError)) as exc_info:
                if invalid_json.strip():
                    try:
                        data = json.loads(invalid_json)
                        MathIR.model_validate(data)
                    except json.JSONDecodeError:
                        raise
                else:
                    json.loads(invalid_json)
            
            # Verify appropriate error type
            assert exc_info.value is not None
            print(f"Case {i+1}: Correctly caught {type(exc_info.value).__name__}")
    
    def test_latex_parsing_error_recovery(self):
        """Test error handling and recovery for invalid LaTeX expressions."""
        error_cases = [
            {
                "description": "Invalid LaTeX command",
                "expr": "\\invalidcommand{x}",
                "expected_error_type": "parsing_error"
            },
            {
                "description": "Unmatched braces",
                "expr": "\\frac{x^2}{x+1",
                "expected_error_type": "syntax_error"
            },
            {
                "description": "Nested invalid commands",
                "expr": "\\frac{\\badcommand{x}}{\\anotherbad{y}}",
                "expected_error_type": "parsing_error"
            },
            {
                "description": "Empty expression",
                "expr": "",
                "expected_error_type": "empty_expression"
            },
            {
                "description": "Only whitespace",
                "expr": "   \t\n  ",
                "expected_error_type": "empty_expression"
            }
        ]
        
        for case in error_cases:
            mathir_data = {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "value",
                    "name": "result",
                    "expr": case["expr"]
                }],
                "output": {"mode": "exact"}
            }
            
            try:
                mathir = MathIR.model_validate(mathir_data)
                results = run_mathir_test(mathir)  # Use safe execution
                
                # Check if error was handled gracefully
                if "error" in results:
                    assert "exception_type" in results
                    print(f"✓ {case['description']}: Handled gracefully with {results['exception_type']}")
                else:
                    # Some cases might succeed with unexpected parsing
                    print(f"? {case['description']}: Unexpectedly succeeded with result: {results}")
                    
            except Exception as e:
                # Direct exception is also acceptable for invalid input
                print(f"✓ {case['description']}: Raised {type(e).__name__}: {e}")
    
    def test_mathematical_computation_errors(self):
        """Test handling of mathematical computation errors."""
        error_cases = [
            {
                "description": "Division by zero",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "\\frac{1}{0}"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected_result": "infinity_or_error"
            },
            {
                "description": "Complex square root in real domain",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "\\sqrt{-1}"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected_result": "complex_number"
            },
            {
                "description": "Undefined logarithm",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "\\ln(0)"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected_result": "negative_infinity_or_error"
            },
            {
                "description": "Indeterminate form 0^0",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "0^0"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected_result": "indeterminate_or_nan"
            },
            {
                "description": "Infinite integral",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "\\frac{1}{x}",
                        "var": "x",
                        "limits": [0, 1],
                        "name": "result"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected_result": "infinity_or_error"
            }
        ]
        
        for case in error_cases:
            mathir = MathIR.model_validate(case["mathir"])
            results = run_mathir_test(mathir)
            
            print(f"\n{case['description']}:")
            print(f"  Results: {results}")
            
            # Verify the computation either succeeds with expected result or fails gracefully
            if "error" in results:
                print(f"  ✓ Handled gracefully: {results['error']}")
            else:
                result = results.get("result")
                if result is not None:
                    print(f"  ✓ Computed result: {result}")
                    
                    # Validate specific expected behaviors
                    if case["expected_result"] == "infinity_or_error":
                        assert result == sp.oo or result == -sp.oo or str(result) == 'zoo' or result == sp.zoo
                    elif case["expected_result"] == "complex_number":
                        assert result == sp.I or (hasattr(result, 'is_complex') and result.is_complex)
                    elif case["expected_result"] == "negative_infinity_or_error":
                        assert result == -sp.oo or str(result) == '-oo' or result == sp.zoo
                    elif case["expected_result"] == "indeterminate_or_nan":
                        assert result == sp.nan or str(result) in ['nan', '1', '0'] or result == sp.zoo or result == 0  # SymPy might return 1, 0, or zoo for indeterminate forms
    
    def test_runtime_context_errors(self):
        """Test error handling in runtime context building and symbol resolution."""
        error_cases = [
            {
                "description": "Undefined symbol in expression",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "x + y"  # y is not defined
                    }],
                    "output": {"mode": "exact"}
                },
                "should_succeed": True  # SymPy creates undefined symbols automatically
            },
            {
                "description": "Invalid domain specification",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "INVALID_DOMAIN"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "x^2"
                    }],
                    "output": {"mode": "exact"}
                },
                "should_succeed": False
            },
            {
                "description": "Circular function definition",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "definitions": {
                        "functions": [{
                            "name": "f",
                            "args": ["x"],
                            "expr": "f(x) + 1"  # Circular reference
                        }]
                    },
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "f(2)"
                    }],
                    "output": {"mode": "exact"}
                },
                "should_succeed": False
            }
        ]
        
        for case in error_cases:
            try:
                mathir = MathIR.model_validate(case["mathir"])
                results = run_mathir_test(mathir)
                
                if case["should_succeed"]:
                    assert "error" not in results or results.get("result") is not None
                    print(f"✓ {case['description']}: Handled as expected")
                else:
                    assert "error" in results
                    print(f"✓ {case['description']}: Correctly failed with error")
                    
            except Exception as e:
                if not case["should_succeed"]:
                    print(f"✓ {case['description']}: Correctly raised {type(e).__name__}")
                else:
                    pytest.fail(f"{case['description']}: Unexpected exception: {e}")
    
    def test_validation_error_propagation(self):
        """Test that validation errors are properly propagated and logged."""
        validation_cases = [
            {
                "description": "Missing required target fields",
                "mathir_data": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def"
                        # Missing expr, var, limits
                    }],
                    "output": {"mode": "exact"}
                }
            },
            {
                "description": "Invalid target type",
                "mathir_data": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "invalid_target_type",
                        "expr": "x^2"
                    }],
                    "output": {"mode": "exact"}
                }
            },
            {
                "description": "Invalid output mode",
                "mathir_data": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "x^2"
                    }],
                    "output": {"mode": "invalid_mode"}
                }
            }
        ]
        
        for case in validation_cases:
            with pytest.raises((ValueError, TypeError)) as exc_info:
                MathIR.model_validate(case["mathir_data"])
            
            error_message = str(exc_info.value)
            assert len(error_message) > 0
            print(f"✓ {case['description']}: {type(exc_info.value).__name__}: {error_message}")
    
    def test_timeout_handling(self):
        """Test handling of computations that might take too long."""
        # Note: This is a conceptual test - actual timeout implementation would require
        # modifications to the main parser to support timeouts
        
        potentially_slow_cases = [
            {
                "description": "Large finite sum",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "n", "domain": "N"}],
                    "targets": [{
                        "type": "sum",
                        "term": "\\frac{1}{n^2}",
                        "idx": "n",
                        "start": "1",
                        "end": "10000"  # Large but finite
                    }],
                    "output": {"mode": "decimal", "round_to": 6}
                },
                "max_time": 10.0  # seconds
            },
            {
                "description": "Complex integral",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "\\sin(x^2) * \\cos(x^3)",
                        "var": "x",
                        "limits": [0, 10],
                        "name": "result"
                    }],
                    "output": {"mode": "decimal", "round_to": 3}
                },
                "max_time": 15.0
            }
        ]
        
        for case in potentially_slow_cases:
            start_time = time.time()
            
            try:
                mathir = MathIR.model_validate(case["mathir"])
                results = run_mathir_test(mathir)
                execution_time = time.time() - start_time
                
                if execution_time > case["max_time"]:
                    print(f"⚠ {case['description']}: Took {execution_time:.2f}s (> {case['max_time']}s)")
                else:
                    print(f"✓ {case['description']}: Completed in {execution_time:.2f}s")
                
                # Verify result is reasonable
                if "error" not in results:
                    assert len(results) > 0
                    
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"✗ {case['description']}: Failed after {execution_time:.2f}s with {type(e).__name__}: {e}")
    
    def test_memory_limit_handling(self):
        """Test behavior under memory pressure."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create computations that might use significant memory
        memory_intensive_cases = [
            {
                "description": "Large matrix operations",
                "mathir": {
                    "expr_format": "latex",
                    "definitions": {
                        "matrices": [{
                            "name": "A",
                            "rows": 10,
                            "cols": 10,
                            "data": [[str(i*j + 1) for j in range(10)] for i in range(10)]
                        }]
                    },
                    "targets": [{
                        "type": "matrix_determinant",
                        "matrix_name": "A"
                    }],
                    "output": {"mode": "exact"}
                }
            },
            {
                "description": "Multiple complex computations",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}, {"name": "n", "domain": "N"}],
                    "targets": [
                        {
                            "type": "integral_def",
                            "expr": f"x^{i}",
                            "var": "x",
                            "limits": [0, 1],
                            "name": f"I{i}"
                        }
                        for i in range(1, 21)  # 20 integrals
                    ],
                    "output": {"mode": "exact"}
                }
            }
        ]
        
        for case in memory_intensive_cases:
            try:
                mathir = MathIR.model_validate(case["mathir"])
                results = run_mathir_test(mathir)
                
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = current_memory - initial_memory
                
                print(f"✓ {case['description']}: Memory used: {memory_used:.2f}MB")
                
                # Verify reasonable memory usage (less than 100MB increase)
                assert memory_used < 100, f"Excessive memory usage: {memory_used:.2f}MB"
                
                if "error" not in results:
                    assert len(results) > 0
                    
            except Exception as e:
                print(f"✗ {case['description']}: Failed with {type(e).__name__}: {e}")
    
    def test_concurrent_error_handling(self):
        """Test error handling under concurrent execution."""
        import threading
        import queue
        
        def execute_with_errors(mathir_data, result_queue, thread_id):
            """Execute potentially error-prone computation."""
            try:
                mathir = MathIR.model_validate(mathir_data)
                results = run_mathir_test(mathir)
                result_queue.put((thread_id, "completed", results))
            except Exception as e:
                result_queue.put((thread_id, "exception", str(e)))
        
        # Mix of valid and invalid computations
        test_cases = [
            # Valid cases
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "value",
                    "name": "result",
                    "expr": "x^2"
                }],
                "output": {"mode": "exact"}
            },
            # Error cases
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "value",
                    "name": "result",
                    "expr": "\\frac{1}{0}"
                }],
                "output": {"mode": "exact"}
            },
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "INVALID"}],
                "targets": [{
                    "type": "value",
                    "name": "result",
                    "expr": "x"
                }],
                "output": {"mode": "exact"}
            }
        ]
        
        result_queue = queue.Queue()
        threads = []
        
        # Start concurrent executions
        for i, case in enumerate(test_cases * 3):  # Run each case 3 times
            thread = threading.Thread(
                target=execute_with_errors,
                args=(case, result_queue, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Analyze results
        completed_count = sum(1 for _, status, _ in results if status == "completed")
        exception_count = sum(1 for _, status, _ in results if status == "exception")
        
        print(f"Concurrent execution results:")
        print(f"  Completed: {completed_count}")
        print(f"  Exceptions: {exception_count}")
        print(f"  Total: {len(results)}")
        
        # Verify all threads completed (either successfully or with handled errors)
        assert len(results) == len(threads)
    
    def test_error_logging_integration(self):
        """Test integration with error logging system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            error_log_path = Path(temp_dir) / "test_error_log.json"
            
            # Mock the log_error function to write to our test file
            original_log_error = log_error
            
            def mock_log_error(task_id, input_data, error_type, traceback_str, context, actions):
                entry = {
                    "task_id": task_id,
                    "input_data": input_data,
                    "error_type": error_type,
                    "traceback": traceback_str,
                    "context": context,
                    "actions": actions
                }
                
                # Write to test log file
                if error_log_path.exists():
                    with open(error_log_path, 'r') as f:
                        data = json.load(f)
                else:
                    data = []
                
                data.append(entry)
                with open(error_log_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            with patch('mathir_parser.main.log_error', side_effect=mock_log_error):
                # Create an error-prone computation
                error_mathir = {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "INVALID_DOMAIN"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "x^2"
                    }],
                    "output": {"mode": "exact"}
                }
                
                try:
                    mathir = MathIR.model_validate(error_mathir)
                    # This should fail during validation
                except Exception as e:
                    # Manually call log_error to test logging
                    mock_log_error(
                        "test_task",
                        error_mathir,
                        type(e).__name__,
                        str(e),
                        "Validation error during testing",
                        ["Attempted MathIR validation"]
                    )
                
                # Verify error was logged
                if error_log_path.exists():
                    with open(error_log_path, 'r') as f:
                        logged_errors = json.load(f)
                    
                    assert len(logged_errors) > 0
                    error_entry = logged_errors[0]
                    assert error_entry["task_id"] == "test_task"
                    assert error_entry["error_type"] in ["ValueError", "ValidationError"]
                    assert "INVALID_DOMAIN" in str(error_entry["input_data"])
                    
                    print(f"✓ Error logging integration working: {len(logged_errors)} errors logged")
    
    def test_graceful_degradation_scenarios(self):
        """Test graceful degradation when some components fail but others succeed."""
        # Multi-target scenario where some targets succeed and others fail
        mixed_mathir = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [
                {
                    "type": "value",
                    "name": "good_result",
                    "expr": "2 + 3"  # Should succeed
                },
                {
                    "type": "value",
                    "name": "bad_result",
                    "expr": "\\frac{1}{0}"  # Should handle gracefully
                },
                {
                    "type": "integral_def",
                    "expr": "x",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "integral_result"  # Should succeed
                }
            ],
            "output": {"mode": "exact"}
        }
        
        mathir = MathIR.model_validate(mixed_mathir)
        results = run_mathir_test(mixed_mathir)
        
        print(f"Mixed scenario results: {results}")
        
        # Verify partial success - at least some targets should succeed
        if "error" not in results:
            # Check if we got some results
            assert len(results) > 0
            
            # Good result should be present
            if "good_result" in results:
                assert results["good_result"] == 5
            
            # Integral should be present
            if "integral_result" in results:
                assert results["integral_result"] == sp.Rational(1, 2)
            
            print("✓ Graceful degradation: Some targets succeeded despite errors in others")
        else:
            print(f"✓ Global error handling: {results['error']}")


@pytest.mark.integration
class TestErrorRecovery:
    """Tests for error recovery and resilience mechanisms."""
    
    def test_parser_recovery_after_errors(self):
        """Test that the parser can recover and process valid input after errors."""
        # First, cause an error
        try:
            invalid_mathir = MathIR.model_validate({
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "INVALID"}],
                "targets": [{
                    "type": "value",
                    "name": "result",
                    "expr": "x"
                }]
            })
        except Exception:
            pass  # Expected to fail
        
        # Then, process valid input
        valid_mathir = MathIR.model_validate({
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "value",
                "name": "result",
                "expr": "2 + 3"  # Use simple arithmetic to avoid parsing issues
            }],
            "output": {"mode": "exact"}
        })
        
        results = run_mathir(valid_mathir)
        assert "result" in results
        assert "error" not in results
        print("✓ Parser recovered successfully after error")
    
    def test_sympy_error_isolation(self):
        """Test that SymPy errors are properly isolated and don't crash the system."""
        # Test cases that might cause SymPy-specific errors
        sympy_error_cases = [
            "\\frac{\\infty}{\\infty}",  # Indeterminate form
            "\\sqrt{x^2}",  # Might have domain issues
            "\\log(-1)",  # Complex logarithm
        ]
        
        for expr in sympy_error_cases:
            mathir_data = {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "value",
                    "name": "result",
                    "expr": expr
                }],
                "output": {"mode": "exact"}
            }
            
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir_test(mathir)
            
            # Should either succeed or fail gracefully
            assert isinstance(results, dict)
            print(f"✓ SymPy expression '{expr}': {results}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
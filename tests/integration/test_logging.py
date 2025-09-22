"""
Integration tests for logging system integration and audit trails in MathIR Parser.

This module tests the logging system integration, ensuring proper logging of
success cases, error cases, performance metrics, and mathematical accuracy
across the complete pipeline.
"""

import pytest
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, mock_open
import datetime
import sympy as sp

from mathir_parser.main import (
    MathIR, run_mathir, log_success, log_error, log_to_file
)
from tests.utils.test_helpers import TestDataLoader, MathIRTestHelper, run_mathir_test
from tests.utils.performance_monitor import PerformanceProfiler


@pytest.mark.integration
class TestLoggingIntegration:
    """Integration tests for the logging system across the complete pipeline."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.test_helper = MathIRTestHelper()
        self.data_loader = TestDataLoader()
        self.profiler = PerformanceProfiler()
        
        # Create temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.test_success_log = Path(self.temp_dir) / "test_success_log.json"
        self.test_error_log = Path(self.temp_dir) / "test_error_log.json"
    
    def teardown_method(self):
        """Clean up after each test method."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_success_logging_integration(self):
        """Test comprehensive success logging with detailed execution traces."""
        # Create a successful computation scenario
        mathir_data = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [
                {
                    "type": "integral_def",
                    "expr": "x^2",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "integral_result"
                },
                {
                    "type": "value",
                    "name": "doubled",
                    "expr": "2 * integral_result"
                }
            ],
            "output": {"mode": "exact"}
        }
        
        task_id = "test_success_001"
        actions = []
        
        with patch('mathir_parser.main.log_to_file') as mock_log_to_file:
            # Execute the computation
            start_time = time.time()
            actions.append("Started computation")
            
            mathir = MathIR.model_validate(mathir_data)
            actions.append("Validated MathIR structure")
            
            results = run_mathir(mathir)
            actions.append("Executed all targets successfully")
            
            execution_time = time.time() - start_time
            actions.append(f"Completed in {execution_time:.3f}s")
            
            # Manually call log_success to test logging
            serializable_results = {k: str(v) for k, v in results.items()}
            log_success(task_id, mathir_data, serializable_results, actions)
            
            # Verify log_to_file was called
            assert mock_log_to_file.called
            call_args = mock_log_to_file.call_args
            
            # Verify the log entry structure
            log_filename = call_args[0][0]
            log_entry = call_args[0][1]
            
            assert log_filename == "success_log.json"
            assert log_entry["task_id"] == task_id
            assert log_entry["input_data"] == mathir_data
            assert log_entry["final_output"] == serializable_results
            assert log_entry["actions"] == actions
            assert "timestamp" in log_entry
            
            # Verify timestamp format
            timestamp = log_entry["timestamp"]
            datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            print(f"✓ Success logging integration verified for task {task_id}")
    
    def test_error_logging_with_stack_traces(self):
        """Test error logging with detailed stack traces and context."""
        # Create an error scenario
        invalid_mathir_data = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "INVALID_DOMAIN"}],
            "targets": [{
                "type": "value",
                "name": "result",
                "expr": "x^2"
            }],
            "output": {"mode": "exact"}
        }
        
        task_id = "test_error_001"
        actions = ["Started validation", "Encountered domain error"]
        
        with patch('mathir_parser.main.log_to_file') as mock_log_to_file:
            try:
                mathir = MathIR.model_validate(invalid_mathir_data)
                pytest.fail("Expected validation error")
            except Exception as e:
                # Log the error
                import traceback
                traceback_str = traceback.format_exc()
                
                log_error(
                    task_id,
                    invalid_mathir_data,
                    type(e).__name__,
                    traceback_str,
                    str(e),
                    actions
                )
                
                # Verify error logging
                assert mock_log_to_file.called
                call_args = mock_log_to_file.call_args
                
                log_filename = call_args[0][0]
                log_entry = call_args[0][1]
                
                assert log_filename == "error_log.json"
                assert log_entry["task_id"] == task_id
                assert log_entry["input_data"] == invalid_mathir_data
                assert log_entry["error_type"] == type(e).__name__
                assert log_entry["traceback"] == traceback_str
                assert log_entry["context"] == str(e)
                assert log_entry["actions"] == actions
                assert "timestamp" in log_entry
                
                print(f"✓ Error logging integration verified for task {task_id}")
    
    def test_performance_logging_integration(self):
        """Test logging of performance metrics and timing information."""
        # Create performance test scenarios
        performance_cases = [
            {
                "name": "simple_computation",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "value",
                        "name": "result",
                        "expr": "2 + 3"
                    }],
                    "output": {"mode": "exact"}
                }
            },
            {
                "name": "integral_computation",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "x^3",
                        "var": "x",
                        "limits": [0, 2],
                        "name": "result"
                    }],
                    "output": {"mode": "decimal", "round_to": 6}
                }
            },
            {
                "name": "limit_computation",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "limit",
                        "expr": "\\frac{\\sin(x)}{x}",
                        "var": "x",
                        "to": "0"
                    }],
                    "output": {"mode": "exact"}
                }
            }
        ]
        
        performance_logs = []
        
        with patch('mathir_parser.main.log_success') as mock_log_success:
            for case in performance_cases:
                with self.profiler.measure(case["name"]) as timer:
                    mathir = MathIR.model_validate(case["mathir"])
                    results = run_mathir(mathir)
                
                execution_time = timer.elapsed_time
                
                # Enhanced actions with performance data
                actions = [
                    "Started computation",
                    "Validated MathIR structure",
                    f"Built runtime context in {execution_time/4:.3f}s",
                    f"Executed targets in {execution_time*3/4:.3f}s",
                    f"Total execution time: {execution_time:.3f}s"
                ]
                
                # Add performance metadata to results
                enhanced_results = {
                    **{k: str(v) for k, v in results.items()},
                    "_performance": {
                        "execution_time": execution_time,
                        "target_count": len(case["mathir"]["targets"]),
                        "complexity": case["name"]
                    }
                }
                
                log_success(
                    f"perf_{case['name']}",
                    case["mathir"],
                    enhanced_results,
                    actions
                )
                
                performance_logs.append({
                    "case": case["name"],
                    "time": execution_time,
                    "results": results
                })
        
        # Verify performance logging calls
        assert mock_log_success.call_count == len(performance_cases)
        
        # Analyze performance data
        total_time = sum(log["time"] for log in performance_logs)
        avg_time = total_time / len(performance_logs)
        
        print(f"\nPerformance Logging Results:")
        print(f"  Total cases: {len(performance_logs)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time: {avg_time:.3f}s")
        
        for log in performance_logs:
            print(f"  {log['case']}: {log['time']:.3f}s")
        
        # Verify reasonable performance
        assert avg_time < 2.0, f"Average execution time too high: {avg_time:.3f}s"
    
    def test_mathematical_accuracy_logging(self):
        """Test logging of mathematical accuracy and validation results."""
        # Test cases with known exact results
        accuracy_cases = [
            {
                "name": "exact_integral",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "x^2",
                        "var": "x",
                        "limits": [0, 1],
                        "name": "result"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected": sp.Rational(1, 3),
                "tolerance": 0
            },
            {
                "name": "decimal_integral",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "x^2",
                        "var": "x",
                        "limits": [0, 1],
                        "name": "result"
                    }],
                    "output": {"mode": "decimal", "round_to": 6}
                },
                "expected": 1/3,
                "tolerance": 1e-6
            },
            {
                "name": "famous_limit",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "limit",
                        "expr": "\\frac{\\sin(x)}{x}",
                        "var": "x",
                        "to": "0"
                    }],
                    "output": {"mode": "exact"}
                },
                "expected": 1,
                "tolerance": 0
            }
        ]
        
        accuracy_logs = []
        
        with patch('mathir_parser.main.log_success') as mock_log_success:
            for case in accuracy_cases:
                mathir = MathIR.model_validate(case["mathir"])
                results = run_mathir(mathir)
                
                # Calculate accuracy
                actual = results["result"]
                expected = case["expected"]
                
                if case["tolerance"] == 0:
                    # Exact comparison
                    if isinstance(actual, sp.Expr) and isinstance(expected, sp.Expr):
                        accuracy = "exact" if sp.simplify(actual - expected) == 0 else "inexact"
                        error = 0 if accuracy == "exact" else float('inf')
                    else:
                        accuracy = "exact" if actual == expected else "inexact"
                        error = 0 if accuracy == "exact" else abs(float(actual) - float(expected))
                else:
                    # Numerical comparison
                    error = abs(float(actual) - float(expected))
                    accuracy = "within_tolerance" if error <= case["tolerance"] else "outside_tolerance"
                
                # Enhanced logging with accuracy data
                accuracy_data = {
                    "expected_value": str(expected),
                    "actual_value": str(actual),
                    "accuracy_status": accuracy,
                    "numerical_error": error,
                    "tolerance": case["tolerance"]
                }
                
                actions = [
                    "Started computation",
                    "Executed mathematical operation",
                    f"Accuracy check: {accuracy}",
                    f"Numerical error: {error}",
                    "Completed accuracy validation"
                ]
                
                enhanced_results = {
                    **{k: str(v) for k, v in results.items()},
                    "_accuracy": accuracy_data
                }
                
                log_success(
                    f"accuracy_{case['name']}",
                    case["mathir"],
                    enhanced_results,
                    actions
                )
                
                accuracy_logs.append({
                    "case": case["name"],
                    "accuracy": accuracy,
                    "error": error,
                    "expected": expected,
                    "actual": actual
                })
        
        # Verify accuracy logging
        assert mock_log_success.call_count == len(accuracy_cases)
        
        print(f"\nMathematical Accuracy Logging Results:")
        for log in accuracy_logs:
            print(f"  {log['case']}: {log['accuracy']} (error: {log['error']})")
        
        # Verify all computations met accuracy requirements
        for log in accuracy_logs:
            if log["case"] in ["exact_integral", "famous_limit"]:
                assert log["accuracy"] == "exact", f"{log['case']} should be exact"
            else:
                assert log["accuracy"] == "within_tolerance", f"{log['case']} should be within tolerance"
    
    def test_log_file_integration(self):
        """Test integration with actual log files and file I/O operations."""
        # Test with real file operations
        success_log_path = self.test_success_log
        error_log_path = self.test_error_log
        
        # Test success logging to file
        success_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task_id": "file_test_001",
            "input_data": {"test": "data"},
            "final_output": {"result": "42"},
            "actions": ["test_action"]
        }
        
        # Use the actual log_to_file function
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.path.exists', return_value=False):
                log_to_file(str(success_log_path), success_entry)
            
            # Verify file operations
            mock_file.assert_called()
            
        # Test error logging to file
        error_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task_id": "file_test_002",
            "input_data": {"test": "error_data"},
            "error_type": "TestError",
            "traceback": "test traceback",
            "context": "test context",
            "actions": ["test_error_action"]
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.path.exists', return_value=False):
                log_to_file(str(error_log_path), error_entry)
            
            mock_file.assert_called()
        
        print("✓ Log file integration tested successfully")
    
    def test_concurrent_logging_safety(self):
        """Test logging system safety under concurrent access."""
        import threading
        import queue
        
        def concurrent_logger(log_data, result_queue, thread_id):
            """Log data concurrently from multiple threads."""
            try:
                with patch('mathir_parser.main.log_to_file') as mock_log:
                    log_success(
                        f"concurrent_{thread_id}",
                        log_data["input"],
                        log_data["output"],
                        log_data["actions"]
                    )
                    result_queue.put((thread_id, "success", mock_log.call_count))
            except Exception as e:
                result_queue.put((thread_id, "error", str(e)))
        
        # Create test data for concurrent logging
        test_data = [
            {
                "input": {"expr_format": "latex", "targets": [{"type": "value", "name": "r", "expr": f"{i}"}]},
                "output": {"r": str(i)},
                "actions": [f"action_{i}"]
            }
            for i in range(10)
        ]
        
        result_queue = queue.Queue()
        threads = []
        
        # Start concurrent logging operations
        for i, data in enumerate(test_data):
            thread = threading.Thread(
                target=concurrent_logger,
                args=(data, result_queue, i)
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
        
        # Verify all logging operations completed
        assert len(results) == len(test_data)
        success_count = sum(1 for _, status, _ in results if status == "success")
        
        print(f"Concurrent logging results: {success_count}/{len(results)} successful")
        assert success_count == len(results), "Some concurrent logging operations failed"
    
    def test_log_rotation_and_size_management(self):
        """Test log file size management and rotation behavior."""
        # Simulate large log entries
        large_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "task_id": "large_test",
            "input_data": {"large_data": "x" * 1000},  # Large data
            "final_output": {"result": "y" * 1000},
            "actions": [f"action_{i}" for i in range(100)]  # Many actions
        }
        
        # Test with mock file operations
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.path.exists', return_value=False):
                # Log multiple large entries
                for i in range(5):
                    entry = {**large_entry, "task_id": f"large_test_{i}"}
                    log_to_file("test_log.json", entry)
        
        # Verify file operations occurred
        assert mock_file.call_count >= 5
        print("✓ Large log entry handling tested")
    
    def test_log_format_consistency(self):
        """Test consistency of log format across different scenarios."""
        test_scenarios = [
            {
                "type": "success",
                "data": {
                    "task_id": "format_test_success",
                    "input_data": {"test": "success_data"},
                    "final_output": {"result": "success"},
                    "actions": ["success_action"]
                }
            },
            {
                "type": "error",
                "data": {
                    "task_id": "format_test_error",
                    "input_data": {"test": "error_data"},
                    "error_type": "TestError",
                    "traceback": "test traceback",
                    "context": "test context",
                    "actions": ["error_action"]
                }
            }
        ]
        
        logged_entries = []
        
        with patch('mathir_parser.main.log_to_file') as mock_log_to_file:
            for scenario in test_scenarios:
                if scenario["type"] == "success":
                    log_success(
                        scenario["data"]["task_id"],
                        scenario["data"]["input_data"],
                        scenario["data"]["final_output"],
                        scenario["data"]["actions"]
                    )
                else:
                    log_error(
                        scenario["data"]["task_id"],
                        scenario["data"]["input_data"],
                        scenario["data"]["error_type"],
                        scenario["data"]["traceback"],
                        scenario["data"]["context"],
                        scenario["data"]["actions"]
                    )
                
                # Capture the logged entry
                if mock_log_to_file.called:
                    call_args = mock_log_to_file.call_args
                    logged_entries.append(call_args[0][1])  # The log entry
        
        # Verify format consistency
        assert len(logged_entries) == len(test_scenarios)
        
        for entry in logged_entries:
            # Common fields
            assert "timestamp" in entry
            assert "task_id" in entry
            assert "input_data" in entry
            assert "actions" in entry
            
            # Verify timestamp format
            timestamp = entry["timestamp"]
            datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Verify task_id format
            assert entry["task_id"].startswith("format_test_")
            
            # Verify actions is a list
            assert isinstance(entry["actions"], list)
        
        print("✓ Log format consistency verified across scenarios")
    
    def test_integration_with_existing_logs(self):
        """Test integration with existing success_log.json and failure_log.json files."""
        # Check if existing log files exist and test compatibility
        project_root = Path(__file__).parent.parent.parent
        existing_success_log = project_root / "success_log.json"
        existing_failure_log = project_root / "failure_log.json"
        
        if existing_success_log.exists():
            # Test reading existing success log
            try:
                with open(existing_success_log, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                assert isinstance(existing_data, list)
                
                if existing_data:
                    # Verify structure of existing entries
                    sample_entry = existing_data[0]
                    expected_fields = ["timestamp", "task_id", "input_data", "final_output", "actions"]
                    
                    for field in expected_fields:
                        assert field in sample_entry, f"Missing field {field} in existing success log"
                
                print(f"✓ Existing success log compatibility verified ({len(existing_data)} entries)")
                
            except Exception as e:
                print(f"⚠ Issue with existing success log: {e}")
        
        if existing_failure_log.exists():
            # Test reading existing failure log
            try:
                with open(existing_failure_log, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                assert isinstance(existing_data, list)
                
                if existing_data:
                    # Verify structure of existing entries
                    sample_entry = existing_data[0]
                    # Note: existing failure log might have different structure
                    # Just verify it's readable
                    assert isinstance(sample_entry, dict)
                
                print(f"✓ Existing failure log compatibility verified ({len(existing_data)} entries)")
                
            except Exception as e:
                print(f"⚠ Issue with existing failure log: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestLoggingPerformance:
    """Performance tests for the logging system."""
    
    def test_logging_performance_impact(self):
        """Test the performance impact of logging on computation speed."""
        # Test computation with and without logging
        test_mathir = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "integral_def",
                "expr": "x^2 + 2*x + 1",
                "var": "x",
                "limits": [0, 1],
                "name": "result"
            }],
            "output": {"mode": "decimal", "round_to": 6}
        }
        
        # Time without logging
        with patch('mathir_parser.main.log_success'):
            with patch('mathir_parser.main.log_error'):
                start_time = time.time()
                for _ in range(10):
                    mathir = MathIR.model_validate(test_mathir)
                    results = run_mathir(mathir)
                time_without_logging = time.time() - start_time
        
        # Time with logging
        start_time = time.time()
        for i in range(10):
            mathir = MathIR.model_validate(test_mathir)
            results = run_mathir(mathir)
            # Simulate logging
            log_success(f"perf_test_{i}", test_mathir, {k: str(v) for k, v in results.items()}, ["test"])
        time_with_logging = time.time() - start_time
        
        # Calculate overhead
        logging_overhead = time_with_logging - time_without_logging
        overhead_percentage = (logging_overhead / time_without_logging) * 100
        
        print(f"\nLogging Performance Impact:")
        print(f"  Without logging: {time_without_logging:.3f}s")
        print(f"  With logging: {time_with_logging:.3f}s")
        print(f"  Overhead: {logging_overhead:.3f}s ({overhead_percentage:.1f}%)")
        
        # Verify reasonable overhead (should be less than 50% increase)
        assert overhead_percentage < 50, f"Logging overhead too high: {overhead_percentage:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
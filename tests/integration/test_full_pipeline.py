"""
Comprehensive integration tests for the MathIR Parser full pipeline.

This module tests the complete JSON → MathIR → Runtime → Execution → Output pipeline,
validating component interactions and end-to-end workflows across all mathematical
target types and processing modes.
"""

import pytest
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import sympy as sp

from mathir_parser.main import MathIR, run_mathir, build_runtime
from tests.utils.test_helpers import TestDataLoader, MathIRTestHelper, run_mathir_test
from tests.utils.assertion_helpers import (
    assert_integral_equal, assert_limit_equal, assert_sum_equal,
    assert_solve_equal, assert_numerical_close, assert_symbolic_equivalent
)
from tests.utils.performance_monitor import PerformanceProfiler


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for complete mathematical processing pipeline."""
    
    def setup_method(self):
        """Set up test environment for each test method."""
        self.profiler = PerformanceProfiler()
        self.test_helper = MathIRTestHelper()
        self.data_loader = TestDataLoader()
    
    @pytest.mark.parametrize("output_mode,expected_type", [
        ("exact", (sp.Expr, int, float)),
        ("decimal", (float, int, sp.Float)),
    ])
    def test_complete_integral_pipeline(self, output_mode, expected_type):
        """Test complete integral computation pipeline with different output modes."""
        # Test data: integral of x^2 from 0 to 1 = 1/3
        mathir_data = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "integral_def",
                "expr": "x^2",
                "var": "x", 
                "limits": [0, 1],
                "name": "integral_result"
            }],
            "output": {"mode": output_mode, "round_to": 6}
        }
        
        with self.profiler.measure("integral_pipeline"):
            # Step 1: JSON validation and MathIR creation
            mathir = MathIR.model_validate(mathir_data)
            assert mathir.expr_format == "latex"
            assert len(mathir.targets) == 1
            
            # Step 2: Runtime context building
            runtime = build_runtime(mathir)
            assert "x" in runtime.symtab
            assert runtime.symtab["x"].is_real
            
            # Step 3: Complete execution
            results = run_mathir(mathir)
            
            # Step 4: Result validation
            assert "integral_result" in results
            result = results["integral_result"]
            assert isinstance(result, expected_type)
            
            if output_mode == "exact":
                assert_symbolic_equivalent(result, sp.Rational(1, 3))
            else:
                assert_numerical_close(float(result), 1/3, tolerance=1e-5)
    
    def test_multi_target_pipeline_with_dependencies(self):
        """Test pipeline with multiple targets where later targets depend on earlier ones."""
        mathir_data = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [
                {
                    "type": "integral_def",
                    "expr": "x",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I1"
                },
                {
                    "type": "integral_def", 
                    "expr": "x^2",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I2"
                },
                {
                    "type": "value",
                    "name": "sum_result",
                    "expr": "I1 + I2"
                },
                {
                    "type": "value",
                    "name": "product_result", 
                    "expr": "I1 * I2"
                }
            ],
            "output": {"mode": "exact"}
        }
        
        with self.profiler.measure("multi_target_pipeline"):
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            
            # Validate all results
            assert len(results) == 4
            assert "I1" in results
            assert "I2" in results
            assert "sum_result" in results
            assert "product_result" in results
            
            # I1 = 1/2, I2 = 1/3
            assert_symbolic_equivalent(results["I1"], sp.Rational(1, 2))
            assert_symbolic_equivalent(results["I2"], sp.Rational(1, 3))
            assert_symbolic_equivalent(results["sum_result"], sp.Rational(5, 6))  # 1/2 + 1/3
            assert_symbolic_equivalent(results["product_result"], sp.Rational(1, 6))  # 1/2 * 1/3
    
    def test_complex_mathematical_workflow(self):
        """Test complex workflow combining multiple mathematical operations."""
        mathir_data = {
            "expr_format": "latex",
            "symbols": [
                {"name": "x", "domain": "R"},
                {"name": "n", "domain": "N"}
            ],
            "targets": [
                {
                    "type": "limit",
                    "expr": "\\frac{\\sin(x)}{x}",
                    "var": "x",
                    "to": "0"
                },
                {
                    "type": "sum",
                    "term": "\\frac{1}{n^2}",
                    "idx": "n",
                    "start": "1",
                    "end": "10"
                },
                {
                    "type": "solve_for",
                    "unknowns": ["x"],
                    "equations": ["x^2 - 4 = 0"]
                }
            ],
            "output": {"mode": "exact"}
        }
        
        with self.profiler.measure("complex_workflow"):
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            
            # Validate limit result
            assert "limit" in results
            assert_limit_equal(results["limit"], 1)
            
            # Validate sum result (partial sum of Basel problem)
            assert "sum" in results
            expected_sum = sum(sp.Rational(1, n**2) for n in range(1, 11))
            assert_sum_equal(results["sum"], expected_sum)
            
            # Validate solve result
            assert "solve" in results
            solutions = results["solve"]
            assert len(solutions) == 2
            solution_values = {sol[sp.Symbol('x', real=True)] for sol in solutions}
            assert solution_values == {-2, 2}
    
    def test_matrix_operations_pipeline(self):
        """Test pipeline with matrix operations and solving."""
        mathir_data = {
            "expr_format": "latex",
            "definitions": {
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
            "conditions": [{
                "type": "matrix_equation",
                "expr": "A*X = B"
            }],
            "targets": [
                {
                    "type": "matrix_determinant",
                    "matrix_name": "A"
                },
                {
                    "type": "matrix_inverse",
                    "matrix_name": "A"
                },
                {
                    "type": "solve_for_matrix",
                    "unknown": "X"
                }
            ],
            "output": {"mode": "exact"}
        }
        
        with self.profiler.measure("matrix_pipeline"):
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            
            # Validate determinant
            assert "determinant" in results
            assert results["determinant"] == -2  # det([[1,2],[3,4]]) = 4-6 = -2
            
            # Validate inverse
            assert "inverse" in results
            inverse = results["inverse"]
            expected_inverse = sp.Matrix([[-2, 1], [sp.Rational(3,2), sp.Rational(-1,2)]])
            assert inverse.equals(expected_inverse)
            
            # Validate matrix solve
            assert "matrix" in results
            solution = results["matrix"]
            # A*X = B where A=[[1,2],[3,4]], B=[[5],[11]]
            # Solution should be X = [[1],[2]]
            assert solution[0, 0] == 1
            assert solution[1, 0] == 2
    
    def test_function_and_sequence_definitions_pipeline(self):
        """Test pipeline with custom function and sequence definitions."""
        mathir_data = {
            "expr_format": "latex",
            "symbols": [
                {"name": "x", "domain": "R"},
                {"name": "n", "domain": "N"}
            ],
            "definitions": {
                "functions": [{
                    "name": "f",
                    "args": ["x"],
                    "expr": "x^2 + 2*x + 1"
                }],
                "sequences": [{
                    "name": "a",
                    "args": ["n"],
                    "expr": "2*n + 1"
                }]
            },
            "targets": [
                {
                    "type": "value",
                    "name": "f_at_3",
                    "expr": "f(3)"
                },
                {
                    "type": "sum",
                    "term": "a(n)",
                    "idx": "n",
                    "start": "1",
                    "end": "5"
                },
                {
                    "type": "integral_def",
                    "expr": "f(x)",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "integral_f"
                }
            ],
            "output": {"mode": "exact"}
        }
        
        with self.profiler.measure("definitions_pipeline"):
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            
            # f(3) = 3^2 + 2*3 + 1 = 9 + 6 + 1 = 16
            assert results["f_at_3"] == 16
            
            # Sum of a(n) = 2n+1 for n=1 to 5: 3+5+7+9+11 = 35
            assert results["sum"] == 35
            
            # Integral of x^2 + 2x + 1 from 0 to 1 = [x^3/3 + x^2 + x] from 0 to 1 = 1/3 + 1 + 1 = 7/3
            assert_integral_equal(results["integral_f"], sp.Rational(7, 3))
    
    @pytest.mark.parametrize("test_file", [
        "task_000001.json",
        "task_000002.json", 
        "task_000003.json"
    ])
    def test_real_world_integration_scenarios(self, test_file):
        """Test integration with real-world test data from individual tasks."""
        test_path = Path("tests/individual") / test_file
        if not test_path.exists():
            pytest.skip(f"Test file {test_file} not found")
            
        with self.profiler.measure(f"real_world_{test_file}"):
            # Load test data
            test_data = self.data_loader.load_json_file(test_path)
            
            if "parsed_json" not in test_data:
                pytest.skip(f"No parsed_json in {test_file}")
            
            # Execute pipeline
            mathir = MathIR.model_validate(test_data["parsed_json"])
            results = run_mathir(mathir)
            
            # Basic validation
            assert isinstance(results, dict)
            assert len(results) > 0
            assert "error" not in results
            
            # Validate result structure based on targets
            for target in mathir.targets:
                if hasattr(target, 'name') and target.name:
                    assert target.name in results
                elif target.type == 'limit':
                    assert 'limit' in results
                elif target.type == 'sum':
                    assert 'sum' in results
                elif target.type == 'solve_for':
                    assert 'solve' in results
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for complete pipelines."""
        test_cases = [
            {
                "name": "simple_integral",
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
                    "output": {"mode": "decimal"}
                },
                "max_time": 1.0  # seconds
            },
            {
                "name": "complex_limit",
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
                "max_time": 2.0
            },
            {
                "name": "finite_sum",
                "mathir": {
                    "expr_format": "latex",
                    "symbols": [{"name": "n", "domain": "N"}],
                    "targets": [{
                        "type": "sum",
                        "term": "n^2",
                        "idx": "n", 
                        "start": "1",
                        "end": "100"
                    }],
                    "output": {"mode": "exact"}
                },
                "max_time": 3.0
            }
        ]
        
        performance_results = {}
        
        for test_case in test_cases:
            with self.profiler.measure(test_case["name"]) as timer:
                mathir = MathIR.model_validate(test_case["mathir"])
                results = run_mathir(mathir)
                
            execution_time = timer.elapsed_time
            performance_results[test_case["name"]] = execution_time
            
            # Validate performance
            assert execution_time < test_case["max_time"], \
                f"Performance test failed for {test_case['name']}: {execution_time:.3f}s > {test_case['max_time']}s"
            
            # Validate correctness
            assert isinstance(results, dict)
            assert len(results) > 0
            assert "error" not in results
        
        # Log performance results
        print("\nPerformance Benchmark Results:")
        for name, time_taken in performance_results.items():
            print(f"  {name}: {time_taken:.3f}s")
    
    def test_memory_usage_pipeline(self):
        """Test memory usage during pipeline execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a moderately complex computation
        mathir_data = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}, {"name": "n", "domain": "N"}],
            "targets": [
                {
                    "type": "integral_def",
                    "expr": "x^3 + 2*x^2 + x + 1",
                    "var": "x",
                    "limits": [0, 2],
                    "name": "I1"
                },
                {
                    "type": "sum",
                    "term": "\\frac{1}{n^2}",
                    "idx": "n",
                    "start": "1", 
                    "end": "50"
                },
                {
                    "type": "limit",
                    "expr": "\\frac{x^2 - 1}{x - 1}",
                    "var": "x",
                    "to": "1"
                }
            ],
            "output": {"mode": "exact"}
        }
        
        # Execute multiple times to check for memory leaks
        for i in range(10):
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            assert len(results) == 3
            assert "error" not in results
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50, f"Excessive memory usage: {memory_increase:.2f}MB increase"
    
    def test_concurrent_pipeline_execution(self):
        """Test concurrent execution of multiple pipelines."""
        import threading
        import queue
        
        def execute_pipeline(mathir_data, result_queue, thread_id):
            """Execute pipeline in a separate thread."""
            try:
                mathir = MathIR.model_validate(mathir_data)
                results = run_mathir(mathir)
                result_queue.put((thread_id, "success", results))
            except Exception as e:
                result_queue.put((thread_id, "error", str(e)))
        
        # Create different pipeline configurations
        pipeline_configs = [
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "integral_def",
                    "expr": f"x^{i+1}",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "result"
                }],
                "output": {"mode": "exact"}
            }
            for i in range(5)
        ]
        
        result_queue = queue.Queue()
        threads = []
        
        # Start concurrent executions
        for i, config in enumerate(pipeline_configs):
            thread = threading.Thread(
                target=execute_pipeline,
                args=(config, result_queue, i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Validate all executions succeeded
        assert len(results) == len(pipeline_configs)
        for thread_id, status, result in results:
            assert status == "success", f"Thread {thread_id} failed: {result}"
            assert isinstance(result, dict)
            assert "result" in result
    
    def test_output_format_consistency(self):
        """Test consistency of output formats across different modes."""
        base_mathir = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "integral_def",
                "expr": "x^2 + 1",
                "var": "x",
                "limits": [0, 1],
                "name": "result"
            }]
        }
        
        output_modes = [
            {"mode": "exact"},
            {"mode": "decimal", "round_to": 3},
            {"mode": "decimal", "round_to": 6},
            {"mode": "decimal", "round_to": 10}
        ]
        
        results_by_mode = {}
        
        for output_config in output_modes:
            mathir_data = {**base_mathir, "output": output_config}
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            
            mode_key = f"{output_config['mode']}"
            if 'round_to' in output_config:
                mode_key += f"_{output_config['round_to']}"
            
            results_by_mode[mode_key] = results["result"]
        
        # Validate exact result
        exact_result = results_by_mode["exact"]
        expected_exact = sp.Rational(4, 3)  # Integral of x^2+1 from 0 to 1
        assert_symbolic_equivalent(exact_result, expected_exact)
        
        # Validate decimal results are consistent
        expected_decimal = float(expected_exact)
        for mode, result in results_by_mode.items():
            if mode.startswith("decimal"):
                assert isinstance(result, (int, float))
                assert abs(float(result) - expected_decimal) < 1e-2
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Print performance summary if profiler was used
        if hasattr(self, 'profiler') and self.profiler.measurements:
            print(f"\nPerformance Summary:")
            for name, duration in self.profiler.measurements.items():
                print(f"  {name}: {duration:.3f}s")


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScalePipeline:
    """Integration tests for large-scale pipeline operations."""
    
    def test_batch_processing_pipeline(self):
        """Test processing multiple MathIR objects in batch."""
        # Create a batch of different mathematical problems
        batch_data = [
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "integral_def",
                    "expr": f"x^{i}",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "result"
                }],
                "output": {"mode": "exact"}
            }
            for i in range(1, 11)  # x^1 to x^10
        ]
        
        batch_results = []
        total_start_time = time.time()
        
        for i, mathir_data in enumerate(batch_data):
            start_time = time.time()
            mathir = MathIR.model_validate(mathir_data)
            results = run_mathir(mathir)
            end_time = time.time()
            
            batch_results.append({
                "index": i,
                "results": results,
                "execution_time": end_time - start_time
            })
        
        total_time = time.time() - total_start_time
        
        # Validate all results
        assert len(batch_results) == 10
        for i, batch_result in enumerate(batch_results):
            power = i + 1
            expected = sp.Rational(1, power + 1)  # Integral of x^n from 0 to 1 = 1/(n+1)
            actual = batch_result["results"]["result"]
            assert_symbolic_equivalent(actual, expected)
        
        # Performance validation
        avg_time = total_time / len(batch_data)
        assert avg_time < 1.0, f"Average processing time too high: {avg_time:.3f}s"
        
        print(f"\nBatch Processing Results:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per item: {avg_time:.3f}s")
        print(f"  Items processed: {len(batch_results)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
#!/usr/bin/env python3
"""
Performance End-to-End Tests for MathIR Parser

This module provides comprehensive performance validation and benchmarking for the MathIR Parser.
It tests execution time, memory usage, throughput, scalability, and resource cleanup to ensure
the system meets production performance requirements.

Performance Requirements:
- Execution time: <30 seconds per mathematical task
- Memory usage: <500MB per task
- Throughput: >100 tasks/hour capability
- Scalability: linear time growth with complexity
- Resource cleanup and memory leak detection

Test Categories:
1. Execution Time Benchmarks - Validate response times for different operation types
2. Memory Usage Monitoring - Track memory consumption and detect leaks
3. Throughput Testing - Measure tasks processed per unit time
4. Scalability Analysis - Test performance scaling with problem complexity
5. Resource Cleanup Validation - Ensure proper cleanup after operations
6. Stress Testing - Test system behavior under high load
7. Concurrent Processing - Test parallel task execution
"""

import pytest
import time
import psutil
import os
import gc
import threading
import concurrent.futures
import sys
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import statistics
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mathir_parser.main import MathIR, run_mathir
from tests.utils.mathematical_accuracy_validator import (
    MathematicalAccuracyValidator, PerformanceMetrics
)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark test."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    throughput_tasks_per_second: Optional[float] = None


@dataclass
class PerformanceBenchmark:
    """Performance benchmark specification."""
    name: str
    mathir_config: Dict[str, Any]
    max_time_seconds: float
    max_memory_mb: float
    description: str
    complexity_level: str  # "simple", "medium", "complex", "extreme"


class PerformanceTestSuite:
    """
    Comprehensive performance test suite for MathIR Parser.
    
    This suite validates that the MathIR Parser meets all performance requirements
    for production use, including response times, memory usage, throughput, and
    scalability characteristics.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up performance testing infrastructure."""
        cls.validator = MathematicalAccuracyValidator()
        cls.benchmark_results: List[BenchmarkResult] = []
        cls.process = psutil.Process(os.getpid())
        cls.baseline_memory = cls.process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance benchmarks with different complexity levels
        cls.performance_benchmarks = [
            # Simple operations (should complete in <1 second, <50MB)
            PerformanceBenchmark(
                name="simple_integral",
                mathir_config={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "x",
                        "var": "x",
                        "limits": [0, 1],
                        "name": "I"
                    }],
                    "output": {"mode": "exact"}
                },
                max_time_seconds=1.0,
                max_memory_mb=50.0,
                description="Simple definite integral",
                complexity_level="simple"
            ),
            
            PerformanceBenchmark(
                name="simple_limit",
                mathir_config={
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
                max_time_seconds=1.0,
                max_memory_mb=50.0,
                description="Simple limit calculation",
                complexity_level="simple"
            ),
            
            # Medium complexity operations (should complete in <5 seconds, <100MB)
            PerformanceBenchmark(
                name="polynomial_solve",
                mathir_config={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "solve_for",
                        "unknowns": ["x"],
                        "equations": ["x^4 - 5*x^3 + 6*x^2 + 4*x - 8 = 0"]
                    }],
                    "output": {"mode": "exact"}
                },
                max_time_seconds=5.0,
                max_memory_mb=100.0,
                description="Fourth-degree polynomial solving",
                complexity_level="medium"
            ),
            
            PerformanceBenchmark(
                name="matrix_operations_3x3",
                mathir_config={
                    "expr_format": "latex",
                    "definitions": {
                        "matrices": [{
                            "name": "A",
                            "rows": 3,
                            "cols": 3,
                            "data": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
                        }]
                    },
                    "targets": [{
                        "type": "matrix_determinant",
                        "matrix_name": "A"
                    }],
                    "output": {"mode": "exact"}
                },
                max_time_seconds=2.0,
                max_memory_mb=100.0,
                description="3x3 matrix determinant",
                complexity_level="medium"
            ),
            
            # Complex operations (should complete in <15 seconds, <300MB)
            PerformanceBenchmark(
                name="complex_integral",
                mathir_config={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "\\frac{\\sin(x^2)}{x^2 + 1}",
                        "var": "x",
                        "limits": [0, 10],
                        "name": "I"
                    }],
                    "output": {"mode": "decimal", "round_to": 10}
                },
                max_time_seconds=15.0,
                max_memory_mb=300.0,
                description="Complex integral with oscillatory integrand",
                complexity_level="complex"
            ),
            
            PerformanceBenchmark(
                name="large_matrix_5x5",
                mathir_config={
                    "expr_format": "latex",
                    "definitions": {
                        "matrices": [{
                            "name": "A",
                            "rows": 5,
                            "cols": 5,
                            "data": [
                                ["1", "2", "3", "4", "5"],
                                ["6", "7", "8", "9", "10"],
                                ["11", "12", "13", "14", "15"],
                                ["16", "17", "18", "19", "20"],
                                ["21", "22", "23", "24", "26"]
                            ]
                        }]
                    },
                    "targets": [{
                        "type": "matrix_inverse",
                        "matrix_name": "A"
                    }],
                    "output": {"mode": "exact"}
                },
                max_time_seconds=10.0,
                max_memory_mb=200.0,
                description="5x5 matrix inversion",
                complexity_level="complex"
            ),
            
            # Extreme operations (should complete in <30 seconds, <500MB)
            PerformanceBenchmark(
                name="high_degree_polynomial",
                mathir_config={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "solve_for",
                        "unknowns": ["x"],
                        "equations": ["x^6 - 21*x^5 + 175*x^4 - 735*x^3 + 1624*x^2 - 1764*x + 720 = 0"]
                    }],
                    "output": {"mode": "exact"}
                },
                max_time_seconds=30.0,
                max_memory_mb=500.0,
                description="Sixth-degree polynomial solving",
                complexity_level="extreme"
            )
        ]
    
    @classmethod
    def teardown_class(cls):
        """Generate comprehensive performance report."""
        cls._generate_performance_report()
    
    @contextmanager
    def performance_monitor(self, test_name: str):
        """Context manager for detailed performance monitoring."""
        # Force garbage collection before test
        gc.collect()
        
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        # Track peak memory during execution
        peak_memory = start_memory
        
        def memory_tracker():
            nonlocal peak_memory
            while True:
                try:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)  # Check every 100ms
                except:
                    break
        
        # Start memory tracking thread
        tracker_thread = threading.Thread(target=memory_tracker, daemon=True)
        tracker_thread.start()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = max(start_cpu, end_cpu)
            
            # Record performance metrics
            result = BenchmarkResult(
                test_name=test_name,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                cpu_percent=cpu_usage,
                success=True
            )
            self.benchmark_results.append(result)
            
            # Force cleanup
            gc.collect()
    
    # ========================================================================
    # EXECUTION TIME BENCHMARKS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_simple_operations_performance(self):
        """Test that simple operations complete within strict time limits."""
        simple_benchmarks = [b for b in self.performance_benchmarks if b.complexity_level == "simple"]
        
        for benchmark in simple_benchmarks:
            with self.performance_monitor(benchmark.name):
                ir = MathIR.model_validate(benchmark.mathir_config)
                results = run_mathir(ir)
                
                # Verify we got results
                assert len(results) > 0, f"No results returned for {benchmark.name}"
                
                # Check performance constraints
                latest_result = self.benchmark_results[-1]
                assert latest_result.execution_time <= benchmark.max_time_seconds, \
                    f"{benchmark.name} took {latest_result.execution_time:.3f}s, exceeds limit {benchmark.max_time_seconds}s"
                assert latest_result.memory_usage_mb <= benchmark.max_memory_mb, \
                    f"{benchmark.name} used {latest_result.memory_usage_mb:.1f}MB, exceeds limit {benchmark.max_memory_mb}MB"
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_medium_complexity_performance(self):
        """Test that medium complexity operations complete within reasonable time limits."""
        medium_benchmarks = [b for b in self.performance_benchmarks if b.complexity_level == "medium"]
        
        for benchmark in medium_benchmarks:
            with self.performance_monitor(benchmark.name):
                ir = MathIR.model_validate(benchmark.mathir_config)
                results = run_mathir(ir)
                
                # Verify we got results
                assert len(results) > 0, f"No results returned for {benchmark.name}"
                
                # Check performance constraints
                latest_result = self.benchmark_results[-1]
                assert latest_result.execution_time <= benchmark.max_time_seconds, \
                    f"{benchmark.name} took {latest_result.execution_time:.3f}s, exceeds limit {benchmark.max_time_seconds}s"
                assert latest_result.memory_usage_mb <= benchmark.max_memory_mb, \
                    f"{benchmark.name} used {latest_result.memory_usage_mb:.1f}MB, exceeds limit {benchmark.max_memory_mb}MB"
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_complex_operations_performance(self):
        """Test that complex operations complete within acceptable time limits."""
        complex_benchmarks = [b for b in self.performance_benchmarks if b.complexity_level == "complex"]
        
        for benchmark in complex_benchmarks:
            with self.performance_monitor(benchmark.name):
                ir = MathIR.model_validate(benchmark.mathir_config)
                results = run_mathir(ir)
                
                # Verify we got results
                assert len(results) > 0, f"No results returned for {benchmark.name}"
                
                # Check performance constraints
                latest_result = self.benchmark_results[-1]
                assert latest_result.execution_time <= benchmark.max_time_seconds, \
                    f"{benchmark.name} took {latest_result.execution_time:.3f}s, exceeds limit {benchmark.max_time_seconds}s"
                assert latest_result.memory_usage_mb <= benchmark.max_memory_mb, \
                    f"{benchmark.name} used {latest_result.memory_usage_mb:.1f}MB, exceeds limit {benchmark.max_memory_mb}MB"
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    @pytest.mark.slow
    def test_extreme_operations_performance(self):
        """Test that extreme operations complete within maximum time limits."""
        extreme_benchmarks = [b for b in self.performance_benchmarks if b.complexity_level == "extreme"]
        
        for benchmark in extreme_benchmarks:
            with self.performance_monitor(benchmark.name):
                ir = MathIR.model_validate(benchmark.mathir_config)
                results = run_mathir(ir)
                
                # Verify we got results
                assert len(results) > 0, f"No results returned for {benchmark.name}"
                
                # Check performance constraints
                latest_result = self.benchmark_results[-1]
                assert latest_result.execution_time <= benchmark.max_time_seconds, \
                    f"{benchmark.name} took {latest_result.execution_time:.3f}s, exceeds limit {benchmark.max_time_seconds}s"
                assert latest_result.memory_usage_mb <= benchmark.max_memory_mb, \
                    f"{benchmark.name} used {latest_result.memory_usage_mb:.1f}MB, exceeds limit {benchmark.max_memory_mb}MB"
    
    # ========================================================================
    # MEMORY USAGE MONITORING
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_memory_usage_limits(self):
        """Test that memory usage stays within acceptable limits."""
        test_cases = [
            {
                "name": "memory_intensive_integral",
                "config": {
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "x^3 * \\sin(x^2)",
                        "var": "x",
                        "limits": [0, 5],
                        "name": "I"
                    }],
                    "output": {"mode": "decimal", "round_to": 8}
                },
                "max_memory_mb": 200.0
            },
            {
                "name": "memory_intensive_matrix",
                "config": {
                    "expr_format": "latex",
                    "definitions": {
                        "matrices": [{
                            "name": "A",
                            "rows": 4,
                            "cols": 4,
                            "data": [
                                ["1", "2", "3", "4"],
                                ["5", "6", "7", "8"],
                                ["9", "10", "11", "12"],
                                ["13", "14", "15", "17"]
                            ]
                        }]
                    },
                    "targets": [{
                        "type": "matrix_determinant",
                        "matrix_name": "A"
                    }],
                    "output": {"mode": "exact"}
                },
                "max_memory_mb": 150.0
            }
        ]
        
        for test_case in test_cases:
            with self.performance_monitor(test_case["name"]):
                ir = MathIR.model_validate(test_case["config"])
                results = run_mathir(ir)
                
                # Verify we got results
                assert len(results) > 0, f"No results returned for {test_case['name']}"
                
                # Check memory usage
                latest_result = self.benchmark_results[-1]
                assert latest_result.peak_memory_mb <= test_case["max_memory_mb"] + self.baseline_memory, \
                    f"{test_case['name']} peak memory {latest_result.peak_memory_mb:.1f}MB exceeds limit {test_case['max_memory_mb']}MB"
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks by running multiple operations and checking memory growth."""
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Run the same operation multiple times
        test_config = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "integral_def",
                "expr": "x^2",
                "var": "x",
                "limits": [0, 1],
                "name": "I"
            }],
            "output": {"mode": "exact"}
        }
        
        memory_measurements = []
        
        for i in range(10):  # Run 10 iterations
            with self.performance_monitor(f"memory_leak_test_{i}"):
                ir = MathIR.model_validate(test_config)
                results = run_mathir(ir)
                
                # Force garbage collection
                gc.collect()
                
                # Measure memory after cleanup
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
        
        # Check for significant memory growth (more than 50MB increase)
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        assert memory_growth < 50.0, f"Potential memory leak detected: {memory_growth:.1f}MB growth over 10 iterations"
    
    # ========================================================================
    # THROUGHPUT TESTING
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_throughput_simple_operations(self):
        """Test throughput for simple operations - should achieve >100 tasks/hour."""
        simple_config = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "value",
                "name": "result",
                "expr": "2 + 3"
            }],
            "output": {"mode": "exact"}
        }
        
        num_tasks = 20
        start_time = time.time()
        
        for i in range(num_tasks):
            ir = MathIR.model_validate(simple_config)
            results = run_mathir(ir)
            assert len(results) > 0, f"No results for task {i}"
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput_per_second = num_tasks / total_time
        throughput_per_hour = throughput_per_second * 3600
        
        # Should achieve at least 100 tasks per hour (very conservative for simple operations)
        assert throughput_per_hour >= 100, \
            f"Throughput {throughput_per_hour:.1f} tasks/hour is below minimum requirement of 100 tasks/hour"
        
        # Record throughput result
        result = BenchmarkResult(
            test_name="throughput_simple",
            execution_time=total_time,
            memory_usage_mb=0,  # Not measured in this test
            peak_memory_mb=0,
            cpu_percent=0,
            success=True,
            throughput_tasks_per_second=throughput_per_second
        )
        self.benchmark_results.append(result)
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_throughput_mixed_operations(self):
        """Test throughput for mixed complexity operations."""
        mixed_configs = [
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "integral_def",
                    "expr": "x",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I"
                }],
                "output": {"mode": "exact"}
            },
            {
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
            {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "solve_for",
                    "unknowns": ["x"],
                    "equations": ["x^2 - 4 = 0"]
                }],
                "output": {"mode": "exact"}
            }
        ]
        
        num_iterations = 5
        start_time = time.time()
        total_tasks = 0
        
        for _ in range(num_iterations):
            for config in mixed_configs:
                ir = MathIR.model_validate(config)
                results = run_mathir(ir)
                assert len(results) > 0, "No results returned"
                total_tasks += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput_per_second = total_tasks / total_time
        throughput_per_hour = throughput_per_second * 3600
        
        # Should achieve reasonable throughput for mixed operations
        assert throughput_per_hour >= 50, \
            f"Mixed operations throughput {throughput_per_hour:.1f} tasks/hour is below minimum requirement"
        
        # Record throughput result
        result = BenchmarkResult(
            test_name="throughput_mixed",
            execution_time=total_time,
            memory_usage_mb=0,
            peak_memory_mb=0,
            cpu_percent=0,
            success=True,
            throughput_tasks_per_second=throughput_per_second
        )
        self.benchmark_results.append(result)
    
    # ========================================================================
    # SCALABILITY ANALYSIS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_scalability_polynomial_degree(self):
        """Test how execution time scales with polynomial degree."""
        degrees = [2, 3, 4, 5]
        execution_times = []
        
        for degree in degrees:
            # Create polynomial equation of given degree
            coeffs = [1] * (degree + 1)  # Simple polynomial with all coefficients = 1
            equation = " + ".join([f"x^{degree-i}" if degree-i > 1 else "x" if degree-i == 1 else "1" 
                                 for i in range(degree)]) + " + 1 = 0"
            
            config = {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "solve_for",
                    "unknowns": ["x"],
                    "equations": [equation]
                }],
                "output": {"mode": "exact"}
            }
            
            with self.performance_monitor(f"scalability_degree_{degree}"):
                ir = MathIR.model_validate(config)
                results = run_mathir(ir)
                assert len(results) > 0, f"No results for degree {degree}"
            
            execution_times.append(self.benchmark_results[-1].execution_time)
        
        # Check that growth is reasonable (not exponential)
        # Time should not increase by more than 10x from degree 2 to 5
        time_ratio = execution_times[-1] / execution_times[0]
        assert time_ratio < 10.0, \
            f"Execution time scaling is too steep: {time_ratio:.2f}x increase from degree 2 to 5"
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_scalability_matrix_size(self):
        """Test how execution time scales with matrix size."""
        sizes = [2, 3, 4]
        execution_times = []
        
        for size in sizes:
            # Create matrix of given size
            matrix_data = [[str(i * size + j + 1) for j in range(size)] for i in range(size)]
            # Make it non-singular by adjusting the last diagonal element
            matrix_data[-1][-1] = str(int(matrix_data[-1][-1]) + 1)
            
            config = {
                "expr_format": "latex",
                "definitions": {
                    "matrices": [{
                        "name": "A",
                        "rows": size,
                        "cols": size,
                        "data": matrix_data
                    }]
                },
                "targets": [{
                    "type": "matrix_determinant",
                    "matrix_name": "A"
                }],
                "output": {"mode": "exact"}
            }
            
            with self.performance_monitor(f"scalability_matrix_{size}x{size}"):
                ir = MathIR.model_validate(config)
                results = run_mathir(ir)
                assert len(results) > 0, f"No results for {size}x{size} matrix"
            
            execution_times.append(self.benchmark_results[-1].execution_time)
        
        # Check that growth follows expected O(n³) pattern for determinant calculation
        # Time should increase roughly as n³, so 4x4 should take ~8x longer than 2x2
        if len(execution_times) >= 2:
            time_ratio = execution_times[-1] / execution_times[0]
            expected_ratio = (sizes[-1] / sizes[0]) ** 3
            # Allow for some variance in the scaling
            assert time_ratio < expected_ratio * 2, \
                f"Matrix determinant scaling is worse than expected: {time_ratio:.2f}x vs expected ~{expected_ratio:.2f}x"
    
    # ========================================================================
    # CONCURRENT PROCESSING
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    def test_concurrent_processing(self):
        """Test concurrent processing of multiple mathematical tasks."""
        def process_task(task_id: int) -> Tuple[int, float, bool]:
            """Process a single mathematical task."""
            config = {
                "expr_format": "latex",
                "symbols": [{"name": "x", "domain": "R"}],
                "targets": [{
                    "type": "integral_def",
                    "expr": f"x^{task_id % 3 + 1}",  # Vary complexity
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I"
                }],
                "output": {"mode": "exact"}
            }
            
            start_time = time.time()
            try:
                ir = MathIR.model_validate(config)
                results = run_mathir(ir)
                success = len(results) > 0
            except Exception:
                success = False
            
            execution_time = time.time() - start_time
            return task_id, execution_time, success
        
        # Test with 5 concurrent tasks
        num_concurrent_tasks = 5
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_tasks) as executor:
            start_time = time.time()
            
            # Submit all tasks
            futures = [executor.submit(process_task, i) for i in range(num_concurrent_tasks)]
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                task_id, exec_time, success = future.result()
                results.append((task_id, exec_time, success))
                assert success, f"Concurrent task {task_id} failed"
            
            total_time = time.time() - start_time
        
        # All tasks should complete successfully
        assert len(results) == num_concurrent_tasks, "Not all concurrent tasks completed"
        
        # Total time should be less than sum of individual times (parallelization benefit)
        individual_times_sum = sum(exec_time for _, exec_time, _ in results)
        parallelization_efficiency = individual_times_sum / total_time
        
        # Should achieve some parallelization benefit (efficiency > 1.5)
        assert parallelization_efficiency > 1.5, \
            f"Poor parallelization efficiency: {parallelization_efficiency:.2f}"
        
        # Record concurrent processing result
        result = BenchmarkResult(
            test_name="concurrent_processing",
            execution_time=total_time,
            memory_usage_mb=0,
            peak_memory_mb=0,
            cpu_percent=0,
            success=True,
            throughput_tasks_per_second=num_concurrent_tasks / total_time
        )
        self.benchmark_results.append(result)
    
    # ========================================================================
    # STRESS TESTING
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_continuous_operations(self):
        """Stress test with continuous operations over extended period."""
        test_duration_seconds = 30  # Run for 30 seconds
        start_time = time.time()
        task_count = 0
        failures = 0
        
        config = {
            "expr_format": "latex",
            "symbols": [{"name": "x", "domain": "R"}],
            "targets": [{
                "type": "integral_def",
                "expr": "x^2",
                "var": "x",
                "limits": [0, 1],
                "name": "I"
            }],
            "output": {"mode": "exact"}
        }
        
        while time.time() - start_time < test_duration_seconds:
            try:
                ir = MathIR.model_validate(config)
                results = run_mathir(ir)
                if len(results) == 0:
                    failures += 1
                task_count += 1
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception:
                failures += 1
                task_count += 1
        
        total_time = time.time() - start_time
        success_rate = (task_count - failures) / task_count if task_count > 0 else 0
        throughput = task_count / total_time
        
        # Should maintain high success rate under stress
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} is below 95% under stress"
        
        # Should maintain reasonable throughput
        assert throughput >= 5.0, f"Throughput {throughput:.1f} tasks/sec is too low under stress"
        
        # Record stress test result
        result = BenchmarkResult(
            test_name="stress_continuous",
            execution_time=total_time,
            memory_usage_mb=0,
            peak_memory_mb=0,
            cpu_percent=0,
            success=success_rate >= 0.95,
            throughput_tasks_per_second=throughput
        )
        self.benchmark_results.append(result)
    
    # ========================================================================
    # PERFORMANCE REPORTING
    # ========================================================================
    
    @classmethod
    def _generate_performance_report(cls):
        """Generate comprehensive performance report."""
        if not cls.benchmark_results:
            print("No performance benchmark results available")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("="*80)
        
        # Overall statistics
        total_tests = len(cls.benchmark_results)
        successful_tests = sum(1 for r in cls.benchmark_results if r.success)
        
        print(f"Total Performance Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Failed Tests: {total_tests - successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
        
        # Execution time statistics
        execution_times = [r.execution_time for r in cls.benchmark_results if r.execution_time > 0]
        if execution_times:
            print(f"\nExecution Time Statistics:")
            print(f"  Average: {statistics.mean(execution_times):.3f}s")
            print(f"  Median: {statistics.median(execution_times):.3f}s")
            print(f"  Min: {min(execution_times):.3f}s")
            print(f"  Max: {max(execution_times):.3f}s")
            print(f"  95th Percentile: {sorted(execution_times)[int(0.95*len(execution_times))]:.3f}s")
        
        # Memory usage statistics
        memory_usage = [r.memory_usage_mb for r in cls.benchmark_results if r.memory_usage_mb > 0]
        if memory_usage:
            print(f"\nMemory Usage Statistics:")
            print(f"  Average: {statistics.mean(memory_usage):.1f}MB")
            print(f"  Median: {statistics.median(memory_usage):.1f}MB")
            print(f"  Min: {min(memory_usage):.1f}MB")
            print(f"  Max: {max(memory_usage):.1f}MB")
            print(f"  95th Percentile: {sorted(memory_usage)[int(0.95*len(memory_usage))]:.1f}MB")
        
        # Throughput statistics
        throughput_results = [r for r in cls.benchmark_results if r.throughput_tasks_per_second is not None]
        if throughput_results:
            throughputs = [r.throughput_tasks_per_second for r in throughput_results]
            print(f"\nThroughput Statistics:")
            print(f"  Average: {statistics.mean(throughputs):.1f} tasks/second")
            print(f"  Peak: {max(throughputs):.1f} tasks/second")
            print(f"  Hourly Capacity: {max(throughputs)*3600:.0f} tasks/hour")
        
        # Performance requirements compliance
        print(f"\nPerformance Requirements Compliance:")
        
        # Check execution time requirement (<30s per task)
        slow_tasks = [r for r in cls.benchmark_results if r.execution_time > 30.0]
        print(f"  Tasks exceeding 30s limit: {len(slow_tasks)}")
        
        # Check memory requirement (<500MB per task)
        memory_heavy_tasks = [r for r in cls.benchmark_results if r.peak_memory_mb > 500.0]
        print(f"  Tasks exceeding 500MB limit: {len(memory_heavy_tasks)}")
        
        # Check throughput requirement (>100 tasks/hour)
        if throughput_results:
            max_hourly_throughput = max(r.throughput_tasks_per_second * 3600 for r in throughput_results)
            throughput_ok = max_hourly_throughput >= 100
            print(f"  Throughput requirement (>100/hour): {'✓ PASS' if throughput_ok else '✗ FAIL'}")
        
        # Individual test results
        print(f"\nIndividual Test Results:")
        print(f"{'Test Name':<30} {'Time (s)':<10} {'Memory (MB)':<12} {'Status':<8}")
        print("-" * 70)
        
        for result in cls.benchmark_results:
            status = "PASS" if result.success else "FAIL"
            print(f"{result.test_name:<30} {result.execution_time:<10.3f} {result.memory_usage_mb:<12.1f} {status:<8}")
        
        # Performance grade
        overall_grade = cls._calculate_performance_grade()
        print(f"\nOverall Performance Grade: {overall_grade}")
        
        print("="*80)
    
    @classmethod
    def _calculate_performance_grade(cls) -> str:
        """Calculate overall performance grade based on test results."""
        if not cls.benchmark_results:
            return "INCOMPLETE"
        
        success_rate = sum(1 for r in cls.benchmark_results if r.success) / len(cls.benchmark_results)
        
        # Check specific requirements
        execution_times = [r.execution_time for r in cls.benchmark_results if r.execution_time > 0]
        memory_usage = [r.peak_memory_mb for r in cls.benchmark_results if r.peak_memory_mb > 0]
        
        time_compliance = sum(1 for t in execution_times if t <= 30.0) / len(execution_times) if execution_times else 1.0
        memory_compliance = sum(1 for m in memory_usage if m <= 500.0) / len(memory_usage) if memory_usage else 1.0
        
        # Calculate overall score
        overall_score = (success_rate * 0.4 + time_compliance * 0.3 + memory_compliance * 0.3)
        
        if overall_score >= 0.95:
            return "EXCELLENT (A+)"
        elif overall_score >= 0.90:
            return "VERY GOOD (A)"
        elif overall_score >= 0.85:
            return "GOOD (B+)"
        elif overall_score >= 0.80:
            return "ACCEPTABLE (B)"
        elif overall_score >= 0.70:
            return "NEEDS IMPROVEMENT (C)"
        else:
            return "POOR (D)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
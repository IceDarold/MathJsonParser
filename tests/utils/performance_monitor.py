"""
Performance monitoring utilities for MathIR Parser testing.

This module provides tools for measuring execution time, memory usage,
and other performance metrics during test execution.
"""

import time
import psutil
import os
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading


@dataclass
class PerformanceMeasurement:
    """Data class for storing performance measurement results."""
    name: str
    start_time: float
    end_time: float
    elapsed_time: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_percent: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Performance profiler for measuring execution metrics."""
    
    def __init__(self):
        """Initialize the performance profiler."""
        self.measurements: Dict[str, float] = {}
        self.detailed_measurements: Dict[str, PerformanceMeasurement] = {}
        self.process = psutil.Process(os.getpid())
        self._lock = threading.Lock()
    
    @contextmanager
    def measure(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for measuring performance of a code block.
        
        Args:
            name: Name of the measurement
            metadata: Additional metadata to store with the measurement
            
        Yields:
            Timer object with elapsed_time property
        """
        if metadata is None:
            metadata = {}
        
        # Get initial measurements
        start_time = time.time()
        memory_start = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_start = self.process.cpu_percent()
        
        class Timer:
            def __init__(self):
                self.elapsed_time = 0
        
        timer = Timer()
        
        try:
            yield timer
        finally:
            # Get final measurements
            end_time = time.time()
            elapsed_time = end_time - start_time
            timer.elapsed_time = elapsed_time
            
            memory_end = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_end = self.process.cpu_percent()
            
            # Store measurements
            with self._lock:
                self.measurements[name] = elapsed_time
                
                self.detailed_measurements[name] = PerformanceMeasurement(
                    name=name,
                    start_time=start_time,
                    end_time=end_time,
                    elapsed_time=elapsed_time,
                    memory_start=memory_start,
                    memory_end=memory_end,
                    memory_peak=max(memory_start, memory_end),
                    cpu_percent=(cpu_start + cpu_end) / 2,
                    metadata=metadata
                )
    
    def get_measurement(self, name: str) -> Optional[PerformanceMeasurement]:
        """Get detailed measurement by name."""
        return self.detailed_measurements.get(name)
    
    def get_elapsed_time(self, name: str) -> Optional[float]:
        """Get elapsed time for a measurement by name."""
        return self.measurements.get(name)
    
    def get_all_measurements(self) -> Dict[str, PerformanceMeasurement]:
        """Get all detailed measurements."""
        return self.detailed_measurements.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.measurements:
            return {}
        
        times = list(self.measurements.values())
        return {
            "total_measurements": len(times),
            "total_time": sum(times),
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "measurements": dict(self.measurements)
        }
    
    def clear(self):
        """Clear all measurements."""
        with self._lock:
            self.measurements.clear()
            self.detailed_measurements.clear()
    
    def print_summary(self):
        """Print a formatted summary of all measurements."""
        summary = self.get_summary()
        if not summary:
            print("No performance measurements recorded.")
            return
        
        print("\nPerformance Summary:")
        print(f"  Total measurements: {summary['total_measurements']}")
        print(f"  Total time: {summary['total_time']:.3f}s")
        print(f"  Average time: {summary['average_time']:.3f}s")
        print(f"  Min time: {summary['min_time']:.3f}s")
        print(f"  Max time: {summary['max_time']:.3f}s")
        print("\nIndividual measurements:")
        for name, time_taken in summary['measurements'].items():
            print(f"    {name}: {time_taken:.3f}s")


class BenchmarkRunner:
    """Runner for performance benchmarks."""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        """
        Initialize benchmark runner.
        
        Args:
            profiler: Performance profiler to use (creates new one if None)
        """
        self.profiler = profiler or PerformanceProfiler()
        self.benchmarks: List[Dict[str, Any]] = []
    
    def add_benchmark(self, name: str, func, *args, **kwargs):
        """
        Add a benchmark function to run.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        self.benchmarks.append({
            "name": name,
            "func": func,
            "args": args,
            "kwargs": kwargs
        })
    
    def run_benchmarks(self, iterations: int = 1) -> Dict[str, List[float]]:
        """
        Run all benchmarks.
        
        Args:
            iterations: Number of times to run each benchmark
            
        Returns:
            Dictionary mapping benchmark names to lists of execution times
        """
        results = {}
        
        for benchmark in self.benchmarks:
            name = benchmark["name"]
            func = benchmark["func"]
            args = benchmark["args"]
            kwargs = benchmark["kwargs"]
            
            times = []
            
            for i in range(iterations):
                with self.profiler.measure(f"{name}_iter_{i}"):
                    try:
                        func(*args, **kwargs)
                        times.append(self.profiler.get_elapsed_time(f"{name}_iter_{i}"))
                    except Exception as e:
                        print(f"Benchmark {name} iteration {i} failed: {e}")
                        times.append(float('inf'))
            
            results[name] = times
        
        return results
    
    def run_single_benchmark(self, name: str, func, iterations: int = 1, *args, **kwargs) -> List[float]:
        """
        Run a single benchmark function.
        
        Args:
            name: Name of the benchmark
            func: Function to benchmark
            iterations: Number of iterations
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            List of execution times
        """
        times = []
        
        for i in range(iterations):
            with self.profiler.measure(f"{name}_iter_{i}"):
                try:
                    func(*args, **kwargs)
                    times.append(self.profiler.get_elapsed_time(f"{name}_iter_{i}"))
                except Exception as e:
                    print(f"Benchmark {name} iteration {i} failed: {e}")
                    times.append(float('inf'))
        
        return times
    
    def compare_benchmarks(self, results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Compare benchmark results and generate statistics.
        
        Args:
            results: Results from run_benchmarks()
            
        Returns:
            Dictionary with statistics for each benchmark
        """
        stats = {}
        
        for name, times in results.items():
            valid_times = [t for t in times if t != float('inf')]
            
            if valid_times:
                stats[name] = {
                    "min": min(valid_times),
                    "max": max(valid_times),
                    "avg": sum(valid_times) / len(valid_times),
                    "median": sorted(valid_times)[len(valid_times) // 2],
                    "total": sum(valid_times),
                    "iterations": len(valid_times),
                    "failures": len(times) - len(valid_times)
                }
            else:
                stats[name] = {
                    "min": float('inf'),
                    "max": float('inf'),
                    "avg": float('inf'),
                    "median": float('inf'),
                    "total": float('inf'),
                    "iterations": 0,
                    "failures": len(times)
                }
        
        return stats
    
    def print_benchmark_results(self, results: Dict[str, List[float]]):
        """Print formatted benchmark results."""
        stats = self.compare_benchmarks(results)
        
        print("\nBenchmark Results:")
        print("-" * 80)
        print(f"{'Benchmark':<30} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Iterations':<12}")
        print("-" * 80)
        
        for name, stat in stats.items():
            if stat['avg'] != float('inf'):
                print(f"{name:<30} {stat['avg']:<10.3f} {stat['min']:<10.3f} {stat['max']:<10.3f} {stat['iterations']:<12}")
            else:
                print(f"{name:<30} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {stat['failures']:<12}")


class MemoryMonitor:
    """Monitor memory usage during test execution."""
    
    def __init__(self):
        """Initialize memory monitor."""
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.get_current_memory()
        self.peak_memory = self.baseline_memory
        self.measurements = []
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def record_measurement(self, label: str = ""):
        """Record current memory usage with optional label."""
        current_memory = self.get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        measurement = {
            "timestamp": time.time(),
            "memory_mb": current_memory,
            "delta_from_baseline": current_memory - self.baseline_memory,
            "label": label
        }
        
        self.measurements.append(measurement)
        return measurement
    
    @contextmanager
    def monitor_block(self, label: str = ""):
        """Context manager to monitor memory usage of a code block."""
        start_memory = self.get_current_memory()
        self.record_measurement(f"{label}_start")
        
        try:
            yield
        finally:
            end_memory = self.get_current_memory()
            self.record_measurement(f"{label}_end")
            
            delta = end_memory - start_memory
            print(f"Memory usage for '{label}': {delta:+.2f} MB")
    
    def get_memory_summary(self) -> Dict[str, float]:
        """Get memory usage summary."""
        current_memory = self.get_current_memory()
        
        return {
            "baseline_memory": self.baseline_memory,
            "current_memory": current_memory,
            "peak_memory": self.peak_memory,
            "total_increase": current_memory - self.baseline_memory,
            "peak_increase": self.peak_memory - self.baseline_memory
        }
    
    def print_memory_summary(self):
        """Print formatted memory usage summary."""
        summary = self.get_memory_summary()
        
        print("\nMemory Usage Summary:")
        print(f"  Baseline: {summary['baseline_memory']:.2f} MB")
        print(f"  Current:  {summary['current_memory']:.2f} MB")
        print(f"  Peak:     {summary['peak_memory']:.2f} MB")
        print(f"  Total increase: {summary['total_increase']:+.2f} MB")
        print(f"  Peak increase:  {summary['peak_increase']:+.2f} MB")


# Convenience functions
def time_function(func, *args, **kwargs) -> tuple:
    """
    Time a function execution.
    
    Args:
        func: Function to time
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time


def benchmark_function(func, iterations: int = 10, *args, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function over multiple iterations.
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations to run
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    for _ in range(iterations):
        _, exec_time = time_function(func, *args, **kwargs)
        times.append(exec_time)
    
    return {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times),
        "total": sum(times),
        "iterations": iterations
    }
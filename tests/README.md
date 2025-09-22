# MathIR Parser Test Infrastructure

This directory contains the comprehensive testing infrastructure for the MathIR Parser system, designed to support unit, integration, end-to-end, regression, and stress testing of mathematical computations.

## Directory Structure

```
tests/
├── unit/                           # Unit tests for individual components
├── integration/                    # Integration tests for component interactions
├── end_to_end/                    # End-to-end workflow tests
├── regression/                    # Regression tests for previously fixed issues
├── stress/                        # Stress and performance tests
├── fixtures/                      # Test fixtures and data
│   ├── mathematical_expressions/  # Mathematical expression test cases
│   ├── expected_results/          # Expected computation results
│   └── edge_cases/               # Edge case and error condition tests
├── utils/                         # Test utilities and helper functions
├── logs/                          # Test execution logs
├── conftest.py                    # Shared pytest fixtures and configuration
└── README.md                      # This file
```

## Core Test Utilities

### Test Helpers (`tests/utils/test_helpers.py`)
- **TestDataLoader**: Load test data from JSON files and directories
- **MathIRTestHelper**: Create and manipulate MathIR objects for testing
- **TestResultValidator**: Validate test results and structure

### Mathematical Assertions (`tests/utils/assertion_helpers.py`)
- **PrecisionAssertion**: Precision-aware numerical comparisons
- **MathematicalValidator**: Specialized validation for different mathematical operations
- Convenience functions for common mathematical assertions

### Data Generators (`tests/utils/data_generators.py`)
- **TestDataGenerator**: Generate mathematical expressions and test cases
- **MathIRGenerator**: Generate complete MathIR objects for testing
- Support for polynomial, trigonometric, exponential, and complex expressions

### Mathematical Validation Framework (`tests/utils/mathematical_validator.py`)
- **MathematicalAccuracyValidator**: Comprehensive validation for mathematical computations
- Support for exact symbolic and numerical result validation
- Specialized validation for integrals, limits, sums, equation solving, and matrix operations

### Performance Monitoring (`tests/utils/performance_monitor.py`)
- **PerformanceProfiler**: Measure execution time, memory usage, and CPU utilization
- **BenchmarkRunner**: Run performance benchmarks and comparisons
- **MemoryMonitor**: Track memory usage during test execution

### Fixture Loading (`tests/utils/fixture_loader.py`)
- **FixtureLoader**: Load test fixtures from various file formats
- **FixtureManager**: Organize and manage test fixtures by category
- **TestFixture**: Container for fixture data with metadata

## Configuration

### pytest.ini
- Test discovery configuration
- Markers for test categorization
- Coverage settings
- Logging configuration
- Timeout and parallel execution settings

### conftest.py
- Shared pytest fixtures
- Test environment setup
- Custom pytest hooks and markers
- Mathematical computation fixtures

## Usage Examples

### Basic Test Setup

```python
import pytest
from tests.utils.test_helpers import MathIRTestHelper
from tests.utils.assertion_helpers import assert_integral_equal

def test_simple_integral(mathir_helper):
    # Create a simple integral test
    mathir = mathir_helper.create_integral_mathir(
        expr="x^2",
        var="x", 
        limits=[0, 1],
        name="result"
    )
    
    # Run computation
    results = run_mathir(mathir)
    
    # Validate result
    assert_integral_equal(results["result"], sp.Rational(1, 3))
```

### Using Fixtures

```python
def test_with_fixture(fixture_loader):
    # Load a test fixture
    fixture = fixture_loader.load_test_fixture("integral_x_squared")
    mathir = fixture_loader.create_mathir_from_fixture(fixture)
    
    # Run test
    results = run_mathir(mathir)
    
    # Validate against expected results
    expected = fixture.expected_results
    assert results["integral_result"] == expected["integral_result"]
```

### Performance Testing

```python
from tests.utils.performance_monitor import performance_test

@performance_test(timeout=30.0)
def test_complex_computation_performance():
    # Test code here
    pass

def test_benchmark_comparison(benchmark_runner):
    # Run benchmark
    result = benchmark_runner.run_benchmark(
        "integral_computation",
        run_mathir,
        iterations=100,
        mathir_object
    )
    
    # Check performance metrics
    assert result["avg_execution_time"] < 0.1  # 100ms threshold
```

### Mathematical Validation

```python
from tests.utils.mathematical_validator import validate_mathematical_result

def test_limit_computation():
    actual_result = compute_limit("sin(x)/x", "x", "0")
    expected_result = 1
    
    # Validate with specialized limit validation
    assert validate_mathematical_result(
        actual_result, 
        expected_result,
        operation_type="limit",
        limit_point="0",
        tolerance=1e-10
    )
```

## Test Categories and Markers

### Available Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.end_to_end`: End-to-end tests
- `@pytest.mark.regression`: Regression tests
- `@pytest.mark.stress`: Stress tests
- `@pytest.mark.mathematical`: Mathematical computation tests
- `@pytest.mark.symbolic`: Symbolic mathematics tests
- `@pytest.mark.numerical`: Numerical computation tests
- `@pytest.mark.edge_case`: Edge case tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow-running tests

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run mathematical computation tests
pytest -m mathematical

# Run tests excluding slow ones
pytest -m "not slow"

# Run integration and end-to-end tests
pytest -m "integration or end_to_end"
```

## Mathematical Operation Support

### Supported Operations
- **Integrals**: Definite and indefinite integrals with symbolic and numerical validation
- **Limits**: Limits to finite points, infinity, and indeterminate forms
- **Summations**: Finite sums and infinite series with convergence testing
- **Equation Solving**: Algebraic, transcendental, and differential equations
- **Matrix Operations**: Matrix arithmetic, eigenvalues, determinants, and inverses
- **Complex Numbers**: Complex arithmetic and special functions

### Validation Features
- Exact symbolic comparison using SymPy
- Numerical comparison with configurable tolerance
- Special value handling (infinity, NaN, complex numbers)
- Mathematical constant recognition (π, e, i)
- Error condition testing

## Performance Testing

### Metrics Collected
- Execution time (wall clock and CPU time)
- Memory usage (RSS and peak memory)
- CPU utilization percentage
- Success/failure rates

### Benchmarking Features
- Multiple iteration support
- Statistical analysis (mean, min, max)
- Comparison between different implementations
- Timeout handling for long-running tests

## Fixture Management

### Fixture Categories
- **Mathematical Expressions**: Basic and complex mathematical expressions
- **Expected Results**: Known correct results for validation
- **Edge Cases**: Error conditions and boundary cases

### Fixture Format
```json
{
  "description": "Test description",
  "mathir": {
    "expr_format": "latex",
    "symbols": [...],
    "targets": [...],
    "output": {...}
  },
  "expected_results": {
    "result_name": "expected_value"
  },
  "metadata": {
    "category": "test_category",
    "difficulty": "basic|intermediate|advanced"
  }
}
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mathir_parser

# Run specific test file
pytest tests/unit/test_integrals.py

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

### Advanced Options
```bash
# Run slow tests
pytest --runslow

# Set custom tolerance
pytest --tolerance=1e-12

# Generate HTML coverage report
pytest --cov=mathir_parser --cov-report=html

# Run with specific log level
pytest --log-cli-level=DEBUG
```

## Contributing

When adding new tests:

1. **Choose the appropriate directory** based on test type
2. **Use existing utilities** from `tests/utils/` when possible
3. **Add appropriate markers** to categorize tests
4. **Include docstrings** explaining test purpose
5. **Use fixtures** for reusable test data
6. **Validate results** using mathematical validators
7. **Consider performance** implications for complex tests

### Test Naming Conventions
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`
- Fixtures: descriptive names without `test_` prefix

### Mathematical Test Guidelines
- Use exact symbolic comparison when possible
- Specify appropriate numerical tolerance for floating-point comparisons
- Test edge cases and error conditions
- Include performance tests for complex computations
- Validate both symbolic and numerical output modes

## Dependencies

The test infrastructure requires:
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pytest-xdist`: Parallel test execution
- `pytest-timeout`: Test timeout handling
- `pytest-benchmark`: Performance benchmarking
- `sympy`: Symbolic mathematics
- `numpy`: Numerical computations
- `psutil`: System monitoring
- `jsonschema`: JSON validation
- `pyyaml`: YAML fixture support

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-benchmark psutil jsonschema pyyaml
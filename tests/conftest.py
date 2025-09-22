"""
Shared fixtures and test configuration for MathIR Parser testing.

This module provides pytest fixtures, configuration, and utilities that are
shared across all test modules in the MathIR Parser test suite.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
import sympy as sp
from mathir_parser.main import MathIR, run_mathir

# Import test utilities
from tests.utils.test_helpers import (
    TestDataLoader, MathIRTestHelper, TestResultValidator,
    run_mathir_test, get_test_data_path
)
from tests.utils.assertion_helpers import (
    PrecisionAssertion, MathematicalValidator
)
from tests.utils.data_generators import (
    TestDataGenerator, MathIRGenerator
)
from tests.utils.performance_monitor import (
    PerformanceProfiler, BenchmarkRunner
)
from tests.utils.fixture_loader import (
    FixtureLoader, FixtureManager, TestFixture
)


# Test configuration constants
DEFAULT_TOLERANCE = 1e-10
DEFAULT_TIMEOUT = 30.0
TEST_DATA_ROOT = Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_root() -> Path:
    """Provide the root directory for test data."""
    return TEST_DATA_ROOT


@pytest.fixture(scope="session")
def fixtures_root() -> Path:
    """Provide the root directory for test fixtures."""
    return TEST_DATA_ROOT / "fixtures"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Data loading fixtures
@pytest.fixture(scope="session")
def test_data_loader() -> TestDataLoader:
    """Provide a TestDataLoader instance."""
    return TestDataLoader()


@pytest.fixture(scope="session")
def fixture_loader(fixtures_root: Path) -> FixtureLoader:
    """Provide a FixtureLoader instance."""
    return FixtureLoader(fixtures_root)


@pytest.fixture(scope="session")
def fixture_manager(fixtures_root: Path) -> FixtureManager:
    """Provide a FixtureManager instance."""
    return FixtureManager(fixtures_root)


# Test helper fixtures
@pytest.fixture
def mathir_helper() -> MathIRTestHelper:
    """Provide a MathIRTestHelper instance."""
    return MathIRTestHelper()


@pytest.fixture
def result_validator() -> TestResultValidator:
    """Provide a TestResultValidator instance."""
    return TestResultValidator()


@pytest.fixture
def precision_assertion() -> PrecisionAssertion:
    """Provide a PrecisionAssertion instance with default tolerance."""
    return PrecisionAssertion(default_tolerance=DEFAULT_TOLERANCE)


@pytest.fixture
def mathematical_validator() -> MathematicalValidator:
    """Provide a MathematicalValidator instance."""
    return MathematicalValidator()


# Data generation fixtures
@pytest.fixture
def data_generator() -> TestDataGenerator:
    """Provide a TestDataGenerator instance with fixed seed for reproducibility."""
    return TestDataGenerator(seed=42)


@pytest.fixture
def mathir_generator(data_generator: TestDataGenerator) -> MathIRGenerator:
    """Provide a MathIRGenerator instance."""
    return MathIRGenerator(data_generator)


# Performance testing fixtures
@pytest.fixture
def performance_profiler() -> PerformanceProfiler:
    """Provide a PerformanceProfiler instance."""
    return PerformanceProfiler()


@pytest.fixture
def benchmark_runner() -> BenchmarkRunner:
    """Provide a BenchmarkRunner instance."""
    return BenchmarkRunner()


# Mathematical computation fixtures
@pytest.fixture
def simple_mathir() -> MathIR:
    """Provide a simple MathIR object for basic testing."""
    return MathIR(
        expr_format="latex",
        symbols=[{"name": "x", "domain": "R"}],
        targets=[{
            "type": "value",
            "name": "result",
            "expr": "2*x + 1"
        }],
        output={"mode": "exact"}
    )


@pytest.fixture
def integral_mathir() -> MathIR:
    """Provide a MathIR object for integral testing."""
    return MathIR(
        expr_format="latex",
        symbols=[{"name": "x", "domain": "R"}],
        targets=[{
            "type": "integral_def",
            "expr": "x^2",
            "var": "x",
            "limits": [0, 1],
            "name": "integral_result"
        }],
        output={"mode": "exact"}
    )


@pytest.fixture
def limit_mathir() -> MathIR:
    """Provide a MathIR object for limit testing."""
    return MathIR(
        expr_format="latex",
        symbols=[{"name": "x", "domain": "R"}],
        targets=[{
            "type": "limit",
            "expr": "\\frac{\\sin(x)}{x}",
            "var": "x",
            "to": "0"
        }],
        output={"mode": "exact"}
    )


@pytest.fixture
def sum_mathir() -> MathIR:
    """Provide a MathIR object for summation testing."""
    return MathIR(
        expr_format="latex",
        symbols=[{"name": "n", "domain": "N"}],
        targets=[{
            "type": "sum",
            "term": "n",
            "idx": "n",
            "start": "1",
            "end": "5"
        }],
        output={"mode": "exact"}
    )


@pytest.fixture
def solve_mathir() -> MathIR:
    """Provide a MathIR object for equation solving testing."""
    return MathIR(
        expr_format="latex",
        symbols=[
            {"name": "x", "domain": "R"},
            {"name": "y", "domain": "R"}
        ],
        targets=[{
            "type": "solve_for",
            "unknowns": ["x", "y"],
            "equations": ["x + y = 3", "x - y = 1"]
        }],
        output={"mode": "exact"}
    )


# Test data fixtures
@pytest.fixture
def sample_expressions() -> List[str]:
    """Provide sample mathematical expressions for testing."""
    return [
        "x^2 + 1",
        "\\sin(x)",
        "\\cos(x)",
        "e^x",
        "\\ln(x)",
        "\\frac{1}{x}",
        "\\sqrt{x}",
        "x^3 - 2*x + 1"
    ]


@pytest.fixture
def known_integrals() -> List[Dict[str, Any]]:
    """Provide known integral test cases with solutions."""
    return [
        {"expr": "x", "limits": [0, 1], "solution": sp.Rational(1, 2)},
        {"expr": "x^2", "limits": [0, 1], "solution": sp.Rational(1, 3)},
        {"expr": "2*x", "limits": [0, 2], "solution": 4},
        {"expr": "x^2 + 1", "limits": [0, 1], "solution": sp.Rational(4, 3)}
    ]


@pytest.fixture
def known_limits() -> List[Dict[str, Any]]:
    """Provide known limit test cases with solutions."""
    return [
        {"expr": "\\frac{\\sin(x)}{x}", "var": "x", "to": "0", "solution": 1},
        {"expr": "\\frac{1}{x}", "var": "x", "to": "oo", "solution": 0},
        {"expr": "\\frac{x^2 - 1}{x - 1}", "var": "x", "to": "1", "solution": 2}
    ]


@pytest.fixture
def edge_cases() -> List[Dict[str, Any]]:
    """Provide edge case test scenarios."""
    return [
        {"expr": "\\frac{1}{0}", "expected_error": "division_by_zero"},
        {"expr": "\\sqrt{-1}", "expected_error": "complex_result"},
        {"expr": "\\ln(0)", "expected_error": "undefined"},
        {"expr": "0^0", "expected_error": "indeterminate"}
    ]


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[
    {"mode": "exact"},
    {"mode": "decimal", "round_to": 3},
    {"mode": "decimal", "round_to": 6}
])
def output_modes(request) -> Dict[str, Any]:
    """Provide different output mode configurations."""
    return request.param


@pytest.fixture(params=[
    {"name": "x", "domain": "R"},
    {"name": "n", "domain": "N"},
    {"name": "t", "domain": "R+"}
])
def symbol_configs(request) -> Dict[str, str]:
    """Provide different symbol configurations."""
    return request.param


# Utility fixtures
@pytest.fixture
def json_schema_validator():
    """Provide JSON schema validator for MathIR validation."""
    import jsonschema
    
    schema_path = Path(__file__).parent.parent / "llm_parser" / "schema" / "mathir.schema.json"
    if schema_path.exists():
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return jsonschema.Draft202012Validator(schema)
    return None


# Test execution helpers
@pytest.fixture
def run_mathir_safely():
    """Provide a safe MathIR execution function that handles errors."""
    def _run_safely(mathir: MathIR) -> Dict[str, Any]:
        try:
            return run_mathir(mathir)
        except Exception as e:
            return {"error": str(e), "exception_type": type(e).__name__}
    
    return _run_safely


@pytest.fixture
def assert_mathir_result():
    """Provide a function to assert MathIR computation results."""
    def _assert_result(mathir: MathIR, 
                      expected: Dict[str, Any], 
                      tolerance: float = DEFAULT_TOLERANCE):
        results = run_mathir(mathir)
        
        # Check that all expected keys are present
        for key in expected.keys():
            assert key in results, f"Expected key '{key}' not found in results"
        
        # Validate results with tolerance
        validator = MathematicalValidator()
        for key, expected_value in expected.items():
            actual_value = results[key]
            
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                assert abs(actual_value - expected_value) <= tolerance, \
                    f"Result mismatch for '{key}': {actual_value} != {expected_value}"
            elif isinstance(expected_value, sp.Expr) and isinstance(actual_value, sp.Expr):
                assert sp.simplify(actual_value - expected_value) == 0, \
                    f"Symbolic result mismatch for '{key}': {actual_value} != {expected_value}"
            else:
                assert actual_value == expected_value, \
                    f"Result mismatch for '{key}': {actual_value} != {expected_value}"
    
    return _assert_result


# Session-level setup and teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Create logs directory
    logs_dir = TEST_DATA_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Set SymPy printing options for consistent output
    sp.init_printing(use_unicode=False, wrap_line=False)
    
    # Configure SymPy assumptions
    sp.assumptions.global_assumptions.add(sp.Q.real(sp.Symbol('x')))
    sp.assumptions.global_assumptions.add(sp.Q.real(sp.Symbol('y')))
    sp.assumptions.global_assumptions.add(sp.Q.real(sp.Symbol('t')))
    sp.assumptions.global_assumptions.add(sp.Q.integer(sp.Symbol('n')))
    
    yield
    
    # Cleanup
    sp.assumptions.global_assumptions.clear()


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "mathematical: mark test as mathematical computation")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.end_to_end)
        elif "regression" in str(item.fspath):
            item.add_marker(pytest.mark.regression)
        elif "stress" in str(item.fspath):
            item.add_marker(pytest.mark.stress)
            item.add_marker(pytest.mark.slow)
        
        # Add mathematical marker for tests involving computations
        if any(keyword in item.name.lower() for keyword in 
               ["integral", "limit", "sum", "solve", "mathematical"]):
            item.add_marker(pytest.mark.mathematical)


def pytest_runtest_setup(item):
    """Set up individual test runs."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--runslow", default=False):
        pytest.skip("need --runslow option to run")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--tolerance", action="store", default=str(DEFAULT_TOLERANCE),
        help="set numerical tolerance for mathematical comparisons"
    )


@pytest.fixture
def tolerance(request):
    """Provide configurable tolerance for numerical comparisons."""
    return float(request.config.getoption("--tolerance"))
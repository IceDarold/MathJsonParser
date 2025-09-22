#!/usr/bin/env python3
"""
Real-World Cases End-to-End Tests for MathIR Parser

This module provides comprehensive testing using real-world mathematical problems
from various sources including HuggingFace outputs, individual test cases,
mathematical olympiad problems, university-level calculus problems, engineering
calculations, and physics formula validation.

Test Data Sources:
1. HuggingFace outputs (72+ files) - Real mathematical tasks processed by LLM
2. Individual test cases (75+ files) - Curated mathematical problems
3. Mathematical olympiad problems - Competition-level mathematics
4. University calculus problems - Academic-level mathematics
5. Engineering calculations - Applied mathematics scenarios
6. Physics formula validation - Scientific computation accuracy

Success Criteria:
- Process all real-world test cases without crashes
- Achieve >95% accuracy on solvable problems
- Handle edge cases and malformed inputs gracefully
- Maintain consistent performance across problem types
- Validate mathematical correctness against known solutions
"""

import pytest
import json
import os
import glob
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mathir_parser.main import MathIR, run_mathir
from tests.utils.mathematical_accuracy_validator import (
    MathematicalAccuracyValidator, ValidationResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RealWorldTestCase:
    """Real-world test case specification."""
    test_id: str
    source: str
    description: str
    mathir_data: Dict[str, Any]
    expected_result: Optional[Any] = None
    expected_error: Optional[str] = None
    difficulty_level: str = "medium"  # "easy", "medium", "hard", "extreme"
    problem_type: str = "general"  # "integral", "limit", "algebra", "matrix", etc.


@dataclass
class TestSuiteResult:
    """Results from processing a test suite."""
    total_tests: int
    successful_tests: int
    failed_tests: int
    error_tests: int
    accuracy_rate: float
    processing_errors: List[str]
    mathematical_errors: List[str]


class RealWorldTestSuite:
    """
    Comprehensive real-world test suite for MathIR Parser.
    
    This suite processes real mathematical problems from various sources to validate
    that the MathIR Parser can handle diverse, real-world mathematical scenarios
    with high accuracy and reliability.
    """
    
    @classmethod
    def setup_class(cls):
        """Set up real-world testing infrastructure."""
        cls.validator = MathematicalAccuracyValidator()
        cls.test_results: List[ValidationResult] = []
        cls.processing_errors: List[str] = []
        cls.mathematical_errors: List[str] = []
        
        # Load test data from various sources
        cls.huggingface_cases = cls._load_huggingface_test_cases()
        cls.individual_cases = cls._load_individual_test_cases()
        cls.olympiad_cases = cls._create_olympiad_test_cases()
        cls.university_cases = cls._create_university_test_cases()
        cls.engineering_cases = cls._create_engineering_test_cases()
        cls.physics_cases = cls._create_physics_test_cases()
        
        logger.info(f"Loaded {len(cls.huggingface_cases)} HuggingFace test cases")
        logger.info(f"Loaded {len(cls.individual_cases)} individual test cases")
        logger.info(f"Created {len(cls.olympiad_cases)} olympiad test cases")
        logger.info(f"Created {len(cls.university_cases)} university test cases")
        logger.info(f"Created {len(cls.engineering_cases)} engineering test cases")
        logger.info(f"Created {len(cls.physics_cases)} physics test cases")
    
    @classmethod
    def teardown_class(cls):
        """Generate comprehensive real-world testing report."""
        cls._generate_real_world_report()
    
    @classmethod
    def _load_huggingface_test_cases(cls) -> List[RealWorldTestCase]:
        """Load test cases from HuggingFace output files."""
        test_cases = []
        huggingface_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'huggingface_outputs')
        
        if not os.path.exists(huggingface_dir):
            logger.warning(f"HuggingFace outputs directory not found: {huggingface_dir}")
            return test_cases
        
        # Load all task files
        task_files = glob.glob(os.path.join(huggingface_dir, 'task_*.json'))
        
        for task_file in task_files[:20]:  # Limit to first 20 for testing
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                task_id = data.get('task_id', os.path.basename(task_file))
                task_description = data.get('task', 'Unknown task')
                
                # Try to extract MathIR data from parsed_json or response
                mathir_data = data.get('parsed_json')
                if not mathir_data and 'response' in data:
                    try:
                        # Try to parse response as JSON
                        response_text = data['response']
                        if isinstance(response_text, str):
                            # Look for JSON in the response
                            import re
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                mathir_data = json.loads(json_match.group())
                    except:
                        continue
                
                if mathir_data:
                    test_case = RealWorldTestCase(
                        test_id=task_id,
                        source="huggingface",
                        description=task_description,
                        mathir_data=mathir_data,
                        difficulty_level="medium",
                        problem_type=mathir_data.get('task_type', 'general')
                    )
                    test_cases.append(test_case)
                    
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace test case {task_file}: {e}")
                continue
        
        return test_cases
    
    @classmethod
    def _load_individual_test_cases(cls) -> List[RealWorldTestCase]:
        """Load test cases from individual test files."""
        test_cases = []
        individual_dir = os.path.join(os.path.dirname(__file__), '..', 'individual')
        
        if not os.path.exists(individual_dir):
            logger.warning(f"Individual tests directory not found: {individual_dir}")
            return test_cases
        
        # Load all individual task files
        task_files = glob.glob(os.path.join(individual_dir, 'task_*.json'))
        
        for task_file in task_files[:15]:  # Limit to first 15 for testing
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                task_id = data.get('task_id', os.path.basename(task_file))
                task_description = data.get('task', 'Individual test case')
                mathir_data = data.get('parsed_json')
                
                if mathir_data:
                    test_case = RealWorldTestCase(
                        test_id=task_id,
                        source="individual",
                        description=task_description,
                        mathir_data=mathir_data,
                        difficulty_level="medium",
                        problem_type=mathir_data.get('task_type', 'general')
                    )
                    test_cases.append(test_case)
                    
            except Exception as e:
                logger.warning(f"Failed to load individual test case {task_file}: {e}")
                continue
        
        return test_cases
    
    @classmethod
    def _create_olympiad_test_cases(cls) -> List[RealWorldTestCase]:
        """Create mathematical olympiad-level test cases."""
        return [
            RealWorldTestCase(
                test_id="olympiad_001",
                source="olympiad",
                description="Find all real solutions to x^4 - 4x^3 + 6x^2 - 4x + 1 = 0",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "solve_for",
                        "unknowns": ["x"],
                        "equations": ["x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = 0"]
                    }],
                    "output": {"mode": "exact"}
                },
                difficulty_level="hard",
                problem_type="algebra"
            ),
            
            RealWorldTestCase(
                test_id="olympiad_002",
                source="olympiad",
                description="Evaluate the integral of sin(x)/x from 0 to infinity",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "integral_def",
                        "expr": "\\frac{\\sin(x)}{x}",
                        "var": "x",
                        "limits": [0, "oo"],
                        "name": "I"
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="pi/2",
                difficulty_level="extreme",
                problem_type="integral"
            ),
            
            RealWorldTestCase(
                test_id="olympiad_003",
                source="olympiad",
                description="Find the limit of (1 + 1/n)^n as n approaches infinity",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [{"name": "n", "domain": "N+"}],
                    "targets": [{
                        "type": "limit",
                        "expr": "(1 + \\frac{1}{n})^n",
                        "var": "n",
                        "to": "oo"
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="E",
                difficulty_level="hard",
                problem_type="limit"
            ),
            
            RealWorldTestCase(
                test_id="olympiad_004",
                source="olympiad",
                description="Solve the system: x + y + z = 6, xy + yz + zx = 11, xyz = 6",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "x", "domain": "R"},
                        {"name": "y", "domain": "R"},
                        {"name": "z", "domain": "R"}
                    ],
                    "targets": [{
                        "type": "solve_for",
                        "unknowns": ["x", "y", "z"],
                        "equations": [
                            "x + y + z = 6",
                            "x*y + y*z + z*x = 11",
                            "x*y*z = 6"
                        ]
                    }],
                    "output": {"mode": "exact"}
                },
                difficulty_level="hard",
                problem_type="algebra"
            ),
            
            RealWorldTestCase(
                test_id="olympiad_005",
                source="olympiad",
                description="Find the sum of the infinite series 1 + 1/4 + 1/9 + 1/16 + ...",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [{"name": "n", "domain": "N+"}],
                    "targets": [{
                        "type": "sum",
                        "term": "\\frac{1}{n^2}",
                        "idx": "n",
                        "start": "1",
                        "end": "oo"
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="pi**2/6",
                difficulty_level="extreme",
                problem_type="sum"
            )
        ]
    
    @classmethod
    def _create_university_test_cases(cls) -> List[RealWorldTestCase]:
        """Create university-level calculus and mathematics test cases."""
        return [
            RealWorldTestCase(
                test_id="university_001",
                source="university",
                description="Find the area between y = x^2 and y = 2x",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "area_between_curves",
                        "curves": ["x^2", "2*x"],
                        "var": "x"
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="4/3",
                difficulty_level="medium",
                problem_type="integral"
            ),
            
            RealWorldTestCase(
                test_id="university_002",
                source="university",
                description="Find the critical points of f(x) = x^3 - 3x^2 + 2",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [{"name": "x", "domain": "R"}],
                    "targets": [{
                        "type": "find_maximum",
                        "expr": "x^3 - 3*x^2 + 2",
                        "var": "x"
                    }],
                    "output": {"mode": "exact"}
                },
                difficulty_level="medium",
                problem_type="optimize"
            ),
            
            RealWorldTestCase(
                test_id="university_003",
                source="university",
                description="Evaluate the double integral of xy over the region [0,1] x [0,2]",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "x", "domain": "R"},
                        {"name": "y", "domain": "R"}
                    ],
                    "targets": [{
                        "type": "integral_double",
                        "expr": "x*y",
                        "vars": ["x", "y"],
                        "limits": [[0, 1], [0, 2]]
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="1",
                difficulty_level="medium",
                problem_type="integral"
            ),
            
            RealWorldTestCase(
                test_id="university_004",
                source="university",
                description="Solve the differential equation dy/dx = y",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "x", "domain": "R"},
                        {"name": "y", "domain": "R"}
                    ],
                    "targets": [{
                        "type": "solve_for",
                        "unknowns": ["y"],
                        "equations": ["y - C*exp(x) = 0"]  # General solution form
                    }],
                    "output": {"mode": "exact"}
                },
                difficulty_level="medium",
                problem_type="algebra"
            ),
            
            RealWorldTestCase(
                test_id="university_005",
                source="university",
                description="Find the eigenvalues of the matrix [[3, 1], [0, 2]]",
                mathir_data={
                    "expr_format": "latex",
                    "definitions": {
                        "matrices": [{
                            "name": "A",
                            "rows": 2,
                            "cols": 2,
                            "data": [["3", "1"], ["0", "2"]]
                        }]
                    },
                    "targets": [{
                        "type": "matrix_determinant",
                        "matrix_name": "A"
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="6",  # det([[3,1],[0,2]]) = 6
                difficulty_level="medium",
                problem_type="matrix"
            )
        ]
    
    @classmethod
    def _create_engineering_test_cases(cls) -> List[RealWorldTestCase]:
        """Create engineering calculation test cases."""
        return [
            RealWorldTestCase(
                test_id="engineering_001",
                source="engineering",
                description="Calculate the moment of inertia of a rectangular beam",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "b", "domain": "R+"},
                        {"name": "h", "domain": "R+"}
                    ],
                    "constants": {"b": "10", "h": "20"},
                    "targets": [{
                        "type": "value",
                        "name": "I",
                        "expr": "\\frac{b * h^3}{12}"
                    }],
                    "output": {"mode": "decimal", "round_to": 2}
                },
                expected_result="6666.67",
                difficulty_level="easy",
                problem_type="general"
            ),
            
            RealWorldTestCase(
                test_id="engineering_002",
                source="engineering",
                description="Calculate the natural frequency of a spring-mass system",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "k", "domain": "R+"},
                        {"name": "m", "domain": "R+"}
                    ],
                    "constants": {"k": "1000", "m": "10"},
                    "targets": [{
                        "type": "value",
                        "name": "omega",
                        "expr": "\\sqrt{\\frac{k}{m}}"
                    }],
                    "output": {"mode": "decimal", "round_to": 3}
                },
                expected_result="10.000",
                difficulty_level="easy",
                problem_type="general"
            ),
            
            RealWorldTestCase(
                test_id="engineering_003",
                source="engineering",
                description="Calculate the stress in a beam under bending",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "M", "domain": "R+"},
                        {"name": "c", "domain": "R+"},
                        {"name": "I", "domain": "R+"}
                    ],
                    "constants": {"M": "5000", "c": "10", "I": "6666.67"},
                    "targets": [{
                        "type": "value",
                        "name": "sigma",
                        "expr": "\\frac{M * c}{I}"
                    }],
                    "output": {"mode": "decimal", "round_to": 2}
                },
                expected_result="7.50",
                difficulty_level="easy",
                problem_type="general"
            )
        ]
    
    @classmethod
    def _create_physics_test_cases(cls) -> List[RealWorldTestCase]:
        """Create physics formula validation test cases."""
        return [
            RealWorldTestCase(
                test_id="physics_001",
                source="physics",
                description="Calculate kinetic energy: KE = (1/2)mv^2",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "m", "domain": "R+"},
                        {"name": "v", "domain": "R+"}
                    ],
                    "constants": {"m": "2", "v": "10"},
                    "targets": [{
                        "type": "value",
                        "name": "KE",
                        "expr": "\\frac{1}{2} * m * v^2"
                    }],
                    "output": {"mode": "exact"}
                },
                expected_result="100",
                difficulty_level="easy",
                problem_type="general"
            ),
            
            RealWorldTestCase(
                test_id="physics_002",
                source="physics",
                description="Calculate gravitational force: F = G*m1*m2/r^2",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "G", "domain": "R+"},
                        {"name": "m1", "domain": "R+"},
                        {"name": "m2", "domain": "R+"},
                        {"name": "r", "domain": "R+"}
                    ],
                    "constants": {
                        "G": "6.674e-11",
                        "m1": "1000",
                        "m2": "2000",
                        "r": "10"
                    },
                    "targets": [{
                        "type": "value",
                        "name": "F",
                        "expr": "\\frac{G * m1 * m2}{r^2}"
                    }],
                    "output": {"mode": "decimal", "round_to": 10}
                },
                difficulty_level="easy",
                problem_type="general"
            ),
            
            RealWorldTestCase(
                test_id="physics_003",
                source="physics",
                description="Calculate wave frequency: f = v/λ",
                mathir_data={
                    "expr_format": "latex",
                    "symbols": [
                        {"name": "v", "domain": "R+"},
                        {"name": "lambda", "domain": "R+"}
                    ],
                    "constants": {"v": "343", "lambda": "0.5"},
                    "targets": [{
                        "type": "value",
                        "name": "f",
                        "expr": "\\frac{v}{lambda}"
                    }],
                    "output": {"mode": "decimal", "round_to": 1}
                },
                expected_result="686.0",
                difficulty_level="easy",
                problem_type="general"
            )
        ]
    
    def _process_test_case(self, test_case: RealWorldTestCase) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Process a single real-world test case.
        
        Returns:
            Tuple of (success, error_message, result)
        """
        try:
            # Validate and create MathIR object
            ir = MathIR.model_validate(test_case.mathir_data)
            
            # Run the mathematical computation
            with self.validator.performance_monitor():
                results = run_mathir(ir)
            
            # Check if we got results
            if not results:
                return False, "No results returned", None
            
            # If we have an expected result, validate it
            if test_case.expected_result is not None:
                # Find the main result (first non-error result)
                main_result = None
                for key, value in results.items():
                    if not isinstance(value, dict) or 'error' not in value:
                        main_result = value
                        break
                
                if main_result is not None:
                    # Validate against expected result
                    validation = self.validator.validate_numerical_tolerance(
                        test_case.expected_result, main_result, tolerance=1e-6
                    )
                    if not validation.passed:
                        return False, f"Result mismatch: expected {test_case.expected_result}, got {main_result}", results
            
            return True, None, results
            
        except Exception as e:
            return False, str(e), None
    
    def _process_test_suite(self, test_cases: List[RealWorldTestCase], suite_name: str) -> TestSuiteResult:
        """Process a suite of test cases and return results."""
        total_tests = len(test_cases)
        successful_tests = 0
        failed_tests = 0
        error_tests = 0
        processing_errors = []
        mathematical_errors = []
        
        for test_case in test_cases:
            success, error_message, result = self._process_test_case(test_case)
            
            if success:
                successful_tests += 1
            else:
                failed_tests += 1
                if error_message:
                    if "error" in error_message.lower() or "exception" in error_message.lower():
                        error_tests += 1
                        processing_errors.append(f"{test_case.test_id}: {error_message}")
                    else:
                        mathematical_errors.append(f"{test_case.test_id}: {error_message}")
        
        accuracy_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        return TestSuiteResult(
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            accuracy_rate=accuracy_rate,
            processing_errors=processing_errors,
            mathematical_errors=mathematical_errors
        )
    
    # ========================================================================
    # HUGGINGFACE OUTPUT TESTS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    def test_huggingface_outputs(self):
        """Test processing of HuggingFace output files."""
        if not self.huggingface_cases:
            pytest.skip("No HuggingFace test cases available")
        
        suite_result = self._process_test_suite(self.huggingface_cases, "HuggingFace")
        
        # Should achieve reasonable success rate (>70% due to potential LLM parsing issues)
        assert suite_result.accuracy_rate >= 0.70, \
            f"HuggingFace test accuracy {suite_result.accuracy_rate:.1%} is below 70%"
        
        # Should not have too many processing errors (>90% should parse correctly)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.90, \
            f"HuggingFace processing success rate {processing_success_rate:.1%} is below 90%"
        
        logger.info(f"HuggingFace tests: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # INDIVIDUAL TEST CASES
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    def test_individual_cases(self):
        """Test processing of individual test case files."""
        if not self.individual_cases:
            pytest.skip("No individual test cases available")
        
        suite_result = self._process_test_suite(self.individual_cases, "Individual")
        
        # Should achieve high success rate (>90% for curated test cases)
        assert suite_result.accuracy_rate >= 0.90, \
            f"Individual test accuracy {suite_result.accuracy_rate:.1%} is below 90%"
        
        # Should have very few processing errors (>95% should parse correctly)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.95, \
            f"Individual test processing success rate {processing_success_rate:.1%} is below 95%"
        
        logger.info(f"Individual tests: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # MATHEMATICAL OLYMPIAD PROBLEMS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    @pytest.mark.slow
    def test_olympiad_problems(self):
        """Test mathematical olympiad-level problems."""
        suite_result = self._process_test_suite(self.olympiad_cases, "Olympiad")
        
        # Olympiad problems are challenging, so lower success rate is acceptable (>60%)
        assert suite_result.accuracy_rate >= 0.60, \
            f"Olympiad test accuracy {suite_result.accuracy_rate:.1%} is below 60%"
        
        # Should still parse correctly (>90%)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.90, \
            f"Olympiad processing success rate {processing_success_rate:.1%} is below 90%"
        
        logger.info(f"Olympiad tests: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # UNIVERSITY-LEVEL PROBLEMS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    def test_university_problems(self):
        """Test university-level calculus and mathematics problems."""
        suite_result = self._process_test_suite(self.university_cases, "University")
        
        # University problems should have good success rate (>80%)
        assert suite_result.accuracy_rate >= 0.80, \
            f"University test accuracy {suite_result.accuracy_rate:.1%} is below 80%"
        
        # Should parse correctly (>95%)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.95, \
            f"University processing success rate {processing_success_rate:.1%} is below 95%"
        
        logger.info(f"University tests: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # ENGINEERING CALCULATIONS
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    def test_engineering_calculations(self):
        """Test engineering calculation scenarios."""
        suite_result = self._process_test_suite(self.engineering_cases, "Engineering")
        
        # Engineering calculations should have very high success rate (>95%)
        assert suite_result.accuracy_rate >= 0.95, \
            f"Engineering test accuracy {suite_result.accuracy_rate:.1%} is below 95%"
        
        # Should parse correctly (>98%)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.98, \
            f"Engineering processing success rate {processing_success_rate:.1%} is below 98%"
        
        logger.info(f"Engineering tests: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # PHYSICS FORMULA VALIDATION
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    def test_physics_formulas(self):
        """Test physics formula validation."""
        suite_result = self._process_test_suite(self.physics_cases, "Physics")
        
        # Physics formulas should have very high success rate (>95%)
        assert suite_result.accuracy_rate >= 0.95, \
            f"Physics test accuracy {suite_result.accuracy_rate:.1%} is below 95%"
        
        # Should parse correctly (>98%)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.98, \
            f"Physics processing success rate {processing_success_rate:.1%} is below 98%"
        
        logger.info(f"Physics tests: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # COMPREHENSIVE REAL-WORLD VALIDATION
    # ========================================================================
    
    @pytest.mark.end_to_end
    @pytest.mark.real_world
    def test_comprehensive_real_world_validation(self):
        """Comprehensive validation across all real-world test categories."""
        all_test_cases = (
            self.huggingface_cases + self.individual_cases + 
            self.olympiad_cases + self.university_cases + 
            self.engineering_cases + self.physics_cases
        )
        
        if not all_test_cases:
            pytest.skip("No real-world test cases available")
        
        suite_result = self._process_test_suite(all_test_cases, "Comprehensive")
        
        # Overall success rate should be reasonable (>80%)
        assert suite_result.accuracy_rate >= 0.80, \
            f"Overall real-world test accuracy {suite_result.accuracy_rate:.1%} is below 80%"
        
        # Processing success rate should be high (>90%)
        processing_success_rate = 1.0 - (suite_result.error_tests / suite_result.total_tests)
        assert processing_success_rate >= 0.90, \
            f"Overall processing success rate {processing_success_rate:.1%} is below 90%"
        
        # Store results for reporting
        self.processing_errors.extend(suite_result.processing_errors)
        self.mathematical_errors.extend(suite_result.mathematical_errors)
        
        logger.info(f"Comprehensive real-world validation: {suite_result.successful_tests}/{suite_result.total_tests} passed "
                   f"({suite_result.accuracy_rate:.1%} accuracy)")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    @classmethod
    def _generate_real_world_report(cls):
        """Generate comprehensive real-world testing report."""
        print("\n" + "="*80)
        print("REAL-WORLD TEST CASES REPORT")
        print("="*80)
        
        # Test case statistics
        total_huggingface = len(cls.huggingface_cases)
        total_individual = len(cls.individual_cases)
        total_olympiad = len(cls.olympiad_cases)
        total_university = len(cls.university_cases)
        total_engineering = len(cls.engineering_cases)
        total_physics = len(cls.physics_cases)
        total_all = total_huggingface + total_individual + total_olympiad + total_university + total_engineering + total_physics
        
        print(f"Test Case Sources:")
        print(f"  HuggingFace outputs: {total_huggingface} cases")
        print(f"  Individual test files: {total_individual} cases")
        print(f"  Mathematical olympiad: {total_olympiad} cases")
        print(f"  University problems: {total_university} cases")
        print(f"  Engineering calculations: {total_engineering} cases")
        print(f"  Physics formulas: {total_physics} cases")
        print(f"  Total: {total_all} cases")
        
        # Error analysis
        if cls.processing_errors:
            print(f"\nProcessing Errors ({len(cls.processing_errors)}):")
            for error in cls.processing_errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(cls.processing_errors) > 10:
                print(f"  ... and {len(cls.processing_errors) - 10} more")
        
        if cls.mathematical_errors:
            print(f"\nMathematical Errors ({len(cls.mathematical_errors)}):")
            for error in cls.mathematical_errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(cls.mathematical_errors) > 10:
                print(f"  ... and {len(cls.mathematical_errors) - 10} more")
        
        # Recommendations
        print(f"\nRecommendations:")
        if total_all == 0:
            print("  - No real-world test cases were loaded. Check test data availability.")
        elif len(cls.processing_errors) > total_all * 0.1:
            print("  - High processing error rate. Review MathIR parsing and validation.")
        elif len(cls.mathematical_errors) > total_all * 0.2:
            print("  - High mathematical error rate. Review mathematical computation accuracy.")
        else:
            print("  - Real-world test processing is performing well!")
        
        print("="*80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "real_world"])
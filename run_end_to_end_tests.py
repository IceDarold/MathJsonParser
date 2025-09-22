#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Runner for MathIR Parser

This script runs the complete end-to-end test suite for mathematical accuracy validation.
It executes all three critical test categories:
1. Mathematical Accuracy Tests - Golden standard mathematical correctness
2. Performance Tests - Benchmarking and resource monitoring  
3. Real-World Cases - Processing of real mathematical problems

The script provides detailed reporting and validates that the MathIR Parser meets
all production readiness criteria for mathematical correctness and performance.

Usage:
    python run_end_to_end_tests.py [options]
    
Options:
    --accuracy-only     Run only mathematical accuracy tests
    --performance-only  Run only performance tests
    --real-world-only   Run only real-world case tests
    --quick            Run a subset of tests for quick validation
    --full             Run all tests including slow tests
    --report-only      Generate reports from previous test runs
    --verbose          Enable verbose output
"""

import sys
import os
import subprocess
import argparse
import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tests/logs/end_to_end_tests.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestSuiteResult:
    """Results from running a test suite."""
    name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    success_rate: float
    exit_code: int
    output: str
    errors: List[str]


class EndToEndTestRunner:
    """
    Comprehensive end-to-end test runner for MathIR Parser.
    
    This runner orchestrates the execution of all end-to-end test suites,
    collects results, generates comprehensive reports, and validates that
    the system meets all production readiness criteria.
    """
    
    def __init__(self, args):
        """Initialize the test runner with command line arguments."""
        self.args = args
        self.results: List[TestSuiteResult] = []
        self.start_time = time.time()
        
        # Test suite configurations
        self.test_suites = {
            'accuracy': {
                'name': 'Mathematical Accuracy Tests',
                'path': 'tests/end_to_end/test_mathematical_accuracy.py',
                'markers': 'end_to_end and mathematical',
                'critical': True,
                'description': 'Validates mathematical correctness against golden standards'
            },
            'performance': {
                'name': 'Performance Tests',
                'path': 'tests/end_to_end/test_performance.py',
                'markers': 'end_to_end and performance',
                'critical': True,
                'description': 'Validates performance requirements and resource usage'
            },
            'real_world': {
                'name': 'Real-World Cases',
                'path': 'tests/end_to_end/test_real_world_cases.py',
                'markers': 'end_to_end and real_world',
                'critical': False,
                'description': 'Processes real mathematical problems from various sources'
            }
        }
    
    def run_test_suite(self, suite_key: str) -> TestSuiteResult:
        """Run a specific test suite and return results."""
        suite_config = self.test_suites[suite_key]
        logger.info(f"Running {suite_config['name']}...")
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test file or markers
        if os.path.exists(suite_config['path']):
            cmd.append(suite_config['path'])
        else:
            cmd.extend(['-m', suite_config['markers']])
        
        # Add common options
        cmd.extend([
            '-v',
            '--tb=short',
            '--durations=10',
            '--color=yes'
        ])
        
        # Add specific options based on arguments
        if self.args.quick and suite_key == 'performance':
            cmd.extend(['-m', 'not slow'])
        elif self.args.full:
            cmd.extend(['-m', 'not skip'])
        
        if self.args.verbose:
            cmd.append('-s')
        
        # Add output capture
        cmd.extend(['--tb=line', '--no-header'])
        
        # Run the test suite
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            execution_time = time.time() - start_time
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            total_tests, passed_tests, failed_tests, skipped_tests = self._parse_pytest_output(output_lines)
            
            success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            
            # Extract errors from stderr
            errors = []
            if result.stderr:
                errors = [line.strip() for line in result.stderr.split('\n') if line.strip()]
            
            return TestSuiteResult(
                name=suite_config['name'],
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                execution_time=execution_time,
                success_rate=success_rate,
                exit_code=result.returncode,
                output=result.stdout,
                errors=errors
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestSuiteResult(
                name=suite_config['name'],
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=execution_time,
                success_rate=0.0,
                exit_code=-1,
                output="",
                errors=["Test suite timed out after 30 minutes"]
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return TestSuiteResult(
                name=suite_config['name'],
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=execution_time,
                success_rate=0.0,
                exit_code=-1,
                output="",
                errors=[f"Test suite execution failed: {str(e)}"]
            )
    
    def _parse_pytest_output(self, output_lines: List[str]) -> tuple:
        """Parse pytest output to extract test statistics."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for line in output_lines:
            line = line.strip()
            
            # Look for summary line like "5 passed, 2 failed, 1 skipped in 10.5s"
            if ' passed' in line or ' failed' in line or ' skipped' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            passed_tests = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            failed_tests = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            skipped_tests = int(parts[i-1])
                        except ValueError:
                            pass
        
        total_tests = passed_tests + failed_tests + skipped_tests
        return total_tests, passed_tests, failed_tests, skipped_tests
    
    def run_all_tests(self) -> bool:
        """Run all configured test suites."""
        logger.info("Starting comprehensive end-to-end test execution...")
        
        # Determine which test suites to run
        suites_to_run = []
        if self.args.accuracy_only:
            suites_to_run = ['accuracy']
        elif self.args.performance_only:
            suites_to_run = ['performance']
        elif self.args.real_world_only:
            suites_to_run = ['real_world']
        else:
            suites_to_run = ['accuracy', 'performance', 'real_world']
        
        # Run each test suite
        overall_success = True
        for suite_key in suites_to_run:
            result = self.run_test_suite(suite_key)
            self.results.append(result)
            
            # Check if critical test suite failed
            if self.test_suites[suite_key]['critical'] and result.exit_code != 0:
                overall_success = False
                logger.error(f"Critical test suite '{result.name}' failed!")
            
            # Log immediate results
            logger.info(f"{result.name}: {result.passed_tests}/{result.total_tests} passed "
                       f"({result.success_rate:.1%}) in {result.execution_time:.1f}s")
        
        return overall_success
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test execution report."""
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall statistics
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed_tests for r in self.results)
        total_failed = sum(r.failed_tests for r in self.results)
        total_skipped = sum(r.skipped_tests for r in self.results)
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Determine overall status
        critical_failures = sum(1 for r in self.results 
                              if r.exit_code != 0 and 
                              any(suite['critical'] for suite in self.test_suites.values() 
                                  if suite['name'] == r.name))
        
        overall_status = "PASS" if critical_failures == 0 else "FAIL"
        
        # Mathematical accuracy assessment
        accuracy_result = next((r for r in self.results if 'Accuracy' in r.name), None)
        mathematical_grade = "UNKNOWN"
        if accuracy_result:
            if accuracy_result.success_rate >= 0.999:
                mathematical_grade = "EXCELLENT"
            elif accuracy_result.success_rate >= 0.99:
                mathematical_grade = "VERY_GOOD"
            elif accuracy_result.success_rate >= 0.95:
                mathematical_grade = "GOOD"
            elif accuracy_result.success_rate >= 0.90:
                mathematical_grade = "ACCEPTABLE"
            else:
                mathematical_grade = "POOR"
        
        # Performance assessment
        performance_result = next((r for r in self.results if 'Performance' in r.name), None)
        performance_grade = "UNKNOWN"
        if performance_result:
            if performance_result.success_rate >= 0.95:
                performance_grade = "EXCELLENT"
            elif performance_result.success_rate >= 0.90:
                performance_grade = "GOOD"
            elif performance_result.success_rate >= 0.80:
                performance_grade = "ACCEPTABLE"
            else:
                performance_grade = "POOR"
        
        # Production readiness assessment
        production_ready = (
            overall_status == "PASS" and
            mathematical_grade in ["EXCELLENT", "VERY_GOOD", "GOOD"] and
            performance_grade in ["EXCELLENT", "GOOD", "ACCEPTABLE"]
        )
        
        report = {
            "execution_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_execution_time": total_execution_time,
                "overall_status": overall_status,
                "production_ready": production_ready
            },
            "test_statistics": {
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "skipped_tests": total_skipped,
                "overall_success_rate": overall_success_rate,
                "critical_failures": critical_failures
            },
            "quality_assessment": {
                "mathematical_accuracy": mathematical_grade,
                "performance": performance_grade,
                "real_world_compatibility": "GOOD" if any("Real-World" in r.name and r.success_rate >= 0.8 for r in self.results) else "NEEDS_IMPROVEMENT"
            },
            "test_suite_results": [
                {
                    "name": r.name,
                    "total_tests": r.total_tests,
                    "passed_tests": r.passed_tests,
                    "failed_tests": r.failed_tests,
                    "skipped_tests": r.skipped_tests,
                    "success_rate": r.success_rate,
                    "execution_time": r.execution_time,
                    "status": "PASS" if r.exit_code == 0 else "FAIL",
                    "errors": r.errors
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check mathematical accuracy
        accuracy_result = next((r for r in self.results if 'Accuracy' in r.name), None)
        if accuracy_result:
            if accuracy_result.success_rate < 0.95:
                recommendations.append("Mathematical accuracy is below 95%. Review failed test cases and improve mathematical computation correctness.")
            elif accuracy_result.success_rate < 0.99:
                recommendations.append("Mathematical accuracy is good but could be improved. Consider enhancing precision handling for edge cases.")
        
        # Check performance
        performance_result = next((r for r in self.results if 'Performance' in r.name), None)
        if performance_result:
            if performance_result.success_rate < 0.80:
                recommendations.append("Performance tests show significant issues. Review execution times and memory usage.")
            elif performance_result.success_rate < 0.95:
                recommendations.append("Some performance tests failed. Consider optimizing slow operations.")
        
        # Check real-world compatibility
        real_world_result = next((r for r in self.results if 'Real-World' in r.name), None)
        if real_world_result:
            if real_world_result.success_rate < 0.70:
                recommendations.append("Real-world case processing has low success rate. Review input parsing and error handling.")
        
        # Overall recommendations
        critical_failures = sum(1 for r in self.results if r.exit_code != 0)
        if critical_failures == 0:
            recommendations.append("All critical test suites passed! System appears ready for production use.")
        else:
            recommendations.append(f"{critical_failures} critical test suite(s) failed. Address these issues before production deployment.")
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print comprehensive test report to console."""
        print("\n" + "="*100)
        print("MATHIR PARSER END-TO-END TEST EXECUTION REPORT")
        print("="*100)
        
        # Execution summary
        summary = report["execution_summary"]
        print(f"Execution Time: {summary['total_execution_time']:.1f} seconds")
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Production Ready: {'✓ YES' if summary['production_ready'] else '✗ NO'}")
        
        # Test statistics
        stats = report["test_statistics"]
        print(f"\nTest Statistics:")
        print(f"  Total Tests: {stats['total_tests']}")
        print(f"  Passed: {stats['passed_tests']} ({stats['overall_success_rate']:.1%})")
        print(f"  Failed: {stats['failed_tests']}")
        print(f"  Skipped: {stats['skipped_tests']}")
        print(f"  Critical Failures: {stats['critical_failures']}")
        
        # Quality assessment
        quality = report["quality_assessment"]
        print(f"\nQuality Assessment:")
        print(f"  Mathematical Accuracy: {quality['mathematical_accuracy']}")
        print(f"  Performance: {quality['performance']}")
        print(f"  Real-World Compatibility: {quality['real_world_compatibility']}")
        
        # Individual test suite results
        print(f"\nTest Suite Results:")
        print(f"{'Suite Name':<30} {'Tests':<8} {'Passed':<8} {'Failed':<8} {'Rate':<8} {'Time':<8} {'Status':<8}")
        print("-" * 90)
        
        for suite in report["test_suite_results"]:
            status_symbol = "✓" if suite["status"] == "PASS" else "✗"
            print(f"{suite['name']:<30} {suite['total_tests']:<8} {suite['passed_tests']:<8} "
                  f"{suite['failed_tests']:<8} {suite['success_rate']:<8.1%} {suite['execution_time']:<8.1f} "
                  f"{status_symbol} {suite['status']:<7}")
        
        # Recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        # Final assessment
        print(f"\n" + "="*100)
        if summary['production_ready']:
            print("🎉 CONGRATULATIONS! MathIR Parser has passed comprehensive end-to-end validation!")
            print("   The system demonstrates excellent mathematical accuracy and performance.")
            print("   It is ready for production deployment.")
        else:
            print("⚠️  ATTENTION REQUIRED! MathIR Parser needs improvements before production use.")
            print("   Please address the issues identified in the recommendations above.")
        print("="*100)
    
    def save_report(self, report: Dict[str, Any], filename: str = "tests/logs/end_to_end_report.json"):
        """Save report to JSON file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Detailed report saved to {filename}")


def main():
    """Main entry point for the end-to-end test runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive End-to-End Test Runner for MathIR Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_end_to_end_tests.py                    # Run all tests
  python run_end_to_end_tests.py --accuracy-only    # Run only accuracy tests
  python run_end_to_end_tests.py --quick            # Run quick test subset
  python run_end_to_end_tests.py --full --verbose   # Run all tests with verbose output
        """
    )
    
    parser.add_argument('--accuracy-only', action='store_true',
                       help='Run only mathematical accuracy tests')
    parser.add_argument('--performance-only', action='store_true',
                       help='Run only performance tests')
    parser.add_argument('--real-world-only', action='store_true',
                       help='Run only real-world case tests')
    parser.add_argument('--quick', action='store_true',
                       help='Run a subset of tests for quick validation')
    parser.add_argument('--full', action='store_true',
                       help='Run all tests including slow tests')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate reports from previous test runs')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = EndToEndTestRunner(args)
    
    if args.report_only:
        # Load previous results and generate report
        try:
            with open('tests/logs/end_to_end_report.json', 'r') as f:
                report = json.load(f)
            runner.print_report(report)
        except FileNotFoundError:
            print("No previous test results found. Run tests first.")
            sys.exit(1)
    else:
        # Run tests
        success = runner.run_all_tests()
        
        # Generate and display report
        report = runner.generate_comprehensive_report()
        runner.print_report(report)
        runner.save_report(report)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
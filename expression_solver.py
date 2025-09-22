import argparse
import configparser
import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, TextIO

from sympy import solve, sympify, SympifyError

@dataclass
class Task:
    task_id: str
    expression: str

@dataclass
class SuccessResult:
    task_id: str
    original_expression: str
    final_result: Any
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class FailureResult:
    task_id: str
    expression: str
    error_type: str
    error_message: str
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())

def setup_logging(level: str) -> None:
    """Configures the logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] - %(message)s",
    )

def parse_ndjson(file: TextIO) -> List[Task]:
    """Parses an NDJSON file line by line, creating a list of tasks."""
    tasks = []
    for line in file:
        try:
            data = json.loads(line)
            tasks.append(Task(task_id=data["id"], expression=data["expression"]))
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Skipping corrupted line: {line.strip()} - {type(e).__name__}")
    return tasks

def solve_expression(expression: str) -> Any:
    """
    Solves a mathematical expression using sympy, handling potential errors.
    """
    try:
        if "solve" in expression:
            return solve(sympify(expression.replace("solve(", "").replace(")", "")))
        else:
            return sympify(expression)
    except (SympifyError, SyntaxError, TypeError, ZeroDivisionError) as e:
        logging.error(f"Could not solve expression '{expression}': {e}")
        raise

def write_success_report(file_path: str, results: List[SuccessResult]) -> None:
    """Writes a list of successful results to a CSV file."""
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "original_expression", "final_result", "timestamp_utc"])
        for r in results:
            writer.writerow([r.task_id, r.original_expression, str(r.final_result), r.timestamp_utc])

def write_failure_report(file_path: str, results: List[FailureResult]) -> None:
    """Writes a list of failed tasks to a JSON file."""
    with open(file_path, "w") as f:
        json.dump([res.__dict__ for res in results], f, indent=4)

def main():
    """Main function to orchestrate the expression solving pipeline."""
    parser = argparse.ArgumentParser(description="Solve symbolic math expressions.")
    parser.add_argument(
        "--config", type=str, default="config.ini", help="Path to the configuration file."
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    paths = config["paths"]
    settings = config_get(config, "settings", {})


    setup_logging(settings.get("log_level", "INFO"))

    input_file = paths.get("input_file")
    success_file = paths.get("success_file")
    failure_file = paths.get("failure_file")

    if not input_file:
        logging.error("Input file path not found in configuration.")
        return

    successes: List[SuccessResult] = []
    failures: List[FailureResult] = []

    try:
        with open(input_file, "r") as f:
            tasks = parse_ndjson(f)
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_file}")
        tasks = []

    for task in tasks:
        try:
            result = solve_expression(task.expression)
            successes.append(
                SuccessResult(
                    task_id=task.task_id,
                    original_expression=task.expression,
                    final_result=result,
                )
            )
        except Exception as e:
            failures.append(
                FailureResult(
                    task_id=task.task_id,
                    expression=task.expression,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )

    if success_file:
        write_success_report(success_file, successes)
    if failure_file:
        write_failure_report(failure_file, failures)

    logging.info(f"Processing complete. Solved: {len(successes)}, Failed: {len(failures)}")

def config_get(config, section, fallback):
    if section in config:
        return config[section]
    
    section_lower = section.lower()
    if section_lower in config:
        return config[section_lower]
        
    return fallback

if __name__ == "__main__":
    main()
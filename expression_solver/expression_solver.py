import argparse
import configparser
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, List, TextIO, Union

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
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class FailureResult:
    task_id: str
    expression: str
    error_type: str
    error_message: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

def setup_logging(level: str) -> None:
    """Configures the logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] - %(message)s",
    )

def parse_ndjson(file: TextIO) -> List[Union[Task, FailureResult]]:
    """
    Parses an NDJSON file line by line.
    Returns a list of Task objects for valid lines and FailureResult for corrupted ones.
    """
    items: List[Union[Task, FailureResult]] = []
    for i, line in enumerate(file):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # Use .get() to avoid KeyErrors for missing fields
            task_id = data.get("id")
            expression = data.get("expression")
            if task_id is None or expression is None:
                raise KeyError("Line is missing 'id' or 'expression'")
            items.append(Task(task_id=task_id, expression=expression))
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Skipping corrupted line: {line}", exc_info=True)
            # Create a FailureResult for the malformed line
            items.append(
                FailureResult(
                    task_id=f"malformed_line_{i+1}",
                    expression=line,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )
    return items

def solve_expression(expression: str) -> Any:
    """Solves a mathematical expression using sympy, handling potential errors."""
    try:
        parsed_expr = sympify(expression, locals={"solve": solve})
        try:
            return parsed_expr.doit()
        except AttributeError:
            return parsed_expr # Typically happens when solve returns a list
    except Exception as e:
        logging.error(f"A general exception occurred while solving: '{expression}'", exc_info=True)
        raise

def write_success_report(file_path: str, results: List[SuccessResult]) -> None:
    """Writes a list of successful results to a CSV file."""
    with open(file_path, "w", newline="") as f:
        # Use asdict to handle dataclass conversion
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

def write_failure_report(file_path: str, results: List[FailureResult]) -> None:
    """Writes a list of failed tasks to a JSON file."""
    with open(file_path, "w") as f:
        json.dump([asdict(res) for res in results], f, indent=4)

def main():
    """Main function to orchestrate the expression solving pipeline."""
    parser = argparse.ArgumentParser(description="Solve symbolic math expressions.")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config.")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    paths = config["paths"]
    settings = config_get(config, "settings", {})

    setup_logging(settings.get("log_level", "INFO"))

    input_file = paths.get("input_file")
    success_file = paths.get("success_file")
    failure_file = paths.get("failure_file")

    if not all([input_file, success_file, failure_file]):
        logging.error("A required file path is missing in the configuration.")
        return

    successes: List[SuccessResult] = []
    failures: List[FailureResult] = []

    try:
        with open(input_file, "r") as f:
            items = parse_ndjson(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        items = []

    for item in items:
        if isinstance(item, FailureResult):
            failures.append(item)
            continue
            
        try:
            result = solve_expression(item.expression)
            successes.append(
                SuccessResult(
                    task_id=item.task_id,
                    original_expression=item.expression,
                    final_result=str(result),
                )
            )
        except Exception as e:
            failures.append(
                FailureResult(
                    task_id=item.task_id,
                    expression=item.expression,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            )

    if successes:
        write_success_report(success_file, successes)
    else: # Create empty file with headers if no successes
        with open(success_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_id", "original_expression", "final_result", "timestamp_utc"])

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

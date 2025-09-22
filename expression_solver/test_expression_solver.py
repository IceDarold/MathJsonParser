import sys
import json
import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from expression_solver.expression_solver import main

@pytest.fixture
def setup_test_environment(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    config_path = tmp_path / "config.ini"
    input_file_path = input_dir / "symbolic_tasks.json"
    success_file_path = output_dir / "solved_tasks_report.csv"
    failure_file_path = output_dir / "failed_tasks_log.json"

    config_content = f"""
[paths]
input_file = {input_file_path}
success_file = {success_file_path}
failure_file = {failure_file_path}
"""
    config_path.write_text(config_content)

    return {
        "config_path": str(config_path),
        "input_file_path": input_file_path,
        "success_file_path": success_file_path,
        "failure_file_path": failure_file_path
    }

def run_main_with_args(config_path):
    with patch.object(sys, 'argv', ['expression_solver.py', '--config', config_path]):
        main()

def test_pipeline_runs_successfully(setup_test_environment):
    env = setup_test_environment
    tasks = [
        {"id": "task_001", "expression": "2 + 2"},
        {"id": "task_002", "expression": "solve(x**2 - 4, x)"}
    ]
    with open(env["input_file_path"], "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    run_main_with_args(env["config_path"])

    assert env["success_file_path"].exists()
    assert env["failure_file_path"].exists()

    success_df = pd.read_csv(env["success_file_path"])
    with open(env["failure_file_path"], "r") as f:
        failed_logs = json.load(f)

    assert len(success_df) == 2
    assert len(failed_logs) == 0

def test_handles_malformed_json(setup_test_environment):
    env = setup_test_environment
    malformed_json = '{"id": "task_001", "expression": "2 + 2"}\n{invalid_json}\n'
    env["input_file_path"].write_text(malformed_json)

    run_main_with_args(env["config_path"])

    assert env["failure_file_path"].exists()
    with open(env["failure_file_path"], "r") as f:
        failed_logs = json.load(f)

    assert len(failed_logs) == 1
    assert failed_logs[0]['error_type'] == 'JSONDecodeError'

def test_handles_sympy_errors(setup_test_environment):
    env = setup_test_environment
    tasks = [
        {"id": "task_001", "expression": "solve(sin(x), x)"},
        {"id": "task_002", "expression": "1 / 0"}
    ]
    with open(env["input_file_path"], "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    run_main_with_args(env["config_path"])

    assert env["failure_file_path"].exists()
    with open(env["failure_file_path"], "r") as f:
        failed_logs = json.load(f)

    assert len(failed_logs) == 2

def test_empty_input_file(setup_test_environment):
    env = setup_test_environment
    env["input_file_path"].touch()

    run_main_with_args(env["config_path"])

    assert env["success_file_path"].exists()
    assert env["failure_file_path"].exists()

    success_df = pd.read_csv(env["success_file_path"])
    with open(env["failure_file_path"], "r") as f:
        failed_logs = json.load(f)

    assert len(success_df) == 0
    assert len(failed_logs) == 0
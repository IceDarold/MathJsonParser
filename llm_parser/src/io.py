import pandas as pd
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class DataLoader:
    """Handles CSV reading and basic preprocessing."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_data(self) -> pd.DataFrame:
        """Load CSV and normalize whitespace."""
        df = pd.read_csv(self.csv_path)
        # Normalize whitespace but preserve LaTeX
        df['task'] = df['task'].str.strip()
        df['answer'] = df['answer'].str.strip()
        return df


class IDGenerator:
    """Generates unique IDs for tasks."""

    @staticmethod
    def generate_id(row_index: int) -> str:
        """Generate ID like task_000123."""
        return f"task_{row_index:06d}"


class Writer:
    """Handles writing JSON files, JSONL, and error files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "outputs.jsonl"

    def write_json(self, task_id: str, data: Dict[str, Any]) -> None:
        """Write individual JSON file."""
        file_path = self.output_dir / f"{task_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def append_jsonl(self, data: Dict[str, Any]) -> None:
        """Append to JSONL file."""
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    def write_error(self, task_id: str, raw_response: str, errors: List[str]) -> None:
        """Write error file for invalid JSON."""
        error_data = {
            "status": "invalid",
            "errors": errors,
            "raw": raw_response[:1000]  # Truncate for brevity
        }
        file_path = self.output_dir / f"{task_id}.error.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

    def write_report(self, report: Dict[str, Any]) -> None:
        """Write run report."""
        report_path = self.output_dir / "run_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
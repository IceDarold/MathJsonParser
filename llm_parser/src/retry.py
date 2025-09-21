from typing import List, Optional
from pathlib import Path


class RetryHandler:
    """Handles retry logic with delta-feedback hints."""

    def __init__(self, hint_template_path: str, max_retries: int = 2):
        self.hint_template = self._load_template(hint_template_path)
        self.max_retries = max_retries

    def _load_template(self, path: str) -> str:
        """Load retry hint template."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def generate_hint(self, errors: List[str]) -> str:
        """Generate retry hint by filling template with errors."""
        error_list = "\n".join(f"- {error}" for error in errors)
        return self.hint_template.replace("{{ERROR_LIST}}", error_list)

    def should_retry(self, attempt_count: int) -> bool:
        """Check if should retry based on attempt count."""
        return attempt_count < self.max_retries
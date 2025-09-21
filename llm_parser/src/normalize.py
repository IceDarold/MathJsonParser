from typing import Dict, Any


class Normalizer:
    """Post-normalization of MathIR JSON."""

    def __init__(self, default_lang: str = "ru", default_expr_format: str = "latex"):
        self.default_lang = default_lang
        self.default_expr_format = default_expr_format

    def normalize(self, data: Dict[str, Any], task_id: str, gold_answer: str, source: str = "csv") -> Dict[str, Any]:
        """Normalize the MathIR data by adding/updating meta fields."""
        # Ensure meta exists
        if 'meta' not in data:
            data['meta'] = {}

        # Add required meta fields
        data['meta']['id'] = task_id
        data['meta']['lang'] = self.default_lang
        data['meta']['gold_answer'] = gold_answer
        data['meta']['source'] = source

        # Set default expr_format if not present
        if 'expr_format' not in data:
            data['expr_format'] = self.default_expr_format

        # Ensure notes is present (even if empty)
        if 'notes' not in data['meta']:
            data['meta']['notes'] = ""

        return data
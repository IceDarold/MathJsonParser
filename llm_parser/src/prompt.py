import json
from typing import List, Dict, Any
from pathlib import Path


class PromptBuilder:
    """Assembles prompts from system, fewshot, and user components."""

    def __init__(self, system_path: str, fewshot_path: str):
        self.system_prompt = self._load_system(system_path)
        self.fewshot_examples = self._load_fewshot(fewshot_path)

    def _load_system(self, path: str) -> str:
        """Load system prompt."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _load_fewshot(self, path: str) -> List[Dict[str, Any]]:
        """Load fewshot examples from JSONL."""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line.strip()))
        return examples

    def build_prompt(self, user_task: str) -> str:
        """Build complete prompt: system + fewshot + user."""
        prompt_parts = []

        # System prompt
        prompt_parts.append(self.system_prompt)

        # Fewshot examples
        for example in self.fewshot_examples:
            # Assuming fewshot is in JSONL with input/output, but in the file it's just output JSON.
            # The spec says fewshot.jsonl has {input, output} pairs, but the file has only JSON outputs.
            # Perhaps it's just the outputs, and input is implied or separate.
            # Looking back: "fewshot.jsonl — 5–8 примеров: {input, output}"
            # But the file has only JSON lines. Maybe it's outputs only, and inputs are separate.
            # For now, assume the JSONL has the output examples, and we format as fewshot.
            prompt_parts.append("Example:")
            prompt_parts.append(json.dumps(example, ensure_ascii=False))

        # User task
        prompt_parts.append("Task:")
        prompt_parts.append(user_task)

        return "\n\n".join(prompt_parts)
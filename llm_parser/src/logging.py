import logging
import json
import sys
from typing import Dict, Any
from pythonjsonlogger import jsonlogger


class StructuredLogger:
    """Structured JSON logging."""

    def __init__(self, level: str = "INFO", log_file: str = None):
        self.logger = logging.getLogger("llm_parser")
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers.clear()

        # JSON formatter
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_task_start(self, task_id: str, task_text: str):
        """Log task processing start."""
        self.logger.info("Task processing started", extra={
            "task_id": task_id,
            "event": "task_start",
            "task_text": task_text[:100]  # Truncate for brevity
        })

    def log_llm_call(self, task_id: str, prompt: str, response: str, usage: Dict[str, Any], latency: float):
        """Log LLM call details."""
        # Truncate for readability if too long
        max_length = 1000
        truncated_prompt = prompt[:max_length] + ("..." if len(prompt) > max_length else "")
        truncated_response = response[:max_length] + ("..." if len(response) > max_length else "")

        self.logger.info("LLM call completed", extra={
            "task_id": task_id,
            "event": "llm_call",
            "prompt": truncated_prompt,
            "response": truncated_response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "usage": usage,
            "latency": latency
        })

    def log_validation(self, task_id: str, valid: bool, errors: list, retries: int):
        """Log validation result."""
        self.logger.info("Validation completed", extra={
            "task_id": task_id,
            "event": "validation",
            "valid": valid,
            "errors": errors,
            "retries": retries
        })

    def log_task_end(self, task_id: str, success: bool, total_latency: float):
        """Log task processing end."""
        self.logger.info("Task processing ended", extra={
            "task_id": task_id,
            "event": "task_end",
            "success": success,
            "total_latency": total_latency
        })

    def log_error(self, task_id: str, error: str, context: Dict[str, Any] = None):
        """Log errors."""
        self.logger.error("Error occurred", extra={
            "task_id": task_id,
            "event": "error",
            "error": error,
            **(context or {})
        })
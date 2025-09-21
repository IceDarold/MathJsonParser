from typing import Dict, Any, List
from dataclasses import dataclass, field
from statistics import mean, median


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_id: str
    valid: bool
    retries_used: int
    tokens_used: int
    latency: float
    errors: List[str] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for the run."""
    total_tasks: int = 0
    valid_count: int = 0
    total_retries: int = 0
    total_tokens: int = 0
    latencies: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def valid_rate(self) -> float:
        return self.valid_count / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def avg_retries_per_task(self) -> float:
        return self.total_retries / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def avg_tokens_per_task(self) -> float:
        return self.total_tokens / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0

    @property
    def median_latency(self) -> float:
        return median(self.latencies) if self.latencies else 0.0

    def to_report(self) -> Dict[str, Any]:
        """Convert to report dict."""
        return {
            "total_tasks": self.total_tasks,
            "valid_rate": round(self.valid_rate, 4),
            "avg_retries_per_task": round(self.avg_retries_per_task, 2),
            "avg_tokens_per_task": round(self.avg_tokens_per_task, 2),
            "avg_latency": round(self.avg_latency, 2),
            "median_latency": round(self.median_latency, 2),
            "error_distribution": self.error_counts
        }


class MetricsCollector:
    """Collects and aggregates metrics."""

    def __init__(self):
        self.tasks: List[TaskMetrics] = []
        self.aggregated = AggregatedMetrics()

    def add_task(self, task: TaskMetrics):
        """Add metrics for a task."""
        self.tasks.append(task)
        self.aggregated.total_tasks += 1
        if task.valid:
            self.aggregated.valid_count += 1
        self.aggregated.total_retries += task.retries_used
        self.aggregated.total_tokens += task.tokens_used
        self.aggregated.latencies.append(task.latency)
        for error in task.errors:
            self.aggregated.error_counts[error] = self.aggregated.error_counts.get(error, 0) + 1

    def get_report(self) -> Dict[str, Any]:
        """Get the final report."""
        return self.aggregated.to_report()
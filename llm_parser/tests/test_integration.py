import pytest
import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock
from src.prompt import PromptBuilder
from src.client import ConcurrentClient
from src.guard import JSONValidator
from src.retry import RetryHandler
from src.normalize import Normalizer
from src.metrics import MetricsCollector
from src.logging import StructuredLogger
from src.io import Writer


class StubLLMClient:
    """Stub LLM client that returns fewshot examples."""

    def __init__(self, fewshot_path: str):
        self.fewshot_examples = self._load_fewshot(fewshot_path)
        self.call_count = 0

    def _load_fewshot(self, path: str):
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line.strip()))
        return examples

    async def generate_json(self, prompt: str, retry_hint=None):
        # Return a fewshot example, cycling through them
        example = self.fewshot_examples[self.call_count % len(self.fewshot_examples)]
        self.call_count += 1
        raw_response = json.dumps(example, ensure_ascii=False)
        usage = {'total_tokens': 100, 'input_tokens': 50, 'output_tokens': 50}
        latency = 0.5
        return raw_response, usage, latency


@pytest.fixture
def stub_client():
    fewshot_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'fewshot.jsonl')
    return StubLLMClient(fewshot_path)


@pytest.fixture
def concurrent_client(stub_client):
    return ConcurrentClient(stub_client, concurrency=1)


@pytest.fixture
def validator():
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'schema', 'mathir.schema.json')
    return JSONValidator(schema_path)


@pytest.fixture
def prompt_builder():
    system_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system.txt')
    fewshot_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'fewshot.jsonl')
    return PromptBuilder(system_path, fewshot_path)


@pytest.fixture
def retry_handler():
    hint_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'retry_hint.txt')
    return RetryHandler(hint_path, 2)


@pytest.fixture
def normalizer():
    return Normalizer()


@pytest.fixture
def logger():
    return StructuredLogger('INFO')


@pytest.fixture
def sample_tasks():
    with open(os.path.join(os.path.dirname(__file__), 'data', 'sample_tasks.json'), 'r') as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_process_task_integration(concurrent_client, validator, prompt_builder, retry_handler, normalizer, logger, sample_tasks):
    """Integration test for processing tasks with stub LLM."""
    metrics_collector = MetricsCollector()

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as output_dir:
        writer = Writer(output_dir)

        # Test first 5 tasks
        for i, task_data in enumerate(sample_tasks[:5]):
            task_id = f"test_task_{i}"
            user_task = task_data['task']
            gold_answer = task_data['answer']

            # Process the task
            await process_task_stub(
                task_id, user_task, gold_answer, prompt_builder, concurrent_client,
                validator, retry_handler, normalizer, writer, metrics_collector, logger
            )

        # Check that tasks were processed
        report = metrics_collector.get_report()
        assert report['total_tasks'] == 5
        assert report['valid_rate'] >= 0.6  # At least 60% should succeed with stub


async def process_task_stub(
    task_id: str,
    user_task: str,
    gold_answer: str,
    prompt_builder: PromptBuilder,
    concurrent_client: ConcurrentClient,
    validator: JSONValidator,
    retry_handler: RetryHandler,
    normalizer: Normalizer,
    writer: Writer,
    metrics_collector: MetricsCollector,
    logger: StructuredLogger
):
    """Stub version of process_task for testing."""
    logger.log_task_start(task_id, user_task)

    prompt = prompt_builder.build_prompt(user_task)
    retries = 0
    valid = False
    parsed_data = None
    errors = []
    total_latency = 0.0
    total_tokens = 0

    while not valid and retry_handler.should_retry(retries):
        retry_hint = retry_handler.generate_hint(errors) if retries > 0 else None
        try:
            raw_response, usage, latency = await concurrent_client.generate_json(prompt, retry_hint)
            total_latency += latency
            total_tokens = usage.get('total_tokens', usage.get('input_tokens', 0) + usage.get('output_tokens', 0))

            logger.log_llm_call(task_id, len(prompt), len(raw_response), usage, latency)

            valid, parsed_data, errors = validator.validate(raw_response)

            if not valid:
                retries += 1
            else:
                break
        except Exception as e:
            errors = [f"LLM call failed: {str(e)}"]
            retries += 1
            if not retry_handler.should_retry(retries):
                break

    if valid and parsed_data:
        try:
            normalized = normalizer.normalize(parsed_data, task_id, gold_answer)
            writer.write_json(task_id, normalized)
            writer.append_jsonl(normalized)
        except Exception as e:
            valid = False
            errors = [f"Normalization failed: {str(e)}"]
            writer.write_error(task_id, raw_response if 'raw_response' in locals() else "", errors)

    if not valid:
        writer.write_error(task_id, raw_response if 'raw_response' in locals() else "", errors)

    from src.metrics import TaskMetrics
    task_metrics = TaskMetrics(task_id, valid, retries, total_tokens, total_latency, errors)
    metrics_collector.add_task(task_metrics)

    logger.log_validation(task_id, valid, errors, retries)
    logger.log_task_end(task_id, valid, total_latency)


@pytest.mark.asyncio
async def test_full_pipeline_with_10_tasks(sample_tasks, concurrent_client, validator, prompt_builder, retry_handler, normalizer, logger):
    """Test the full pipeline with 10 sample tasks."""
    metrics_collector = MetricsCollector()

    with tempfile.TemporaryDirectory() as output_dir:
        writer = Writer(output_dir)

        tasks = []
        for i, task_data in enumerate(sample_tasks):
            task_id = f"integration_task_{i}"
            user_task = task_data['task']
            gold_answer = task_data['answer']

            task = process_task_stub(
                task_id, user_task, gold_answer, prompt_builder, concurrent_client,
                validator, retry_handler, normalizer, writer, metrics_collector, logger
            )
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        report = metrics_collector.get_report()
        assert report['total_tasks'] == 10
        # With stub returning valid JSON, most should succeed
        assert report['valid_rate'] >= 0.8
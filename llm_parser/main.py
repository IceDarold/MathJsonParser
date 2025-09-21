import asyncio
import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

from src.io import DataLoader, IDGenerator, Writer
from src.prompt import PromptBuilder
from src.client import create_client, ConcurrentClient
from src.guard import JSONValidator
from src.retry import RetryHandler
from src.normalize import Normalizer
from src.metrics import MetricsCollector, TaskMetrics
from src.logging import StructuredLogger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM MathIR Parser')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--provider', help='Override provider (openai, anthropic, local)')
    parser.add_argument('--model', help='Override model')
    parser.add_argument('--local', action='store_true', help='Use local OpenAI-compatible API')
    parser.add_argument('--api-key', help='API key for provider')
    parser.add_argument('--base-url', help='Base URL for local API')
    return parser.parse_args()


async def process_task(
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
) -> None:
    """Process a single task."""
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

    task_metrics = TaskMetrics(task_id, valid, retries, total_tokens, total_latency, errors)
    metrics_collector.add_task(task_metrics)

    logger.log_validation(task_id, valid, errors, retries)
    logger.log_task_end(task_id, valid, total_latency)


async def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply overrides
    if args.provider:
        config['provider'] = args.provider
    if args.model:
        config['model'] = args.model
    if args.local:
        config['provider'] = 'local'
    if args.api_key:
        config['api_key'] = args.api_key
    if args.base_url:
        config['base_url'] = args.base_url

    # Initialize components
    try:
        logger = StructuredLogger(config.get('log_level', 'INFO'))

        data_loader = DataLoader(config['input_csv'])
        id_gen = IDGenerator()
        prompt_builder = PromptBuilder(config['system_prompt'], config['fewshot_path'])
        validator = JSONValidator(config['schema_path'])
        retry_handler = RetryHandler(config['retry_hint_template'], config['retries'])
        normalizer = Normalizer()
        writer = Writer(config['output_dir'])
        client = create_client(config)
        concurrent_client = ConcurrentClient(client, config['concurrency'])
        metrics_collector = MetricsCollector()

        # Load data
        df = data_loader.load_data()
        if df.empty:
            logger.logger.error("No data loaded from CSV")
            sys.exit(1)

        # Process tasks concurrently
        tasks = []
        for index, row in df.iterrows():
            task_id = id_gen.generate_id(index)
            user_task = row['task']
            gold_answer = row['answer']
            task = process_task(
                task_id, user_task, gold_answer, prompt_builder, concurrent_client,
                validator, retry_handler, normalizer, writer, metrics_collector, logger
            )
            tasks.append(task)

        # Run all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        # Generate report
        report = metrics_collector.get_report()
        writer.write_report(report)

        # Close client if needed
        if hasattr(client, 'close'):
            await client.close()

        logger.logger.info("Processing completed", extra={"report": report})

    except Exception as e:
        logger.logger.error("Fatal error", extra={"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
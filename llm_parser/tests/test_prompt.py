import pytest
import os
from src.prompt import PromptBuilder


@pytest.fixture
def prompt_builder():
    system_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system.txt')
    fewshot_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'fewshot.jsonl')
    return PromptBuilder(system_path, fewshot_path)


def test_build_prompt_includes_system(prompt_builder):
    user_task = "Solve x^2 = 4"
    prompt = prompt_builder.build_prompt(user_task)
    assert prompt_builder.system_prompt in prompt


def test_build_prompt_includes_user_task(prompt_builder):
    user_task = "Solve x^2 = 4"
    prompt = prompt_builder.build_prompt(user_task)
    assert user_task in prompt


def test_build_prompt_includes_fewshot_examples(prompt_builder):
    user_task = "Solve x^2 = 4"
    prompt = prompt_builder.build_prompt(user_task)
    # Check that fewshot examples are included
    assert "Example:" in prompt
    # Assuming fewshot has JSON, check for some key
    assert "expr_format" in prompt


def test_build_prompt_no_answer_leak(prompt_builder):
    user_task = "Solve x^2 = 4"
    gold_answer = "x = ±2"
    # Simulate if answer was somehow in prompt - but it shouldn't be
    prompt = prompt_builder.build_prompt(user_task)
    # Ensure gold_answer is not in prompt
    assert gold_answer not in prompt
    # Ensure the user_task does not contain the answer
    assert "±2" not in user_task


def test_build_prompt_structure(prompt_builder):
    user_task = "Solve x^2 = 4"
    prompt = prompt_builder.build_prompt(user_task)
    parts = prompt.split("\n\n")
    assert len(parts) >= 3  # system, fewshot(s), task
    assert parts[-2] == "Task:"
    assert parts[-1] == user_task


def test_load_system(prompt_builder):
    assert isinstance(prompt_builder.system_prompt, str)
    assert len(prompt_builder.system_prompt) > 0


def test_load_fewshot(prompt_builder):
    assert isinstance(prompt_builder.fewshot_examples, list)
    assert len(prompt_builder.fewshot_examples) > 0
    # Check each is dict
    for ex in prompt_builder.fewshot_examples:
        assert isinstance(ex, dict)
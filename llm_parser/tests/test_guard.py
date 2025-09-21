import pytest
import json
import os
from src.guard import JSONValidator


@pytest.fixture
def validator():
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'schema', 'mathir.schema.json')
    return JSONValidator(schema_path)


@pytest.fixture
def valid_json():
    with open(os.path.join(os.path.dirname(__file__), 'data', 'valid_mathir.json'), 'r') as f:
        return json.dumps(json.load(f))


@pytest.fixture
def invalid_json():
    with open(os.path.join(os.path.dirname(__file__), 'data', 'invalid_mathir.json'), 'r') as f:
        return json.dumps(json.load(f))


def test_validate_valid_json(validator, valid_json):
    is_valid, parsed, errors = validator.validate(valid_json)
    assert is_valid is True
    assert parsed is not None
    assert errors == []


def test_validate_invalid_json(validator, invalid_json):
    is_valid, parsed, errors = validator.validate(invalid_json)
    assert is_valid is False
    assert parsed is None
    assert len(errors) > 0


def test_validate_invalid_json_format(validator):
    invalid_json_str = "{invalid json"
    is_valid, parsed, errors = validator.validate(invalid_json_str)
    assert is_valid is False
    assert parsed is None
    assert "Invalid JSON" in errors[0]


def test_validate_with_extra_fields(validator):
    # Valid JSON with extra fields should be invalid due to additionalProperties: false
    extra_json = json.dumps({
        "expr_format": "latex",
        "targets": [{"type": "integral_def", "expr": "x", "var": "x", "limits": ["0", "1"]}],
        "extra_field": "should not be allowed"
    })
    is_valid, parsed, errors = validator.validate(extra_json)
    assert is_valid is False
    assert parsed is None
    assert len(errors) > 0


def test_validate_missing_required_field(validator):
    missing_required = json.dumps({
        "expr_format": "latex"
        # missing targets
    })
    is_valid, parsed, errors = validator.validate(missing_required)
    assert is_valid is False
    assert parsed is None
    assert len(errors) > 0
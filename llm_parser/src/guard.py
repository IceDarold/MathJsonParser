import json
import jsonschema
from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path


# Pydantic models based on MathIR schema

class SymbolSpec(BaseModel):
    name: str
    domain: Literal['R', 'R+', 'Z', 'N', 'N+', 'C'] = 'R'

class FunctionDef(BaseModel):
    name: str
    args: List[str]
    expr: str

class SequenceDef(BaseModel):
    name: str
    args: List[str]
    expr: str

class MatrixDef(BaseModel):
    name: str
    rows: int
    cols: int
    data: List[List[str]]

class DistributionDef(BaseModel):
    name: str
    kind: Literal['bernoulli', 'binomial', 'geometric', 'poisson', 'hypergeom']
    params: Dict[str, str]

class GeometryDef(BaseModel):
    id: str
    kind: Literal['line', 'circle', 'parabola', 'ellipse', 'polyline']
    equation: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class Definitions(BaseModel):
    functions: List[FunctionDef] = []
    sequences: List[SequenceDef] = []
    matrices: List[MatrixDef] = []
    distributions: List[DistributionDef] = []
    geometry: List[GeometryDef] = []

class TransformSpec(BaseModel):
    target: str
    type: Literal['rotation', 'translation', 'scaling', 'substitution']
    angle_deg: Optional[float] = None
    center: Optional[List[float]] = None
    vector: Optional[List[float]] = None
    factor: Optional[float] = None
    subs: Optional[Dict[str, str]] = None

class Condition(BaseModel):
    type: str
    expr: Optional[str] = None
    object: Optional[str] = None
    point: Optional[List[float]] = None
    value: Optional[int] = None

class TargetIntegral(BaseModel):
    type: Literal['integral_def']
    expr: str
    var: str
    limits: List[Union[str, float]]
    name: Optional[str] = None

class TargetLimit(BaseModel):
    type: Literal['limit']
    expr: str
    var: str
    to: str

class TargetSum(BaseModel):
    type: Literal['sum']
    term: str
    idx: str
    start: str
    end: str

class TargetSolve(BaseModel):
    type: Literal['solve_for']
    unknowns: List[str]
    equations: List[str]

class TargetIneq(BaseModel):
    type: Literal['inequalities']
    inequalities: List[str]

class TargetMatrixSolve(BaseModel):
    type: Literal['solve_for_matrix']
    unknown: str

class TargetProbability(BaseModel):
    type: Literal['probability']
    event_expr: str

class TargetValue(BaseModel):
    type: Literal['value']
    name: str
    expr: str

Target = Union[TargetIntegral, TargetLimit, TargetSum, TargetSolve, TargetIneq, TargetMatrixSolve, TargetProbability, TargetValue]

class OutputSpec(BaseModel):
    mode: Optional[Literal['exact', 'decimal']] = 'decimal'
    round_to: Optional[int] = 3
    simplify: bool = True
    rationalize: bool = False

class ValidationSpec(BaseModel):
    tolerance_abs: float = 1e-9
    check_domain_violations: bool = True

class MathIR(BaseModel):
    meta: Dict[str, Any] = {}
    task_type: Literal['auto', 'integral', 'limit', 'sum', 'algebra', 'matrix', 'probability', 'geometry', 'optimize'] = 'auto'
    expr_format: Literal['latex', 'sympy', 'infix'] = 'latex'
    assumptions: Dict[str, Any] = {"reals_default": True}
    constants: Dict[str, str] = {}
    symbols: List[SymbolSpec] = []
    definitions: Definitions = Definitions()
    transforms: List[TransformSpec] = []
    conditions: List[Condition] = []
    targets: List[Target]
    output: OutputSpec = OutputSpec()
    validation: ValidationSpec = ValidationSpec()


class JSONValidator:
    """Validates JSON against schema using jsonschema and Pydantic."""

    def __init__(self, schema_path: str):
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)

    def validate(self, json_str: str) -> tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """Validate JSON string. Returns (is_valid, parsed_data, error_list)."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return False, None, [f"Invalid JSON: {str(e)}"]

        # Validate against JSON schema
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError as e:
            errors = [f"{e.absolute_path}: {e.message}"] if e.absolute_path else [e.message]
            return False, None, errors

        # Validate with Pydantic
        try:
            validated_data = MathIR(**data)
            return True, validated_data.model_dump(), []
        except Exception as e:
            return False, None, [f"Pydantic validation error: {str(e)}"]
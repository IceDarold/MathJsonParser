# mathir_parser_v01.py
from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel
from dataclasses import dataclass
import sympy as sp
import logging
import json
import sys

from sympy.parsing.latex import parse_latex

# === Helpers: LaTeX → SymPy ===
def to_sympy_expr(s: str) -> sp.Expr:
    """Parse a LaTeX string to a SymPy expression without context."""
    return parse_latex(s)

# === Core IR ===
class SymbolSpec(BaseModel):
    name: str
    domain: Literal['R','R+','Z','N','N+','C'] = 'R'

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
    kind: Literal['bernoulli','binomial','geometric','poisson','hypergeom']
    params: Dict[str, str]

class GeometryDef(BaseModel):
    id: str
    kind: Literal['line','circle','parabola','ellipse','polyline']
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
    type: Literal['rotation','translation','scaling','substitution']
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
    limits: List[Any]
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
    mode: Literal['exact','decimal'] = 'decimal'
    round_to: Optional[int] = 3
    simplify: bool = True
    rationalize: bool = False

class ValidationSpec(BaseModel):
    tolerance_abs: float = 1e-9
    check_domain_violations: bool = True

class MathIR(BaseModel):
    meta: Dict[str, Any] = {}
    task_type: Literal['auto','integral','limit','sum','algebra','matrix','probability','geometry','optimize'] = 'auto'
    expr_format: Literal['latex','sympy','infix'] = 'latex'
    assumptions: Dict[str, Any] = {}
    constants: Dict[str, str] = {}
    symbols: List[SymbolSpec] = []
    definitions: Definitions = Definitions()
    transforms: List[TransformSpec] = []
    conditions: List[Condition] = []
    targets: List[Target]
    output: OutputSpec = OutputSpec()
    validation: ValidationSpec = ValidationSpec()

# === Runtime context ===
@dataclass
class Runtime:
    symtab: Dict[str, sp.Symbol]
    funcs: Dict[str, sp.Lambda]
    sequences: Dict[str, sp.Lambda]
    matrices: Dict[str, sp.Matrix]
    distributions: Dict[str, Any]
    geometry: Dict[str, Any]
    # Master context for parsing
    context: Dict[str, Any]

# === Builders ===
DOMAIN_MAP = {
    'R': sp.S.Reals,
    'R+': sp.Interval.Ropen(0, sp.oo),
    'Z': sp.S.Integers,
    'N': sp.S.Naturals0,
    'N+': sp.S.Naturals,
    'C': sp.S.Complexes,
}

def build_runtime(ir: MathIR) -> Runtime:
    symtab: Dict[str, sp.Symbol] = {}
    for s in ir.symbols:
        # Simplified domain handling for now
        symtab[s.name] = sp.Symbol(s.name, real=True)
    
    # Build master context for parsing
    context: Dict[str, Any] = {**symtab}
    context['pi'] = sp.pi
    context['e'] = sp.E
    context['i'] = sp.I

    funcs: Dict[str, sp.Lambda] = {}
    for f_def in ir.definitions.functions:
        args = [sp.Symbol(a) for a in f_def.args]
        # For function definitions, the context is only the function's own arguments
        func_arg_context = {a.name: a for a in args}
        expr_template = to_sympy_expr(f_def.expr)
        expr = expr_template.subs({sp.Symbol(k): v for k, v in func_arg_context.items()})
        funcs[f_def.name] = sp.Lambda(tuple(args), expr)

    sequences: Dict[str, sp.Lambda] = {}
    for s_def in ir.definitions.sequences:
        args = [sp.Symbol(a) for a in s_def.args]
        # Similar for sequences
        seq_arg_context = {a.name: a for a in args}
        expr_template = to_sympy_expr(s_def.expr)
        expr = expr_template.subs({sp.Symbol(k): v for k, v in seq_arg_context.items()})
        sequences[s_def.name] = sp.Lambda(tuple(args), expr)
        
    matrices: Dict[str, sp.Matrix] = {}
    for m in ir.definitions.matrices:
        # Matrix elements are parsed with the main context
        mat = sp.Matrix([[to_sympy_expr(v).subs({sp.Symbol(k): v for k, v in context.items()}) for v in row] for row in m.data])
        matrices[m.name] = mat

    # Add newly defined items to context
    context.update(funcs)
    context.update(matrices)

    return Runtime(symtab, funcs, sequences, matrices, {}, {}, context)

# === Node executors ===

def exec_integral(rt: Runtime, t: TargetIntegral) -> Tuple[str, sp.Expr]:
    # This is the correct way to handle the integration variable
    integration_var = sp.Symbol(t.var, real=True)

    # 1. Parse expressions first from raw LaTeX
    a_template = to_sympy_expr(str(t.limits[0]))
    b_template = to_sympy_expr(str(t.limits[1]))
    expr_template = to_sympy_expr(t.expr)

    # 2. Create the substitution dictionary from the runtime context
    # We must map the generic symbols from parsing to the real context symbols
    # e.g. {Symbol('x'): context['x'], Symbol('pi'): context['pi']}
    subs_dict = {sp.Symbol(k): v for k, v in rt.context.items()}
    
    # 3. Substitute the context into the templates
    a = a_template.subs(subs_dict)
    b = b_template.subs(subs_dict)
    expr = expr_template.subs(subs_dict)

    val = sp.integrate(expr, (integration_var, a, b))
    # .doit() is crucial for evaluating definite integrals
    return (t.name or 'I', val.doit())

def exec_value(rt: Runtime, t: TargetValue, store: Dict[str, sp.Expr]) -> Tuple[str, sp.Expr]:
    # For value, the context must also include the intermediate results from the store
    value_context = {**rt.context, **store}
    expr_template = to_sympy_expr(t.expr)
    
    # Create substitution dict from symbols and stored values
    subs_dict = {sp.Symbol(k): v for k, v in value_context.items()}
    
    expr = expr_template.subs(subs_dict)
    return (t.name, expr)

def exec_limit(rt: Runtime, t: TargetLimit) -> Tuple[str, sp.Expr]:
    limit_var = sp.Symbol(t.var, real=True)
    subs_dict = {sp.Symbol(k): v for k, v in rt.context.items()}

    expr_template = to_sympy_expr(t.expr)
    expr = expr_template.subs(subs_dict)

    to_template = to_sympy_expr(t.to)
    to = to_template.subs(subs_dict)

    if str(to) == 'oo':
        to = sp.oo
    elif str(to) == '-oo':
        to = -sp.oo

    val = sp.limit(expr, limit_var, to)
    return ('limit', val.doit())

def exec_sum(rt: Runtime, t: TargetSum) -> Tuple[str, sp.Expr]:
    import uuid
    # Create a completely unique symbol for the index to guarantee no conflicts
    unique_idx_name = f"idx_{uuid.uuid4().hex[:8]}"
    idx_var = sp.Symbol(unique_idx_name, integer=True)

    # Substitute the user's index name (e.g., 'n') with our unique one
    term_template = to_sympy_expr(t.term)
    # This replaces 'n' with 'idx_...'
    term_for_sum = term_template.subs({sp.Symbol(t.idx): idx_var})

    # Now substitute the rest of the context
    subs_dict = {sp.Symbol(k): v for k, v in rt.context.items() if k != t.idx}
    term = term_for_sum.subs(subs_dict)
    
    start = to_sympy_expr(t.start).subs(subs_dict)
    end = to_sympy_expr(t.end).subs(subs_dict)
    
    if str(end) == 'oo':
        end = sp.oo

    val = sp.summation(term, (idx_var, start, end))
    return ('sum', val.doit())


def exec_solve(rt: Runtime, t: TargetSolve) -> Tuple[str, Any]:
    unknowns = [rt.context[u] for u in t.unknowns if u in rt.context]
    subs_dict = {sp.Symbol(k): v for k, v in rt.context.items()}
    
    eqs = []
    for eq_str in t.equations:
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=', 1)
            lhs_expr = to_sympy_expr(lhs).subs(subs_dict)
            rhs_expr = to_sympy_expr(rhs).subs(subs_dict)
            eqs.append(sp.Eq(lhs_expr, rhs_expr))
        else:
            # Implicit equation, expr = 0
            expr = to_sympy_expr(eq_str).subs(subs_dict)
            eqs.append(sp.Eq(expr, 0))
            
    solution = sp.solve(eqs, unknowns)
    return ('solve', solution)

def exec_ineq(rt: Runtime, t: TargetIneq) -> Tuple[str, Any]:
    subs_dict = {sp.Symbol(k): v for k, v in rt.context.items()}

    # Assuming a single variable for now, as reduce_inequalities is tricky with multiple
    all_symbols = set()
    parsed_ineqs = []
    for ineq_str in t.inequalities:
        ineq_expr = to_sympy_expr(ineq_str).subs(subs_dict)
        all_symbols.update(ineq_expr.free_symbols)
        parsed_ineqs.append(ineq_expr)

    # Convert set to list for sympy
    symbols_list = list(all_symbols)

    solution = sp.reduce_inequalities(parsed_ineqs, symbols_list)
    return ('inequalities', solution)

def parse_matrix_expr(expr_str: str, matrices: Dict[str, sp.Matrix | sp.MatrixSymbol]) -> sp.Matrix | sp.MatrixSymbol:
    """Safely parse a matrix expression like 'A*B.T' without eval."""
    # Split on '*' and handle each part
    parts = expr_str.replace(' ', '').split('*')
    result = None

    for part in parts:
        is_transpose = part.endswith('.T')
        matrix_name = part.replace('.T', '')

        if matrix_name not in matrices:
            raise ValueError(f"Matrix '{matrix_name}' not defined")

        matrix = matrices[matrix_name]
        if is_transpose:
            matrix = matrix.T

        if result is None:
            result = matrix
        else:
            result = result * matrix

    return result

def exec_matrix(rt: Runtime, t: TargetMatrixSolve, ir: MathIR) -> Tuple[str, Any]:
    # Find the matrix equation in conditions
    eq = None
    for c in ir.conditions:
        if c.type == 'matrix_equation' and c.expr:
            eq = c.expr
            break
    if not eq:
        return ('matrix', {'error': 'no_matrix_equation'})

    try:
        left, right = map(str.strip, eq.split('='))

        # Parse right side first to determine dimensions
        R = parse_matrix_expr(right, rt.matrices)

        # For now, let's handle the simple case A*X = B where A is invertible
        # This is a placeholder implementation - full matrix solving would be more complex
        if '*' in left and t.unknown in left:
            # Extract the coefficient matrix name
            parts = left.split('*')
            if len(parts) == 2 and parts[1] == t.unknown:
                coeff_name = parts[0]
                if coeff_name in rt.matrices:
                    A = rt.matrices[coeff_name]
                    # Solve A*X = R for X using matrix inverse
                    try:
                        X = A.inv() * R
                        return ('matrix', X)
                    except Exception as e:
                        return ('matrix', {'error': f'Matrix is not invertible: {e}'})
                else:
                    return ('matrix', {'error': f'Coefficient matrix {coeff_name} not found'})
            else:
                return ('matrix', {'error': 'Unsupported matrix equation format'})
        else:
            return ('matrix', {'error': 'Unsupported matrix equation format'})

    except Exception as e:
        return ('matrix', {'error': f"Failed to solve matrix equation: {e}"})


# Other exec functions would follow a similar pattern, using rt.context
# For brevity, they are omitted as the main bug is in integral/value flow

# === Main runner ===
def run_mathir(ir: MathIR) -> Dict[str, Any]:
    rt = build_runtime(ir)
    store: Dict[str, sp.Expr] = {}
    results: Dict[str, Any] = {}

    for tgt in ir.targets:
        k: Optional[str] = None
        v: Any = None

        if tgt.type == 'integral_def':
            k, v = exec_integral(rt, tgt)
            if k: store[k] = v
        elif tgt.type == 'value':
            k, v = exec_value(rt, tgt, store)
        elif tgt.type == 'limit':
            k, v = exec_limit(rt, tgt)
        elif tgt.type == 'sum':
            k, v = exec_sum(rt, tgt)
        elif tgt.type == 'solve_for':
            k, v = exec_solve(rt, tgt)
        elif tgt.type == 'inequalities':
            k, v = exec_ineq(rt, tgt)
        elif tgt.type == 'solve_for_matrix':
            k, v = exec_matrix(rt, tgt, ir)
        else:
            results[f"{tgt.type}"] = {"error": "unsupported_target"}
            continue
        
        if k is None: continue

        # Post-processing
        if isinstance(v, sp.Expr):
            if ir.output.simplify:
                v = sp.simplify(v)
            if ir.output.mode == 'decimal':
                try:
                    # Use sp.N for robust numerical evaluation
                    numerical_value = sp.N(v)
                    if ir.output.round_to is not None:
                        # round() works on SymPy numeric types
                        v = round(numerical_value, ir.output.round_to)
                    else:
                        v = numerical_value
                except (TypeError, ValueError):
                    logging.warning(f"Could not convert expression '{v}' to a decimal value.")
        
        results[k] = v

    return results

# === Main execution block ===
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_json_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading or parsing JSON file: {e}")
        sys.exit(1)

    try:
        math_ir = MathIR.model_validate(json_data)
        results = run_mathir(math_ir)
        
        serializable_results = {}
        for key, value in results.items():
            serializable_results[key] = str(value)

        print(json.dumps(serializable_results, indent=2))

    except Exception as e:
        logging.exception(f"An exception occurred during processing: {e}")
        sys.exit(1)
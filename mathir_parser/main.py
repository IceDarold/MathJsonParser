# mathir_parser_v01.py
from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel
from dataclasses import dataclass
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

def parse_any(s: str, applied_subs: dict, context: dict):
    s = s.strip()
    if '\\' in s or '|' in s or '{' in s:
        expr = parse_latex(s)
        # Replace symbols with those from context to match assumptions
        symbol_subs = {sp.Symbol(k): v for k, v in context.items() if isinstance(k, str)}
        expr = expr.subs(symbol_subs)
    else:
        # Replace ^ with ** for infix expressions to ensure exponentiation
        s = s.replace('^', '**')
        # Pass context as local_dict for parse_expr
        local_dict = {k: v for k, v in context.items() if isinstance(k, str)}
        expr = parse_expr(s, local_dict=local_dict, transformations='all')
    # Substitute symbols from context
    symbol_subs = {sp.Symbol(k): v for k, v in context.items() if isinstance(k, str)}
    return expr.xreplace(applied_subs).subs(symbol_subs)
import logging
import json
import sys
import datetime
import traceback
import multiprocessing
import queue


# === Helpers: LaTeX → SymPy ===

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
    name: Optional[str] = None

class TargetValue(BaseModel):
    type: Literal['value']
    name: str
    expr: str

class TargetMatrixInverse(BaseModel):
    type: Literal['matrix_inverse']
    matrix_name: str

class TargetMatrixDeterminant(BaseModel):
    type: Literal['matrix_determinant']
    matrix_name: str

class TargetSequenceLimitCondition(BaseModel):
    type: Literal['sequence_limit_condition']
    sequence_expr: str
    var: str
    inequality: str
    epsilon: str

class TargetFindMaximum(BaseModel):
    type: Literal['find_maximum']
    expr: str
    var: str
    domain: Optional[str] = None

class TargetAreaBetweenCurves(BaseModel):
    type: Literal['area_between_curves']
    curves: List[str]
    var: str
    limits: Optional[List[Any]] = None

class TargetDoubleIntegral(BaseModel):
    type: Literal['integral_double']
    expr: str
    vars: List[str]
    limits: List[List[Any]]

Target = Union[
    TargetIntegral, TargetLimit, TargetSum, TargetSolve, TargetIneq,
    TargetMatrixSolve, TargetProbability, TargetValue,
    TargetMatrixInverse, TargetMatrixDeterminant, TargetSequenceLimitCondition,
    TargetFindMaximum, TargetAreaBetweenCurves, TargetDoubleIntegral
]

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
    task_type: Literal['auto','integral','limit','sum','algebra','matrix','probability','geometry','optimize','solve_for'] = 'auto'
    expr_format: Literal['latex','sympy','infix'] = 'latex'
    prob_space: Optional[Dict[str, Any]] = None
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
    funcs: Dict[sp.Function, sp.Lambda]
    sequences: Dict[sp.Function, sp.Lambda]
    matrices: Dict[str, sp.Matrix]
    distributions: Dict[str, Any]
    geometry: Dict[str, Any]
    # Master context for parsing
    context: Dict[Any, Any]
    applied_subs: Dict[sp.Expr, sp.Expr]

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
        # Domain handling with assumptions
        if s.domain == 'R':
            symtab[s.name] = sp.Symbol(s.name, real=True)
        elif s.domain == 'R+':
            symtab[s.name] = sp.Symbol(s.name, real=True, positive=True)
        elif s.domain == 'Z':
            symtab[s.name] = sp.Symbol(s.name, integer=True)
        elif s.domain == 'N':
            symtab[s.name] = sp.Symbol(s.name, integer=True, nonnegative=True)
        elif s.domain == 'N+':
            symtab[s.name] = sp.Symbol(s.name, integer=True, positive=True)
        elif s.domain == 'C':
            symtab[s.name] = sp.Symbol(s.name, complex=True)
        else:
            symtab[s.name] = sp.Symbol(s.name, real=True)  # default to real
    
    # Build master context for parsing
    context: Dict[str, Any] = {**symtab}
    context['pi'] = sp.pi
    context['e'] = sp.E
    context['i'] = sp.I
    
    # Add constants from the IR
    for const_name, const_value in ir.constants.items():
        try:
            # Try to parse as a number first
            if '.' in const_value:
                context[const_name] = sp.Float(const_value)
            else:
                context[const_name] = sp.Integer(const_value)
        except:
            # If parsing fails, treat as a symbol
            context[const_name] = sp.Symbol(const_name)

    funcs: Dict[sp.Function, sp.Lambda] = {}
    applied_subs: Dict[sp.Expr, sp.Expr] = {}
    for f_def in ir.definitions.functions:
        func_symbol = sp.Function(f_def.name)
        args = [sp.Symbol(a) for a in f_def.args]
        # For function definitions, the context is only the function's own arguments
        func_arg_context = {a.name: a for a in args}
        expr = parse_any(f_def.expr, {}, func_arg_context)
        lambda_expr = sp.Lambda(tuple(args), expr)
        funcs[func_symbol] = lambda_expr
        applied_subs[func_symbol(*args)] = expr

    sequences: Dict[sp.Function, sp.Lambda] = {}
    for s_def in ir.definitions.sequences:
        seq_symbol = sp.Function(s_def.name)
        args = [sp.Symbol(a) for a in s_def.args]
        # Similar for sequences
        seq_arg_context = {a.name: a for a in args}
        expr = parse_any(s_def.expr, {}, seq_arg_context)
        lambda_expr = sp.Lambda(tuple(args), expr)
        sequences[seq_symbol] = lambda_expr
        applied_subs[seq_symbol(*args)] = expr

    matrices: Dict[str, sp.Matrix] = {}
    for m in ir.definitions.matrices:
        # Matrix elements are parsed with the main context
        mat = sp.Matrix([[parse_any(v, applied_subs, context) for v in row] for row in m.data])
        matrices[m.name] = mat

    # Add string names to context for functions and sequences
    for f_def in ir.definitions.functions:
        context[f_def.name] = sp.Function(f_def.name)
    for s_def in ir.definitions.sequences:
        context[s_def.name] = sp.Function(s_def.name)

    # Add newly defined items to context
    context.update(funcs.items())
    context.update(sequences.items())
    context.update(matrices)

    distributions: Dict[str, Any] = {}
    for d in ir.definitions.distributions:
        if d.kind == 'binomial':
            from sympy.stats import Binomial
            n = parse_any(d.params['n'], {}, context)
            p = parse_any(d.params['p'], {}, context)
            distributions[d.name] = Binomial(d.name, n, p)
        elif d.kind == 'bernoulli':
            from sympy.stats import Bernoulli
            p = parse_any(d.params['p'], {}, context)
            distributions[d.name] = Bernoulli(d.name, p)
        elif d.kind == 'geometric':
            from sympy.stats import Geometric
            p = parse_any(d.params['p'], {}, context)
            distributions[d.name] = Geometric(d.name, p)
        elif d.kind == 'poisson':
            from sympy.stats import Poisson
            lam = parse_any(d.params['lambda'], {}, context)
            distributions[d.name] = Poisson(d.name, lam)
        elif d.kind == 'hypergeom':
            from sympy.stats import Hypergeometric
            N = parse_any(d.params['N'], {}, context)
            K = parse_any(d.params['K'], {}, context)
            n = parse_any(d.params['n'], {}, context)
            distributions[d.name] = Hypergeometric(d.name, N, K, n)
        # Add more as needed

    return Runtime(symtab, funcs, sequences, matrices, distributions, {}, context, applied_subs)

# === Node executors ===

def exec_integral(rt: Runtime, t: TargetIntegral) -> Tuple[str, sp.Expr]:
    # This is the correct way to handle the integration variable
    integration_var = rt.context[t.var]

    # 1. Parse expressions first from raw LaTeX
    a = parse_any(str(t.limits[0]), rt.applied_subs, rt.context)
    b = parse_any(str(t.limits[1]), rt.applied_subs, rt.context)
    expr = parse_any(t.expr, rt.applied_subs, rt.context)

    val = sp.integrate(expr, (integration_var, a, b))
    if isinstance(val, sp.Integral):
        # Could not compute symbolically, return error to avoid hanging on numerical
        return (t.name or 'I', {'error': 'could not compute integral'})
    return (t.name or 'I', val)

def exec_value(rt: Runtime, t: TargetValue, expr_store: Dict[str, sp.Expr]) -> Tuple[str, sp.Expr]:
    # For value, the context must also include the intermediate results from the expr_store
    expr = parse_any(t.expr, rt.applied_subs, {**rt.context, **expr_store}).doit()
    return (t.name, expr)

def exec_limit(rt: Runtime, t: TargetLimit) -> Tuple[str, sp.Expr]:
    # Ensure the limit variable is in context
    if t.var not in rt.context:
        rt.context[t.var] = sp.Symbol(t.var, real=True)
    limit_var = rt.context[t.var]
    subs_dict = rt.context.copy()

    expr = parse_any(t.expr, rt.applied_subs, subs_dict)

    if t.to == "oo" or t.to == "+\\infty":
        to = sp.oo
    elif t.to == "-oo":
        to = -sp.oo
    else:
        to = parse_any(t.to, rt.applied_subs, subs_dict)

    val = sp.limit(expr, limit_var, to)
    simplified = sp.simplify(val.doit())
    return ('limit', simplified)

def exec_sum(rt: Runtime, t: TargetSum) -> Tuple[str, sp.Expr]:
    import uuid
    # Create a completely unique symbol for the index to guarantee no conflicts
    unique_idx_name = f"idx_{uuid.uuid4().hex[:8]}"
    idx_var = sp.Symbol(unique_idx_name, integer=True)

    # Parse the term using parse_any for consistent handling
    if '\\' in t.term:
        # LaTeX expressions
        local_dict = rt.context.copy()
        term_template = parse_any(t.term, rt.applied_subs, local_dict)
    else:
        # For non-LaTeX expressions, use parse_any with local_dict
        local_dict = rt.context.copy()
        local_dict[t.idx] = rt.context[t.idx]
        term_template = parse_any(t.term, rt.applied_subs, local_dict)

    # This replaces the index variable (e.g., 'n') with the unique index variable
    term_for_sum = term_template.subs({rt.context[t.idx]: idx_var})

    # Now substitute the rest of the context
    subs_dict = {k: v for k, v in rt.context.items() if k != t.idx}
    term = term_for_sum.subs(subs_dict).doit()

    start = parse_any(t.start, rt.applied_subs, rt.context)
    if str(t.end) == 'oo':
        end = sp.oo
    else:
        end = parse_any(t.end, rt.applied_subs, rt.context)

    val = sp.summation(term, (idx_var, start, end))
    return ('sum', val.doit())


def exec_solve(rt: Runtime, t: TargetSolve) -> Tuple[str, Any]:
    unknowns = [rt.context[u] for u in t.unknowns if u in rt.context]
    subs_dict = rt.context.copy()

    eqs = []
    for eq_str in t.equations:
        if '=' in eq_str:
            lhs, rhs = eq_str.split('=', 1)
            lhs_expr = parse_any(lhs, rt.applied_subs, subs_dict)
            rhs_expr = parse_any(rhs, rt.applied_subs, subs_dict)
            eqs.append(sp.Eq(lhs_expr, rhs_expr))
        else:
            # Implicit equation, expr = 0
            expr = parse_any(eq_str, rt.applied_subs, subs_dict)
            eqs.append(sp.Eq(expr, 0))
    if not eqs:
        return ('solve', {'error': 'no equations found'})

    try:
        solution = sp.solve(eqs, unknowns)
        if isinstance(solution, dict):
            solution = [solution]
        elif isinstance(solution, list):
            if solution and isinstance(solution[0], dict):
                # Already list of dicts
                pass
            else:
                # List of tuples (for multiple unknowns) or values (for single)
                solution = [dict(zip(unknowns, val if isinstance(val, tuple) else (val,))) for val in solution]
    except NotImplementedError:
        solution = {'error': 'could not solve'}
    return ('solve', solution)

def exec_ineq(rt: Runtime, t: TargetIneq) -> Tuple[str, Any]:
    subs_dict = rt.context.copy()

    # Collect all free symbols
    all_symbols = set()
    parsed_ineqs = []
    for ineq_str in t.inequalities:
        ineq_expr = parse_any(ineq_str, rt.applied_subs, subs_dict)
        all_symbols.update(ineq_expr.free_symbols)
        parsed_ineqs.append(ineq_expr)

    # Order symbols stably: first those in IR symbols, then others
    ir_symbols = {sp.Symbol(s.name) for s in ir.symbols}
    ir_symbols_list = [s for s in ir_symbols if s in all_symbols]
    other_symbols = [s for s in all_symbols if s not in ir_symbols]
    symbols_list = ir_symbols_list + other_symbols

    solution = sp.reduce_inequalities(parsed_ineqs, symbols_list)
    return ('inequalities', solution)

def parse_matrix_expr(expr_str: str, matrices: Dict[str, sp.Matrix | sp.MatrixSymbol]) -> sp.Matrix | sp.MatrixSymbol:
    """Safely parse a matrix expression like 'A*(B + C.T)' without eval."""
    expr_str = expr_str.replace(' ', '')
    tokens = tokenize_matrix_expr(expr_str)
    result, _ = parse_expr_matrix(tokens, 0, matrices)
    return result

def tokenize_matrix_expr(s: str) -> list:
    tokens = []
    i = 0
    while i < len(s):
        if s[i] in '+-*().':
            tokens.append(s[i])
            i += 1
        elif s[i].isalpha():
            start = i
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                i += 1
            tokens.append(s[start:i])
        elif s[i] == '.' and i + 1 < len(s) and s[i+1] == 'T':
            tokens.append('.T')
            i += 2
        else:
            raise ValueError(f"Invalid character '{s[i]}' in matrix expression")
    return tokens

def parse_expr_matrix(tokens: list, pos: int, matrices: dict) -> (sp.Matrix | sp.MatrixSymbol, int):
    left, pos = parse_term_matrix(tokens, pos, matrices)
    while pos < len(tokens) and tokens[pos] in '+-':
        op = tokens[pos]
        pos += 1
        right, pos = parse_term_matrix(tokens, pos, matrices)
        if op == '+':
            left = left + right
        elif op == '-':
            left = left - right
    return left, pos

def parse_term_matrix(tokens: list, pos: int, matrices: dict) -> (sp.Matrix | sp.MatrixSymbol, int):
    left, pos = parse_factor_matrix(tokens, pos, matrices)
    while pos < len(tokens) and tokens[pos] == '*':
        pos += 1
        right, pos = parse_factor_matrix(tokens, pos, matrices)
        left = left * right
    return left, pos

def parse_factor_matrix(tokens: list, pos: int, matrices: dict) -> (sp.Matrix | sp.MatrixSymbol, int):
    if tokens[pos] == '(':
        pos += 1
        expr, pos = parse_expr_matrix(tokens, pos, matrices)
        if pos >= len(tokens) or tokens[pos] != ')':
            raise ValueError("Mismatched parentheses in matrix expression")
        pos += 1
    else:
        matrix_name = tokens[pos]
        pos += 1
        if matrix_name not in matrices:
            raise ValueError(f"Matrix '{matrix_name}' not defined")
        expr = matrices[matrix_name]
    if pos < len(tokens) and tokens[pos] == '.T':
        expr = expr.T
        pos += 1
    return expr, pos

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

        # Advanced parsing for matrix equations
        # Define the unknown as a matrix symbol
        R_shape = R.shape
        unknown_matrix = sp.MatrixSymbol(t.unknown, R_shape[0], R_shape[1])
        
        # Build up a dictionary of all matrices (both defined and the unknown)
        all_matrices = rt.matrices.copy()
        all_matrices[t.unknown] = unknown_matrix

        # Parse both sides of the equation
        L_expr = parse_matrix_expr(left, all_matrices)

        # Create the equation and solve for the unknown matrix symbol
        eq = sp.Eq(L_expr, R)
        solution = sp.solve(eq, unknown_matrix)
        
        return ('matrix', solution)

    except Exception as e:
        # It's useful to log the actual exception for debugging
        logging.error(f"Matrix solver failed: {e}")
        return ('matrix', {'error': f"Failed to solve matrix equation: {e}"})


def exec_matrix_inverse(rt: Runtime, t: TargetMatrixInverse) -> Tuple[str, Any]:
    """Computes the inverse of a given matrix."""
    if t.matrix_name not in rt.matrices:
        return ('inverse', {'error': f"Matrix '{t.matrix_name}' not defined"})
    
    matrix = rt.matrices[t.matrix_name]
    try:
        inverse = matrix.inv()
        return ('inverse', inverse)
    except Exception as e:
        return ('inverse', {'error': f'Failed to compute inverse: {e}'})

def exec_matrix_determinant(rt: Runtime, t: TargetMatrixDeterminant) -> Tuple[str, Any]:
    """Computes the determinant of a given matrix."""
    if t.matrix_name not in rt.matrices:
        return ('determinant', {'error': f"Matrix '{t.matrix_name}' not defined"})
        
    matrix = rt.matrices[t.matrix_name]
    try:
        det = matrix.det()
        return ('determinant', det)
    except Exception as e:
        return ('determinant', {'error': f'Failed to compute determinant: {e}'})

def exec_sequence_limit_condition(rt: Runtime, t: TargetSequenceLimitCondition) -> Tuple[str, Any]:
    """Solves an inequality like |a_n| < epsilon for n."""
    try:
        var = rt.context[t.var]
        epsilon = parse_any(t.epsilon, rt.applied_subs, rt.context)
        seq_expr = parse_any(t.sequence_expr, rt.applied_subs, rt.context)

        # Assuming the inequality is of the form |expr| < epsilon
        inequality = sp.Abs(seq_expr) < epsilon
        
        # Solve for the variable
        solution_set = sp.solve_univariate_inequality(inequality, var, relational=False)
        
        # We are looking for N such that for all n > N, the condition holds.
        # This typically means we need the upper bound of the solution for 1/n or similar.
        # A more general approach is needed. Let's assume for a_n = 1/n, solve |1/n| < eps -> n > 1/eps
        if solution_set.is_Interval and solution_set.start.is_finite:
             # For inequalities like n > some_value
            N = sp.floor(solution_set.start) + 1
            return ('sequence_limit_N', N)
        else:
            # Fallback for more complex cases
            return ('sequence_limit_solution', solution_set)

    except Exception as e:
        return ('sequence_limit', {'error': f'Failed to solve sequence inequality: {e}'})

def exec_find_maximum(rt: Runtime, t: TargetFindMaximum) -> Tuple[str, Any]:
    """Finds the maximum of an expression by finding critical points."""
    try:
        var = rt.context[t.var]
        expr = parse_any(t.expr, rt.applied_subs, rt.context)
        
        # Find the derivative
        f_diff = sp.diff(expr, var)
        
        # Find critical points by solving f'(x) = 0
        critical_points = sp.solve(f_diff, var)
        
        # For now, we return the critical points. A full implementation would check the second derivative.
        return ('critical_points', critical_points)
        
    except Exception as e:
        return ('maximum', {'error': f'Failed to find maximum: {e}'})

def exec_area_between_curves(rt: Runtime, t: TargetAreaBetweenCurves) -> Tuple[str, Any]:
    """Calculates the area between two curves."""
    try:
        var = rt.context[t.var]
        
        if len(t.curves) != 2:
            return ('area', {'error': 'area_between_curves requires exactly two curves.'})

        curve1_expr = parse_any(t.curves[0], rt.applied_subs, rt.context)
        curve2_expr = parse_any(t.curves[1], rt.applied_subs, rt.context)
        
        limits = t.limits
        if not limits:
            # If no limits are given, find intersection points
            intersections = sp.solve(sp.Eq(curve1_expr, curve2_expr), var)
            intersections = [i for i in intersections if i.is_real]
            if len(intersections) < 2:
                return ('area', {'error': 'Could not find at least two real intersection points.'})
            limits = [min(intersections), max(intersections)]

        # Integrate the absolute difference between the curves
        integrand = sp.Abs(curve1_expr - curve2_expr)
        area = sp.integrate(integrand, (var, limits[0], limits[1]))
        
        return ('area', area.doit())

    except Exception as e:
        return ('area', {'error': f'Failed to calculate area: {e}'})

def exec_probability(rt: Runtime, t: TargetProbability) -> Tuple[str, Any]:
    try:
        # Special case for !D1 & !D2
        if '!' in t.event_expr and '&' in t.event_expr and '|' not in t.event_expr:
            parts = [p.strip().lstrip('!') for p in t.event_expr.split('&')]
            if len(parts) == 2 and all(p in rt.distributions for p in parts):
                dists = [rt.distributions[p] for p in parts]
                if all(hasattr(d.pspace.distribution, 'p') for d in dists):
                    prob = sp.prod([sp.Rational(str(1 - d.pspace.distribution.p)) for d in dists])
                    name = t.name or 'probability'
                    return (name, prob)
        # General case
        event_str = t.event_expr.replace('!', '~').replace('&', '&').replace('|', '|')
        # For probability events, use sympy's parse_expr without transformations to avoid misinterpretation
        local_dict = {k: v for k, v in rt.context.items() if isinstance(k, str)}
        event = parse_expr(event_str, local_dict=local_dict)  # Default transformations
        from sympy.stats import P
        prob = P(event).doit()
        name = t.name or 'probability'
        return (name, prob)
    except Exception as e:
        name = t.name or 'probability'
        return (name, {'error': str(e)})

def exec_double_integral(rt: Runtime, t: TargetDoubleIntegral) -> Tuple[str, Any]:
    """Computes a double integral."""
    try:
        if len(t.vars) != 2 or len(t.limits) != 2:
            return ('double_integral', {'error': 'Double integral requires two variables and two sets of limits.'})

        var1 = sp.Symbol(t.vars[0])
        var2 = sp.Symbol(t.vars[1])

        expr = parse_any(t.expr, rt.applied_subs, rt.context)

        # Parse limits
        lim1_start = parse_any(str(t.limits[0][0]), rt.applied_subs, rt.context)
        lim1_end = parse_any(str(t.limits[0][1]), rt.applied_subs, rt.context)
        lim2_start = parse_any(str(t.limits[1][0]), rt.applied_subs, rt.context)
        lim2_end = parse_any(str(t.limits[1][1]), rt.applied_subs, rt.context)

        # Integrate step-by-step
        inner_integral = sp.integrate(expr, (var2, lim2_start, lim2_end))
        outer_integral = sp.integrate(inner_integral, (var1, lim1_start, lim1_end))
        
        return ('double_integral', outer_integral.doit())

    except Exception as e:
        return ('double_integral', {'error': f'Failed to compute double integral: {e}'})


# Other exec functions would follow a similar pattern, using rt.context
# For brevity, they are omitted as the main bug is in integral/value flow

def run_mathir_worker(ir: MathIR, result_queue: multiprocessing.Queue):
    """Worker function to run mathir in a separate process."""
    try:
        rt = build_runtime(ir)
        expr_store: Dict[str, sp.Expr] = {}
        results: Dict[str, Any] = {}

        for tgt in ir.targets:
            k: Optional[str] = None
            v: Any = None

            if tgt.type == 'integral_def':
                k, v = exec_integral(rt, tgt)
                if k: expr_store[k] = v
            elif tgt.type == 'value':
                k, v = exec_value(rt, tgt, expr_store)
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
            elif tgt.type == 'matrix_inverse':
                k, v = exec_matrix_inverse(rt, tgt)
            elif tgt.type == 'matrix_determinant':
                k, v = exec_matrix_determinant(rt, tgt)
            elif tgt.type == 'sequence_limit_condition':
                k, v = exec_sequence_limit_condition(rt, tgt)
            elif tgt.type == 'find_maximum':
                k, v = exec_find_maximum(rt, tgt)
            elif tgt.type == 'probability':
                k, v = exec_probability(rt, tgt)
            elif tgt.type == 'area_between_curves':
                k, v = exec_area_between_curves(rt, tgt)
            elif tgt.type == 'integral_double':
                k, v = exec_double_integral(rt, tgt)
            else:
                results[f"{tgt.type}"] = {"error": "unsupported_target"}
                continue

            if k is None: continue

            # Store expr for later use
            if isinstance(v, sp.Expr):
                expr_store[k] = v

            # Post-processing for output
            output_v = v
            if isinstance(output_v, sp.Expr):
                if ir.output.simplify:
                    output_v = sp.simplify(output_v)
                if ir.output.mode == 'decimal':
                    # Only attempt numerical evaluation if the expression is a number or can be evaluated to one
                    if output_v.is_Number or not output_v.free_symbols: # Check if it's a number or has no free symbols
                        try:
                            # Use sp.N for robust numerical evaluation
                            numerical_value = sp.N(output_v)
                            round_to = ir.output.round_to if ir.output.round_to is not None else 3
                            # Format as string with fixed decimal places
                            output_v = f"{float(numerical_value):.{round_to}f}"
                        except (TypeError, ValueError):
                            logging.warning(f"Could not convert expression '{output_v}' to a decimal value.")
                    else:
                        logging.info(f"Skipping decimal conversion for symbolic expression: '{output_v}'")

            results[k] = output_v

        # Debug logging
        logging.debug(f"Raw results: {results}")

        result_queue.put(results)
    except Exception as e:
        result_queue.put({'error': str(e)})

# === Main runner ===
def run_mathir(ir: MathIR) -> Dict[str, Any]:
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_mathir_worker, args=(ir, result_queue))
    process.start()
    process.join(timeout=3)  # 3 seconds timeout

    if process.is_alive():
        process.terminate()
        process.join()
        return {'error': 'timeout: computation took longer than 3 seconds'}

    try:
        results = result_queue.get_nowait()
        return results
    except queue.Empty:
        return {'error': 'no result from worker'}

# === Logging functions ===
def log_success(task_id, input_data, final_output, actions):
    """Log successful execution to success_log.json"""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_id": task_id,
        "input_data": input_data,
        "final_output": final_output,
        "actions": actions
    }
    log_to_file("success_log.json", entry)

def log_error(task_id, input_data, error_type, traceback_str, context, actions):
    """Log error to error_log.json and console"""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task_id": task_id,
        "input_data": input_data,
        "error_type": error_type,
        "traceback": traceback_str,
        "context": context,
        "actions": actions
    }
    log_to_file("error_log.json", entry)
    logging.error(f"Error processing task {task_id}: {error_type} - {context}")

def log_to_file(filename, entry):
    """Append entry to JSONL file (one JSON per line)"""
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

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
        # Check if json_data is a list of tasks or a single MathIR object
        if isinstance(json_data, list):
            # Process each task in the list
            all_results = []
            for item in json_data:
                task_id = item.get('task_id', 'unknown')
                input_data = item  # Full item as input data
                actions = []

                # Use parsed_json if available, otherwise parse response
                math_ir_data = item.get('parsed_json')
                if math_ir_data is None and 'response' in item:
                    actions.append("Parsed response JSON string")
                    try:
                        math_ir_data = json.loads(item['response'])
                    except json.JSONDecodeError as e:
                        log_error(task_id, input_data, "JSONDecodeError", str(e), "Failed to parse response string", actions)
                        all_results.append({
                            'task_id': task_id,
                            'error': str(e)
                        })
                        continue

                if math_ir_data:
                    actions.append("Validated MathIR data")
                    try:
                        math_ir = MathIR.model_validate(math_ir_data)
                        actions.append("Built runtime context")
                        results = run_mathir(math_ir)
                        actions.append("Executed all targets")

                        serializable_results = {}
                        for key, value in results.items():
                            serializable_results[key] = str(value)

                        log_success(task_id, input_data, serializable_results, actions)
                        all_results.append({
                            'task_id': task_id,
                            'results': serializable_results
                        })
                    except Exception as e:
                        tb = traceback.format_exc()
                        log_error(task_id, input_data, type(e).__name__, tb, str(e), actions)
                        all_results.append({
                            'task_id': task_id,
                            'error': str(e)
                        })
            print(json.dumps(all_results, indent=2))
        else:
            # Single MathIR object
            task_id = 'single_task'
            input_data = json_data
            actions = []

            actions.append("Validated MathIR data")
            math_ir = MathIR.model_validate(json_data)
            actions.append("Built runtime context")
            results = run_mathir(math_ir)
            actions.append("Executed all targets")

            serializable_results = {}
            for key, value in results.items():
                serializable_results[key] = str(value)

            log_success(task_id, input_data, serializable_results, actions)
            print(json.dumps(serializable_results, indent=2))

    except Exception as e:
        tb = traceback.format_exc()
        log_error('global', json_data, type(e).__name__, tb, str(e), ["Global processing"])
        logging.exception(f"An exception occurred during processing: {e}")
        sys.exit(1)
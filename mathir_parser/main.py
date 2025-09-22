# mathir_parser_v01.py
from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel
from dataclasses import dataclass
import sympy as sp
import logging
import json
import sys
import re # Import the regex module
import datetime
import os
import traceback


# === Helpers: LaTeX → SymPy ===
def to_sympy_expr(s: str) -> sp.Expr:
    """Parse a LaTeX string to a SymPy expression with proper LaTeX command handling."""
    
    # Step 1: Handle LaTeX commands first (before any character processing)
    # This prevents the character-by-character processing from destroying LaTeX commands
    
    # Handle LaTeX commands with a more robust approach that handles nested braces
    def extract_braced_content(text, start_pos):
        """Extract content from braces starting at start_pos, handling nesting"""
        if start_pos >= len(text) or text[start_pos] != '{':
            return "", start_pos
        
        brace_count = 0
        i = start_pos
        while i < len(text):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_pos + 1:i], i + 1
            i += 1
        return text[start_pos + 1:], len(text)
    
    # Handle fractions: \frac{a}{b} -> (a)/(b)
    def replace_fractions(text):
        result = ""
        i = 0
        while i < len(text):
            if text[i:i+5] == r'\frac' and i + 5 < len(text) and text[i+5] == '{':
                # Found \frac{
                numerator, next_pos = extract_braced_content(text, i + 5)
                if next_pos < len(text) and text[next_pos] == '{':
                    denominator, final_pos = extract_braced_content(text, next_pos)
                    result += f'({numerator})/({denominator})'
                    i = final_pos
                else:
                    result += text[i]
                    i += 1
            else:
                result += text[i]
                i += 1
        return result
    
    # Handle square roots: \sqrt{expr} -> sqrt(expr)
    def replace_square_roots(text):
        result = ""
        i = 0
        while i < len(text):
            if text[i:i+5] == r'\sqrt' and i + 5 < len(text) and text[i+5] == '{':
                # Found \sqrt{
                content, next_pos = extract_braced_content(text, i + 5)
                result += f'sqrt({content})'
                i = next_pos
            else:
                result += text[i]
                i += 1
        return result
    
    # Apply replacements multiple times to handle nested cases
    prev_s = ""
    while prev_s != s:
        prev_s = s
        s = replace_fractions(s)
        s = replace_square_roots(s)
    
    # Handle absolute values: |expr| -> Abs(expr)
    # First handle nested absolute values by counting | characters
    abs_count = s.count('|')
    if abs_count >= 2 and abs_count % 2 == 0:
        # Simple case: replace pairs of | with Abs()
        parts = s.split('|')
        if len(parts) >= 3:  # At least |something|
            result_parts = []
            i = 0
            while i < len(parts):
                if i % 2 == 0:  # Outside absolute value
                    result_parts.append(parts[i])
                else:  # Inside absolute value
                    result_parts.append(f'Abs({parts[i]})')
                    i += 1  # Skip the closing |
                i += 1
            s = ''.join(result_parts)
    
    # Handle Greek letters and constants
    greek_letters = {
        r'\\pi': 'pi',
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\gamma': 'gamma',
        r'\\delta': 'delta',
        r'\\epsilon': 'epsilon',
        r'\\zeta': 'zeta',
        r'\\eta': 'eta',
        r'\\theta': 'theta',
        r'\\iota': 'iota',
        r'\\kappa': 'kappa',
        r'\\lambda': 'lambda',
        r'\\mu': 'mu',
        r'\\nu': 'nu',
        r'\\xi': 'xi',
        r'\\omicron': 'omicron',
        r'\\rho': 'rho',
        r'\\sigma': 'sigma',
        r'\\tau': 'tau',
        r'\\upsilon': 'upsilon',
        r'\\phi': 'phi',
        r'\\chi': 'chi',
        r'\\psi': 'psi',
        r'\\omega': 'omega',
        r'\\infty': 'oo'
    }
    
    for latex_cmd, sympy_equiv in greek_letters.items():
        s = re.sub(latex_cmd, sympy_equiv, s)
    
    # Handle trigonometric and other functions
    function_replacements = {
        r'\\cos': 'cos',
        r'\\sin': 'sin',
        r'\\tan': 'tan',
        r'\\tg': 'tan',  # Russian notation
        r'\\log': 'log',
        r'\\ln': 'ln',
        r'\\exp': 'exp'
    }
    
    for latex_func, sympy_func in function_replacements.items():
        s = re.sub(latex_func, sympy_func, s)
    
    # Handle other LaTeX constructs
    s = s.replace('\\cdot', '*')
    s = s.replace('^\\circ', '*(pi/180)')  # Degrees to radians
    
    # Handle function spacing: \sin 2x -> \sin(2x), \cos 3y -> \cos(3y), etc.
    # This needs to be done before other processing
    function_spacing_patterns = [
        (r'\\sin\s+([^()\s]+)', r'\\sin(\1)'),
        (r'\\cos\s+([^()\s]+)', r'\\cos(\1)'),
        (r'\\tan\s+([^()\s]+)', r'\\tan(\1)'),
        (r'\\log\s+([^()\s]+)', r'\\log(\1)'),
        (r'\\ln\s+([^()\s]+)', r'\\ln(\1)'),
        (r'\\exp\s+([^()\s]+)', r'\\exp(\1)'),
        (r'\\sqrt\s+([^()\s]+)', r'\\sqrt(\1)'),
    ]
    
    for pattern, replacement in function_spacing_patterns:
        s = re.sub(pattern, replacement, s)
    
    # Also handle cases where function is followed directly by a number/variable without space
    # sin 2x -> sin(2x), but also sin2x -> sin(2x)
    direct_function_patterns = [
        (r'sin\s*([0-9]+[a-zA-Z]*)', r'sin(\1)'),
        (r'cos\s*([0-9]+[a-zA-Z]*)', r'cos(\1)'),
        (r'tan\s*([0-9]+[a-zA-Z]*)', r'tan(\1)'),
        (r'log\s*([0-9]+[a-zA-Z]*)', r'log(\1)'),
        (r'ln\s*([0-9]+[a-zA-Z]*)', r'ln(\1)'),
        (r'exp\s*([0-9]+[a-zA-Z]*)', r'exp(\1)'),
    ]
    
    for pattern, replacement in direct_function_patterns:
        s = re.sub(pattern, replacement, s)
    
    # Handle exponentiation with braces: e^{2x} -> e**(2x)
    # This needs to be done before implicit multiplication to avoid conflicts
    def replace_exponentiation(text):
        result = ""
        i = 0
        while i < len(text):
            if text[i] == '^' and i + 1 < len(text) and text[i+1] == '{':
                # Found ^{
                content, next_pos = extract_braced_content(text, i + 1)
                result += f'**({content})'
                i = next_pos
            else:
                result += text[i]
                i += 1
        return result
    
    s = replace_exponentiation(s)
    
    # Step 2: Handle implicit multiplication (only after LaTeX commands are processed)
    # Use a more sophisticated approach that preserves function names and multi-letter variables
    
    # Define known function names and multi-letter variables that should not be split
    known_names = [
        'cos', 'sin', 'tan', 'log', 'ln', 'exp', 'sqrt', 'Abs', 'floor',
        'pi', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
        'mu', 'sigma', 'phi', 'omega', 'oo'
    ]
    
    # Use regex to handle implicit multiplication more intelligently
    # This approach preserves known function/variable names
    
    # First, protect known names by temporarily replacing them with placeholders
    protected_names = {}
    placeholder_counter = 0
    
    for name in sorted(known_names, key=len, reverse=True):  # Sort by length to handle longer names first
        if name in s:
            placeholder = f"__PROTECTED_{placeholder_counter}__"
            protected_names[placeholder] = name
            s = s.replace(name, placeholder)
            placeholder_counter += 1
    
    # Now apply implicit multiplication rules on the protected string
    processed_s = []
    i = 0
    while i < len(s):
        processed_s.append(s[i])
        if i + 1 < len(s):
            current_char = s[i]
            next_char = s[i+1]

            # Case 1: Digit followed by letter or opening parenthesis (e.g., 2x, 2(x+y))
            if current_char.isdigit() and (next_char.isalpha() or next_char == '('):
                processed_s.append('*')
            # Case 2: Closing parenthesis followed by letter, digit, or opening parenthesis (e.g., (x+y)z, (x+y)2, (x+1)(x+2))
            elif current_char == ')' and (next_char.isalpha() or next_char.isdigit() or next_char == '('):
                processed_s.append('*')
            # Case 3: Letter followed by opening parenthesis - check if it's a protected function
            elif current_char.isalpha() and next_char == '(':
                # Look ahead to see if this is part of a protected function name
                is_protected_function = False
                for placeholder in protected_names:
                    if placeholder in s[max(0, i-20):i+20] and placeholder.endswith('__'):
                        # This is likely part of a protected function, don't add multiplication
                        is_protected_function = True
                        break
                if not is_protected_function:
                    processed_s.append('*')
            # Case 3b: Function name followed by space and then digit/letter (e.g., "sin 2x")
            elif current_char == ' ' and i > 0:
                # Check if we just finished a function name
                func_end_pos = i
                for func_name in ['cos', 'sin', 'tan', 'log', 'ln', 'exp', 'sqrt']:
                    if func_end_pos >= len(func_name) and s[func_end_pos - len(func_name):func_end_pos] == func_name:
                        # This is a function followed by space, replace space with opening parenthesis
                        # and find the end of the argument to add closing parenthesis
                        if next_char.isalnum():
                            # Replace the space with '('
                            processed_s[-1] = '('  # Replace the space we just added
                            # We need to find where to put the closing parenthesis
                            # For now, let's add it at the end of the current "word"
                            j = i + 1
                            while j < len(s) and (s[j].isalnum() or s[j] in '*+-^{}()'):
                                j += 1
                            # Insert closing paren at position j
                            s = s[:j] + ')' + s[j:]
                        break
            # Case 4: Letter followed by letter (e.g., xy -> x*y)
            elif current_char.isalpha() and next_char.isalpha():
                # Only add multiplication if we're not inside a protected name
                in_protected_name = False
                for placeholder in protected_names:
                    if placeholder in s[max(0, i-20):i+20]:
                        in_protected_name = True
                        break
                if not in_protected_name:
                    processed_s.append('*')
        i += 1
    
    s = "".join(processed_s)
    
    # Restore protected names
    for placeholder, original_name in protected_names.items():
        s = s.replace(placeholder, original_name)

    # Step 3: Use sympify with a comprehensive local dictionary
    local_dict = {
        'pi': sp.pi,
        'e': sp.E,
        'i': sp.I,
        'oo': sp.oo,
        'cos': sp.cos,
        'sin': sp.sin,
        'tan': sp.tan,
        'tg': sp.tan,
        'log': sp.log,
        'ln': sp.ln,
        'exp': sp.exp,
        'floor': sp.floor,
        'sqrt': sp.sqrt,
        'Abs': sp.Abs,
        # Greek letters
        'alpha': sp.Symbol('alpha'),
        'beta': sp.Symbol('beta'),
        'gamma': sp.Symbol('gamma'),
        'delta': sp.Symbol('delta'),
        'epsilon': sp.Symbol('epsilon'),
        'theta': sp.Symbol('theta'),
        'lambda': sp.Symbol('lambda'),
        'mu': sp.Symbol('mu'),
        'sigma': sp.Symbol('sigma'),
        'phi': sp.Symbol('phi'),
        'omega': sp.Symbol('omega')
    }
    
    # Add any undefined symbols as SymPy symbols
    # This ensures that variables like 'N', 'x', 'n', etc. are treated as symbols
    # Find all alphabetic identifiers in the string
    identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', s)
    for identifier in identifiers:
        if identifier not in local_dict and not identifier.isdigit():
            local_dict[identifier] = sp.Symbol(identifier)
    
    return sp.sympify(s, locals=local_dict)

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
    kind: Literal['bernoulli','binomial','geometric','poisson','hypergeom','uniform']
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
    task_type: Literal['auto','integral','limit','sum','algebra','matrix','probability','geometry','optimize'] = 'auto'
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
        if s.domain == 'R':
            symtab[s.name] = sp.Symbol(s.name, real=True)
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
    for f_def in ir.definitions.functions:
        func_symbol = sp.Function(f_def.name)
        args = [sp.Symbol(a) for a in f_def.args]
        # For function definitions, the context is only the function's own arguments
        func_arg_context = {a.name: a for a in args}
        expr_template = to_sympy_expr(f_def.expr)
        expr = expr_template.subs(func_arg_context)
        funcs[func_symbol] = sp.Lambda(tuple(args), expr)

    sequences: Dict[sp.Function, sp.Lambda] = {}
    for s_def in ir.definitions.sequences:
        seq_symbol = sp.Function(s_def.name)
        args = [sp.Symbol(a) for a in s_def.args]
        # Similar for sequences
        seq_arg_context = {a.name: a for a in args}
        expr_template = to_sympy_expr(s_def.expr)
        expr = expr_template.subs(seq_arg_context)
        sequences[seq_symbol] = sp.Lambda(tuple(args), expr)
        
    matrices: Dict[str, sp.Matrix] = {}
    for m in ir.definitions.matrices:
        # Matrix elements are parsed with the main context
        mat = sp.Matrix([[to_sympy_expr(v).subs(context) for v in row] for row in m.data])
        matrices[m.name] = mat

    # Add newly defined items to context
    context.update(funcs.items())
    context.update(sequences.items())
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
    subs_dict = rt.context.copy()

    # 3. Substitute the context into the templates
    a = a_template.subs(subs_dict)
    b = b_template.subs(subs_dict)
    expr = expr_template.subs(subs_dict)

    val = sp.integrate(expr, (integration_var, a, b))
    # .doit() is crucial for evaluating definite integrals
    return (t.name or 'I', val.doit())

def exec_value(rt: Runtime, t: TargetValue, store: Dict[str, sp.Expr]) -> Tuple[str, sp.Expr]:
    # For value, the context must also include the intermediate results from the store
    if '\\' in t.expr or '|' in t.expr:
        # Use to_sympy_expr for LaTeX expressions or expressions with absolute values
        expr_template = to_sympy_expr(t.expr)
        subs_dict = {k: v for k, v in {**rt.context, **store}.items()}
        expr = expr_template.subs(subs_dict).doit()
    else:
        local_dict = {}
        for k, v in rt.context.items():
            if isinstance(k, str):
                local_dict[k] = v
            elif hasattr(k, 'name'):
                local_dict[k.name] = v
        for k in store:
            if k not in local_dict:
                local_dict[k] = sp.Symbol(k)
        expr_template = sp.parse_expr(t.expr, local_dict=local_dict)
        expr = expr_template.subs(store).doit()
    return (t.name, expr)

def exec_limit(rt: Runtime, t: TargetLimit) -> Tuple[str, sp.Expr]:
    limit_var = sp.Symbol(t.var, real=True)
    subs_dict = rt.context.copy()

    expr_template = to_sympy_expr(t.expr)
    expr = expr_template.subs(subs_dict)

    if t.to == "oo":
        to = sp.oo
    elif t.to == "-oo":
        to = -sp.oo
    else:
        to_template = to_sympy_expr(t.to)
        to = to_template.subs(subs_dict)

    val = sp.limit(expr, limit_var, to)
    simplified = sp.simplify(val.doit())
    return ('limit', simplified)

def exec_sum(rt: Runtime, t: TargetSum) -> Tuple[str, sp.Expr]:
    import uuid
    # Create a completely unique symbol for the index to guarantee no conflicts
    unique_idx_name = f"idx_{uuid.uuid4().hex[:8]}"
    idx_var = sp.Symbol(unique_idx_name, integer=True)

    # Parse the term using to_sympy_expr for proper LaTeX handling
    if '\\' in t.term:
        # Use to_sympy_expr for LaTeX expressions
        term_template = to_sympy_expr(t.term)
    else:
        # For non-LaTeX expressions, use parse_expr with proper local_dict
        local_dict = {t.idx: rt.context[t.idx]}
        for k, v in rt.context.items():
            if isinstance(k, str):
                local_dict[k] = v
            elif hasattr(k, 'name'):
                local_dict[k.name] = v
        term_template = sp.parse_expr(t.term, local_dict=local_dict)
    
    # This replaces the index variable (e.g., 'n') with the unique index variable
    term_for_sum = term_template.subs({rt.context[t.idx]: idx_var})

    # Now substitute the rest of the context
    subs_dict = {k: v for k, v in rt.context.items() if k != t.idx}
    term = term_for_sum.subs(subs_dict).doit()

    start = sp.Integer(t.start)
    if str(t.end) == 'oo':
        end = sp.oo
    else:
        end = sp.Integer(t.end)

    val = sp.summation(term, (idx_var, start, end))
    return ('sum', val.doit())


def exec_solve(rt: Runtime, t: TargetSolve) -> Tuple[str, Any]:
    unknowns = [rt.context[u] for u in t.unknowns if u in rt.context]
    subs_dict = rt.context.copy()

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
    if not eqs:
        return ('solve', {'error': 'no equations found'})

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
    return ('solve', solution)

def exec_ineq(rt: Runtime, t: TargetIneq) -> Tuple[str, Any]:
    subs_dict = rt.context.copy()

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
        epsilon = to_sympy_expr(t.epsilon).subs(rt.context)
        seq_expr = to_sympy_expr(t.sequence_expr).subs(rt.context)

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
        expr = to_sympy_expr(t.expr).subs(rt.context)
        
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

        curve1_expr = to_sympy_expr(t.curves[0]).subs(rt.context)
        curve2_expr = to_sympy_expr(t.curves[1]).subs(rt.context)
        
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

def exec_double_integral(rt: Runtime, t: TargetDoubleIntegral) -> Tuple[str, Any]:
    """Computes a double integral."""
    try:
        if len(t.vars) != 2 or len(t.limits) != 2:
            return ('double_integral', {'error': 'Double integral requires two variables and two sets of limits.'})

        var1 = sp.Symbol(t.vars[0])
        var2 = sp.Symbol(t.vars[1])
        
        expr = to_sympy_expr(t.expr).subs(rt.context)
        
        # Parse limits
        lim1_start = to_sympy_expr(str(t.limits[0][0])).subs(rt.context)
        lim1_end = to_sympy_expr(str(t.limits[0][1])).subs(rt.context)
        lim2_start = to_sympy_expr(str(t.limits[1][0])).subs(rt.context)
        lim2_end = to_sympy_expr(str(t.limits[1][1])).subs(rt.context)

        # Integrate step-by-step
        inner_integral = sp.integrate(expr, (var2, lim2_start, lim2_end))
        outer_integral = sp.integrate(inner_integral, (var1, lim1_start, lim1_end))
        
        return ('double_integral', outer_integral.doit())

    except Exception as e:
        return ('double_integral', {'error': f'Failed to compute double integral: {e}'})


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
        elif tgt.type == 'matrix_inverse':
            k, v = exec_matrix_inverse(rt, tgt)
        elif tgt.type == 'matrix_determinant':
            k, v = exec_matrix_determinant(rt, tgt)
        elif tgt.type == 'sequence_limit_condition':
            k, v = exec_sequence_limit_condition(rt, tgt)
        elif tgt.type == 'find_maximum':
            k, v = exec_find_maximum(rt, tgt)
        elif tgt.type == 'area_between_curves':
            k, v = exec_area_between_curves(rt, tgt)
        elif tgt.type == 'integral_double':
            k, v = exec_double_integral(rt, tgt)
        else:
            results[f"{tgt.type}"] = {"error": "unsupported_target"}
            continue
        
        if k is None: continue

        # Post-processing
        if isinstance(v, sp.Expr):
            if ir.output.simplify:
                v = sp.simplify(v)
            if ir.output.mode == 'decimal':
                # Only attempt numerical evaluation if the expression is a number or can be evaluated to one
                if v.is_Number or not v.free_symbols: # Check if it's a number or has no free symbols
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
                else:
                    logging.info(f"Skipping decimal conversion for symbolic expression: '{v}'")
        
        results[k] = v

    # Debug logging
    logging.debug(f"Raw results: {results}")

    return results

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
    """Append entry to JSON file (create if not exists)"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
    else:
        data = []
    data.append(entry)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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
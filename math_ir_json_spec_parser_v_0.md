# MathIR‑JSON: универсальный формат задачи → детерминированное решение в SymPy (v0.1)

Ниже — спецификация универсального JSON-формата, который LLM генерирует строго по схеме, и скелет пайплайна, который валидирует JSON, нормализует сущности и строит вычислительный граф → SymPy‑решение.

---

## 0) TL;DR

- **LLM только парсит** исходный текст → **строгий JSON** по схеме.
- Мы валидируем JSON (Pydantic/JSON Schema), затем **детерминированно** генерируем SymPy‑код.
- Поддерживаем подтипы: интегралы/пределы/ряды, уравнения/неравенства, матрицы/линал, вероятность/комбинаторика, аналитическая геометрия (с поворотами/касательными/областями), оптимизация.

---

## 1) Верхнеуровневая структура MathIR‑JSON

```json
{
  "meta": {"id": "task_001", "lang": "ru", "notes": ""},
  "task_type": "auto",
  "expr_format": "latex",
  "assumptions": {"reals_default": true, "positivity": {"n": "N+", "a": "R", "b": "R+"}},
  "constants": {"pi": "\\pi", "e": "e"},
  "symbols": [{"name": "x", "domain": "R"}, {"name": "n", "domain": "N+"}],
  "definitions": {
    "functions": [{"name": "f", "args": ["x"], "expr": "\\sin(2x)"}],
    "sequences": [{"name": "a", "args": ["n"], "expr": "\\frac{3n-2}{2n}"}],
    "matrices": [{"name": "A", "rows": 2, "cols": 2, "data": [["1","2"],["3","4"]]}],
    "distributions": [{"name": "B1", "kind": "bernoulli", "params": {"p": "0.7"}}],
    "geometry": [{"id":"C1","kind":"parabola","equation":"y = x^2 - 4"}]
  },
  "transforms": [{"target":"C1","type":"rotation","angle_deg":60,"center":[0,0]}],
  "conditions": [
    {"type":"equation","expr":"\\int_{0}^{\\pi} \\sin(2x) dx = I"},
    {"type":"tangent_through","object":"C1","point":[0,0]},
    {"type":"quadrant","object":"C1","value":4}
  ],
  "targets": [
    {"type":"integral_def","expr":"\\sin(2x)","var":"x","limits":[0,"\\pi"],"name":"I"},
    {"type":"value","name":"answer","expr":"I"}
  ],
  "output": {"mode":"decimal","round_to":3,"simplify":true},
  "validation": {"tolerance_abs":1e-9,"check_domain_violations":true}
}
```

> Примечание: `task_type: "auto"` — маршрутизатор на стороне парсера сам определяет подтип из содержимого.

---

## 2) JSON Schema (укорочено; реальные проверки — в Pydantic)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.org/mathir.schema.json",
  "type": "object",
  "required": ["expr_format", "targets"],
  "properties": {
    "expr_format": {"enum": ["latex", "sympy", "infix"]},
    "symbols": {
      "type": "array",
      "items": {"type": "object", "required": ["name","domain"],
        "properties": {"name": {"type":"string"}, "domain": {"enum":["R","R+","Z","N","N+","C"]}}}
    },
    "definitions": {"type": "object"},
    "transforms": {"type": "array"},
    "conditions": {"type": "array"},
    "targets": {"type": "array", "minItems": 1}
  }
}
```

---

## 3) Скелет пайплайна (Python)

> Ниже: Pydantic‑модели, нормализация, маршрутизация по `targets`, генерация SymPy, вычисление и пост‑валидация. Код ориентирован на расширение — добавляйте типы условий/целей как новые узлы.

```python
# mathir_parser_v01.py
from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from dataclasses import dataclass
import sympy as sp

# === Helpers: LaTeX → SymPy ===
# Предпочтительно использовать ваш tsm (text2sympy). Здесь fallback на sympy.parsing.latex
try:
    import text2sympy as tsm
    def to_sympy_expr(s: str, symbols: Dict[str, sp.Symbol]) -> sp.Expr:
        return tsm.to_sympy_expr(s)  # ваша функция
except Exception:
    from sympy.parsing.latex import parse_latex
    def to_sympy_expr(s: str, symbols: Dict[str, sp.Symbol]) -> sp.Expr:
        return parse_latex(s)

# === Core IR ===
class SymbolSpec(BaseModel):
    name: str
    domain: Literal['R','R+','Z','N','N+','C'] = 'R'

class FunctionDef(BaseModel):
    name: str
    args: List[str]
    expr: str  # latex/sympy string

class SequenceDef(BaseModel):
    name: str
    args: List[str]
    expr: str

class MatrixDef(BaseModel):
    name: str
    rows: int
    cols: int
    data: List[List[str]]  # элементы как строки выражений

class DistributionDef(BaseModel):
    name: str
    kind: Literal['bernoulli','binomial','geometric','poisson','hypergeom']
    params: Dict[str, str]

class GeometryDef(BaseModel):
    id: str
    kind: Literal['line','circle','parabola','ellipse','polyline']
    equation: Optional[str] = None  # latex/infix eq like 'y = x^2 - 4'
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
    subs: Optional[Dict[str, str]] = None  # var->expr

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
    limits: List[str] | List[float] | List[Any]
    name: Optional[str] = None

class TargetLimit(BaseModel):
    type: Literal['limit']
    expr: str
    var: str
    to: str  # 'oo', '-oo', '0', 'a', etc.

class TargetSum(BaseModel):
    type: Literal['sum']
    term: str
    idx: str
    start: str
    end: str  # number or 'oo'

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
    event_expr: str  # e.g., 'H1 | H2'

class TargetValue(BaseModel):
    type: Literal['value']
    name: str
    expr: str

Target = TargetIntegral | TargetLimit | TargetSum | TargetSolve | TargetIneq | TargetMatrixSolve | TargetProbability | TargetValue

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
    assumptions: Dict[str, Any] = {"reals_default": True}
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
    funcs: Dict[str, sp.Function]
    sequences: Dict[str, sp.Lambda]
    matrices: Dict[str, sp.Matrix]
    distributions: Dict[str, Any]
    geometry: Dict[str, Any]

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
        dom = DOMAIN_MAP.get(s.domain, sp.S.Reals)
        symtab[s.name] = sp.Symbol(s.name, real=True) if dom == sp.S.Reals else sp.Symbol(s.name)
    funcs: Dict[str, sp.Function] = {}
    for f in ir.definitions.functions:
        args = [symtab[a] for a in f.args]
        expr = to_sympy_expr(f.expr, symtab)
        funcs[f.name] = sp.Lambda(tuple(args), expr)
    sequences: Dict[str, sp.Lambda] = {}
    for a in ir.definitions.sequences:
        args = [symtab[x] for x in a.args]
        expr = to_sympy_expr(a.expr, symtab)
        sequences[a.name] = sp.Lambda(tuple(args), expr)
    matrices: Dict[str, sp.Matrix] = {}
    for m in ir.definitions.matrices:
        mat = sp.Matrix([[to_sympy_expr(v, symtab) for v in row] for row in m.data])
        matrices[m.name] = mat
    # TODO: distributions/geometry as needed
    return Runtime(symtab, funcs, sequences, matrices, {}, {})

# === Node executors ===

def exec_integral(rt: Runtime, t: TargetIntegral) -> Tuple[str, sp.Expr]:
    x = rt.symtab.get(t.var, sp.Symbol(t.var, real=True))
    a, b = t.limits
    a = to_sympy_expr(str(a), rt.symtab)
    b = to_sympy_expr(str(b), rt.symtab)
    expr = to_sympy_expr(t.expr, rt.symtab)
    val = sp.integrate(expr, (x, a, b))
    return (t.name or 'I', val)

def exec_limit(rt: Runtime, t: TargetLimit) -> Tuple[str, sp.Expr]:
    x = rt.symtab.get(t.var, sp.Symbol(t.var, real=True))
    expr = to_sympy_expr(t.expr, rt.symtab)
    to = to_sympy_expr(t.to, rt.symtab) if t.to not in ['oo','-oo'] else (sp.oo if t.to=='oo' else -sp.oo)
    return ('limit', sp.limit(expr, x, to))

def exec_sum(rt: Runtime, t: TargetSum) -> Tuple[str, sp.Expr]:
    n = rt.symtab.get(t.idx, sp.Symbol(t.idx, integer=True))
    term = to_sympy_expr(t.term, rt.symtab)
    a = to_sympy_expr(t.start, rt.symtab)
    b = sp.oo if t.end=='oo' else to_sympy_expr(t.end, rt.symtab)
    return ('sum', sp.summation(term, (n, a, b)))

def exec_solve(rt: Runtime, t: TargetSolve) -> Tuple[str, Any]:
    unknowns = [rt.symtab.get(u, sp.Symbol(u)) for u in t.unknowns]
    eqs = [sp.Eq(*map(lambda e: to_sympy_expr(e, rt.symtab), expr.split('='))) if '=' in expr else sp.Eq(to_sympy_expr(expr, rt.symtab), 0) for expr in t.equations]
    sol = sp.solve(eqs, unknowns, dict=True)
    return ('solve', sol)

def exec_ineq(rt: Runtime, t: TargetIneq) -> Tuple[str, Any]:
    x = list(rt.symtab.values())[0] if rt.symtab else sp.Symbol('x', real=True)
    ineqs = [to_sympy_expr(s, rt.symtab) for s in t.inequalities]
    sol = sp.reduce_inequalities(ineqs, list(rt.symtab.values()))
    return ('inequalities', sol)

def exec_matrix(rt: Runtime, t: TargetMatrixSolve) -> Tuple[str, Any]:
    # Поиск уравнения вида A*A.T*X = B в conditions
    # Для v0.1: берём первый matrix_equation
    # Пример expr: "A*A.T*X = B"
    eq = None
    for c in ir_global.conditions:
        if c.type == 'matrix_equation' and c.expr:
            eq = c.expr
            break
    if not eq:
        return ('matrix', {'error':'no_matrix_equation'})
    left, right = map(str.strip, eq.split('='))
    # Грубый парсинг (v0.1): заменим имена на реальные Matrix
    env = {**rt.matrices}
    env['T'] = lambda M: M.T
    # WARNING: eval — только в доверенной среде / заменить на безопасный парсер
    L = eval(left.replace('.T','').replace('T(','('), {}, env)
    R = rt.matrices[right]
    X = sp.Matrix(sp.symbols('x11 x12 x21 x22')).reshape(R.shape[0], R.shape[1])
    sol = sp.Eq(L*X, R)
    # Решаем покомпонентно
    eqs = list((L*X - R).reshape(R.rows*R.cols, 1))
    vars = list(X)
    out = sp.solve([sp.Eq(e,0) for e in eqs], vars, dict=True)
    return ('matrix', out)

def exec_value(rt: Runtime, t: TargetValue, store: Dict[str, sp.Expr]) -> Tuple[str, sp.Expr]:
    # Позволяет ссылаться на ранее посчитанные имена (напр. I)
    symtab_ext = {**rt.symtab, **store}
    expr = to_sympy_expr(t.expr, symtab_ext)
    return (t.name, expr)

EXEC_MAP = {
    'integral_def': exec_integral,
    'limit': exec_limit,
    'sum': exec_sum,
    'solve_for': exec_solve,
    'inequalities': exec_ineq,
    'solve_for_matrix': exec_matrix,
}

# === Main runner ===

def run_mathir(ir: MathIR) -> Dict[str, Any]:
    global ir_global
    ir_global = ir  # для exec_matrix v0.1
    rt = build_runtime(ir)
    store: Dict[str, sp.Expr] = {}
    results: Dict[str, Any] = {}

    for tgt in ir.targets:
        if isinstance(tgt, TargetValue):
            k, v = exec_value(rt, tgt, store)
        else:
            fn = EXEC_MAP.get(tgt.type)
            if fn is None:
                results[f"{tgt.type}"] = {"error":"unsupported_target"}
                continue
            if tgt.type == 'integral_def':
                k, v = fn(rt, tgt)
                store[k] = v
            else:
                k, v = fn(rt, tgt)
        # simplify/format
        if isinstance(v, sp.Expr) and ir.output.simplify:
            v = sp.simplify(v)
        if isinstance(v, sp.Expr) and ir.output.mode == 'decimal':
            v = sp.N(v)
            if ir.output.round_to is not None:
                v = sp.nsimplify(round(float(v), ir.output.round_to))
        results[k] = v

    return results
```

> В реальном коде: убрать `eval`, добавить безопасный парсер матричных выражений, реализовать геометрию (повороты/касательные) и вероятность (комбинаторные модели/дистрибутивы) как отдельные узлы.

---

## 4) Примеры JSON → результат

### A) Интеграл

```json
{
  "expr_format":"latex",
  "symbols":[{"name":"x","domain":"R"}],
  "targets":[{"type":"integral_def","expr":"\\sin(2x)","var":"x","limits":[0,"\\pi"],"name":"I"},
             {"type":"value","name":"answer","expr":"I"}],
  "output":{"mode":"decimal","round_to":3}
}
```

Ожидаемо: `answer = 0.000`.

### B) Два охотника (вероятность)

```json
{
  "definitions":{ "distributions":[
    {"name":"H1","kind":"bernoulli","params":{"p":"0.7"}},
    {"name":"H2","kind":"bernoulli","params":{"p":"0.6"}}
  ]},
  "prob_space":{"independence":[["H1","H2"]]},
  "targets":[{"type":"probability","event_expr":"H1 | H2"}],
  "output":{"mode":"decimal","round_to":3}
}
```

В v0.1 узел `probability` не реализован — добавить: `P(H1|H2)=P(H1∩H2)/P(H2)` и набор примитивов для бинома/гипергеом.

### C) Матричное уравнение

```json
{
  "definitions":{"matrices":[
    {"name":"A","rows":2,"cols":2,"data":[["1","2"],["3","4"]]},
    {"name":"B","rows":2,"cols":2,"data":[["1","3"],["2","4"]]}
  ]},
  "conditions":[{"type":"matrix_equation","expr":"A*A.T*X = B"}],
  "targets":[{"type":"solve_for_matrix","unknown":"X"}],
  "output":{"mode":"exact"}
}
```

---

## 5) Правила для LLM‑парсера (системный промпт, кратко)

- Ты — **строгий парсер**. Возвращай **только валидный JSON** по схеме MathIR‑JSON.
- **Не решай** задачу и **не добавляй** фактов. Если данных не хватает — ставь `unknown/ambiguous` и коротко поясняй в `meta.notes`.
- Формулы — в `expr_format` (предпочт. LaTeX без `$`).
- Геометрию опиши через `definitions.geometry` + `transforms` + `conditions`.
- Цели — только через `targets` (никаких текстовых ответов).

Рекомендуется: JSON‑mode/grammar‑constrained decoding + валидация Pydantic + авто‑ремаршаллинг ошибок.

---

## 6) Дорожная карта до v0.2–v0.3

- Реализовать безопасный парсер матричных выражений и геометрии (повороты: подстановка `x,y ← R(θ)^T·[x;y]`).
- Узел `probability`: биномиальная, гипергеометрическая, условные события, независимость.
- Узлы `area_between`, `optimize` (Лагранж/границы/выпуклость), `base_conversion`.
- Юнит‑тесты на 30–50 примеров из твоего датасета (особенно «Нужно смотреть»).
- Метрики: % валидного JSON, полнота полей, доля задач с корректным ответом downstream.

---

## 7) Анти‑хаки и надёжность

- Запрет свободного текста вне JSON.
- Pydantic‑валидация + строгая схема типов (enums, обязательные поля).
- Детект «галлюцинаций»: если объект/параметр не упомянут в исходном тексте — перегенерация с подсказкой.
- Решение всегда проверяет домены/условия и сигналит о противоречиях структурированно.

---

**Готово.** Этот документ — отправная точка. Добавляем/уточняем узлы по мере обкатки на твоём датасете `test_private.csv` (особенно геометрию и вероятность).


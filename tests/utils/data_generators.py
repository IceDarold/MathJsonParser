"""
Test data generators for MathIR Parser testing.

This module provides utilities for generating test data, including MathIR objects,
mathematical expressions, and test scenarios for comprehensive testing.
"""

import random
import string
from typing import Dict, Any, List, Optional, Union, Tuple
import sympy as sp
from mathir_parser.main import MathIR


class TestDataGenerator:
    """Generator for various types of test data."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize test data generator.
        
        Args:
            seed: Random seed for reproducible test data
        """
        if seed is not None:
            random.seed(seed)
        
        self.variable_names = ['x', 'y', 'z', 't', 'u', 'v', 'w']
        self.index_names = ['n', 'm', 'k', 'i', 'j']
        self.function_names = ['f', 'g', 'h', 'p', 'q']
        self.matrix_names = ['A', 'B', 'C', 'D', 'M', 'N']
    
    def generate_random_string(self, length: int = 10) -> str:
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def generate_symbol_spec(self, name: Optional[str] = None, domain: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a symbol specification.
        
        Args:
            name: Symbol name (random if None)
            domain: Symbol domain (random if None)
            
        Returns:
            Symbol specification dictionary
        """
        if name is None:
            name = random.choice(self.variable_names)
        
        if domain is None:
            domain = random.choice(['R', 'R+', 'Z', 'N', 'N+', 'C'])
        
        return {"name": name, "domain": domain}
    
    def generate_polynomial_expression(self, variable: str = 'x', max_degree: int = 5) -> str:
        """
        Generate a random polynomial expression.
        
        Args:
            variable: Variable name
            max_degree: Maximum degree of polynomial
            
        Returns:
            LaTeX polynomial expression
        """
        degree = random.randint(1, max_degree)
        terms = []
        
        for i in range(degree + 1):
            coeff = random.randint(-10, 10)
            if coeff == 0:
                continue
            
            power = degree - i
            
            if power == 0:
                terms.append(str(coeff))
            elif power == 1:
                if coeff == 1:
                    terms.append(variable)
                elif coeff == -1:
                    terms.append(f"-{variable}")
                else:
                    terms.append(f"{coeff}*{variable}")
            else:
                if coeff == 1:
                    terms.append(f"{variable}^{power}")
                elif coeff == -1:
                    terms.append(f"-{variable}^{power}")
                else:
                    terms.append(f"{coeff}*{variable}^{power}")
        
        if not terms:
            return "1"
        
        expression = terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                expression += f" {term}"
            else:
                expression += f" + {term}"
        
        return expression
    
    def generate_trigonometric_expression(self, variable: str = 'x') -> str:
        """
        Generate a random trigonometric expression.
        
        Args:
            variable: Variable name
            
        Returns:
            LaTeX trigonometric expression
        """
        functions = ['\\sin', '\\cos', '\\tan']
        func = random.choice(functions)
        
        # Generate argument
        arg_type = random.choice(['simple', 'linear', 'polynomial'])
        
        if arg_type == 'simple':
            arg = variable
        elif arg_type == 'linear':
            coeff = random.randint(1, 5)
            const = random.randint(-5, 5)
            if const == 0:
                arg = f"{coeff}*{variable}"
            elif const > 0:
                arg = f"{coeff}*{variable} + {const}"
            else:
                arg = f"{coeff}*{variable} - {abs(const)}"
        else:  # polynomial
            arg = self.generate_polynomial_expression(variable, max_degree=2)
        
        return f"{func}({arg})"
    
    def generate_rational_expression(self, variable: str = 'x') -> str:
        """
        Generate a random rational expression.
        
        Args:
            variable: Variable name
            
        Returns:
            LaTeX rational expression
        """
        numerator = self.generate_polynomial_expression(variable, max_degree=3)
        denominator = self.generate_polynomial_expression(variable, max_degree=2)
        
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    
    def generate_integration_limits(self) -> List[Union[int, str]]:
        """
        Generate random integration limits.
        
        Returns:
            List of [lower_limit, upper_limit]
        """
        limit_types = [
            [0, 1],
            [-1, 1],
            [0, 2],
            [-2, 2],
            [1, 5],
            [0, "\\pi"],
            ["-\\pi", "\\pi"],
            [0, "oo"],
            [1, "oo"]
        ]
        
        return random.choice(limit_types)
    
    def generate_matrix_data(self, rows: int, cols: int) -> List[List[str]]:
        """
        Generate random matrix data.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            Matrix data as list of lists of strings
        """
        matrix = []
        for i in range(rows):
            row = []
            for j in range(cols):
                value = random.randint(-10, 10)
                row.append(str(value))
            matrix.append(row)
        return matrix


class MathIRGenerator:
    """Generator for MathIR objects with various configurations."""
    
    def __init__(self, data_generator: Optional[TestDataGenerator] = None):
        """
        Initialize MathIR generator.
        
        Args:
            data_generator: Test data generator to use
        """
        self.data_gen = data_generator or TestDataGenerator()
    
    def generate_simple_value_mathir(self, expression: Optional[str] = None) -> MathIR:
        """
        Generate a simple value computation MathIR.
        
        Args:
            expression: Expression to use (random if None)
            
        Returns:
            MathIR object
        """
        if expression is None:
            expression = self.data_gen.generate_polynomial_expression()
        
        return MathIR(
            expr_format="latex",
            symbols=[self.data_gen.generate_symbol_spec()],
            targets=[{
                "type": "value",
                "name": "result",
                "expr": expression
            }],
            output={"mode": "exact"}
        )
    
    def generate_integral_mathir(self, 
                               expression: Optional[str] = None,
                               variable: str = 'x',
                               limits: Optional[List[Union[int, str]]] = None) -> MathIR:
        """
        Generate an integral computation MathIR.
        
        Args:
            expression: Expression to integrate (random if None)
            variable: Integration variable
            limits: Integration limits (random if None)
            
        Returns:
            MathIR object
        """
        if expression is None:
            expr_type = random.choice(['polynomial', 'trigonometric', 'rational'])
            if expr_type == 'polynomial':
                expression = self.data_gen.generate_polynomial_expression(variable)
            elif expr_type == 'trigonometric':
                expression = self.data_gen.generate_trigonometric_expression(variable)
            else:
                expression = self.data_gen.generate_rational_expression(variable)
        
        if limits is None:
            limits = self.data_gen.generate_integration_limits()
        
        return MathIR(
            expr_format="latex",
            symbols=[{"name": variable, "domain": "R"}],
            targets=[{
                "type": "integral_def",
                "expr": expression,
                "var": variable,
                "limits": limits,
                "name": "integral_result"
            }],
            output={"mode": random.choice(["exact", "decimal"])}
        )
    
    def generate_limit_mathir(self,
                            expression: Optional[str] = None,
                            variable: str = 'x',
                            to: Optional[str] = None) -> MathIR:
        """
        Generate a limit computation MathIR.
        
        Args:
            expression: Expression for limit (random if None)
            variable: Variable approaching limit
            to: Limit point (random if None)
            
        Returns:
            MathIR object
        """
        if expression is None:
            # Generate expressions suitable for limits
            expressions = [
                f"\\frac{{\\sin({variable})}}{{{variable}}}",
                f"\\frac{{{variable}^2 - 1}}{{{variable} - 1}}",
                f"\\frac{{1}}{{{variable}}}",
                f"\\frac{{{variable}^2}}{{{variable} + 1}}",
                f"(1 + \\frac{{1}}{{{variable}}})^{variable}"
            ]
            expression = random.choice(expressions)
        
        if to is None:
            to = random.choice(["0", "1", "-1", "oo", "-oo"])
        
        return MathIR(
            expr_format="latex",
            symbols=[{"name": variable, "domain": "R"}],
            targets=[{
                "type": "limit",
                "expr": expression,
                "var": variable,
                "to": to
            }],
            output={"mode": "exact"}
        )
    
    def generate_sum_mathir(self,
                          term: Optional[str] = None,
                          index: str = 'n',
                          start: str = "1",
                          end: Optional[str] = None) -> MathIR:
        """
        Generate a summation MathIR.
        
        Args:
            term: Term to sum (random if None)
            index: Summation index
            start: Start value
            end: End value (random if None)
            
        Returns:
            MathIR object
        """
        if term is None:
            terms = [
                index,
                f"{index}^2",
                f"\\frac{{1}}{{{index}}}",
                f"\\frac{{1}}{{{index}^2}}",
                f"2^{index}",
                f"(-1)^{index} * \\frac{{1}}{{{index}}}"
            ]
            term = random.choice(terms)
        
        if end is None:
            end = random.choice(["5", "10", "100", "oo"])
        
        return MathIR(
            expr_format="latex",
            symbols=[{"name": index, "domain": "N"}],
            targets=[{
                "type": "sum",
                "term": term,
                "idx": index,
                "start": start,
                "end": end
            }],
            output={"mode": random.choice(["exact", "decimal"])}
        )
    
    def generate_solve_mathir(self,
                            equations: Optional[List[str]] = None,
                            unknowns: Optional[List[str]] = None) -> MathIR:
        """
        Generate an equation solving MathIR.
        
        Args:
            equations: Equations to solve (random if None)
            unknowns: Unknown variables (random if None)
            
        Returns:
            MathIR object
        """
        if unknowns is None:
            num_unknowns = random.randint(1, 3)
            unknowns = random.sample(self.data_gen.variable_names, num_unknowns)
        
        if equations is None:
            equations = []
            for i, var in enumerate(unknowns):
                if len(unknowns) == 1:
                    # Single variable equations
                    eq_types = [
                        f"{var}^2 - 4 = 0",
                        f"2*{var} + 3 = 7",
                        f"{var}^2 + {var} - 6 = 0"
                    ]
                    equations.append(random.choice(eq_types))
                else:
                    # Multi-variable system
                    if i == 0:
                        equations.append(f"{unknowns[0]} + {unknowns[1]} = 5")
                    elif i == 1:
                        equations.append(f"{unknowns[0]} - {unknowns[1]} = 1")
        
        symbols = [{"name": var, "domain": "R"} for var in unknowns]
        
        return MathIR(
            expr_format="latex",
            symbols=symbols,
            targets=[{
                "type": "solve_for",
                "unknowns": unknowns,
                "equations": equations
            }],
            output={"mode": "exact"}
        )
    
    def generate_matrix_mathir(self,
                             matrix_size: Optional[Tuple[int, int]] = None,
                             operation: str = "determinant") -> MathIR:
        """
        Generate a matrix operation MathIR.
        
        Args:
            matrix_size: Matrix dimensions (random if None)
            operation: Matrix operation type
            
        Returns:
            MathIR object
        """
        if matrix_size is None:
            size = random.choice([(2, 2), (3, 3), (2, 3), (3, 2)])
        else:
            size = matrix_size
        
        rows, cols = size
        matrix_name = random.choice(self.data_gen.matrix_names)
        matrix_data = self.data_gen.generate_matrix_data(rows, cols)
        
        mathir_data = {
            "expr_format": "latex",
            "definitions": {
                "matrices": [{
                    "name": matrix_name,
                    "rows": rows,
                    "cols": cols,
                    "data": matrix_data
                }]
            },
            "output": {"mode": "exact"}
        }
        
        if operation == "determinant":
            mathir_data["targets"] = [{
                "type": "matrix_determinant",
                "matrix_name": matrix_name
            }]
        elif operation == "inverse":
            mathir_data["targets"] = [{
                "type": "matrix_inverse",
                "matrix_name": matrix_name
            }]
        
        return MathIR.model_validate(mathir_data)
    
    def generate_multi_target_mathir(self, num_targets: int = 3) -> MathIR:
        """
        Generate a MathIR with multiple targets.
        
        Args:
            num_targets: Number of targets to generate
            
        Returns:
            MathIR object with multiple targets
        """
        symbols = [{"name": "x", "domain": "R"}, {"name": "n", "domain": "N"}]
        targets = []
        
        target_types = ["integral_def", "limit", "sum", "value"]
        
        for i in range(num_targets):
            target_type = random.choice(target_types)
            
            if target_type == "integral_def":
                targets.append({
                    "type": "integral_def",
                    "expr": self.data_gen.generate_polynomial_expression(),
                    "var": "x",
                    "limits": [0, 1],
                    "name": f"I{i+1}"
                })
            elif target_type == "limit":
                targets.append({
                    "type": "limit",
                    "expr": "\\frac{\\sin(x)}{x}",
                    "var": "x",
                    "to": "0"
                })
            elif target_type == "sum":
                targets.append({
                    "type": "sum",
                    "term": f"n^{i+1}",
                    "idx": "n",
                    "start": "1",
                    "end": "5"
                })
            else:  # value
                targets.append({
                    "type": "value",
                    "name": f"V{i+1}",
                    "expr": f"{i+1} + {i+2}"
                })
        
        return MathIR(
            expr_format="latex",
            symbols=symbols,
            targets=targets,
            output={"mode": "exact"}
        )
    
    def generate_complex_workflow_mathir(self) -> MathIR:
        """
        Generate a complex MathIR with interdependent targets.
        
        Returns:
            Complex MathIR object
        """
        return MathIR(
            expr_format="latex",
            symbols=[
                {"name": "x", "domain": "R"},
                {"name": "n", "domain": "N"}
            ],
            definitions={
                "functions": [{
                    "name": "f",
                    "args": ["x"],
                    "expr": "x^2 + 1"
                }]
            },
            targets=[
                {
                    "type": "integral_def",
                    "expr": "f(x)",
                    "var": "x",
                    "limits": [0, 1],
                    "name": "I"
                },
                {
                    "type": "value",
                    "name": "doubled_I",
                    "expr": "2 * I"
                },
                {
                    "type": "sum",
                    "term": "\\frac{1}{n^2}",
                    "idx": "n",
                    "start": "1",
                    "end": "10"
                },
                {
                    "type": "value",
                    "name": "final_result",
                    "expr": "doubled_I + sum"
                }
            ],
            output={"mode": "exact"}
        )


# Convenience functions for quick test data generation
def generate_test_mathir_batch(count: int = 10, 
                              types: Optional[List[str]] = None) -> List[MathIR]:
    """
    Generate a batch of test MathIR objects.
    
    Args:
        count: Number of MathIR objects to generate
        types: Types of MathIR to generate (all types if None)
        
    Returns:
        List of MathIR objects
    """
    if types is None:
        types = ["value", "integral", "limit", "sum", "solve"]
    
    generator = MathIRGenerator()
    mathirs = []
    
    for _ in range(count):
        mathir_type = random.choice(types)
        
        if mathir_type == "value":
            mathir = generator.generate_simple_value_mathir()
        elif mathir_type == "integral":
            mathir = generator.generate_integral_mathir()
        elif mathir_type == "limit":
            mathir = generator.generate_limit_mathir()
        elif mathir_type == "sum":
            mathir = generator.generate_sum_mathir()
        elif mathir_type == "solve":
            mathir = generator.generate_solve_mathir()
        else:
            mathir = generator.generate_simple_value_mathir()
        
        mathirs.append(mathir)
    
    return mathirs


def generate_edge_case_expressions() -> List[str]:
    """
    Generate mathematical expressions that are edge cases.
    
    Returns:
        List of edge case expressions
    """
    return [
        "\\frac{1}{0}",  # Division by zero
        "\\sqrt{-1}",    # Complex square root
        "\\ln(0)",       # Undefined logarithm
        "0^0",           # Indeterminate form
        "\\frac{0}{0}",  # Indeterminate form
        "\\infty - \\infty",  # Indeterminate form
        "\\frac{\\infty}{\\infty}",  # Indeterminate form
        "0 * \\infty",   # Indeterminate form
        "1^\\infty",     # Indeterminate form
        "\\infty^0",     # Indeterminate form
    ]
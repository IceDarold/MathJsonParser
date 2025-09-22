import sympy as sp

n = sp.symbols('n', integer=True, positive=True)
expr = n * sp.cos(n**2 + sp.exp(n)) / (2**n - 1)

# Предел при n → ∞
L = sp.limit(expr, n, sp.oo)

# Перевод в десятичную дробь и округление до 3 знаков
answer = f"{float(sp.N(L)):.3f}"

print("SymPy limit:", L)          # 0
print("Decimal (rounded to 3):", answer)  # 0.000

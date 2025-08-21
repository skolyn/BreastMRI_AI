# Code Style Guide

This repository follows a **consistent Python coding style** to improve readability, maintainability, and collaboration.

---

## ✅ General Guidelines
- Use **PEP 8** as the baseline style guide.
- Always use **4 spaces** for indentation (no tabs).
- Keep line length ≤ **88 characters** (Black formatter default).
- Use **meaningful variable and function names** (avoid single-letter names except in math/loops).
- Use **f-strings** for string formatting.
- Use **type hints** for all function arguments, return types, elementary variables and local module classes.
- No need to use type hints for non-local classes (e.g torch.Optimizer) except np.array and torch.tensor.
- Functions with 3+ arguments should have each of them in a new line
- Organize imports according to **PEP 8**:
  - Standard library
  - Third-party libraries
  - Local modules

---

## 📘 Docstrings
- Use **Google-style docstrings** for functions, classes, and modules.
- Always include **types** in both type hints and docstrings.
- Use the **`typing`** library (e.g., `List`, `Dict`, `Optional`) when applicable.

### Example
```python
from typing import List

def add(a: int, b: int) -> int:
    """
    Add two numbers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: Sum of a and b.

    Raises:
        ValueError: If inputs are not numbers.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Inputs must be integers.")
    return a + b

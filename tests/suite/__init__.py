"""Suite v1 test package.

`tests/` itself is not a package (its modules import siblings by plain module
name, e.g. `from agentic_helpers import ...`), but this subdirectory is: its
modules share the deterministic bundle factory in `conftest.py` through
`from .conftest import ...`, which only resolves when the directory is a real
package. Without this file pytest cannot even collect tests/suite.
"""

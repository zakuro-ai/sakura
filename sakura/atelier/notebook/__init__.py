"""The run's own notebook (CONTRACTS §2.5): built from the code that ran and
the values it measured, re-executed and repaired by the notebook agent, and
shipped as `artifacts/notebook.tar`. Nothing here is imported by the engine
core at module level; every heavy import (nbformat, nbclient, matplotlib) is
inside a function."""

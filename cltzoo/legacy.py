"""Version 1, loaded from legacy/clt_zoo_v1.ipynb without its widgets, so that its own functions (densities,
probability functions, samplers, presets, plotting grid, numerical distribution function) can be run and
compared with version 2. Nothing is copied by hand: the code is read from the notebook.

Author: Muhammed Ikbal Yilmaz
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "legacy" / "clt_zoo_v1.ipynb"


def load_v1():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    src = "".join(next(c for c in cells if c["cell_type"] == "code")["source"])
    src = src.split("# Widgets;")[0]          # everything before the widgets: helpers, PDFs, PMFs, DIST, PRESETS
    src = src.replace("from IPython.display import display\n", "").replace("import ipywidgets as W\n", "")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    saved = plt.rcParams.copy()              # version 1 changes the global figure style; undo that afterwards
    ns = {}
    try:
        exec(compile(src, str(NOTEBOOK), "exec"), ns)
    finally:
        plt.rcParams.update(saved)
    # theory_grid and enforce_param_constraints are defined after the widgets; take them from the source
    rest = "".join(next(c for c in cells if c["cell_type"] == "code")["source"])
    for fname in ("enforce_param_constraints", "theory_grid"):
        start = rest.index(f"def {fname}(")
        end = rest.index("\ndef ", start + 1)
        exec(compile(rest[start:end], str(NOTEBOOK), "exec"), ns)
    return ns

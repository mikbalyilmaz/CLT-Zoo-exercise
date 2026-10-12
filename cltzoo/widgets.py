"""Widget controls for the notebook (needs ipywidgets). The figures themselves come from cltzoo.panels.

explorer_ui() : distribution, seed, sample size, sample and overlay switches, x-padding, one slider per
                parameter (labels, ranges and steps of version 1), presets applied when the distribution
                changes, and a button that saves the current figure as PNG in notebook/renders/.
clt_ui()      : distribution, its parameters, N, M, path length and seed, for the running mean, the sampling
                distribution of the mean and the standardized mean.

Impossible parameter values (Uniform with b <= a) are reported in the output instead of being changed.

Author: Muhammed Ikbal Yilmaz
"""

from pathlib import Path

import ipywidgets as W
import matplotlib.pyplot as plt
from IPython.display import display

from .distributions import NAMES, ZOO
from .panels import clt_figure, explorer_figure

RENDERS = Path(__file__).resolve().parents[1] / "notebook" / "renders"


def _param_sliders(name, use_preset=True):
    d = ZOO[name]
    values = d.preset_params() if use_preset else d.defaults()
    out = []
    for s in d.params:
        cls = W.IntSlider if s.integer else W.FloatSlider
        kw = dict(min=s.lo, max=s.hi, step=s.step) if not s.integer else dict(min=int(s.lo), max=int(s.hi), step=1)
        out.append(cls(value=values[s.key], description=f"{s.label}:", continuous_update=False,
                       layout=W.Layout(width="430px"), style={"description_width": "90px"}, **kw))
    return out


def _values(name, sliders):
    return {s.key: w.value for s, w in zip(ZOO[name].params, sliders)}


def explorer_ui(start="Gamma"):
    dist = W.Dropdown(options=NAMES, value=start, description="Dist:", layout=W.Layout(width="260px"))
    seed = W.IntText(value=42, description="Seed:", layout=W.Layout(width="180px"))
    n = W.IntSlider(value=6000, min=100, max=50_000, step=100, description="Sample n:", continuous_update=False)
    show = W.Checkbox(value=True, description="Show sample", indent=False)
    overlay = W.Checkbox(value=True, description="Overlay theory on sample", indent=False)
    xpad = W.FloatSlider(value=0.3, min=0.0, max=0.8, step=0.05, description="x-pad:", continuous_update=False)
    save = W.Button(description="Save PNG", button_style="success", icon="save")
    info = W.HTML()
    box = W.VBox()
    out = W.Output()
    state = {"fig": None, "busy": False}

    def render(*_):
        if state["busy"]:            # several controls change while a preset is applied: draw once at the end
            return
        with out:
            out.clear_output(wait=True)
            plt.close("all")
            try:
                fig = explorer_figure(dist.value, _values(dist.value, box.children), n_sample=n.value,
                                      seed=seed.value, show_sample=show.value, overlay=overlay.value, xpad=xpad.value)
            except ValueError as e:
                print(e)
                state["fig"] = None
                return
            state["fig"] = fig
            display(fig)          # shown inside the output area, whatever the matplotlib backend
            plt.close(fig)

    def rebuild(*_):
        pre = ZOO[dist.value].preset
        state["busy"] = True
        box.children = tuple(_param_sliders(dist.value))
        for w in box.children:
            w.observe(render, names="value")
        show.value, overlay.value, xpad.value = pre["sample"], pre["overlay"], pre["xpad"]
        n.value = max(n.min, min(n.max, int(pre["n"]) or n.min))
        state["busy"] = False
        render()

    def on_save(_):
        if state["fig"] is None:
            info.value = "<b>Nothing to save.</b>"
            return
        RENDERS.mkdir(parents=True, exist_ok=True)
        path, i = RENDERS / f"{dist.value}.png", 2
        while path.exists():
            path, i = RENDERS / f"{dist.value}_{i:03d}.png", i + 1
        state["fig"].savefig(path, dpi=200)
        info.value = f"Saved: {path}"

    dist.observe(rebuild, names="value")
    for w in (seed, n, show, overlay, xpad):
        w.observe(render, names="value")
    save.on_click(on_save)
    display(W.VBox([W.HBox([dist, seed, save]), W.HBox([n, xpad]), W.HBox([show, overlay]), box, info, out]))
    rebuild()


def clt_ui(start="Exponential"):
    dist = W.Dropdown(options=NAMES, value=start, description="Dist:", layout=W.Layout(width="260px"))
    seed = W.IntText(value=42, description="Seed:", layout=W.Layout(width="180px"))
    N = W.IntSlider(value=30, min=1, max=1000, step=1, description="N:", continuous_update=False)
    M = W.IntSlider(value=5000, min=500, max=50_000, step=500, description="M:", continuous_update=False)
    L = W.IntSlider(value=5000, min=100, max=100_000, step=100, description="Path n:", continuous_update=False)
    box = W.VBox()
    out = W.Output()

    def render(*_):
        with out:
            out.clear_output(wait=True)
            plt.close("all")
            try:
                fig = clt_figure(dist.value, _values(dist.value, box.children), N=N.value, M=M.value,
                                 path_length=L.value, seed=seed.value)
            except ValueError as e:
                print(e)
                return
            display(fig)
            plt.close(fig)

    def rebuild(*_):
        box.children = tuple(_param_sliders(dist.value))
        for w in box.children:
            w.observe(render, names="value")
        render()

    dist.observe(rebuild, names="value")
    for w in (seed, N, M, L):
        w.observe(render, names="value")
    display(W.VBox([W.HBox([dist, seed]), W.HBox([N, M]), L, box, out]))
    rebuild()

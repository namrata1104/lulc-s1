"""Figures for the prototype and the case-study notebook."""
import calendar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import config as C

CMAP = ListedColormap(C.CLASS_COLORS)
N = len(C.CLASSES)


def month_label(month):
    y, m = month.split("-")
    return f"{calendar.month_abbr[int(m)]} {y}"


# Every figure takes an optional class scheme (classes, colors); the default is the 8 project classes.

def class_legend(fig, ncol=None, y=0.0, classes=None, colors=None):
    classes, colors = classes or C.CLASSES, colors or C.CLASS_COLORS
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    fig.legend(handles, [c.replace("_", " ") for c in classes], loc="lower center", ncol=ncol or len(classes),
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=10)


def show_map(ax, label, title=None, title_color="black", colors=None):
    colors = colors or C.CLASS_COLORS
    if label is None:
        ax.text(0.5, 0.5, "no Sentinel-1\nscene this month", ha="center", va="center", color="#6b7280")
        ax.set_facecolor("#f3f4f6")
    else:
        ax.imshow(np.ma.masked_equal(label, C.IGNORE), cmap=ListedColormap(colors), vmin=0, vmax=len(colors) - 1,
                  interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=11, color=title_color)
    ax.set_xticks([]); ax.set_yticks([])


def maps_grid(maps, months, ncols=4, title=None, scenes=None, min_scenes=4, classes=None, colors=None):
    """12 monthly maps. With `scenes` (month -> number of Sentinel-1 scenes), months composited from fewer than
    `min_scenes` scenes get an orange "⚠ n scenes" title: they are noisier and less reliable."""
    rows = int(np.ceil(len(months) / ncols))
    height = 3.9 * rows + 1.3
    fig, axes = plt.subplots(rows, ncols, figsize=(3.6 * ncols, height),
                             gridspec_kw=dict(hspace=0.22, wspace=0.05))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.75 / height, bottom=0.6 / height)
    for ax in axes.ravel()[len(months):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), months):
        low = scenes is not None and 0 < scenes[m] < min_scenes
        show_map(ax, maps[m], f"{month_label(m)}  ⚠ {scenes[m]} scenes" if low else month_label(m),
                 title_color="#c2410c" if low else "black", colors=colors)
    if title:
        fig.suptitle(title, fontsize=13, y=1 - 0.2 / height)
    n = len(classes or C.CLASSES)
    class_legend(fig, ncol=n // 2 if ncols < 4 else n, y=0.05 / height, classes=classes, colors=colors)
    return fig


def s1_grid(s1, months, band=0, ncols=4):
    lo, hi = C.CLIP_DB[band]
    rows = int(np.ceil(len(months) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(3.6 * ncols, 3.6 * rows))
    for ax in axes.ravel()[len(months):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), months):
        if s1[m] is None:
            ax.text(0.5, 0.5, "no scene", ha="center", va="center")
        else:
            ax.imshow(s1[m][..., band], cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"{C.S1_BANDS[band]} · {month_label(m)}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def pct_chart(pct, colors=None):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(pct))
    ax.stackplot(x, pct.fillna(0).T.values, colors=colors or C.CLASS_COLORS, labels=list(pct.columns),
                 edgecolor="white", lw=0.3)
    ax.set_xticks(x, [month_label(m) for m in pct.index], rotation=45, ha="right")
    ax.set_ylabel("% of area"); ax.set_ylim(0, 100); ax.set_xlim(0, len(pct) - 1)
    ax.set_title("Class shares per month")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def total_change_chart(total, colors=None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(total))
    ax.barh(y, total.values, color=colors or C.CLASS_COLORS, edgecolor="#374151", lw=0.5)
    ax.set_yticks(y, [c.replace("_", " ") for c in total.index])
    ax.axvline(0, color="#374151", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("change in percentage points")
    ax.set_title(f"Total change {total.name}")
    for yi, v in zip(y, total.values):
        if np.isfinite(v):
            ax.text(v, yi, f" {v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    lim = max(1.0, np.nanmax(np.abs(total.values)) * 1.3)
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    return fig


def reference_pair(pred, ref, ref_year, months, classes=None, colors=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
    show_map(ax[0], pred, f"Model · majority class {month_label(months[0])} – {month_label(months[-1])}", colors=colors)
    show_map(ax[1], ref, f"Esri 10 m Annual LULC · {ref_year}", colors=colors)
    n = len(classes or C.CLASSES)
    class_legend(fig, ncol=(n + 1) // 2, classes=classes, colors=colors)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig

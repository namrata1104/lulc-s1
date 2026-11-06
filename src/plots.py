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


def class_legend(fig, ncol=N, y=0.0):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in C.CLASS_COLORS]
    fig.legend(handles, [c.replace("_", " ") for c in C.CLASSES], loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=10)


def show_map(ax, label, title=None):
    if label is None:
        ax.text(0.5, 0.5, "no Sentinel-1\nscene this month", ha="center", va="center", color="#6b7280")
        ax.set_facecolor("#f3f4f6")
    else:
        ax.imshow(np.ma.masked_equal(label, C.IGNORE), cmap=CMAP, vmin=0, vmax=N - 1, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def maps_grid(maps, months, ncols=4, title=None):
    rows = int(np.ceil(len(months) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(3.6 * ncols, 4.0 * rows + 0.8))
    for ax in axes.ravel()[len(months):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), months):
        show_map(ax, maps[m], month_label(m))
    if title:
        fig.suptitle(title, fontsize=13)
    class_legend(fig, ncol=N // 2 if ncols < 4 else N)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=1.5)
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


def pct_chart(pct):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(pct))
    ax.stackplot(x, pct.fillna(0).T.values, colors=C.CLASS_COLORS, labels=C.CLASSES, edgecolor="white", lw=0.3)
    ax.set_xticks(x, [month_label(m) for m in pct.index], rotation=45, ha="right")
    ax.set_ylabel("% of area"); ax.set_ylim(0, 100); ax.set_xlim(0, len(pct) - 1)
    ax.set_title("Class shares per month")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    return fig


def total_change_chart(total):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(N)
    ax.barh(y, total.values, color=C.CLASS_COLORS, edgecolor="#374151", lw=0.5)
    ax.set_yticks(y, [c.replace("_", " ") for c in C.CLASSES])
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


def reference_pair(pred, ref, ref_year, months):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
    show_map(ax[0], pred, f"Model · majority class {month_label(months[0])} – {month_label(months[-1])}")
    show_map(ax[1], ref, f"Esri 10 m Annual LULC · {ref_year}")
    class_legend(fig, ncol=N // 2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig

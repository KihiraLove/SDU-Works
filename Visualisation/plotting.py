#!/usr/bin/env python3
"""
plotting.py

All figure creation lives here.

Contents
- Histograms with readable tick formatting + summary annotations
- Bar charts with edge outlines + spacing + value labels
- Scatter plots with labelled best/worst/outliers and light de-overlap
- Choropleths with non-white sequential palettes and improved diverging palette
- PDF export of all generated PNGs

Design notes
- Prefer sequential light→dark palettes without white (truncate low end).
- Clip map ranges to 5–95 percentiles for readability.
- Keep plotting functions pure-ish: write a single file per call, return nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import hashlib
import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from utils import log

try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None


# -----------------------
# Formatting helpers
# -----------------------

def fmt_int(x, _pos=None) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return ""


def fmt_float(x, _pos=None) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return ""


def _apply_nice_axes(ax: plt.Axes, x_money: bool = False, y_money: bool = False) -> None:
    def kfmt(v, _pos=None) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            return f"{v/1000.0:.0f}k"
        except Exception:
            return ""

    if x_money:
        xmin, xmax = ax.get_xlim()
        if max(abs(xmin), abs(xmax)) >= 20_000:
            ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    if y_money:
        ymin, ymax = ax.get_ylim()
        if max(abs(ymin), abs(ymax)) >= 20_000:
            ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.tick_params(axis="x", labelrotation=0)


def _stable_unit_hash(s: str) -> float:
    """Stable hash in [0, 1). Python's built-in hash is salted per process."""
    h = hashlib.md5(s.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], byteorder="big", signed=False)
    return (n % 10_000_000) / 10_000_000.0


def savefig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {outpath}")


# -----------------------
# Color helpers
# -----------------------

def truncate_cmap(cmap_name: str, minval: float = 0.20, maxval: float = 0.95, n: int = 256):
    """
    Truncate colormap to avoid near-white low end for better map readability.
    """
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(minval, maxval, n))
    return mpl.colors.LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors)


def robust_vmin_vmax(values: pd.Series, lo: float = 5, hi: float = 95) -> Tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return (0.0, 1.0)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


# -----------------------
# Label selection logic
# -----------------------

def pick_best_worst_outliers(df: pd.DataFrame, xcol: str, ycol: str, n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Best: high x, low y
    Worst: low x, high y
    Outliers: far from center in z-space
    """
    d = df[["municipality", xcol, ycol]].dropna().copy()
    if d.empty:
        return d.head(0), d.head(0), d.head(0)

    zx = (d[xcol] - d[xcol].mean()) / (d[xcol].std(ddof=0) + 1e-9)
    zy = (d[ycol] - d[ycol].mean()) / (d[ycol].std(ddof=0) + 1e-9)

    score = zx - zy
    dist = np.sqrt(zx**2 + zy**2)

    d["_score"] = score
    d["_dist"] = dist
    best = d.sort_values("_score", ascending=False).head(n)
    worst = d.sort_values("_score", ascending=True).head(n)
    outl = d.sort_values("_dist", ascending=False).head(n)
    return best, worst, outl


def annotate_points(
    ax: plt.Axes,
    pts: pd.DataFrame,
    xcol: str,
    ycol: str,
    prefix: str,
    offsets: Optional[List[Tuple[int, int]]] = None,
    *,
    coord_lookup: Optional[dict[str, tuple[float, float]]] = None,
) -> None:
    if pts.empty:
        return
    offsets = offsets or [(10, 10), (10, -12), (-10, 10), (-10, -12), (14, 0), (-14, 0)]
    for i, r in enumerate(pts.itertuples(index=False)):
        x = getattr(r, xcol)
        y = getattr(r, ycol)
        name = getattr(r, "municipality")

        if coord_lookup is not None and isinstance(name, str) and name in coord_lookup:
            x, y = coord_lookup[name]
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            f"{prefix}: {name}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78),
            arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6),
            zorder=5,
        )


# -----------------------
# Histograms
# -----------------------

def plot_hist(series: pd.Series | None, title: str, xlabel: str, outpath: Path, is_money: bool = False) -> None:
    if series is None:
        log(f"Skip hist: {title} (missing series)")
        return
    v = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 7))

    # If we're displaying large monetary values, use "k" ticks and reflect it in the label.
    if is_money and (not v.empty) and float(v.max()) >= 20_000 and "DKK" in xlabel and "k" not in xlabel.lower():
        xlabel = xlabel.replace("DKK", "k DKK")

    ax.hist(v, bins=18, edgecolor="white", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    _apply_nice_axes(ax, x_money=is_money)

    # Summary text
    if not v.empty:
        q5, q50, q95 = np.percentile(v, [5, 50, 95])
        ax.text(
            0.98, 0.98,
            f"n={len(v)}\nmin={fmt_int(v.min())}\nmedian={fmt_int(q50)}\n95%={fmt_int(q95)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8),
        )

    fig.tight_layout()
    savefig(fig, outpath)


# -----------------------
# Bars
# -----------------------

def plot_bar_top_bottom(
    df: pd.DataFrame,
    col: str,
    title: str,
    outpath: Path,
    xlabel: str,
    *,
    top: bool = True,
    k: int = 20,
    is_money: bool = False,
) -> None:
    """
    Horizontal bar chart for top/bottom municipalities.

    Note: outpath is positional and required; xlabel is also required (forces callsite correctness).
    """
    d = df[["municipality", col]].dropna().copy()
    if d.empty:
        log(f"Skip bar: {title} (no data)")
        return

    d = d.sort_values(col, ascending=not top).head(k)
    d = d.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(
        d["municipality"],
        d[col],
        edgecolor="white",
        linewidth=0.9,
        height=0.72,   # spacing so bars don't touch
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)

    if is_money:
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))

    # Value labels at bar ends
    labels = [fmt_int(v) if is_money else fmt_float(v) for v in d[col].values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)

    fig.tight_layout()
    savefig(fig, outpath)


def plot_hus1_region_bars(df: pd.DataFrame, title: str, outpath: Path) -> None:
    if df.empty:
        log(f"Skip HUS1 bars: {title} (no data)")
        return

    d = df.sort_values("rent_index_private_2021_100")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(d["region"], d["rent_index_private_2021_100"], edgecolor="white", linewidth=0.9, height=0.75)
    ax.set_title(title)
    ax.set_xlabel("Index (2021=100)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)

    ax.bar_label(bars, labels=[fmt_float(v) for v in d["rent_index_private_2021_100"].values], padding=3, fontsize=9)

    fig.tight_layout()
    savefig(fig, outpath)


def plot_group_bars(df: pd.DataFrame, xcol: str, title: str, ylabel: str, outpath: Path, is_pct: bool = False) -> None:
    if df.empty:
        log(f"Skip group bars: {title} (no data)")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(df["group"], df[xcol], edgecolor="white", linewidth=0.9, width=0.75)
    ax.set_title(title)
    ax.set_xlabel("Municipality group")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.0)

    if is_pct:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _p: f"{y:.0f}%"))
        labels = [f"{v:.1f}%" for v in df[xcol].values]
    else:
        labels = [fmt_float(v) for v in df[xcol].values]

    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)

    # Add headroom so labels don't clip.
    try:
        ymax = float(pd.to_numeric(df[xcol], errors="coerce").max())
        if math.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0, ymax * 1.18)
    except Exception:
        pass

    fig.tight_layout()
    savefig(fig, outpath)


# -----------------------
# Scatter
# -----------------------

def plot_scatter_labeled(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    *,
    x_money: bool = False,
    y_money: bool = False,
    jitter: bool = True,
    jitter_frac: float = 0.002,
) -> None:
    d = df[["municipality", xcol, ycol]].dropna().copy()
    if d.empty:
        log(f"Skip scatter: {title} (no data)")
        return

    # Light, deterministic jitter to reduce overplotting without changing the story.
    # (Jitter is tiny relative to the axis range.)
    x = pd.to_numeric(d[xcol], errors="coerce").astype(float).to_numpy()
    y = pd.to_numeric(d[ycol], errors="coerce").astype(float).to_numpy()
    if jitter and len(d) >= 50:
        xr = float(np.nanmax(x) - np.nanmin(x)) or 1.0
        yr = float(np.nanmax(y) - np.nanmin(y)) or 1.0
        dx = np.array([(_stable_unit_hash(str(m)) - 0.5) * 2.0 for m in d["municipality"].astype(str)]) * (jitter_frac * xr)
        dy = np.array([(_stable_unit_hash("y:" + str(m)) - 0.5) * 2.0 for m in d["municipality"].astype(str)]) * (jitter_frac * yr)
        xj = x + dx
        yj = y + dy
    else:
        xj, yj = x, y

    coord_lookup = {str(m): (float(xv), float(yv)) for m, xv, yv in zip(d["municipality"].astype(str), xj, yj)}

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(xj, yj, alpha=0.75, s=45, edgecolors="white", linewidth=0.55)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _apply_nice_axes(ax, x_money=x_money, y_money=y_money)

    best, worst, outl = pick_best_worst_outliers(df, xcol, ycol, n=3)
    annotate_points(ax, best, xcol, ycol, "Best", coord_lookup=coord_lookup)
    annotate_points(ax, worst, xcol, ycol, "Worst", coord_lookup=coord_lookup)
    annotate_points(ax, outl, xcol, ycol, "Outlier", coord_lookup=coord_lookup)

    fig.tight_layout()
    savefig(fig, outpath)


# -----------------------
# Maps
# -----------------------

def plot_map(
    gdf: "gpd.GeoDataFrame",
    col: str,
    title: str,
    outpath: Path,
    *,
    cmap_name: str = "viridis",
    diverging: bool = False,
    cbar_label: Optional[str] = None,
    tick_mode: str | None = None,
) -> None:
    if gpd is None:
        log("geopandas unavailable; skipping maps")
        return

    values = pd.to_numeric(gdf[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 12))

    if diverging:
        vmax = float(np.nanpercentile(values, 95))
        vmin = float(np.nanpercentile(values, 5))
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

        # Discrete diverging bins to avoid a muddy/blended center.
        ncolors = 11
        bounds = np.linspace(vmin, vmax, ncolors + 1)
        base = mpl.colormaps.get_cmap("RdBu_r")(np.linspace(0.12, 0.88, ncolors))
        # Replace the center with a light neutral grey (avoid pure white).
        base[ncolors // 2] = mpl.colors.to_rgba("#e6e6e6")
        cmap = mpl.colors.ListedColormap(base, name="RdBu_r_binned")
        norm = mpl.colors.BoundaryNorm(bounds, ncolors)

        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.30,
            edgecolor="white",
            ax=ax,
            norm=norm,
            missing_kwds={"color": "#d0d0d0"},
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    else:
        vmin, vmax = robust_vmin_vmax(values, 5, 95)
        cmap = truncate_cmap(cmap_name, 0.22, 0.95)  # avoid whites
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.35,
            edgecolor="white",
            ax=ax,
            norm=norm,
            missing_kwds={"color": "#d0d0d0"},
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_axis_off()

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel(cbar_label or col)
    # Tick formatting
    if tick_mode == "k":
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v/1000:.0f}k"))
    elif tick_mode in {"pct", "percent"}:
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}%"))
    elif tick_mode == "int" or (tick_mode is None and (not values.dropna().empty) and values.dropna().max() > 1000):
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))

    fig.tight_layout()
    savefig(fig, outpath)


def try_make_maps(data_dir: Path, muni: pd.DataFrame, figures_dir: Path, gpkg_name: str, *, year: int = 2024) -> None:
    if gpd is None:
        log("geopandas not available in environment; maps skipped")
        return

    gpkg = data_dir / gpkg_name
    if not gpkg.exists():
        log(f"Map file missing (expected in data/): {gpkg_name} -> maps skipped")
        return

    log(f"Loading GPKG: {gpkg}")
    g = gpd.read_file(gpkg)
    if "navn" not in g.columns:
        raise ValueError('GPKG does not contain municipality name column "navn"')

    from utils import muni_key  # local import to avoid circular

    g["muni_key"] = g["navn"].map(muni_key)
    g = g.merge(muni, on="muni_key", how="left")

    # Sequential maps
    if "income_dkk" in g.columns:
        plot_map(
            g,
            "income_dkk",
            f"Disposable family income ({year})",
            figures_dir / f"map_income_{year}.png",
            cmap_name="Blues",
            cbar_label="k DKK",
            tick_mode="k",
        )
    if "rent_dkk_m2" in g.columns:
        plot_map(
            g,
            "rent_dkk_m2",
            f"Rent (DKK/m²) ({year})",
            figures_dir / f"map_rent_{year}.png",
            cmap_name="PuBuGn",
            cbar_label="DKK per m²",
            tick_mode="int",
        )
    if "property_price_dkk_m2" in g.columns:
        plot_map(
            g,
            "property_price_dkk_m2",
            f"Property prices (DKK/m²) ({year})",
            figures_dir / f"map_property_price_{year}.png",
            cmap_name="OrRd",
            cbar_label="k DKK per m²",
            tick_mode="k",
        )
    if "tax_rate_pct" in g.columns:
        plot_map(
            g,
            "tax_rate_pct",
            f"Municipality tax rate (%) ({year})",
            figures_dir / f"map_tax_rate_{year}.png",
            cmap_name="YlOrBr",
            cbar_label="%",
            tick_mode="pct",
        )
    if "income_to_rent" in g.columns:
        plot_map(
            g,
            "income_to_rent",
            f"Affordability: implied m² = income / rent ({year})",
            figures_dir / f"map_income_to_rent_ratio_{year}.png",
            cmap_name="Greens",
            cbar_label="m²",
            tick_mode="int",
            clip_quantiles=(0.05, 0.95),
        )

    # Diverging map
    if "earnings_net_dkk" in g.columns:
        plot_map(
            g,
            "earnings_net_dkk",
            f"Earnings net (workplace - residence) ({year})",
            figures_dir / f"map_earnings_net_{year}.png",
            diverging=True,
            cbar_label="k DKK",
            tick_mode="k",
        )


# -----------------------
# PDF export
# -----------------------

def export_pngs_to_pdf(figures_dir: Path, out_pdf: Path) -> None:
    prefix_order = {
        "hist_": 0,
        "bar_": 1,
        "scatter_": 2,
        "map_": 3,
    }

    def sort_key(p: Path):
        name = p.name.lower()
        for pref, idx in prefix_order.items():
            if name.startswith(pref):
                return (idx, name)
        return (99, name)

    pngs = sorted((p for p in figures_dir.glob("*.png")), key=sort_key)
    if not pngs:
        log("No PNGs found for PDF export")
        return

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for p in pngs:
            # render PNG as an image on a page
            img = plt.imread(p)
            fig, ax = plt.subplots(figsize=(11, 8.5))  # landscape
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(p.name, fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    log(f"Wrote {out_pdf}")


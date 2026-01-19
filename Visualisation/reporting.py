#!/usr/bin/env python3
"""
reporting.py

Creates human-readable debug reports that describe:
- The parsed datasets
- The municipality master table and derived metrics
- The exact data behind each figure (histograms, bars, scatters, maps)

The intent is to make it easy to sanity-check parsing and joins without staring at plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import log
from plotting import robust_vmin_vmax  # keep consistent with plotting


# -----------------------------
# Formatting helpers
# -----------------------------

def _fmt_int(x: float) -> str:
    return f"{x:,.0f}"

def _fmt_float(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"

def _fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}%"

def _safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()

def _series_summary(s: pd.Series) -> Dict[str, float]:
    v = _safe_series(s)
    if v.empty:
        return {"n": 0}
    return {
        "n": int(v.shape[0]),
        "min": float(v.min()),
        "p05": float(np.percentile(v, 5)),
        "p25": float(np.percentile(v, 25)),
        "median": float(np.percentile(v, 50)),
        "mean": float(v.mean()),
        "p75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }

def _hist_data(s: pd.Series, bins: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    v = _safe_series(s)
    if v.empty:
        return np.array([]), np.array([])
    counts, edges = np.histogram(v.to_numpy(), bins=bins)
    return counts, edges

def _warn_out_of_range(name: str, s: pd.Series, lo: Optional[float], hi: Optional[float], df: Optional[pd.DataFrame] = None) -> List[str]:
    v = pd.to_numeric(s, errors="coerce")
    mask = pd.Series(False, index=v.index)
    if lo is not None:
        mask |= v < lo
    if hi is not None:
        mask |= v > hi
    mask &= v.notna()
    if not mask.any():
        return []
    lines = [f"WARNING: {name} has {int(mask.sum())} out-of-range values (expected [{lo},{hi}]):"]
    if df is not None and "municipality" in df.columns:
        bad = df.loc[mask, ["municipality"]].copy()
        bad[name] = v.loc[mask].values
        bad = bad.sort_values(name, ascending=False).head(15)
        for r in bad.itertuples(index=False):
            lines.append(f"  - {r.municipality}: {getattr(r, name)}")
        if int(mask.sum()) > 15:
            lines.append("  - ...")
    else:
        sample = v.loc[mask].head(15).tolist()
        lines.append(f"  Sample: {sample}")
    return lines


# -----------------------------
# Report generators
# -----------------------------

def dataset_report(datasets: Dict[str, pd.DataFrame]) -> List[str]:
    lines: List[str] = []
    lines.append("DATASET SUMMARY")
    lines.append("=" * 60)

    for key in sorted(datasets.keys()):
        df = datasets[key]
        lines.append(f"\n[{key}] rows={len(df):,} cols={df.shape[1]:,}")
        cols = list(df.columns)
        lines.append(f"  columns: {', '.join(cols)}")
        # common diagnostics
        if "muni_key" in df.columns:
            dup = int(df["muni_key"].duplicated().sum())
            missing_key = int(df["muni_key"].isna().sum())
            lines.append(f"  muni_key: missing={missing_key:,}, duplicates={dup:,}, unique={df['muni_key'].nunique(dropna=True):,}")
        if "municipality" in df.columns:
            missing_name = int(df["municipality"].isna().sum())
            lines.append(f"  municipality: missing={missing_name:,}, unique={df['municipality'].nunique(dropna=True):,}")
        # show head/tail small
        head = df.head(5).to_string(index=False)
        tail = df.tail(5).to_string(index=False)
        lines.append("  head(5):")
        lines.extend(["    " + ln for ln in head.splitlines()])
        lines.append("  tail(5):")
        lines.extend(["    " + ln for ln in tail.splitlines()])

    return lines


def master_table_report(muni: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append("\n\nMASTER TABLE SUMMARY")
    lines.append("=" * 60)
    lines.append(f"rows={len(muni):,} cols={muni.shape[1]:,}")

    # Non-null counts
    key_cols = [
        "income_dkk", "population", "tax_rate_pct", "rent_dkk_m2", "property_price_dkk_m2",
        "earnings_residence_dkk", "earnings_workplace_dkk", "earnings_net_dkk", "income_to_rent",
    ]
    avail = [c for c in key_cols if c in muni.columns]
    lines.append("\nNon-null counts:")
    for c in avail:
        lines.append(f"  {c}: {int(muni[c].notna().sum()):,} / {len(muni):,}")

    # Summary statistics
    lines.append("\nSummary statistics (n, min, p05, median, p95, max):")
    fmt_map = {
        "income_dkk": _fmt_int,
        "population": _fmt_int,
        "tax_rate_pct": lambda x: _fmt_float(x, 2),
        "rent_dkk_m2": _fmt_int,
        "property_price_dkk_m2": _fmt_int,
        "earnings_residence_dkk": _fmt_int,
        "earnings_workplace_dkk": _fmt_int,
        "earnings_net_dkk": _fmt_int,
        "income_to_rent": lambda x: _fmt_float(x, 2),
    }
    for c in avail:
        stats = _series_summary(muni[c])
        if stats.get("n", 0) == 0:
            lines.append(f"  {c}: (no data)")
            continue
        f = fmt_map.get(c, _fmt_float)
        lines.append(
            f"  {c}: n={stats['n']:,}, min={f(stats['min'])}, p05={f(stats['p05'])}, "
            f"median={f(stats['median'])}, p95={f(stats['p95'])}, max={f(stats['max'])}"
        )

    # Sanity warnings
    lines.append("\nSanity checks:")
    warn_lines: List[str] = []
    warn_lines += _warn_out_of_range("tax_rate_pct", muni.get("tax_rate_pct"), 10.0, 40.0, muni)
    warn_lines += _warn_out_of_range("rent_dkk_m2", muni.get("rent_dkk_m2"), 200.0, 4000.0, muni)
    warn_lines += _warn_out_of_range("property_price_dkk_m2", muni.get("property_price_dkk_m2"), 1000.0, 200000.0, muni)
    warn_lines += _warn_out_of_range("income_dkk", muni.get("income_dkk"), 50000.0, 3000000.0, muni)
    warn_lines += _warn_out_of_range("income_to_rent", muni.get("income_to_rent"), 50.0, 5000.0, muni)
    if warn_lines:
        lines.extend(warn_lines)
    else:
        lines.append("  OK (no out-of-range values detected with the coarse thresholds).")

    # Filled-from-region flags
    fill_cols = [c for c in muni.columns if c.endswith("_filled_from_region")]
    if fill_cols:
        lines.append("\nValues filled from regional averages:")
        for fc in fill_cols:
            n = int(muni[fc].sum()) if muni[fc].dtype != object else int(pd.to_numeric(muni[fc], errors="coerce").fillna(0).sum())
            lines.append(f"  {fc}: {n:,} municipalities filled")
            if n:
                base_col = fc.replace("_filled_from_region", "")
                sample = muni.loc[muni[fc].astype(bool), ["municipality", "region", base_col]].head(15)
                for r in sample.itertuples(index=False):
                    lines.append(f"    - {r.municipality} ({r.region}): {getattr(r, base_col)}")
                if n > 15:
                    lines.append("    - ...")

    return lines


def figures_report(muni: pd.DataFrame) -> List[str]:
    """
    Produce a text representation of the data underlying each figure.
    (Group-level figures are reported if present as columns or via external callers.)
    """
    lines: List[str] = []
    lines.append("\n\nFIGURE DATA")
    lines.append("=" * 60)

    # Histograms
    hist_specs = [
        ("hist_income_2024", "income_dkk", "Income (DKK)"),
        ("hist_rent_2024", "rent_dkk_m2", "Rent (DKK/m²)"),
        ("hist_property_price_2024", "property_price_dkk_m2", "Property price (DKK/m²)"),
        ("hist_tax_rate_2024", "tax_rate_pct", "Tax rate (%)"),
        ("hist_income_to_rent_2024", "income_to_rent", "Income / Rent"),
    ]
    for fig_name, col, label in hist_specs:
        if col not in muni.columns:
            continue
        v = muni[col]
        counts, edges = _hist_data(v, bins=18)
        lines.append(f"\n[{fig_name}] {label}")
        stats = _series_summary(v)
        if stats.get("n", 0) == 0:
            lines.append("  (no data)")
            continue
        lines.append(
            "  summary: "
            f"n={stats['n']:,}, min={stats['min']:.4g}, median={stats['median']:.4g}, mean={stats['mean']:.4g}, max={stats['max']:.4g}"
        )
        if counts.size:
            lines.append("  histogram bins (edge_left, edge_right, count):")
            for i in range(len(counts)):
                lines.append(f"    [{edges[i]:.4g}, {edges[i+1]:.4g}): {int(counts[i])}")
        else:
            lines.append("  (histogram empty)")

    # Bars
    if "income_to_rent" in muni.columns:
        d = muni[["municipality", "income_to_rent"]].dropna().copy()
        best = d.sort_values("income_to_rent", ascending=False).head(20)
        worst = d.sort_values("income_to_rent", ascending=True).head(20)
        lines.append("\n[bar_income_to_rent_best20_2024] top-20 municipalities by income_to_rent:")
        for r in best.itertuples(index=False):
            lines.append(f"  - {r.municipality}: {r.income_to_rent:.3f}")
        lines.append("\n[bar_income_to_rent_worst20_2024] bottom-20 municipalities by income_to_rent:")
        for r in worst.itertuples(index=False):
            lines.append(f"  - {r.municipality}: {r.income_to_rent:.3f}")

    # Scatter summaries
    scatter_specs = [
        ("scatter_income_vs_rent_2024", "income_dkk", "rent_dkk_m2"),
        ("scatter_income_vs_property_2024", "income_dkk", "property_price_dkk_m2"),
    ]
    for fig_name, xcol, ycol in scatter_specs:
        if xcol not in muni.columns or ycol not in muni.columns:
            continue
        d = muni[["municipality", xcol, ycol]].dropna().copy()
        lines.append(f"\n[{fig_name}] points={len(d):,}")
        if len(d) == 0:
            continue
        corr = float(np.corrcoef(d[xcol].to_numpy(), d[ycol].to_numpy())[0, 1])
        lines.append(f"  Pearson corr({xcol},{ycol}) = {corr:.4f}")
        lines.append(f"  x range: {d[xcol].min():.4g} .. {d[xcol].max():.4g}")
        lines.append(f"  y range: {d[ycol].min():.4g} .. {d[ycol].max():.4g}")

        # crude overlap indicator: count of identical (x,y) pairs
        dup_xy = int(d.duplicated([xcol, ycol]).sum())
        if dup_xy:
            lines.append(f"  NOTE: {dup_xy} duplicate (x,y) points -> visible overlap likely.")
        # show extremes
        lines.append("  top-5 highest y:")
        for r in d.sort_values(ycol, ascending=False).head(5).itertuples(index=False):
            lines.append(f"    - {r.municipality}: x={getattr(r,xcol):.4g}, y={getattr(r,ycol):.4g}")
        lines.append("  top-5 lowest y:")
        for r in d.sort_values(ycol, ascending=True).head(5).itertuples(index=False):
            lines.append(f"    - {r.municipality}: x={getattr(r,xcol):.4g}, y={getattr(r,ycol):.4g}")

    # Map color scales
    map_specs = [
        ("map_income_2024", "income_dkk"),
        ("map_rent_2024", "rent_dkk_m2"),
        ("map_property_price_2024", "property_price_dkk_m2"),
        ("map_tax_rate_2024", "tax_rate_pct"),
        ("map_income_to_rent_ratio_2024", "income_to_rent"),
        ("map_earnings_residence_2024", "earnings_residence_dkk"),
        ("map_earnings_workplace_2024", "earnings_workplace_dkk"),
        ("map_earnings_net_2024", "earnings_net_dkk"),
    ]
    lines.append("\n\nMap scaling diagnostics (uses the same 5–95 percentile clipping as plotting):")
    for fig_name, col in map_specs:
        if col not in muni.columns:
            continue
        v = muni[col]
        vmin, vmax = robust_vmin_vmax(v, 5, 95)
        stats = _series_summary(v)
        lines.append(
            f"  [{fig_name}] {col}: n={stats.get('n',0):,}, vmin(p05)={vmin:.4g}, vmax(p95)={vmax:.4g}, "
            f"min={stats.get('min',np.nan):.4g}, max={stats.get('max',np.nan):.4g}"
        )

    return lines


def group_figures_report(group_data: Dict[str, pd.DataFrame]) -> List[str]:
    lines: List[str] = []
    if not group_data:
        return lines

    lines.append("\n\nGROUP / REGION FIGURES")
    lines.append("=" * 60)
    for name, df in group_data.items():
        lines.append(f"\n[{name}] rows={len(df):,}")
        if df.empty:
            lines.append("  (no data)")
            continue
        lines.append(df.to_string(index=False))
        # sanity warning for percentages
        if "share_renters_pct" in df.columns:
            bad = df[(df["share_renters_pct"] > 100) | (df["share_renters_pct"] < 0)]
            if not bad.empty:
                lines.append("  WARNING: share_renters_pct outside [0,100].")
        if "rent_index_2021_100" in df.columns:
            bad = df[df["rent_index_2021_100"] > 1000]  # extremely high
            if not bad.empty:
                lines.append("  NOTE: rent_index_2021_100 contains very large values (check parsing).")
    return lines


def write_full_report(
    out_txt: Path,
    datasets: Dict[str, pd.DataFrame],
    muni: pd.DataFrame,
    group_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """
    Writes a single text report. Also logs a short pointer.
    """
    lines: List[str] = []
    lines.extend(dataset_report(datasets))
    lines.extend(master_table_report(muni))
    lines.extend(figures_report(muni))
    if group_data:
        lines.extend(group_figures_report(group_data))

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote figure/data debug report: {out_txt}")

#!/usr/bin/env python3
"""
run_pipeline.py

Entry point for the Denmark municipality cost-of-living visualization pipeline.

Default input location:
  <project_root>/data

Outputs:
  <project_root>/figures/*.png
  <project_root>/figures/all_figures_<YEAR>.pdf
  <project_root>/figures/figures_data_report_<YEAR>.txt
  <project_root>/cache/parsed_data_<YEAR>.json

Parsing is the default. Use --use-cache to load the JSON cache instead.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pandas as pd

from config import PlotConfig, make_paths
from utils import log, ensure_dir, set_matplotlib_style
from data_io import (
    build_master_table,
    load_bundle_from_json,
    save_bundle_to_json,
    load_laby32,
    load_laby45,
    load_hus1,
)
from plotting import (
    plot_hist,
    plot_bar_top_bottom,
    plot_scatter_labeled,
    plot_group_bars,
    plot_hus1_region_bars,
    try_make_maps,
    export_pngs_to_pdf,
)
from reporting import write_full_report


YEAR_DEFAULT = 2024
GPKG_DEFAULT = "DAGI_V1_Kommuneinddeling_TotalDownload_gpkg_Current_507.gpkg"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=YEAR_DEFAULT, help="Year label used in filenames/titles.")
    ap.add_argument("--root", type=str, default=None, help="Project root. Defaults to folder containing config.py.")
    ap.add_argument("--use-cache", action="store_true", help="Load cache instead of parsing CSV files.")
    ap.add_argument("--cache-path", type=str, default=None, help="Explicit cache file to use (overrides default).")
    ap.add_argument("--gpkg", type=str, default=GPKG_DEFAULT, help="GPKG filename inside data/ for maps.")
    return ap.parse_args()


def _log_master_quick_stats(muni: pd.DataFrame) -> None:
    log(f"Master table rows: {len(muni)}")
    if "muni_key" in muni.columns:
        dup = muni["muni_key"].duplicated().sum()
        if dup:
            log(f"WARNING: duplicated muni_key rows: {dup}")
    key_cols = [
        "income_dkk",
        "population",
        "tax_rate_pct",
        "rent_dkk_m2",
        "property_price_dkk_m2",
        "earnings_residence_dkk",
        "earnings_workplace_dkk",
        "earnings_net_dkk",
        "income_to_rent",
    ]
    present = [c for c in key_cols if c in muni.columns]
    missing = {c: int(muni[c].isna().sum()) for c in present}
    log("Missing values: " + ", ".join([f"{k}={v}" for k, v in missing.items()]))

    if "income_to_rent" in muni.columns:
        d = muni[["municipality", "income_to_rent"]].dropna().sort_values("income_to_rent", ascending=False)
        if len(d) >= 3:
            log("Top income_to_rent: " + ", ".join([f"{r.municipality}={r.income_to_rent:.1f}" for r in d.head(3).itertuples(index=False)]))
            log("Bottom income_to_rent: " + ", ".join([f"{r.municipality}={r.income_to_rent:.1f}" for r in d.tail(3).itertuples(index=False)]))


def main() -> None:
    args = parse_args()
    year = args.year

    root = Path(args.root).resolve() if args.root else None
    paths = make_paths(root)

    set_matplotlib_style(PlotConfig().dpi)

    ensure_dir(paths.figures_dir)
    ensure_dir(paths.cache_dir)

    cache_path = Path(args.cache_path).resolve() if args.cache_path else (paths.cache_dir / f"parsed_data_{year}.json")

    # Helpful to avoid signature drift bugs
    log(f"plot_bar_top_bottom signature: {inspect.signature(plot_bar_top_bottom)}")

    if args.use_cache and cache_path.exists():
        log(f"Loading JSON cache: {cache_path}")
        bundle = load_bundle_from_json(cache_path)
    else:
        log(f"Parsing CSV files under: {paths.data_dir}")
        bundle = build_master_table(paths.data_dir)
        # (optional) add group datasets into cache for completeness
        group_extra = {}
        try:
            group_extra["LABY32"] = load_laby32(paths.data_dir / "LABY32.csv")
        except Exception as e:
            log(f"LABY32 parse failed: {e}")
        try:
            group_extra["LABY45"] = load_laby45(paths.data_dir / "LABY45.csv")
        except Exception as e:
            log(f"LABY45 parse failed: {e}")
        try:
            group_extra["HUS1"] = load_hus1(paths.data_dir / "HUS1.csv")
        except Exception as e:
            log(f"HUS1 parse failed: {e}")

        bundle.datasets.update(group_extra)
        save_bundle_to_json(bundle, cache_path, extra={"year": year}, extra_datasets=group_extra)

    muni = bundle.muni
    _log_master_quick_stats(muni)

    # Group data (for plots + report)
    group_data = {}
    for k in ["LABY32", "LABY45", "HUS1"]:
        if k in bundle.datasets:
            group_data[k] = bundle.datasets[k]

    # Debug report (text representation of figure data)
    report_path = paths.figures_dir / f"figures_data_report_{year}.txt"
    write_full_report(report_path, datasets=bundle.datasets, muni=muni, group_data=group_data)

    # --- Figures ---
    plot_hist(muni.get("income_dkk"), f"Disposable family income ({year})", "Income (DKK)", paths.figures_dir / f"hist_income_{year}.png", is_money=True)
    plot_hist(muni.get("rent_dkk_m2"), f"Rent ({year})", "Rent (DKK per m²)", paths.figures_dir / f"hist_rent_{year}.png", is_money=True)
    plot_hist(muni.get("property_price_dkk_m2"), f"Property prices ({year})", "Price (DKK per m²)", paths.figures_dir / f"hist_property_price_{year}.png", is_money=True)
    plot_hist(muni.get("tax_rate_pct"), f"Municipality tax rate ({year})", "Tax rate (%)", paths.figures_dir / f"hist_tax_rate_{year}.png", is_money=False)

    if "income_to_rent" in muni.columns:
        plot_hist(
            muni["income_to_rent"],
            f"Affordability: implied area = income / rent ({year})",
            "Implied area (m²)",
            paths.figures_dir / f"hist_income_to_rent_{year}.png",
            is_money=False,
        )
        plot_bar_top_bottom(
            muni,
            "income_to_rent",
            f"Best 20 municipalities by implied area (income / rent) ({year})",
            paths.figures_dir / f"bar_income_to_rent_best20_{year}.png",
            xlabel="Implied area (m²)",
            top=True,
            k=20,
        )
        plot_bar_top_bottom(
            muni,
            "income_to_rent",
            f"Worst 20 municipalities by implied area (income / rent) ({year})",
            paths.figures_dir / f"bar_income_to_rent_worst20_{year}.png",
            xlabel="Implied area (m²)",
            top=False,
            k=20,
        )

    if "income_dkk" in muni.columns and "rent_dkk_m2" in muni.columns:
        plot_scatter_labeled(
            muni,
            "income_dkk",
            "rent_dkk_m2",
            f"Income vs rent ({year})",
            "Income (k DKK)",
            "Rent (DKK per m²)",
            paths.figures_dir / f"scatter_income_vs_rent_{year}.png",
            x_money=True,
            y_money=True,
        )

    if "income_dkk" in muni.columns and "property_price_dkk_m2" in muni.columns:
        plot_scatter_labeled(
            muni,
            "income_dkk",
            "property_price_dkk_m2",
            f"Income vs property prices ({year})",
            "Income (k DKK)",
            "Property price (k DKK per m²)",
            paths.figures_dir / f"scatter_income_vs_property_{year}.png",
            x_money=True,
            y_money=True,
        )

    # Group-level bars if available
    if "LABY32" in group_data and not group_data["LABY32"].empty:
        plot_group_bars(
            group_data["LABY32"],
            "rent_index_2021_100",
            f"LABY32: Rent index by municipality groups ({year})",
            "Index (2021=100)",
            paths.figures_dir / f"bar_laby32_rent_index_groups_{year}.png",
        )

    if "LABY45" in group_data and not group_data["LABY45"].empty:
        plot_group_bars(
            group_data["LABY45"],
            "share_renters_pct",
            f"LABY45: Share of renters by municipality groups ({year})",
            "Percent",
            paths.figures_dir / f"bar_laby45_share_renters_groups_{year}.png",
            is_pct=True,
        )

    if "HUS1" in group_data and not group_data["HUS1"].empty:
        plot_hus1_region_bars(
            group_data["HUS1"],
            f"HUS1: Rent index (private) by region ({year})",
            paths.figures_dir / f"bar_hus1_rent_index_private_{year}.png",
        )

    # Maps
    try:
        try_make_maps(paths.data_dir, muni, paths.figures_dir, gpkg_name=args.gpkg, year=year)
    except Exception as e:
        log(f"Map generation failed: {e}")

    # Combined PDF (all PNGs)
    export_pngs_to_pdf(paths.figures_dir, paths.figures_dir / f"all_figures_{year}.pdf")

    log("Done.")


if __name__ == "__main__":
    main()


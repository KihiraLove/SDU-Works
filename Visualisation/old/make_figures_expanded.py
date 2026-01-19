#!/usr/bin/env python3
"""
make_figures_expanded.py

Fixes vs previous version
- Robust Danish number parsing:
  - Handles decimals with '.' correctly (23.36 stays 23.36; 106.2 stays 106.2)
  - Handles thousands with '.' correctly (1.016 -> 1016)
  - Handles thousands '.' + decimal ',' correctly (1.016,5 -> 1016.5)
  - Treats numeric inputs (int/float) as already-parsed (prevents 780.0 -> 7800)
- Restricts master table to *real municipalities* using nuts_muni_region.csv mapping
  (prevents extra non-municipality rows from outer merges).
- Improved municipality key normalization:
  - Strips “kommune/municipality” suffixes and common noise
  - Adds a couple of practical aliases
  - Greatly improves GPKG join coverage
- Basic sanity scaling:
  - If PSKAT looks like basis-points (median > 100), divide by 100
  - If rent looks x10 inflated (median > 2000), divide by 10 (logged)
- Scatter labeling: ensures annotations are not clipped.
- Map palettes: avoids near-white low end via stronger truncation.
- Exports all PNGs in ./figures into one PDF: figures/all_figures_2024.pdf
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional (maps)
try:
    import geopandas as gpd  # type: ignore
except Exception:
    gpd = None

# Optional (PDF from images)
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


# -----------------------------
# Configuration / styling
# -----------------------------

FIG_DIR_NAME = "../figures"
DPI = 220
SEED = 42


def log(msg: str) -> None:
    print(f"[make_figures_expanded] {msg}")


def set_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Name normalization
# -----------------------------

_SPECIAL = str.maketrans(
    {
        "å": "aa",
        "Å": "aa",
        "æ": "ae",
        "Æ": "ae",
        "ø": "oe",
        "Ø": "oe",
        "é": "e",
        "É": "e",
    }
)

_ENGLISH_TO_DANISH = {
    "copenhagen": "københavn",
    "capital region": "region hovedstaden",
    "capital region of denmark": "region hovedstaden",
}

# Practical aliases that appear in some DK datasets
_KEY_ALIASES = {
    "koebenhavns": "koebenhavn",
    "kobenhavns": "koebenhavn",
}


def muni_key(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = _ENGLISH_TO_DANISH.get(s.lower(), s)

    # normalize diacritics early
    s = s.translate(_SPECIAL)
    s = s.lower()

    # strip common suffixes/noise before removing spaces
    # (important for GPKG values like "Aarhus Kommune")
    s = re.sub(r"\b(kommune|municipality)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(region)\b", "region", s, flags=re.IGNORECASE)  # keep region word if present
    s = s.strip()

    # collapse separators
    s = re.sub(r"[\s\-\–\—/]+", "", s)

    # keep only alnum
    s = re.sub(r"[^a-z0-9]+", "", s)

    # alias last
    s = _KEY_ALIASES.get(s, s)
    return s


# -----------------------------
# Numeric parsing helpers
# -----------------------------

def parse_danish_number(x) -> float:
    """
    Robust parsing for Danish/StatBank-ish numbers.

    Handles:
      - thousands '.' : 1.016 -> 1016
      - thousands '.' + decimal ',' : 1.016,5 -> 1016.5
      - decimal '.' : 23.36 -> 23.36 (NOT 2336)
      - decimal ',' : 25,6 -> 25.6
      - percent strings: '23,6%' -> 23.6
      - numeric inputs: float/int are returned as-is
      - missing tokens: '..', '', etc -> NaN
    """
    if x is None:
        return np.nan

    # critical: if pandas already parsed it, do NOT string-heuristic it again
    if isinstance(x, (int, np.integer)):
        return float(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if not np.isnan(x) else np.nan

    s = str(x).strip()
    if s in {"..", ".", "", "nan", "NaN", "None"}:
        return np.nan

    s = s.replace("\xa0", "").replace(" ", "").replace("%", "").strip()

    # parentheses negatives e.g. (1.234)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # If both separators exist, Danish style is usually thousands '.' and decimal ','
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        # treat as decimal comma
        s = s.replace(",", ".")
    elif "." in s:
        # decide if '.' is thousands separator (groups of 3) or decimal dot
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
        # else keep '.' as decimal

    # strip any remaining junk
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return np.nan

    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


# -----------------------------
# StatBank-like CSV parsing
# -----------------------------

def read_statbank_pivot_csv(path: Path, value_col_index: int = -1) -> pd.DataFrame:
    """
    Reads pivot-style CSV where the last column is numeric and the penultimate is label.

    Returns columns:
      - name
      - value
    """
    raw = None
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            raw = path.read_text(encoding=enc, errors="strict")
            break
        except Exception:
            continue
    if raw is None:
        raw = path.read_text(encoding="latin1", errors="replace")

    rows: List[List[str]] = list(csv.reader(raw.splitlines()))

    names: List[str] = []
    values: List[float] = []

    for r in rows:
        if not r:
            continue
        if len(r) < 2:
            continue

        last = r[value_col_index] if len(r) >= abs(value_col_index) else ""
        v = parse_danish_number(last)
        if np.isnan(v):
            continue

        name = r[value_col_index - 1].strip().strip('"').strip()
        if not name or name in {"Total", "I-alt", "I alt", "I-alt "}:
            continue

        names.append(name)
        values.append(v)

    df = pd.DataFrame({"name": names, "value": values})
    log(f"Loaded {path.name}: {len(df):,} numeric rows")
    return df


# -----------------------------
# NUTS2 mapping (regional averages)
# -----------------------------

def build_muni_to_region_from_nuts(path: Path) -> Dict[str, str]:
    """
    nuts_muni_region.csv is hierarchical with LEVEL.
    We map LEVEL 3 (municipalities) to current LEVEL 1 region (5 DK regions).
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df = df.sort_values("SEQUENCE")
    current_region: Optional[str] = None
    muni_to_region: Dict[str, str] = {}

    for _, row in df.iterrows():
        lvl = int(row["LEVEL"])
        title = str(row["TITLE"]).strip()
        if lvl == 1:
            current_region = title
        elif lvl == 3 and current_region is not None:
            muni_to_region[muni_key(title)] = current_region

    log(f"NUTS mapping: {len(muni_to_region):,} municipalities mapped to LEVEL-1 regions")
    return muni_to_region


# -----------------------------
# Plot helpers (palettes, clipping)
# -----------------------------

def truncate_cmap(cmap_name: str, minval: float = 0.30, maxval: float = 0.95, n: int = 256):
    # avoid deprecated get_cmap
    cmap = mpl.colormaps.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mpl.colors.LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", new_colors)


def robust_vmin_vmax(values: pd.Series, lo: float = 5, hi: float = 95) -> Tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return (0.0, 1.0)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def savefig(fig: plt.Figure, outpath: Path) -> None:
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    try:
        log(f"Wrote {outpath.relative_to(outpath.parents[1])}")
    except Exception:
        log(f"Wrote {outpath}")


# -----------------------------
# Loading domain datasets
# -----------------------------

@dataclass
class DataBundle:
    muni: pd.DataFrame
    muni_to_region: Dict[str, str]


def _dedupe(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    """
    Ensure one row per muni_key. Numeric columns are averaged; municipality name takes first non-null.
    """
    if df.empty:
        return df
    keep_cols = ["muni_key", "municipality"] + value_cols
    d = df[keep_cols].copy()

    def first_nonnull(x: pd.Series):
        for v in x:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return np.nan

    agg = {c: "mean" for c in value_cols}
    agg["municipality"] = first_nonnull
    d = d.groupby("muni_key", as_index=False).agg(agg)
    return d


def load_income_indkf132(path: Path) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": "income_dkk"})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = _dedupe(df, ["income_dkk"])
    log(f"Income: {df['income_dkk'].notna().sum():,} values, range {df['income_dkk'].min():.0f}–{df['income_dkk'].max():.0f}")
    return df


def load_population_folk1a(path: Path) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": "population"})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = _dedupe(df, ["population"])
    log(f"Population: {df['population'].notna().sum():,} values, range {df['population'].min():.0f}–{df['population'].max():.0f}")
    return df


def load_tax_pskat(path: Path) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": "tax_rate_pct"})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = _dedupe(df, ["tax_rate_pct"])

    # sanity: if parsed like 2336..2630, convert to 23.36..26.30
    med = float(df["tax_rate_pct"].median(skipna=True)) if df["tax_rate_pct"].notna().any() else np.nan
    if not np.isnan(med) and med > 100:
        df["tax_rate_pct"] = df["tax_rate_pct"] / 100.0
        log("Tax rate appears scaled (basis points). Dividing by 100.")

    log(f"Tax: {df['tax_rate_pct'].notna().sum():,} values, range {df['tax_rate_pct'].min():.2f}–{df['tax_rate_pct'].max():.2f}")
    return df


def load_earnings_ligelb1(path: Path, colname: str) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": colname})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = _dedupe(df, [colname])
    log(f"{colname}: {df[colname].notna().sum():,} values, range {df[colname].min():.0f}–{df[colname].max():.0f}")
    return df


def load_property_bm010(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="latin1", errors="replace")
    rows = list(csv.reader(raw.splitlines()))

    # If the export is a normal pivot with last-column numeric, fall back to generic reader
    # but prefer a 2-value row format if present.
    data = []
    for r in rows:
        if not r or len(r) < 3:
            continue
        # last two columns numeric?
        v_last = parse_danish_number(r[-1])
        v_prev = parse_danish_number(r[-2]) if len(r) >= 2 else np.nan
        if np.isnan(v_last) and np.isnan(v_prev):
            continue

        # municipality label often near the end; take the nearest non-empty string before numeric tail
        muni = ""
        for j in range(len(r) - 1, -1, -1):
            if parse_danish_number(r[j]) == parse_danish_number(r[j]):  # numeric (not NaN)
                continue
            cand = str(r[j]).strip().strip('"').strip()
            if cand and cand not in {"Total", "I-alt", "I alt"}:
                muni = cand
                break
        if not muni:
            continue

        price = v_prev if not np.isnan(v_prev) else v_last
        data.append((muni, price))

    df = pd.DataFrame(data, columns=["municipality", "property_price_dkk_m2"])
    df["muni_key"] = df["municipality"].map(muni_key)
    df = _dedupe(df, ["property_price_dkk_m2"])
    log(f"Property: {df['property_price_dkk_m2'].notna().sum():,} values, range {df['property_price_dkk_m2'].min():.0f}–{df['property_price_dkk_m2'].max():.0f}")
    return df


def load_rent_husleje_table7(path: Path) -> pd.DataFrame:
    """
    Reads Huslejestatistik_2024_TABLE7.csv.
    IMPORTANT: dtype=str so values are not auto-converted to floats like '780.0' (which used to break parsing).
    """
    df = pd.read_csv(path, encoding="utf-8", dtype=str)
    cols = list(df.columns)
    if len(cols) < 6:
        raise ValueError("Husleje table7: unexpected format/columns")

    df = df.rename(
        columns={
            cols[0]: "municipality",
            cols[1]: "all_dwellings",
            cols[2]: "rent_all_dkk_m2",
            cols[3]: "rent_all_change",
            cols[4]: "family_dwellings",
            cols[5]: "rent_family_dkk_m2",
        }
    )

    for c in ["all_dwellings", "rent_all_dkk_m2", "family_dwellings", "rent_family_dkk_m2"]:
        df[c] = df[c].apply(parse_danish_number)

    df = df[~df["municipality"].astype(str).str.strip().isin(["I-alt", "Total", "I alt", "I-alt "])].copy()

    df["rent_dkk_m2"] = df["rent_family_dkk_m2"]
    df["muni_key"] = df["municipality"].map(muni_key)
    df = _dedupe(df, ["rent_dkk_m2"])

    # sanity: if rent looks inflated by x10 (e.g., ~8000 instead of ~800)
    med = float(df["rent_dkk_m2"].median(skipna=True)) if df["rent_dkk_m2"].notna().any() else np.nan
    if not np.isnan(med) and med > 2000:
        df["rent_dkk_m2"] = df["rent_dkk_m2"] / 10.0
        log("Rent appears scaled x10 (median > 2000). Dividing by 10.")

    log(
        f"Rent: {df['rent_dkk_m2'].notna().sum():,} values, range "
        f"{df['rent_dkk_m2'].min():.0f}–{df['rent_dkk_m2'].max():.0f}"
    )
    return df


def _parse_header_and_row_matrix(path: Path, want_row_contains: str) -> pd.DataFrame:
    """
    For small StatBank matrix exports like LABY32/LABY45/HUS1 where:
      - there is a header row with group names
      - there is a data row with numeric values
    Strategy:
      1) find first row containing want_row_contains (e.g., '2024Q2' or '2024')
      2) find numeric columns in that row
      3) search upward for a header row that has non-numeric labels in those same columns
    """
    raw = path.read_text(encoding="latin1", errors="replace")
    rows = [list(map(lambda x: str(x).strip().strip('"'), r)) for r in csv.reader(raw.splitlines())]
    rows = [r for r in rows if any(c.strip() for c in r)]

    data_i = None
    for i, r in enumerate(rows):
        if any(want_row_contains in c for c in r):
            # require at least 3 numeric cells
            nums = [parse_danish_number(c) for c in r]
            if sum(0 if np.isnan(v) else 1 for v in nums) >= 3:
                data_i = i
                break
    if data_i is None:
        return pd.DataFrame()

    data_row = rows[data_i]
    num_mask = [not np.isnan(parse_danish_number(c)) for c in data_row]
    num_idx = [j for j, m in enumerate(num_mask) if m]

    # find header row above with labels in numeric positions
    header_row = None
    for k in range(data_i - 1, -1, -1):
        r = rows[k]
        if len(r) < max(num_idx) + 1:
            continue
        labels = [r[j].strip() for j in num_idx]
        if sum(1 for t in labels if t and np.isnan(parse_danish_number(t))) >= 3:
            header_row = r
            break

    if header_row is None:
        return pd.DataFrame()

    groups = [header_row[j].strip() for j in num_idx]
    values = [parse_danish_number(data_row[j]) for j in num_idx]
    df = pd.DataFrame({"group": groups, "value": values}).dropna()
    return df


def load_laby32(path: Path) -> pd.DataFrame:
    df = _parse_header_and_row_matrix(path, want_row_contains="2024")
    if df.empty:
        return pd.DataFrame(columns=["group", "rent_index_2021_100"])
    df = df.rename(columns={"value": "rent_index_2021_100"})
    # sanity: if looks like 1060 instead of 106
    med = float(df["rent_index_2021_100"].median(skipna=True)) if df["rent_index_2021_100"].notna().any() else np.nan
    if not np.isnan(med) and med > 200 and med < 2000:
        df["rent_index_2021_100"] = df["rent_index_2021_100"] / 10.0
        log("LABY32 rent index appears scaled x10. Dividing by 10.")
    return df


def load_laby45(path: Path) -> pd.DataFrame:
    df = _parse_header_and_row_matrix(path, want_row_contains="2024")
    if df.empty:
        return pd.DataFrame(columns=["group", "share_renters_pct"])
    df = df.rename(columns={"value": "share_renters_pct"})
    # sanity: if looks like 5700 instead of 57
    med = float(df["share_renters_pct"].median(skipna=True)) if df["share_renters_pct"].notna().any() else np.nan
    if not np.isnan(med) and med > 100 and med < 10000:
        # ambiguous if x100 or x10; most likely x100
        df["share_renters_pct"] = df["share_renters_pct"] / 100.0
        log("LABY45 share appears scaled. Dividing by 100.")
    return df


def load_hus1(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="latin1", errors="replace")
    rows = [list(map(lambda x: str(x).strip().strip('"'), r)) for r in csv.reader(raw.splitlines())]
    data = []
    for r in rows:
        if not r:
            continue
        # find a cell that looks like "Region ..."
        region = next((c for c in r if c.lower().startswith("region ")), None)
        if not region:
            continue
        # take last numeric cell as the index
        nums = [parse_danish_number(c) for c in r]
        vals = [v for v in nums if not np.isnan(v)]
        if not vals:
            continue
        data.append((region, float(vals[-1])))

    df = pd.DataFrame(data, columns=["region", "rent_index_private_2021_100"]).dropna()
    # sanity for x10
    med = float(df["rent_index_private_2021_100"].median(skipna=True)) if df["rent_index_private_2021_100"].notna().any() else np.nan
    if not np.isnan(med) and med > 200 and med < 2000:
        df["rent_index_private_2021_100"] = df["rent_index_private_2021_100"] / 10.0
        log("HUS1 rent index appears scaled x10. Dividing by 10.")
    return df


# -----------------------------
# Master table build
# -----------------------------

def build_master_table(root: Path) -> DataBundle:
    muni_to_region = build_muni_to_region_from_nuts(root / "nuts_muni_region.csv")
    muni_keys = set(muni_to_region.keys())

    income = load_income_indkf132(root / "INDKF132.csv")
    pop = load_population_folk1a(root / "FOLK1A.csv")
    tax = load_tax_pskat(root / "PSKAT.csv")
    rent = load_rent_husleje_table7(root / "Huslejestatistik_2024_TABLE7.csv")
    prop = load_property_bm010(root / "BM010.csv")
    earn_res = load_earnings_ligelb1(root / "LIGELB1_residence.csv", "earnings_residence_dkk")
    earn_work = load_earnings_ligelb1(root / "LIGELB1_workplace.csv", "earnings_workplace_dkk")

    # Build a municipality frame anchored on NUTS mapping to avoid non-muni rows
    muni = pd.DataFrame({"muni_key": sorted(muni_keys)})

    def merge_in(df: pd.DataFrame, label: str) -> None:
        nonlocal muni
        before = len(muni)
        muni = muni.merge(df.drop(columns=["municipality"], errors="ignore"), on="muni_key", how="left")
        after = len(muni)
        log(f"Merge {label}: rows {before:,} -> {after:,}; missing after merge: {muni.isna().sum().to_dict()}")

    # attach names from the richest sources (rent/property/income etc.)
    name_sources = [rent, prop, income, tax, pop, earn_res, earn_work]
    names = pd.DataFrame({"muni_key": sorted(muni_keys)})
    for i, df in enumerate(name_sources):
        tmp = df[["muni_key", "municipality"]].rename(columns={"municipality": f"name_{i}"})
        names = names.merge(tmp, on="muni_key", how="left")

    name_cols = [c for c in names.columns if c.startswith("name_")]

    def pick_name(row):
        for c in name_cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return np.nan

    names["municipality"] = names.apply(pick_name, axis=1)
    muni = muni.merge(names[["muni_key", "municipality"]], on="muni_key", how="left")

    merge_in(income, "income")
    merge_in(pop, "population")
    merge_in(tax, "tax")
    merge_in(rent, "rent")
    merge_in(prop, "property")
    merge_in(earn_res, "earnings_residence")
    merge_in(earn_work, "earnings_workplace")

    # Attach region (should be complete now)
    muni["region"] = muni["muni_key"].map(muni_to_region)
    missing_region = muni["region"].isna().mean() * 100
    log(f"Region attach: missing region for {missing_region:.1f}% of rows")

    # Fill missing rent/property with regional averages (requested behavior)
    for col in ["rent_dkk_m2", "property_price_dkk_m2"]:
        if col not in muni.columns:
            continue
        reg_avg = muni.groupby("region")[col].mean()
        before_missing = int(muni[col].isna().sum())
        muni[col] = muni.apply(lambda r: reg_avg.get(r["region"], np.nan) if pd.isna(r[col]) else r[col], axis=1)
        after_missing = int(muni[col].isna().sum())
        log(f"Filled {col} with regional averages: missing {before_missing:,} -> {after_missing:,}")

    # Additional sanity constraints to prevent absurd ratios:
    if "rent_dkk_m2" in muni.columns:
        muni.loc[muni["rent_dkk_m2"] <= 0, "rent_dkk_m2"] = np.nan
        # If any remaining ultra-tiny values exist, they will explode ratios; treat as invalid
        muni.loc[muni["rent_dkk_m2"] < 100, "rent_dkk_m2"] = np.nan

    if "tax_rate_pct" in muni.columns:
        muni.loc[muni["tax_rate_pct"] > 100, "tax_rate_pct"] = np.nan

    # Derived metrics
    if "earnings_workplace_dkk" in muni.columns and "earnings_residence_dkk" in muni.columns:
        muni["earnings_net_dkk"] = muni["earnings_workplace_dkk"] - muni["earnings_residence_dkk"]

    muni["income_to_rent"] = muni["income_dkk"] / muni["rent_dkk_m2"]

    log(f"Master table rows: {len(muni):,}")
    log(
        "Non-null counts: "
        + ", ".join(
            [
                f"{c}={muni[c].notna().sum():,}"
                for c in ["income_dkk", "rent_dkk_m2", "income_to_rent", "property_price_dkk_m2", "tax_rate_pct"]
                if c in muni.columns
            ]
        )
    )
    return DataBundle(muni=muni, muni_to_region=muni_to_region)


# -----------------------------
# Ranking / labeling logic
# -----------------------------

def pick_best_worst_outliers(df: pd.DataFrame, xcol: str, ycol: str, n: int = 3):
    d = df[["municipality", xcol, ycol]].dropna().copy()
    if d.empty:
        return d.head(0), d.head(0), d.head(0)

    zx = (d[xcol] - d[xcol].mean()) / (d[xcol].std(ddof=0) + 1e-9)
    zy = (d[ycol] - d[ycol].mean()) / (d[ycol].std(ddof=0) + 1e-9)

    score = zx - zy
    dist = np.sqrt(zx**2 + zy**2)

    d = d.assign(_score=score, _dist=dist)
    best = d.sort_values("_score", ascending=False).head(n)
    worst = d.sort_values("_score", ascending=True).head(n)
    outl = d.sort_values("_dist", ascending=False).head(n)
    return best, worst, outl


def annotate_points(ax: plt.Axes, pts: pd.DataFrame, xcol: str, ycol: str, label_prefix: str) -> None:
    for i, r in enumerate(pts.itertuples(index=False)):
        x = getattr(r, xcol)
        y = getattr(r, ycol)
        name = getattr(r, "municipality")

        dx = (i % 3 - 1) * 10
        dy = (i // 3 - 0.5) * 10

        ax.annotate(
            f"{label_prefix}: {name}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.80),
            arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.6),
            annotation_clip=False,  # IMPORTANT: don’t let tight bbox clip labels
        )


# -----------------------------
# Charts
# -----------------------------

def plot_hist(series: pd.Series, title: str, xlabel: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    v = pd.to_numeric(series, errors="coerce").dropna()
    ax.hist(v, bins=18, edgecolor="white", linewidth=1.0, rwidth=0.92)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    savefig(fig, outpath)


def plot_bar_top_bottom(df: pd.DataFrame, col: str, title: str, outpath: Path, top: bool = True, k: int = 20) -> None:
    d = df[["municipality", col]].dropna().copy()
    d = d.sort_values(col, ascending=not top).head(k)
    d = d.sort_values(col, ascending=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(d["municipality"], d[col], edgecolor="white", linewidth=1.1, height=0.80)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)
    fig.tight_layout()
    savefig(fig, outpath)


def plot_scatter_labeled(df: pd.DataFrame, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    d = df[["municipality", xcol, ycol]].dropna().copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(d[xcol], d[ycol], alpha=0.85, s=55, edgecolors="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    best, worst, outl = pick_best_worst_outliers(df, xcol, ycol, n=3)
    annotate_points(ax, best, xcol, ycol, "Best")
    annotate_points(ax, worst, xcol, ycol, "Worst")
    annotate_points(ax, outl, xcol, ycol, "Outlier")

    fig.tight_layout()
    savefig(fig, outpath)


def plot_group_bars(df: pd.DataFrame, xcol: str, title: str, ylabel: str, outpath: Path) -> None:
    if df.empty:
        log(f"Skip {title}: no data")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["group"], df[xcol], edgecolor="white", linewidth=1.1, width=0.75)
    ax.set_title(title)
    ax.set_xlabel("Municipality group")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    savefig(fig, outpath)


# -----------------------------
# Choropleths
# -----------------------------

def plot_map(
    gdf: "gpd.GeoDataFrame",
    col: str,
    title: str,
    outpath: Path,
    cmap_name: str = "viridis",
    diverging: bool = False,
) -> None:
    if gpd is None:
        log("geopandas unavailable; skipping maps")
        return

    values = gdf[col]
    fig, ax = plt.subplots(figsize=(10, 12))

    if diverging:
        vmax = float(np.nanpercentile(values, 95))
        vmin = float(np.nanpercentile(values, 5))
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
        cmap = truncate_cmap("RdBu_r", 0.15, 0.90)
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.4,
            edgecolor="white",
            ax=ax,
            norm=norm,
            missing_kwds={"color": "#bdbdbd"},
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    else:
        vmin, vmax = robust_vmin_vmax(values, 5, 95)
        cmap = truncate_cmap(cmap_name, 0.30, 0.95)  # avoid near-white
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        gdf.plot(
            column=col,
            cmap=cmap,
            linewidth=0.4,
            edgecolor="white",
            ax=ax,
            norm=norm,
            missing_kwds={"color": "#bdbdbd"},
        )
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_axis_off()
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel(col)

    fig.tight_layout()
    savefig(fig, outpath)


def try_make_maps(root: Path, muni: pd.DataFrame, fig_dir: Path) -> None:
    if gpd is None:
        log("geopandas not available in environment; maps skipped")
        return

    gpkg = root / "DAGI_V1_Kommuneinddeling_TotalDownload_gpkg_Current_507.gpkg"
    if not gpkg.exists():
        log(f"Map file missing (expected locally): {gpkg.name} -> maps skipped")
        return

    log(f"Loading GPKG: {gpkg}")
    g = gpd.read_file(gpkg)
    if "navn" not in g.columns:
        raise ValueError('GPKG does not contain municipality name column "navn"')

    g["muni_key"] = g["navn"].map(muni_key)
    m = muni.copy()
    m["muni_key"] = m["muni_key"].astype(str)
    g = g.merge(m, on="muni_key", how="left")

    # Maps (choose palettes that won’t wash out)
    if "income_dkk" in g.columns:
        plot_map(g, "income_dkk", "Disposable family income (2024)", fig_dir / "map_income_2024.png", cmap_name="viridis")
    if "rent_dkk_m2" in g.columns:
        plot_map(g, "rent_dkk_m2", "Rent (DKK/m²) (2024)", fig_dir / "map_rent_2024.png", cmap_name="cividis")
    if "property_price_dkk_m2" in g.columns:
        plot_map(g, "property_price_dkk_m2", "Property prices (DKK/m²) (2024)", fig_dir / "map_property_price_2024.png", cmap_name="magma")
    if "tax_rate_pct" in g.columns:
        plot_map(g, "tax_rate_pct", "Municipality tax rate (%) (2024)", fig_dir / "map_tax_rate_2024.png", cmap_name="viridis")
    if "income_to_rent" in g.columns:
        plot_map(g, "income_to_rent", "Affordability: Income / Rent (2024)", fig_dir / "map_income_to_rent_ratio_2024.png", cmap_name="viridis")
    if "earnings_residence_dkk" in g.columns:
        plot_map(g, "earnings_residence_dkk", "Earnings (residence) (2024)", fig_dir / "map_earnings_residence_2024.png", cmap_name="viridis")
    if "earnings_workplace_dkk" in g.columns:
        plot_map(g, "earnings_workplace_dkk", "Earnings (workplace) (2024)", fig_dir / "map_earnings_workplace_2024.png", cmap_name="viridis")
    if "earnings_net_dkk" in g.columns:
        plot_map(g, "earnings_net_dkk", "Earnings net (workplace - residence) (2024)", fig_dir / "map_earnings_net_2024.png", diverging=True)


# -----------------------------
# PDF export
# -----------------------------

def export_all_pngs_to_pdf(fig_dir: Path, out_pdf: Path) -> None:
    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        log("No PNGs found for PDF export.")
        return

    if Image is None:
        log("PIL not available; skipping PDF export.")
        return

    images = []
    for p in pngs:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            log(f"PDF export: failed to read {p.name}: {e}")

    if not images:
        log("PDF export: no readable PNGs.")
        return

    images[0].save(out_pdf, save_all=True, append_images=images[1:])
    log(f"Wrote {out_pdf.relative_to(out_pdf.parents[1])}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    np.random.seed(SEED)
    set_matplotlib_style()

    root = Path(__file__).resolve().parent
    fig_dir = root / FIG_DIR_NAME
    ensure_dir(fig_dir)

    log(f"Working directory: {root}")
    bundle = build_master_table(root)
    muni = bundle.muni

    # Basic histograms
    if "income_dkk" in muni.columns:
        plot_hist(muni["income_dkk"], "Disposable family income (2024)", "Income (DKK)", fig_dir / "hist_income_2024.png")
    if "rent_dkk_m2" in muni.columns:
        plot_hist(muni["rent_dkk_m2"], "Rent (2024)", "Rent (DKK/m²)", fig_dir / "hist_rent_2024.png")
    if "property_price_dkk_m2" in muni.columns:
        plot_hist(muni["property_price_dkk_m2"], "Property prices (2024)", "Price (DKK/m²)", fig_dir / "hist_property_price_2024.png")
    if "tax_rate_pct" in muni.columns:
        plot_hist(muni["tax_rate_pct"], "Municipality tax rate (2024)", "Tax rate (%)", fig_dir / "hist_tax_rate_2024.png")

    # Affordability index
    if "income_to_rent" in muni.columns:
        plot_hist(muni["income_to_rent"], "Affordability: Income / Rent (2024)", "Income / (DKK/m²)", fig_dir / "hist_income_to_rent_2024.png")
        plot_bar_top_bottom(muni, "income_to_rent", "Best 20 municipalities by Income / Rent (2024)", fig_dir / "bar_income_to_rent_best20_2024.png", top=True, k=20)
        plot_bar_top_bottom(muni, "income_to_rent", "Worst 20 municipalities by Income / Rent (2024)", fig_dir / "bar_income_to_rent_worst20_2024.png", top=False, k=20)

    # Scatterplots
    if "income_dkk" in muni.columns and "rent_dkk_m2" in muni.columns:
        plot_scatter_labeled(
            muni,
            "income_dkk",
            "rent_dkk_m2",
            "Income vs rent (2024)",
            "Income (DKK)",
            "Rent (DKK/m²)",
            fig_dir / "scatter_income_vs_rent_2024.png",
        )

    if "income_dkk" in muni.columns and "property_price_dkk_m2" in muni.columns:
        plot_scatter_labeled(
            muni,
            "income_dkk",
            "property_price_dkk_m2",
            "Income vs property prices (2024)",
            "Income (DKK)",
            "Property price (DKK/m²)",
            fig_dir / "scatter_income_vs_property_2024.png",
        )

    # Group-level bars
    laby32 = load_laby32(root / "LABY32.csv")
    if not laby32.empty:
        plot_group_bars(
            laby32,
            "rent_index_2021_100",
            "LABY32: Rent index by municipality groups (2024Q2)",
            "Index (2021=100)",
            fig_dir / "bar_laby32_rent_index_groups_2024.png",
        )

    laby45 = load_laby45(root / "LABY45.csv")
    if not laby45.empty:
        plot_group_bars(
            laby45,
            "share_renters_pct",
            "LABY45: Share of renters by municipality groups (2024)",
            "Percent",
            fig_dir / "bar_laby45_share_renters_groups_2024.png",
        )

    hus1 = load_hus1(root / "HUS1.csv")
    if not hus1.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        d = hus1.sort_values("rent_index_private_2021_100")
        ax.barh(d["region"], d["rent_index_private_2021_100"], edgecolor="white", linewidth=1.1, height=0.80)
        ax.set_title("HUS1: Rent index (private) by region (2024Q2)")
        ax.set_xlabel("Index (2021=100)")
        ax.set_ylabel("")
        fig.tight_layout()
        savefig(fig, fig_dir / "bar_hus1_rent_index_private_2024.png")

    # Maps (only if GPKG exists locally)
    try_make_maps(root, muni, fig_dir)

    # Export single PDF with all PNGs
    export_all_pngs_to_pdf(fig_dir, fig_dir / "all_figures_2024.pdf")

    log("Done.")


if __name__ == "__main__":
    main()

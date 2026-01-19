"""
data_io.py

Responsible for:
- parsing the input CSVs into tidy pandas DataFrames
- building the municipality master table
- serializing/loading parsed datasets to/from JSON

Default behaviour: parse CSVs.
Optional: --use-json reads the cached JSON instead (still validates schema).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import log, muni_key, parse_danish_number, ensure_dir


@dataclass
class DataBundle:
    muni: pd.DataFrame                       # master municipality table
    muni_to_region: Dict[str, str]           # muni_key -> region name
    datasets: Dict[str, pd.DataFrame]        # raw-ish parsed datasets


# -----------------------------
# Generic CSV parsing helpers
# -----------------------------

def _read_text_best_effort(path: Path) -> str:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return path.read_text(encoding="latin1", errors="replace")


def read_statbank_pivot_csv(path: Path) -> pd.DataFrame:
    """
    Robustly read a StatBank-like "pivot CSV" export.
    For each row, we:
      - find the right-most numeric cell -> value
      - find the closest non-empty text cell to its left -> name
    Returns columns: name, value
    """
    raw = _read_text_best_effort(path)
    rows = list(csv.reader(raw.splitlines()))

    names: List[str] = []
    values: List[float] = []

    for r in rows:
        if not r:
            continue

        # Find right-most numeric value
        val_idx = None
        val = np.nan
        for j in range(len(r) - 1, -1, -1):
            v = parse_danish_number(r[j])
            if not np.isnan(v):
                val_idx = j
                val = v
                break
        if val_idx is None:
            continue

        # Find name to the left (first meaningful text)
        name = ""
        for j in range(val_idx - 1, -1, -1):
            cell = str(r[j]).strip().strip('"')
            if not cell or cell in {" ", "\t"}:
                continue
            if cell.lower() in {"total", "i-alt", "i alt"}:
                name = ""
                break
            name = cell
            break

        if not name:
            continue

        names.append(name)
        values.append(val)

    df = pd.DataFrame({"name": names, "value": values})
    log(f"Loaded {path.name}: {len(df):,} numeric rows")
    return df


# -----------------------------
# NUTS mapping (regional averages)
# -----------------------------

def build_muni_to_region_from_nuts(path: Path) -> Dict[str, str]:
    """
    nuts_muni_region.csv is hierarchical with LEVEL.
    We map municipality entries (LEVEL=3) to the nearest parent region.
    Many exports use REGION at LEVEL=1; if yours uses LEVEL=2, set env var NUTS_REGION_LEVEL=2.
    """
    import os
    region_level = int(os.environ.get("NUTS_REGION_LEVEL", "1"))

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df = df.sort_values("SEQUENCE")

    current_region = None
    muni_to_region: Dict[str, str] = {}

    for _, row in df.iterrows():
        lvl = int(row["LEVEL"])
        title = str(row["TITLE"]).strip()

        if lvl == region_level:
            current_region = title
        elif lvl == 3 and current_region is not None:
            muni_to_region[muni_key(title)] = current_region

    log(f"NUTS mapping: {len(muni_to_region):,} municipalities mapped to LEVEL-{region_level} regions")
    return muni_to_region


# -----------------------------
# Dataset-specific loaders
# -----------------------------

def load_income_indkf132(path: Path) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": "income_dkk"})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = df.drop_duplicates("muni_key", keep="first")
    if df["income_dkk"].notna().any():
        log(f"Income: {df['income_dkk'].notna().sum():,} values, range {df['income_dkk'].min():.0f}–{df['income_dkk'].max():.0f}")
    return df[["muni_key", "municipality", "income_dkk"]]


def load_population_folk1a(path: Path) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": "population"})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = df.drop_duplicates("muni_key", keep="first")
    if df["population"].notna().any():
        log(f"Population: {df['population'].notna().sum():,} values, range {df['population'].min():.0f}–{df['population'].max():.0f}")
    return df[["muni_key", "municipality", "population"]]


def load_tax_pskat(path: Path) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": "tax_rate_pct"})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = df.drop_duplicates("muni_key", keep="first")

    # Auto-fix if rates are e.g. 2336..2630 (basis points) instead of 23.36..26.30
    med = float(df["tax_rate_pct"].median(skipna=True)) if df["tax_rate_pct"].notna().any() else 0.0
    if med > 100:
        df["tax_rate_pct"] = df["tax_rate_pct"] / 100.0

    if df["tax_rate_pct"].notna().any():
        log(f"Tax: {df['tax_rate_pct'].notna().sum():,} values, range {df['tax_rate_pct'].min():.2f}–{df['tax_rate_pct'].max():.2f}")
    return df[["muni_key", "municipality", "tax_rate_pct"]]


def load_earnings_ligelb1(path: Path, colname: str) -> pd.DataFrame:
    df = read_statbank_pivot_csv(path).rename(columns={"name": "municipality", "value": colname})
    df["muni_key"] = df["municipality"].map(muni_key)
    df = df.drop_duplicates("muni_key", keep="first")
    if df[colname].notna().any():
        log(f"{colname}: {df[colname].notna().sum():,} values, range {df[colname].min():.0f}–{df[colname].max():.0f}")
    return df[["muni_key", "municipality", colname]]


def load_property_bm010(path: Path) -> pd.DataFrame:
    """
    BM010.csv can be "pivot" with two numeric columns (house / flat).
    Strategy:
      - read rows
      - use the last two numeric cells in the row
      - municipality label is nearest text cell to the left of those
      - prefer the first numeric (often house) else second
    """
    raw = _read_text_best_effort(path)
    rows = list(csv.reader(raw.splitlines()))

    data: List[Tuple[str, float]] = []

    for r in rows:
        if not r or len(r) < 3:
            continue

        # find last two numeric cells
        nums = []
        num_idxs = []
        for j in range(len(r) - 1, -1, -1):
            v = parse_danish_number(r[j])
            if not np.isnan(v):
                nums.append(v)
                num_idxs.append(j)
            if len(nums) == 2:
                break
        if not nums:
            continue
        nums = list(reversed(nums))
        num_idxs = list(reversed(num_idxs))

        val_idx_left = num_idxs[0]
        # find municipality text to the left
        muni = ""
        for j in range(val_idx_left - 1, -1, -1):
            cell = str(r[j]).strip().strip('"')
            if not cell:
                continue
            if cell.lower() in {"total", "i-alt", "i alt"}:
                muni = ""
                break
            muni = cell
            break
        if not muni:
            continue

        v1 = nums[0]
        v2 = nums[1] if len(nums) > 1 else np.nan
        price = v1 if not np.isnan(v1) else v2
        if np.isnan(price):
            continue

        data.append((muni, float(price)))

    df = pd.DataFrame(data, columns=["municipality", "property_price_dkk_m2"])
    df["muni_key"] = df["municipality"].map(muni_key)
    df = df.drop_duplicates("muni_key", keep="first")

    # Sanity clean: values should be plausible DKK/m²
    df.loc[(df["property_price_dkk_m2"] < 2000) | (df["property_price_dkk_m2"] > 200000), "property_price_dkk_m2"] = np.nan

    if df["property_price_dkk_m2"].notna().any():
        log(f"Property: {df['property_price_dkk_m2'].notna().sum():,} values, range {df['property_price_dkk_m2'].min():.0f}–{df['property_price_dkk_m2'].max():.0f}")
    return df[["muni_key", "municipality", "property_price_dkk_m2"]]


def load_rent_husleje_table7(path: Path) -> pd.DataFrame:
    """
    Huslejestatistik_2024_TABLE7.csv is typically a normal table export.
    We auto-detect separator and then parse numeric columns with parse_danish_number.
    """
    # sep autodetect
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    cols = list(df.columns)
    if len(cols) < 3:
        # fallback: latin1
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
        cols = list(df.columns)

    # Rename by position for robustness
    df = df.rename(
        columns={
            cols[0]: "municipality",
            cols[1]: "all_dwellings",
            cols[2]: "rent_all_dkk_m2",
        }
    )
    # Some variants include family-specific columns at positions 4 and 5
    if len(cols) >= 6:
        df = df.rename(columns={cols[4]: "family_dwellings", cols[5]: "rent_family_dkk_m2"})
    else:
        df["rent_family_dkk_m2"] = np.nan

    for c in ["all_dwellings", "rent_all_dkk_m2", "family_dwellings", "rent_family_dkk_m2"]:
        if c in df.columns:
            df[c] = df[c].apply(parse_danish_number)

    df["municipality"] = df["municipality"].astype(str).str.strip()
    df = df[~df["municipality"].isin(["I-alt", "Total", "I alt", "I-alt "])].copy()

    # Prefer family rent if available, else all rent
    df["rent_dkk_m2"] = df["rent_family_dkk_m2"].where(df["rent_family_dkk_m2"].notna(), df["rent_all_dkk_m2"])

    # Sanity clean: rent should be plausible; drop absurd values and refill later
    df.loc[(df["rent_dkk_m2"] < 100) | (df["rent_dkk_m2"] > 5000), "rent_dkk_m2"] = np.nan

    df["muni_key"] = df["municipality"].map(muni_key)
    df = df.drop_duplicates("muni_key", keep="first")

    if df["rent_dkk_m2"].notna().any():
        log(
            f"Rent: {df['rent_dkk_m2'].notna().sum():,} values, range "
            f"{df['rent_dkk_m2'].min():.0f}–{df['rent_dkk_m2'].max():.0f}"
        )
    return df[["muni_key", "municipality", "rent_dkk_m2"]]


def _pick_header_and_row(rows: List[List[str]], target_year: int, min_groups: int = 4) -> Tuple[List[str], List[str]]:
    """
    For group-level exports (LABY32/LABY45/HUS1-like), pick:
      - a header row with many non-empty text labels
      - a data row for target_year with many numeric cells
    """
    year_patterns = [
        re.compile(rf"^{target_year}$"),
        re.compile(rf"^{target_year}Q[1-4]$"),
        re.compile(rf"^{target_year}K[1-4]$"),
    ]

    # Candidate header rows: lots of text, few numbers
    def row_text_score(r: List[str]) -> int:
        score = 0
        for c in r:
            c2 = str(c).strip().strip('"')
            if not c2:
                continue
            if np.isnan(parse_danish_number(c2)):
                score += 1
        return score

    header_cands = [r for r in rows if row_text_score(r) >= min_groups]
    header = max(header_cands, key=row_text_score, default=[])

    # Candidate data rows: begins with a year token and has numeric cells
    data_cands = []
    for r in rows:
        if not r:
            continue
        first = str(r[0]).strip().strip('"')
        if any(p.match(first) for p in year_patterns):
            numeric_count = sum(0 if np.isnan(parse_danish_number(c)) else 1 for c in r[1:])
            data_cands.append((numeric_count, r))

    data = max(data_cands, key=lambda t: t[0], default=(0, []))[1]
    return header, data


def load_laby32(path: Path, target_year: int = 2024) -> pd.DataFrame:
    """LABY32: Rent indices for housing (2021=100) by municipality groups.

    The StatBank export has the year/quarter token in the 2nd column (e.g. "2024Q2"),
    so the older generic group parser (which assumed the year token is in column 1)
    would fail and yield a single bar / empty data.
    """
    raw = _read_text_best_effort(path)
    rows = list(csv.reader(raw.splitlines()))

    groups = [
        "Capital municipalities",
        "Metropolitan municipalities",
        "Provincial municipalities",
        "Commuter municipalities",
        "Rural municipalities",
    ]

    # Find a header row containing the expected group labels and record their indices.
    header_idx_map: dict[str, int] = {}
    for r in rows:
        toks = [str(c).strip().strip('"') for c in r]
        found = {g: i for i, c in enumerate(toks) if c in groups for g in [c]}
        if len(found) >= 4:
            header_idx_map = found
            break

    if not header_idx_map:
        log(f"LABY32: failed to find group header row in {path.name}")
        return pd.DataFrame(columns=["group", "rent_index_2021_100"])

    group_cols = [(g, header_idx_map[g]) for g in groups if g in header_idx_map]

    # Find the best-matching data row for target_year (any cell contains the year) that
    # has numeric entries in the group columns.
    best_vals: list[float] | None = None
    for r in rows:
        toks = [str(c).strip().strip('"') for c in r]
        if not any(str(target_year) in t for t in toks if t):
            continue
        vals = []
        numeric = 0
        for _g, idx in group_cols:
            v = parse_danish_number(r[idx]) if idx < len(r) else np.nan
            vals.append(v)
            if not np.isnan(v):
                numeric += 1
        if numeric >= 4:
            best_vals = vals
            # keep scanning; later rows often correspond to later quarters

    if best_vals is None:
        log(f"LABY32: no data row found for year {target_year} in {path.name}")
        return pd.DataFrame(columns=["group", "rent_index_2021_100"])

    df = pd.DataFrame(
        {
            "group": [g for g, _ in group_cols],
            "rent_index_2021_100": best_vals[: len(group_cols)],
        }
    ).dropna()

    # Auto-fix if values look like 1067 instead of 106.7
    if not df.empty:
        vmax = float(df["rent_index_2021_100"].max())
        if vmax > 500:
            for div in (10, 100):
                test = df["rent_index_2021_100"] / div
                if float(test.max()) < 200:
                    df["rent_index_2021_100"] = test
                    break

    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"] != ""].copy()
    if df.empty:
        log(f"LABY32: parsed 0 groups from {path.name}")
    else:
        log(f"LABY32: parsed {len(df)} groups; range {df['rent_index_2021_100'].min():.1f}–{df['rent_index_2021_100'].max():.1f}")
        log("  LABY32 groups: " + ", ".join(df["group"].astype(str).tolist()))
    return df


def load_laby45(path: Path, target_year: int = 2024) -> pd.DataFrame:
    """LABY45: Share of population living for rent by municipality groups."""
    raw = _read_text_best_effort(path)
    rows = list(csv.reader(raw.splitlines()))

    groups = [
        "Capital municipalities",
        "Metropolitan municipalities",
        "Provincial municipalities",
        "Commuter municipalities",
        "Rural municipalities",
    ]

    header_idx_map: dict[str, int] = {}
    for r in rows:
        toks = [str(c).strip().strip('"') for c in r]
        found = {g: i for i, c in enumerate(toks) if c in groups for g in [c]}
        if len(found) >= 4:
            header_idx_map = found
            break

    if not header_idx_map:
        log(f"LABY45: failed to find group header row in {path.name}")
        return pd.DataFrame(columns=["group", "share_renters_pct"])

    group_cols = [(g, header_idx_map[g]) for g in groups if g in header_idx_map]

    best_vals: list[float] | None = None
    for r in rows:
        toks = [str(c).strip().strip('"') for c in r]
        if not toks:
            continue
        if not any(str(target_year) == t for t in toks):
            continue
        vals = []
        numeric = 0
        for _g, idx in group_cols:
            v = parse_danish_number(r[idx]) if idx < len(r) else np.nan
            vals.append(v)
            if not np.isnan(v):
                numeric += 1
        if numeric >= 4:
            best_vals = vals
            break

    if best_vals is None:
        log(f"LABY45: no data row found for year {target_year} in {path.name}")
        return pd.DataFrame(columns=["group", "share_renters_pct"])

    df = pd.DataFrame({"group": [g for g, _ in group_cols], "share_renters_pct": best_vals[: len(group_cols)]}).dropna()

    # Scaling fixes
    if not df.empty:
        vmax = float(df["share_renters_pct"].max())
        # If data looks like fractions (0..1), convert to percent
        if vmax <= 1.5:
            df["share_renters_pct"] = df["share_renters_pct"] * 100.0
            vmax = float(df["share_renters_pct"].max())
        # If data looks like 5710 => 57.10
        if vmax > 200:
            for div in (10, 100):
                test = df["share_renters_pct"] / div
                if float(test.max()) <= 100.5:
                    df["share_renters_pct"] = test
                    break

        df["share_renters_pct"] = df["share_renters_pct"].clip(lower=0, upper=100)

    df["group"] = df["group"].astype(str).str.strip()
    df = df[df["group"] != ""].copy()

    if df.empty:
        log(f"LABY45: no group data parsed from {path.name}")
    else:
        log(f"LABY45: parsed {len(df)} groups; range {df['share_renters_pct'].min():.1f}–{df['share_renters_pct'].max():.1f}")
        log("  LABY45 groups: " + ", ".join(df["group"].astype(str).tolist()))
    return df


def load_hus1(path: Path) -> pd.DataFrame:
    """
    HUS1: region-level private rent index.
    We extract any row that contains a "Region ..." label and a numeric.
    """
    raw = _read_text_best_effort(path)
    rows = list(csv.reader(raw.splitlines()))
    data = []
    for r in rows:
        if not r:
            continue
        region = None
        for c in r:
            c2 = str(c).strip().strip('"')
            if c2.lower().startswith("region "):
                region = c2
                break
        if not region:
            continue
        # Prefer the numeric value immediately after the region label (Privately owned housing).
        # Fall back to the last numeric if indexing fails.
        idx_region = next((i for i, c in enumerate(r) if str(c).strip().strip('"').lower().startswith('region ')), None)
        val = np.nan
        if idx_region is not None and idx_region + 1 < len(r):
            val = parse_danish_number(r[idx_region + 1])
        if np.isnan(val):
            nums = [parse_danish_number(c) for c in r]
            nums = [x for x in nums if not np.isnan(x)]
            if not nums:
                continue
            val = float(nums[-1])
        if val > 300:
            for div in (10, 100):
                if val / div < 200:
                    val = val / div
                    break
        data.append((region, val))
    df = pd.DataFrame(data, columns=["region", "rent_index_private_2021_100"]).dropna()
    if df.empty:
        log(f"HUS1: no region data parsed from {path.name}")
    else:
        log(f"HUS1: parsed {len(df)} regions; range {df['rent_index_private_2021_100'].min():.1f}–{df['rent_index_private_2021_100'].max():.1f}")
    return df


# -----------------------------
# Master table build
# -----------------------------

def build_master_table(root: Path) -> DataBundle:
    muni_to_region = build_muni_to_region_from_nuts(root / "nuts_muni_region.csv")

    income = load_income_indkf132(root / "INDKF132.csv")
    pop = load_population_folk1a(root / "FOLK1A.csv")
    tax = load_tax_pskat(root / "PSKAT.csv")
    rent = load_rent_husleje_table7(root / "Huslejestatistik_2024_TABLE7.csv")
    prop = load_property_bm010(root / "BM010.csv")
    earn_res = load_earnings_ligelb1(root / "LIGELB1_residence.csv", "earnings_residence_dkk")
    earn_work = load_earnings_ligelb1(root / "LIGELB1_workplace.csv", "earnings_workplace_dkk")

    # Base index: municipality keys from NUTS mapping (ensures full coverage)
    base = pd.DataFrame({"muni_key": sorted(muni_to_region.keys())})

    muni = base.merge(income.drop(columns=["municipality"]), on="muni_key", how="left")
    log(f"Merge income: rows={len(muni):,}, missing income={muni['income_dkk'].isna().sum():,}")

    for df, label, col in [
        (pop, "population", "population"),
        (tax, "tax", "tax_rate_pct"),
        (rent, "rent", "rent_dkk_m2"),
        (prop, "property", "property_price_dkk_m2"),
        (earn_res, "earnings_residence", "earnings_residence_dkk"),
        (earn_work, "earnings_workplace", "earnings_workplace_dkk"),
    ]:
        muni = muni.merge(df.drop(columns=["municipality"]), on="muni_key", how="left")
        log(f"Merge {label}: missing {col}={muni[col].isna().sum():,}")

    # Attach best-available municipality name
    name_sources = [rent, prop, income, tax, pop, earn_res, earn_work]
    name_map = {}
    for src in name_sources:
        for k, v in src[["muni_key", "municipality"]].dropna().itertuples(index=False):
            if k not in name_map and isinstance(v, str) and v.strip():
                name_map[k] = v.strip()
    muni["municipality"] = muni["muni_key"].map(name_map)

    muni["region"] = muni["muni_key"].map(muni_to_region)

    # Fill missing rent/property by regional average (and record which rows were filled)
    for col in ["rent_dkk_m2", "property_price_dkk_m2"]:
        reg_avg = muni.groupby("region")[col].mean()
        missing_mask = muni[col].isna()
        before = int(missing_mask.sum())

        muni[col] = muni.apply(
            lambda r: reg_avg.get(r["region"], np.nan) if pd.isna(r[col]) else r[col],
            axis=1,
        )

        after = int(muni[col].isna().sum())
        filled_mask = missing_mask & muni[col].notna()
        muni[f"{col}_filled_from_region"] = filled_mask

        log(f"Filled {col} with regional averages: missing {before} -> {after}")

        if before:
            filled_rows = muni.loc[filled_mask, ["municipality", "muni_key", "region", col]].sort_values(["region", "municipality"])
            log(f"  Filled {int(filled_mask.sum())} municipalities for {col} (showing up to 10):")
            for _, r in filled_rows.head(10).iterrows():
                log(f"    - {r['municipality']} (key={r['muni_key']}, {r['region']}): {r[col]}")
            if int(filled_mask.sum()) > 10:
                log("    - ...")

    # Derived metrics
    muni["earnings_net_dkk"] = muni["earnings_workplace_dkk"] - muni["earnings_residence_dkk"]

    muni.loc[(muni["rent_dkk_m2"] <= 0) | muni["rent_dkk_m2"].isna(), "income_to_rent"] = np.nan
    muni.loc[muni["rent_dkk_m2"].notna() & (muni["rent_dkk_m2"] > 0), "income_to_rent"] = muni["income_dkk"] / muni["rent_dkk_m2"]

    keep_cols = [
        "muni_key", "municipality", "region",
        "income_dkk", "population", "tax_rate_pct",
        "rent_dkk_m2", "property_price_dkk_m2",
        "earnings_residence_dkk", "earnings_workplace_dkk", "earnings_net_dkk",
        "income_to_rent",
    ]
    muni = muni[keep_cols]

    log(f"Master table rows: {len(muni):,}")
    log("Non-null counts: " + ", ".join([f"{c}={muni[c].notna().sum():,}" for c in keep_cols if c not in {"muni_key","municipality","region"}]))

    datasets = {
        "income": income,
        "population": pop,
        "tax": tax,
        "rent": rent,
        "property": prop,
        "earnings_residence": earn_res,
        "earnings_workplace": earn_work,
    }
    return DataBundle(muni=muni, muni_to_region=muni_to_region, datasets=datasets)


# -----------------------------
# Serialization
# -----------------------------

def _df_to_records(df: pd.DataFrame) -> List[dict]:
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v) if not np.isnan(v) else None
            elif pd.isna(v):
                v = None
            clean[k] = v
        out.append(clean)
    return out


def save_bundle_to_json(bundle: DataBundle, outpath: Path, extra: dict | None = None, extra_datasets: dict[str, pd.DataFrame] | None = None) -> None:
    ensure_dir(outpath.parent)
    payload = {
        "meta": {
            "generated_utc": pd.Timestamp.utcnow().isoformat(),
            **(extra or {}),
        },
        "muni_to_region": bundle.muni_to_region,
        "muni_master": _df_to_records(bundle.muni),
        "datasets": {k: _df_to_records(v) for k, v in {**bundle.datasets, **(extra_datasets or {})}.items()},
    }
    outpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote JSON cache: {outpath}")


def load_bundle_from_json(path: Path) -> DataBundle:
    payload = json.loads(path.read_text(encoding="utf-8"))
    muni_to_region = {str(k): str(v) for k, v in payload["muni_to_region"].items()}

    muni = pd.DataFrame(payload["muni_master"])
    datasets = {k: pd.DataFrame(v) for k, v in payload.get("datasets", {}).items()}
    return DataBundle(muni=muni, muni_to_region=muni_to_region, datasets=datasets)


"""utils.py

Shared helpers: logging, name normalization, numeric parsing, styling.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import matplotlib as mpl


def log(msg: str) -> None:
    print(f"[cost_viz] {msg}")


def ensure_dir(p: Path) -> None:
    """Create directory if needed.

    Guards against passing a *file* path (e.g. config.py) by mistake.
    """
    if p.suffix:
        raise ValueError(f"ensure_dir() called on file path: {p}")
    p.mkdir(parents=True, exist_ok=True)


# --- Municipality name normalization ---

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


def muni_key(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = _ENGLISH_TO_DANISH.get(s.lower(), s)
    s = s.translate(_SPECIAL).lower()
    s = re.sub(r"[\s\-\–\—/]+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# --- Numeric parsing (Danish-ish exports) ---

def parse_danish_number(x) -> float:
    """Parse numbers that may appear as:
      - 1.016  -> 1016        (thousands separator '.')
      - 106.7  -> 106.7       (decimal '.')
      - 1.234,5 -> 1234.5     (thousands '.', decimal ',')
      - 23,6%  -> 23.6
      - '..' / '' -> NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in {"..", ".", "", "nan", "NaN", "None"}:
        return np.nan

    s = s.replace("%", "").strip()
    s = s.replace(" ", "")

    # Guardrail: tokens with alphabetic characters (e.g. "2024Q2", "Region ...")
    # are not numbers. The older fallback stripped letters and could turn "2024Q2" -> 20242.
    if re.search(r"[A-Za-z]", s):
        return np.nan

    # If both separators exist: assume Danish thousands '.' and decimal ','
    if "." in s and "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            pass

    # Only comma: decimal comma
    if "," in s and "." not in s:
        s2 = s.replace(",", ".")
        try:
            return float(s2)
        except Exception:
            pass

    # Only dot: could be thousands or decimal dot.
    if "." in s and "," not in s:
        parts = s.split(".")
        if len(parts) == 2:
            left, right = parts
            # Heuristic: exactly 3 digits after dot => thousands separator
            if right.isdigit() and len(right) == 3 and left.replace("-", "").isdigit():
                s2 = left + right
                try:
                    return float(s2)
                except Exception:
                    pass
            # Otherwise treat as decimal dot
            try:
                return float(s)
            except Exception:
                pass
        else:
            # multiple dots => thousands separators
            s2 = s.replace(".", "")
            try:
                return float(s2)
            except Exception:
                pass

    # Fallback: strip non-numeric
    s2 = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s2) if s2 else np.nan
    except Exception:
        return np.nan


def set_matplotlib_style(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
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

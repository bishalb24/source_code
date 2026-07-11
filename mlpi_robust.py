"""
MLPI robust four-dimensional junction-link prioritization.

Current workflow:
  1. Build Nepal national-highway graph from Nepal.gpkg layer NH.
  2. Build exhaustive non-overlapping links between road junctions/dead ends.
  3. Load the externally generated ME2 OD matrix and map its district/HQ zones
     onto the road graph.
  4. Remove each link and score road-only Physical, Social and Economic
     consequences plus road access to operational airports.
  5. Render clean maps for each dimension plus the combined MLPI ranking.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import re
import subprocess
import textwrap
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from shapely.geometry import LineString, MultiLineString, Point, Polygon, shape
from shapely.ops import polygonize, unary_union

try:
    import fiona
    from pyproj import CRS
except Exception:  # pragma: no cover - notebook users get a clearer error later
    fiona = None
    CRS = None


# Nepal DoR/SNH source CRS used by the supplied GeoPackages.
NEPAL_TM_PROJ4 = (
    "+proj=tmerc +lat_0=0 +lon_0=84 +k=0.9999 "
    "+x_0=500000 +y_0=0 +a=6377276.345 +b=6356075.413 +units=m +no_defs"
)

PROJECT_DIR = Path("/Users/bishalbhurtel/Desktop/Project")
DEFAULT_INPUT = PROJECT_DIR / "Input.xlsx"
DEFAULT_GPKG_DIR = PROJECT_DIR / "1. Geopkg"
DEFAULT_MASTER_GPKG = PROJECT_DIR / "Nepal.gpkg"
DEFAULT_OUT_DIR = PROJECT_DIR / "output"
DEFAULT_ME2_OD_MATRIX = PROJECT_DIR / "OD" / "outputs" / "od_matrix_names.csv"
DEFAULT_AIRPORT_RTFD = PROJECT_DIR / "Data" / "Airport Table.rtfd"
AIRPORT_REGISTRY_MATCH_KM = 40.0
MAP1_KEY_PLACES = {"Butwal"}
DEFAULT_TERRAIN_SPEED_KMH = {
    "Terai": 60.0,
    "Hills": 40.0,
    "Mountains": 20.0,
}

BUILT_IN_CITY_LOOKUP = [
    (85.316, 27.698, "Kathmandu", 1140), (85.316, 27.665, "Lalitpur", 340),
    (83.986, 28.210, "Pokhara", 518), (84.429, 27.702, "Bharatpur", 395),
    (84.877, 27.014, "Birgunj", 276), (87.272, 26.453, "Biratnagar", 245),
    (83.449, 27.701, "Butwal", 178), (85.034, 27.429, "Hetauda", 130),
    (81.617, 28.050, "Nepalgunj", 112), (85.926, 26.728, "Janakpur", 153),
    (87.274, 26.661, "Itahari", 198), (80.599, 28.682, "Dhangadhi", 203),
    (82.183, 29.274, "Jumla", 30), (81.622, 28.600, "Surkhet", 72),
    (88.082, 26.571, "Kakarbhitta", 78), (85.910, 26.840, "Rajbiraj", 45),
    (87.900, 26.645, "Birtamod", 55), (85.500, 26.800, "Lalbandi", 30),
    (83.449, 27.506, "Bhairahawa", 60), (84.980, 27.160, "Simara", 40),
    (80.548, 29.292, "Dadeldhura", 20), (81.197, 29.462, "Dipayal", 8),
    (81.819, 29.971, "Simikot", 4), (82.298, 28.554, "Chaurjahari", 6),
    (82.481, 28.608, "Rukumkot", 5), (82.895, 28.997, "Dunai", 5),
    (81.706, 29.076, "Kalikot", 5), (82.011, 29.717, "Gamgadhi", 4),
    (83.723, 28.781, "Jomsom", 8), (83.056, 27.574, "Kapilvastu", 22),
    (84.600, 28.100, "Gorkha", 20), (82.600, 28.350, "Baglung", 30),
    (83.100, 28.050, "Tansen", 22), (84.476, 27.874, "Mugling", 18),
    (83.980, 28.700, "Lamjung", 10), (83.440, 28.490, "Syangja", 15),
    (81.190, 29.600, "Bajhang", 12), (87.320, 27.050, "Dhankuta", 40),
    (87.650, 27.080, "Basantapur", 4), (86.500, 26.998, "Gaighat", 8),
    (85.650, 27.150, "Kavrepalanchok", 30), (85.570, 27.950, "Melamchi", 20),
    (84.020, 28.663, "Manang", 5), (86.714, 27.807, "Namche", 6),
    (85.292, 28.098, "Dhunche", 5), (84.897, 27.910, "Malekhu", 4),
    (84.847, 27.880, "Benighat", 3), (84.983, 27.681, "Naubise", 6),
    (85.545, 27.622, "Dhulikhel", 20), (85.830, 27.470, "Nepalthok", 4),
    (85.985, 27.323, "Khurkot", 5), (85.912, 27.256, "Sindhuli", 25),
    (85.890, 26.999, "Bardibas", 38), (87.292, 26.812, "Dharan", 75),
    (87.333, 27.188, "Hile", 7),
    (87.100, 27.300, "Bhojpur", 15), (87.193, 27.315, "Tumlingtar", 4),
    (87.695, 27.351, "Taplejung", 8), (86.730, 27.687, "Lukla", 4),
    (88.080, 26.571, "Bhadrapur", 20),
]

# Named corridor pairs can be used when an explicit section table is supplied.
# The current MLPI scoring is based on generated junction-to-junction links.
NAMED_SECTION_SPECS = [
    ("Dhangadhi", "Dadeldhura"),
    ("Dhangadhi", "Nepalgunj"),
    ("Jumla", "Kalikot"),
    ("Butwal", "Bhairahawa"),
    ("Butwal", "Tansen"),
    ("Tansen", "Pokhara"),
    ("Pokhara", "Syangja"),
    ("Pokhara", "Gorkha"),
    ("Gorkha", "Mugling"),
    ("Pokhara", "Mugling"),
    ("Bharatpur", "Butwal"),
    ("Bharatpur", "Mugling"),
    ("Bharatpur", "Hetauda"),
    ("Hetauda", "Birgunj"),
    ("Birgunj", "Simara"),
    ("Birgunj", "Lalbandi"),
    ("Lalbandi", "Bardibas"),
    ("Mugling", "Naubise"),
    ("Naubise", "Kathmandu"),
    ("Mugling", "Naubise", "Kathmandu"),
    ("Bharatpur", "Mugling", "Naubise", "Kathmandu"),
    ("Kathmandu", "Dhulikhel"),
    ("Dhulikhel", "Khurkot"),
    ("Khurkot", "Bardibas"),
    ("Bardibas", "Janakpur"),
    ("Bardibas", "Gaighat"),
    ("Bardibas", "Itahari"),
    ("Itahari", "Biratnagar"),
    ("Itahari", "Birtamod"),
    ("Birtamod", "Kakarbhitta"),
    ("Birtamod", "Taplejung"),
    ("Rajbiraj", "Gaighat"),
    ("Gaighat", "Dharan"),
    ("Dharan", "Dhankuta"),
    ("Dhankuta", "Hile"),
    ("Kathmandu", "Melamchi"),
    ("Kathmandu", "Dhunche"),
]


# ----------------------------- configuration ---------------------------------


@dataclass
class MLPIConfig:
    input_xlsx: Path = DEFAULT_INPUT
    gpkg_dir: Path = DEFAULT_GPKG_DIR
    master_gpkg_path: Path = DEFAULT_MASTER_GPKG
    out_dir: Path = DEFAULT_OUT_DIR
    od_matrix_path: Path = DEFAULT_ME2_OD_MATRIX
    od_matrix_source: str = "ME2 calibrated OD matrix"
    seed: int = 42
    gen_maps: bool = True
    gen_figs: bool = True
    top_n: int = 20

    road_min_km_score: float = 0.0
    road_min_km_plot: float = 0.0
    road_ref_filter: str = "ALL"
    road_ref_filter_source: str = "code default: ALL for Nepal.gpkg NH"
    road_aadt_field: str = "AADT_2023"
    road_future_aadt_field: str = "AADT_2024_25"
    allow_input_aadt_gap_fill: bool = False
    mean_speed_factor: float = 0.40
    endpoint_precision: int = 5
    topology_connector_max_km: float = 0.50

    terr_grid_deg: float = 0.10
    speed_road: float = 40.0
    air_cost_multiplier: float = 10.0
    speed_air: float = 400.0
    air_fixed_penalty_h: float = 4.0
    section_anchor_max_km: float = 20.0
    section_max_detour_ratio: float = 3.5
    section_max_length_km: float = 300.0

    airport_access_km: float = 5.0

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "physical": 0.35,
            "social": 0.25,
            "economic": 0.25,
            "interconnected": 0.15,
        }
    )
    w_s1: float = 0.40
    w_s2: float = 0.60
    w_s3: float = 0.0
    w_p_fftdi: float = 0.70
    w_p_flow: float = 0.30
    w_e1: float = 0.46
    w_e2: float = 0.54
    w_e3: float = 0.0
    w_i1: float = 0.35
    w_i2: float = 0.65
    w_i3: float = 0.0

    beta: float = 1.5
    alpha_aadt: float = 0.40
    e2_scale: float = 0.10
    e2_regional_balance: bool = True
    od_deterrence_h: float = 8.0
    detour_full_credit_ratio: float = 2.0
    sett_window_km: float = 25.0
    station_match_km: float = 5.0
    fw_max_iter: int = 200
    assignment_rgap_target: float = 0.001
    bpr_alpha: float = 0.0
    bpr_beta_power: float = 4.0
    socioeconomic_nodes_per_district: int = 24
    disconnected_penalty_h: float = 24.0

    aadt_rows: List[Dict[str, Any]] = field(default_factory=list)
    border_trade: List[Tuple[float, float, float, str]] = field(default_factory=list)
    airport_registry: List[Tuple[float, float, str, str]] = field(default_factory=list)
    helipads: List[Tuple[float, float, str, str]] = field(default_factory=list)
    airport_status_lookup: Dict[str, str] = field(default_factory=dict)
    airport_registry_source: str = ""
    population_gdp: pd.DataFrame = field(default_factory=pd.DataFrame)
    place_lookup: List[Tuple[float, float, str, float]] = field(default_factory=list)
    population_by_name: Dict[str, float] = field(default_factory=dict)
    gdp_by_name: Dict[str, float] = field(default_factory=dict)
    named_place_source: str = "built-in city lookup"
    section_specs: List[Tuple[str, ...]] = field(default_factory=lambda: list(NAMED_SECTION_SPECS))
    section_specs_source: str = "built-in named section specs"
    formula_index_rows: List[Dict[str, Any]] = field(default_factory=list)
    formula_index_source: str = "code default formula index"
    od_zone_mapping_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    od_matrix_total_demand: float = 0.0
    od_pairs_loaded: int = 0
    od_pairs_same_node_skipped: int = 0
    od_same_node_demand_skipped: float = 0.0
    od_baseline_connected_pairs: int = 0
    od_baseline_connected_demand: float = 0.0
    od_baseline_disconnected_pairs: int = 0
    od_baseline_disconnected_demand: float = 0.0

    aadt_default: float = 3500.0
    gdp_default: float = 150000.0
    population_total: float = 1.0


def _is_bad_path(path: str) -> bool:
    return not path or path.startswith("/content/") or path.strip().lower() in {"nan", "none"}


def _read_param_sheet(xlsx: Path, sheet: str) -> Dict[str, Any]:
    df = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        name = row.iloc[0] if len(row) else None
        value = row.iloc[1] if len(row) > 1 else None
        if pd.isna(name) or pd.isna(value):
            continue
        name = str(name).strip()
        if not name or name.lower().startswith("parameter") or name.lower().startswith("sum check"):
            continue
        out[name] = value
        if len(row) > 8 and isinstance(row.iloc[8], str) and row.iloc[8].strip():
            out[row.iloc[8].strip()] = value
    return out


def _param_get(params: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in params:
            return params[name]
    normalized = {re.sub(r"\s+", " ", str(k).replace("—", "-").replace("–", "-")).strip().lower(): v for k, v in params.items()}
    for name in names:
        key = re.sub(r"\s+", " ", str(name).replace("—", "-").replace("–", "-")).strip().lower()
        if key in normalized:
            return normalized[key]
    return default


def _sheet_name(available: Sequence[str], *aliases: str) -> Optional[str]:
    lower = {s.lower(): s for s in available}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _header_score(values: Sequence[Any]) -> int:
    tokens = {
        "station no", "road link", "location", "aadt", "lon", "longitude",
        "lat", "latitude", "district", "population", "province", "gdp",
        "border crossing", "trade volume", "airport name", "icao", "code",
        "helipad name", "formula", "term", "dimension", "source url",
    }
    score = 0
    for value in values:
        text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
        if not text:
            continue
        if text in tokens or any(token in text for token in tokens):
            score += 1
    return score


def _read_table(xlsx: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet, header=3)
    df = df.dropna(how="all").reset_index(drop=True)
    unnamed = sum(str(c).lower().startswith("unnamed") for c in df.columns)
    if len(df) and unnamed >= max(1, len(df.columns) // 2):
        first = df.iloc[0].tolist()
        if _header_score(first) > _header_score(list(df.columns)) and _header_score(first) >= 2:
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(x).strip() if pd.notna(x) else f"col_{i}" for i, x in enumerate(first)]
    while len(df) and df.iloc[-1].isna().sum() >= len(df.columns) - 1:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def _split_place_sequence(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = re.sub(r"\s*(?:->|→|–|—|-|>|;|\|)\s*", "|", text)
    return [p.strip() for p in text.split("|") if p.strip()]


def _read_section_specs(xlsx: Path, sheet: str) -> List[Tuple[str, ...]]:
    specs: List[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for header in (0, 3):
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet, header=header).dropna(how="all")
        except Exception:
            continue
        if df.empty:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        origin_col = next((cols[k] for k in cols if k in {"origin", "from", "start", "origin_place"}), None)
        destination_col = next((cols[k] for k in cols if k in {"destination", "to", "end", "destination_place"}), None)
        via_col = next((cols[k] for k in cols if k in {"via", "waypoints", "intermediate", "through"}), None)
        section_col = next((cols[k] for k in cols if k in {"section", "section_name", "corridor", "named_section"}), None)
        include_col = next((cols[k] for k in cols if k in {"include", "active", "use"}), None)
        for _, row in df.iterrows():
            if include_col is not None and not _as_bool(row.get(include_col), True):
                continue
            if origin_col is not None and destination_col is not None:
                sequence = _split_place_sequence(row.get(origin_col))
                sequence += _split_place_sequence(row.get(via_col)) if via_col is not None else []
                sequence += _split_place_sequence(row.get(destination_col))
            elif section_col is not None:
                sequence = _split_place_sequence(row.get(section_col))
            else:
                non_empty = [v for v in row.tolist() if not pd.isna(v) and str(v).strip()]
                sequence = _split_place_sequence(non_empty[0]) if non_empty else []
            if len(sequence) < 2:
                continue
            key = tuple(p.lower() for p in sequence)
            if key not in seen:
                specs.append(tuple(sequence))
                seen.add(key)
        if specs:
            break
    return specs


def _read_formula_index(xlsx: Path, sheet: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for header in (0, 3):
        try:
            df = pd.read_excel(xlsx, sheet_name=sheet, header=header).dropna(how="all")
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in df.columns}
        dim_col = lower.get("dimension")
        term_col = lower.get("term")
        formula_col = lower.get("formula")
        note_col = lower.get("implementation_note") or lower.get("implementation note") or lower.get("notes")
        if not (dim_col and term_col and formula_col):
            continue
        for _, row in df.iterrows():
            term = row.get(term_col)
            formula = row.get(formula_col)
            if pd.isna(term) or pd.isna(formula):
                continue
            rows.append(
                {
                    "Dimension": str(row.get(dim_col, "")).strip(),
                    "Term": str(term).strip(),
                    "Formula": str(formula).strip(),
                    "Implementation_note": "" if note_col is None or pd.isna(row.get(note_col)) else str(row.get(note_col)).strip(),
                }
            )
        if rows:
            break
    return rows


def _plain_rtfd_text(rtfd_path: Path) -> str:
    """Extract plain text from the local RTFD airport table."""
    p = rtfd_path.expanduser()
    if p.is_dir():
        p = p / "TXT.rtf"
    if not p.exists():
        return ""
    try:
        proc = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(p)],
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout
    except Exception:
        raw = p.read_text(errors="ignore")
        raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
        raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw)
        raw = re.sub(r"[{}]", " ", raw)
        return raw


_COORD_RE = re.compile(
    r"(\d{1,3})(?:[^0-9NSEW]+(\d+(?:\.\d+)?))?"
    r"(?:[^0-9NSEW]+(\d+(?:\.\d+)?))?[^0-9NSEW]*([NSEW])",
    re.IGNORECASE,
)


def _dms_to_decimal(deg: str, minute: Optional[str], sec: Optional[str], hemi: str) -> float:
    val = float(deg) + (float(minute) if minute else 0.0) / 60.0 + (float(sec) if sec else 0.0) / 3600.0
    return -val if hemi.upper() in {"S", "W"} else val


def _parse_coord_pair(text: str) -> Optional[Tuple[float, float]]:
    matches = _COORD_RE.findall(text.replace("\xa0", " "))
    lat = lon = None
    for deg, minute, sec, hemi in matches:
        val = _dms_to_decimal(deg, minute or None, sec or None, hemi)
        if hemi.upper() in {"N", "S"}:
            lat = val
        elif hemi.upper() in {"E", "W"}:
            lon = val
    if lat is None or lon is None:
        return None
    return lon, lat


def _clean_airport_text(s: Any) -> str:
    out = str(s).replace("\xa0", " ").strip()
    out = re.sub(r"\[\d+\]", "", out)
    out = re.sub(r"\s+", " ", out)
    return out


def _valid_iata(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}", s.strip())) and s.strip() not in {"NAN", "NAA"}


def _valid_icao(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}", s.strip()))


def _airport_lookup_key(s: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())


def _airport_name_to_code(name: str) -> str:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", re.sub(r"\([^)]*\)", "", name)) if w]
    words = [w for w in words if w.lower() not in {"airport", "international", "domestic", "the"}]
    if not words:
        return "AIR"
    if len(words) >= 2:
        return (words[0][0] + words[1][:2]).upper().ljust(3, "X")[:3]
    return words[0][:3].upper().ljust(3, "X")


def read_airport_rtfd_table(rtfd_path: Path = DEFAULT_AIRPORT_RTFD) -> List[Dict[str, Any]]:
    """Read airport name/IATA/status/coordinates from the user's RTFD table."""
    text = _plain_rtfd_text(rtfd_path)
    if not text:
        return []
    lines = [_clean_airport_text(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    skip = {
        "Airport name",
        "City served",
        "Province",
        "Location",
        "ICAO",
        "IATA",
        "Usage",
        "Runway(s)",
        "Coordinates",
        "Total passengers",
        "Remarks",
        "District",
        "Domestic Airports",
        "Regional Hub Airports",
    }
    records: List[Dict[str, Any]] = []
    section_status = "In operation"
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "In operation Airports":
            section_status = "In operation"
            i += 1
            continue
        if line == "Not in operation Airports":
            section_status = "Not in operation"
            i += 1
            continue
        if line in skip or "Airport" not in line or line.endswith("Airports"):
            i += 1
            continue
        coord_idx = None
        coord = None
        for j in range(i + 1, min(i + 18, len(lines))):
            coord = _parse_coord_pair(lines[j])
            if coord:
                coord_idx = j
                break
        if coord_idx is None or coord is None:
            i += 1
            continue
        fields = lines[i + 1 : coord_idx]
        icao = next((x.strip().upper() for x in fields if _valid_icao(x.strip().upper())), "")
        iata = ""
        if icao and icao in fields:
            start = fields.index(icao) + 1
            iata = next((x.strip().upper() for x in fields[start:] if _valid_iata(x.strip().upper())), "")
        if not iata:
            iata = next((x.strip().upper() for x in fields if _valid_iata(x.strip().upper())), "")
        code = iata or _airport_name_to_code(line)
        city = fields[0] if fields else ""
        province = fields[1] if len(fields) > 1 else ""
        records.append(
            dict(
                lon=coord[0],
                lat=coord[1],
                code=code,
                iata=iata,
                icao=icao,
                name=line,
                city=city,
                province=province,
                status=section_status,
            )
        )
        i = coord_idx + 1
    return records


def _as_bool(v: Any, default: bool = True) -> bool:
    if pd.isna(v):
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in {"yes", "y", "true", "t", "1", "on"}


def _as_list(v: Any, default: Sequence[str]) -> List[str]:
    if pd.isna(v):
        return list(default)
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]


def _normalize_strategy_name(name: str) -> str:
    low = str(name).strip().lower()
    if low in {"hierarchy", "pf_length", "pf-length", "pflength", "hazard-length"}:
        return "PF-Length"
    if low == "quickest":
        return "Quickest"
    if low == "closest":
        return "Closest"
    return str(name).strip()


def _numeric_or_nan(v: Any) -> float:
    try:
        out = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        return float(out) if pd.notna(out) else float("nan")
    except Exception:
        return float("nan")


def _norm_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in d.values())
    if s <= 0:
        return d
    return {k: max(0.0, float(v)) / s for k, v in d.items()}


def load_config(
    input_xlsx: str | Path = DEFAULT_INPUT,
    gpkg_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    od_matrix_path: str | Path | None = None,
) -> MLPIConfig:
    xlsx = Path(input_xlsx).expanduser()
    if not xlsx.exists():
        raise FileNotFoundError(f"Input workbook not found: {xlsx}")

    available = pd.ExcelFile(xlsx).sheet_names
    cfg = MLPIConfig(input_xlsx=xlsx)

    rc_name = _sheet_name(available, "01_Run_Config")
    if rc_name:
        rc = _read_param_sheet(xlsx, rc_name)
        data_dir = str(rc.get("Data directory", cfg.gpkg_dir))
        output_dir = str(rc.get("Output directory", cfg.out_dir))
        master_gpkg_value = _param_get(
            rc,
            "Master GeoPackage",
            "Master GPKG",
            "Road master GeoPackage",
            "Nepal GeoPackage",
            default=cfg.master_gpkg_path,
        )
        od_path_value = _param_get(
            rc,
            "ME2 OD matrix CSV",
            "ME2 OD matrix path",
            "OD matrix CSV",
            "OD matrix path",
            "Calibrated OD matrix CSV",
            default=cfg.od_matrix_path,
        )
        if gpkg_dir is not None:
            cfg.gpkg_dir = Path(gpkg_dir).expanduser()
        elif not _is_bad_path(data_dir) and Path(data_dir).expanduser().is_dir():
            cfg.gpkg_dir = Path(data_dir).expanduser()
        elif DEFAULT_GPKG_DIR.exists():
            cfg.gpkg_dir = DEFAULT_GPKG_DIR
        if not _is_bad_path(str(master_gpkg_value)):
            cfg.master_gpkg_path = Path(master_gpkg_value).expanduser()
        if out_dir is not None:
            cfg.out_dir = Path(out_dir).expanduser()
        elif not _is_bad_path(output_dir):
            cfg.out_dir = Path(output_dir).expanduser()
        else:
            cfg.out_dir = DEFAULT_OUT_DIR
        if od_matrix_path is not None:
            cfg.od_matrix_path = Path(od_matrix_path).expanduser()
        elif not _is_bad_path(str(od_path_value)):
            cfg.od_matrix_path = Path(od_path_value).expanduser()
        cfg.seed = int(rc.get("Random seed", cfg.seed))
        cfg.gen_maps = _as_bool(rc.get("Generate maps", cfg.gen_maps), cfg.gen_maps)
        cfg.gen_figs = _as_bool(rc.get("Generate analytical figures", cfg.gen_figs), cfg.gen_figs)
        cfg.top_n = int(rc.get("Top-N links printed to console", cfg.top_n))
    if gpkg_dir is not None:
        cfg.gpkg_dir = Path(gpkg_dir).expanduser()
    if out_dir is not None:
        cfg.out_dir = Path(out_dir).expanduser()
    if od_matrix_path is not None:
        cfg.od_matrix_path = Path(od_matrix_path).expanduser()

    np_name = _sheet_name(available, "02_Network_Params")
    if np_name:
        np_ = _read_param_sheet(xlsx, np_name)
        # Section paths need the detailed NH geometry.
        cfg.road_min_km_score = min(float(np_.get("Min. road length for MLPI scoring", cfg.road_min_km_score)), 1.0)
        cfg.road_min_km_plot = float(np_.get("Min. road length for background plot", cfg.road_min_km_plot))
        cfg.road_ref_filter = str(_param_get(np_, "Road ref filter", "ROAD_REF_FILTER", default=cfg.road_ref_filter)).strip()
        cfg.road_ref_filter_source = f"Input.xlsx sheet {np_name}"
        cfg.road_aadt_field = str(_param_get(np_, "Road AADT field", "ROAD_AADT_FIELD", default=cfg.road_aadt_field)).strip() or cfg.road_aadt_field
        cfg.road_future_aadt_field = str(_param_get(np_, "Future road AADT field", "ROAD_FUTURE_AADT_FIELD", default=cfg.road_future_aadt_field)).strip() or cfg.road_future_aadt_field
        cfg.mean_speed_factor = float(_param_get(np_, "Mean speed factor", "MEAN_SPEED_FACTOR", default=cfg.mean_speed_factor))
        cfg.allow_input_aadt_gap_fill = _as_bool(
            _param_get(np_, "Allow Input AADT gap fill", "ALLOW_INPUT_AADT_GAP_FILL", default=cfg.allow_input_aadt_gap_fill),
            cfg.allow_input_aadt_gap_fill,
        )
        cfg.speed_road = float(_param_get(np_, "Topology connector speed km/h", "Road free-flow speed km/h", "SPEED_ROAD", default=cfg.speed_road))
        cfg.assignment_rgap_target = float(_param_get(np_, "BFW relative-gap stop", "ASSIGNMENT_RGAP", default=cfg.assignment_rgap_target))
        cfg.fw_max_iter = int(_param_get(np_, "BFW max iterations", "FW_MAX_ITER", default=cfg.fw_max_iter))
        cfg.bpr_alpha = float(_param_get(np_, "BPR congestion alpha", "BPR_ALPHA", default=cfg.bpr_alpha))
        cfg.bpr_beta_power = float(_param_get(np_, "BPR congestion beta power", "BPR_BETA", default=cfg.bpr_beta_power))
        cfg.socioeconomic_nodes_per_district = int(_param_get(np_, "Socioeconomic allocation nodes per district", "SOCIOECONOMIC_NODES_PER_DISTRICT", default=cfg.socioeconomic_nodes_per_district))
        cfg.disconnected_penalty_h = float(_param_get(np_, "Disconnected OD penalty h", "DISCONNECTED_PENALTY_H", default=cfg.disconnected_penalty_h))
        cfg.section_anchor_max_km = float(_param_get(np_, "Named-place anchor tolerance km", "SECTION_ANCHOR_MAX_KM", default=cfg.section_anchor_max_km))
        cfg.section_max_length_km = float(_param_get(np_, "Named-section maximum path length km", "SECTION_MAX_LENGTH_KM", default=cfg.section_max_length_km))
        cfg.section_max_detour_ratio = float(_param_get(np_, "Named-section maximum detour ratio", "SECTION_MAX_DETOUR_RATIO", default=cfg.section_max_detour_ratio))
        cfg.topology_connector_max_km = float(_param_get(np_, "Topology connector maximum gap km", "TOPOLOGY_CONNECTOR_MAX_KM", default=cfg.topology_connector_max_km))

    # The four-dimensional MLPI calculation uses the run-control, network,
    # air-access, weight, coupling, socioeconomic, OD and formula sheets.

    aa_name = _sheet_name(available, "05_Air_Access")
    if aa_name:
        aa = _read_param_sheet(xlsx, aa_name)
        cfg.airport_access_km = float(_param_get(aa, "Airport ground-access radius", default=cfg.airport_access_km))
        cfg.air_cost_multiplier = float(_param_get(aa, "Air cost multiplier", "AIR_COST_MULT", default=cfg.air_cost_multiplier))
        cfg.air_fixed_penalty_h = float(_param_get(aa, "Air fixed penalty h", "AIR_FIXED_PENALTY_H", default=cfg.air_fixed_penalty_h))
        cfg.speed_air = float(_param_get(aa, "Air speed km/h", "SPEED_AIR", default=cfg.speed_air))

    wm_name = _sheet_name(available, "06_MLPI_Weights")
    if wm_name:
        wm = _read_param_sheet(xlsx, wm_name)
        if str(_read_param_sheet(xlsx, rc_name).get("Weight scheme selector", "Tentative")).lower().startswith("ahp"):
            cfg.weights = {
                "physical": float(_param_get(wm, "AHP — Physical", "AHP - Physical", default=0.473)),
                "social": float(_param_get(wm, "AHP — Social", "AHP - Social", default=0.170)),
                "economic": float(_param_get(wm, "AHP — Economic", "AHP - Economic", default=0.284)),
                "interconnected": float(_param_get(wm, "AHP — Interconnected", "AHP - Interconnected", default=0.073)),
            }
        else:
            cfg.weights = {
                "physical": float(_param_get(wm, "Tentative — Physical", "Tentative - Physical", default=0.35)),
                "social": float(_param_get(wm, "Tentative — Social", "Tentative - Social", default=0.25)),
                "economic": float(_param_get(wm, "Tentative — Economic", "Tentative - Economic", default=0.25)),
                "interconnected": float(_param_get(wm, "Tentative — Interconnected", "Tentative - Interconnected", default=0.15)),
            }
        cfg.weights = _norm_weights(cfg.weights)

    sw_name = _sheet_name(available, "07_SubDim_Weights")
    if sw_name:
        sw = _read_param_sheet(xlsx, sw_name)
        cfg.w_p_fftdi = float(_param_get(sw, "Physical P1 — Global FFTDI", "Physical P1 - Global FFTDI", "Physical P1 - AADT detour FFTDI", "w_p_fftdi", default=cfg.w_p_fftdi))
        cfg.w_p_flow = float(_param_get(sw, "Physical P2 — UE flow", "Physical P2 - UE flow", "Physical P2 - AADT exposure", "Physical P2 - detour-weighted OD exposure", "Physical P2 — detour-weighted OD exposure", default=cfg.w_p_flow))
        cfg.w_s1 = float(_param_get(sw, "Social S1 — Newly isolated population", "Social S1 - Newly isolated population", "Social S1 — Vulnerability-weighted isolation", default=cfg.w_s1))
        cfg.w_s2 = float(_param_get(sw, "Social S2 — Healthcare travel-cost increase", "Social S2 - Healthcare travel-cost increase", "Social S2 — Dead-end fraction", default=cfg.w_s2))
        cfg.w_s3 = 0.0
        cfg.w_e1 = float(_param_get(sw, "Economic E1 — OD vehicle-hour/freight loss", "Economic E1 - OD vehicle-hour/freight loss", "Economic E1 - AADT vehicle-hour/freight loss", "Economic E1 — AADT flow loss", default=cfg.w_e1))
        cfg.w_e2 = float(_param_get(sw, "Economic E2 — Border-trade access loss", "Economic E2 - Border-trade access loss", "Economic E2 — Trade-gravity loss", default=cfg.w_e2))
        cfg.w_e3 = 0.0
        cfg.w_i1 = float(_param_get(sw, "Interconnected I1 — Airport generalized-cost increase", "Interconnected I1 - Airport generalized-cost increase", "Interconnected I1 — Airport road-degree loss", default=cfg.w_i1))
        cfg.w_i2 = float(_param_get(sw, "Interconnected I2 - Complete loss of road airport access", "Interconnected I2 — Complete loss of multimodal airport access", "Interconnected I2 - Complete loss of multimodal airport access", "Interconnected I2 — 5 km airport-access indicator", default=cfg.w_i2))
        cfg.w_i3 = 0.0

    cf_name = _sheet_name(available, "08_Coupling_Factors")
    if cf_name:
        cf = _read_param_sheet(xlsx, cf_name)
        cfg.beta = float(_param_get(cf, "Hansen gravity decay β", "Hansen gravity decay beta", "beta", default=cfg.beta))
        cfg.alpha_aadt = float(_param_get(cf, "E2 weight — AADT share of node weight", "E2 weight - AADT share of node weight", default=cfg.alpha_aadt))
        cfg.e2_scale = float(_param_get(cf, "E2 normaliser scale", default=cfg.e2_scale))
        cfg.e2_regional_balance = _as_bool(_param_get(cf, "E2 regional balance", "E2_REGIONAL_BALANCE", default=cfg.e2_regional_balance), cfg.e2_regional_balance)
        cfg.od_deterrence_h = float(_param_get(cf, "Hansen access decay time h", "OD deterrence time h", "OD_DETERRENCE_H", default=cfg.od_deterrence_h))
        cfg.detour_full_credit_ratio = float(_param_get(cf, "Detour full-credit ratio", "DETOUR_FULL_CREDIT_RATIO", default=cfg.detour_full_credit_ratio))
        cfg.sett_window_km = 111.0 * float(_param_get(cf, "Settlement-count proximity window", "SETT_WINDOW", default=0.22))
        cfg.air_cost_multiplier = float(_param_get(cf, "Air cost multiplier", "AIR_COST_MULT", default=cfg.air_cost_multiplier))
        cfg.air_fixed_penalty_h = float(_param_get(cf, "Air fixed penalty h", "AIR_FIXED_PENALTY_H", default=cfg.air_fixed_penalty_h))
        cfg.speed_air = float(_param_get(cf, "Air speed km/h", "SPEED_AIR", default=cfg.speed_air))

    aadt_name = _sheet_name(available, "11_AADT", "11_Corridor_AADT", "9_AADT")
    if cfg.allow_input_aadt_gap_fill and aadt_name:
        df = _read_table(xlsx, aadt_name)
        cfg.aadt_rows = [
            {
                "station_no": r.get("Station No", np.nan),
                "road_link": str(r.get("Road Link", "")),
                "location": str(r.get("Location", "")),
                "aadt": _numeric_or_nan(r.get("AADT", np.nan)),
                "lon": _numeric_or_nan(next((r.get(c) for c in df.columns if "lon" in str(c).lower()), np.nan)),
                "lat": _numeric_or_nan(next((r.get(c) for c in df.columns if "lat" in str(c).lower()), np.nan)),
            }
            for _, r in df.iterrows()
            if _numeric_or_nan(r.get("AADT", np.nan)) > 0
        ]
        if cfg.aadt_rows:
            cfg.aadt_default = float(np.nanmedian([r["aadt"] for r in cfg.aadt_rows]))

    tr_name = _sheet_name(available, "12_Border_Trade", "10_Border_Trade")
    if tr_name:
        df = _read_table(xlsx, tr_name)
        cfg.border_trade = [
            (float(r.iloc[0]), float(r.iloc[1]), float(r.iloc[2]), str(r.iloc[3]))
            for _, r in df.iterrows()
            if pd.notna(r.iloc[0]) and pd.notna(pd.to_numeric(pd.Series([r.iloc[0]]), errors="coerce").iloc[0])
        ]

    ap_name = _sheet_name(available, "13_CAAN_Airports", "11_CAAN_Airports")
    if ap_name:
        df = _read_table(xlsx, ap_name)
        cfg.airport_registry = [
            (float(r.iloc[2]), float(r.iloc[3]), str(r.iloc[0]).strip(), str(r.iloc[1]).strip())
            for _, r in df.iterrows()
            if pd.notna(r.iloc[0]) and pd.notna(pd.to_numeric(pd.Series([r.iloc[2]]), errors="coerce").iloc[0]) and pd.notna(pd.to_numeric(pd.Series([r.iloc[3]]), errors="coerce").iloc[0])
        ]
        if cfg.airport_registry:
            cfg.airport_registry_source = "Input.xlsx sheet 13"

    rtfd_records = read_airport_rtfd_table(DEFAULT_AIRPORT_RTFD)
    if rtfd_records:
        cfg.airport_registry = [
            (float(r["lon"]), float(r["lat"]), str(r["code"]).strip(), str(r["name"]).strip())
            for r in rtfd_records
        ]
        cfg.airport_status_lookup = {}
        for r in rtfd_records:
            status = str(r.get("status", "In operation"))
            for key in (r.get("code"), r.get("iata"), r.get("icao"), r.get("name"), r.get("city")):
                k = _airport_lookup_key(key)
                if k:
                    cfg.airport_status_lookup[k] = status
        cfg.airport_registry_source = str(DEFAULT_AIRPORT_RTFD)

    hp_name = _sheet_name(available, "14_Helipads", "12_Helipads")
    if hp_name:
        df = _read_table(xlsx, hp_name)
        lon_col = next((c for c in df.columns if "lon" in str(c).lower()), None)
        lat_col = next((c for c in df.columns if "lat" in str(c).lower()), None)
        name_col = next((c for c in df.columns if "helipad" in str(c).lower() and "name" in str(c).lower()), None)
        notes_col = next((c for c in df.columns if "district" in str(c).lower() or "note" in str(c).lower()), None)
        if lon_col and lat_col:
            cfg.helipads = [
                (
                    float(pd.to_numeric(pd.Series([r.get(lon_col)]), errors="coerce").iloc[0]),
                    float(pd.to_numeric(pd.Series([r.get(lat_col)]), errors="coerce").iloc[0]),
                    str(r.get(name_col, "Helipad")).strip(),
                    str(r.get(notes_col, "")).strip(),
                )
                for _, r in df.iterrows()
                if pd.notna(pd.to_numeric(pd.Series([r.get(lon_col)]), errors="coerce").iloc[0])
                and pd.notna(pd.to_numeric(pd.Series([r.get(lat_col)]), errors="coerce").iloc[0])
            ]

    pop_name = _sheet_name(available, "15_Population_GDP", "15_City_Lookup", "13_Population_GDP")
    workbook_places: List[Tuple[float, float, str, float]] = []
    if pop_name:
        cfg.population_gdp = _read_table(xlsx, pop_name)
        canonical_population_cols = [
            "District", "Population", "Province", "GDP per capita NPR", "Longitude",
            "Latitude", "OSM type", "OSM ID", "OSM display name", "Source URL",
        ]
        present_norm = {re.sub(r"\s+", " ", str(c).strip().lower()) for c in cfg.population_gdp.columns}
        if not {"district", "population", "gdp per capita npr", "longitude", "latitude"}.issubset(present_norm):
            if cfg.population_gdp.shape[1] >= len(canonical_population_cols):
                cfg.population_gdp = cfg.population_gdp.copy()
                cols = list(cfg.population_gdp.columns)
                cols[: len(canonical_population_cols)] = canonical_population_cols
                cfg.population_gdp.columns = cols
        pop_col = next((c for c in cfg.population_gdp.columns if "Population" in str(c)), None)
        gdp_col = next((c for c in cfg.population_gdp.columns if "GDP" in str(c)), None)
        name_col = next(
            (c for c in cfg.population_gdp.columns if any(token in str(c).lower() for token in ("city", "node", "district"))),
            None,
        )
        lon_col = next((c for c in cfg.population_gdp.columns if "lon" in str(c).lower()), None)
        lat_col = next((c for c in cfg.population_gdp.columns if "lat" in str(c).lower()), None)
        if pop_col:
            vals = pd.to_numeric(cfg.population_gdp[pop_col], errors="coerce").dropna()
            cfg.population_total = float(vals.sum()) if len(vals) else 1.0
        if gdp_col:
            vals = pd.to_numeric(cfg.population_gdp[gdp_col], errors="coerce").dropna()
            cfg.gdp_default = float(vals.median()) if len(vals) else cfg.gdp_default
        if name_col and pop_col:
            for _, r in cfg.population_gdp.iterrows():
                if pd.notna(r.get(name_col)):
                    key = str(r.get(name_col)).strip().lower()
                    if key:
                        pop_val = pd.to_numeric(pd.Series([r.get(pop_col)]), errors="coerce").iloc[0]
                        gdp_val = pd.to_numeric(pd.Series([r.get(gdp_col)]), errors="coerce").iloc[0] if gdp_col else np.nan
                        if pd.notna(pop_val):
                            cfg.population_by_name[key] = float(pop_val)
                        if pd.notna(gdp_val):
                            cfg.gdp_by_name[key] = float(gdp_val)
                        lon_val = pd.to_numeric(pd.Series([r.get(lon_col)]), errors="coerce").iloc[0] if lon_col else np.nan
                        lat_val = pd.to_numeric(pd.Series([r.get(lat_col)]), errors="coerce").iloc[0] if lat_col else np.nan
                        if pd.notna(lon_val) and pd.notna(lat_val):
                            workbook_places.append((float(lon_val), float(lat_val), str(r.get(name_col)).strip(), float(pop_val) if pd.notna(pop_val) else 1000.0))

    named_place_name = _sheet_name(available, "16_Named_Places", "Named_Places")
    if named_place_name:
        named_places = pd.read_excel(xlsx, sheet_name=named_place_name, header=0).dropna(how="all").reset_index(drop=True)
        name_col = next((c for c in named_places.columns if "place" in str(c).lower() and "name" in str(c).lower()), None)
        lon_col = next((c for c in named_places.columns if "lon" in str(c).lower()), None)
        lat_col = next((c for c in named_places.columns if "lat" in str(c).lower()), None)
        pop_col = next((c for c in named_places.columns if "population" in str(c).lower() and "source" not in str(c).lower()), None)
        include_col = next((c for c in named_places.columns if "include" in str(c).lower()), None)
        workbook_places = []
        if name_col and lon_col and lat_col:
            for _, row in named_places.iterrows():
                if include_col and not _as_bool(row.get(include_col), True):
                    continue
                lon = pd.to_numeric(pd.Series([row.get(lon_col)]), errors="coerce").iloc[0]
                lat = pd.to_numeric(pd.Series([row.get(lat_col)]), errors="coerce").iloc[0]
                population = pd.to_numeric(pd.Series([row.get(pop_col)]), errors="coerce").iloc[0] if pop_col else np.nan
                name = str(row.get(name_col, "")).strip()
                if name and pd.notna(lon) and pd.notna(lat):
                    population_value = float(population) if pd.notna(population) else 1000.0
                    workbook_places.append((float(lon), float(lat), name, population_value))
                    cfg.population_by_name[name.lower()] = population_value
        if workbook_places:
            cfg.named_place_source = f"Input.xlsx sheet {named_place_name}"

    cfg.place_lookup = workbook_places if named_place_name and workbook_places else workbook_places + list(BUILT_IN_CITY_LOOKUP)
    seen_places = set()
    dedup_places = []
    for lon, lat, name, pop in cfg.place_lookup:
        key = (round(lon, 4), round(lat, 4), name.lower())
        if key not in seen_places:
            seen_places.add(key)
            dedup_places.append((lon, lat, name, pop))
    cfg.place_lookup = dedup_places

    sec_name = _sheet_name(
        available,
        "17_Section_Candidates",
        "16_Named_Sections",
        "16_Section_Candidates",
        "Section_Candidates",
        "Named_Sections",
        "Corridor_Candidates",
    )
    if sec_name:
        specs = _read_section_specs(xlsx, sec_name)
        if specs:
            cfg.section_specs = specs
            cfg.section_specs_source = f"Input.xlsx sheet {sec_name}"

    formula_name = _sheet_name(available, "18_Formula_Index", "17_Formula_Index", "Formula_Index", "Methodology_Formulas")
    if formula_name:
        formula_rows = _read_formula_index(xlsx, formula_name)
        if formula_rows:
            cfg.formula_index_rows = formula_rows
            cfg.formula_index_source = f"Input.xlsx sheet {formula_name}"

    if not cfg.master_gpkg_path.exists():
        raise FileNotFoundError(
            f"Master GeoPackage not found: {cfg.master_gpkg_path}. "
            "Nepal.gpkg is the required current input for this version."
        )
    if not cfg.od_matrix_path.exists():
        raise FileNotFoundError(
            f"ME2 OD matrix not found: {cfg.od_matrix_path}. "
            "Provide the calibrated ME2 OD CSV before running MLPI."
        )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ------------------------------ geodata --------------------------------------


MASTER_GPKG_LAYERS = {
    "airports": ("airport", "airports"),
    "country": ("country_boundary", "country"),
    "district_hq": ("district_hq", "dist_hq"),
    "healthcare": ("health_facilities", "healthcare"),
    "nh_roads": ("NH", "nh"),
    "settlement": ("settlement", "ettlement"),
}
MASTER_OPTIONAL_KEYS = {
    "contours",
    "districts",
    "district_hq_name",
    "junction",
    "junction_name",
    "nh_name",
    "settlement_name",
}
MASTER_GEODATA_KEYS = (
    "airports",
    "contours",
    "country",
    "districts",
    "district_hq",
    "district_hq_name",
    "healthcare",
    "junction",
    "junction_name",
    "nh_name",
    "nh_roads",
    "settlement",
    "settlement_name",
)


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")


def _read_gpkg(path: Path, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception as e:
        if fiona is None or CRS is None:
            raise
        try:
            first_layer = layer or fiona.listlayers(path)[0]
            with fiona.open(path, layer=first_layer) as src:
                records = [{"geometry": shape(feat["geometry"]), **dict(feat["properties"])} for feat in src]
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS.from_proj4(NEPAL_TM_PROJ4))
        except Exception as e2:
            raise RuntimeError(f"Could not read {path.name}: {e}; alternate reader also failed: {e2}") from e2
    if gdf.empty:
        return gdf.set_crs(epsg=4326, allow_override=True)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS.from_proj4(NEPAL_TM_PROJ4) if CRS is not None else None, allow_override=True)
    try:
        return gdf.to_crs(epsg=4326)
    except Exception:
        if CRS is None:
            raise
        return gdf.set_crs(CRS.from_proj4(NEPAL_TM_PROJ4), allow_override=True).to_crs(epsg=4326)


def _available_layers(path: Path) -> List[str]:
    if fiona is None:
        return []
    try:
        return list(fiona.listlayers(path))
    except Exception:
        return []


def _read_master_layer(master_path: Path, key: str) -> gpd.GeoDataFrame:
    layers = _available_layers(master_path)
    lower = {layer.lower(): layer for layer in layers}
    for candidate in MASTER_GPKG_LAYERS[key]:
        layer = lower.get(candidate.lower())
        if layer:
            return _read_gpkg(master_path, layer=layer)
    raise FileNotFoundError(
        f"{master_path} does not contain a layer for '{key}'. "
        f"Tried {MASTER_GPKG_LAYERS[key]}; available layers: {layers}"
    )


def _harmonize_master_roads(gdf: gpd.GeoDataFrame, cfg: MLPIConfig) -> gpd.GeoDataFrame:
    out = gdf.copy()

    def first_col(*names: str) -> Optional[str]:
        canonical = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in out.columns}
        for name in names:
            key = re.sub(r"[^a-z0-9]+", "", name.lower())
            if key in canonical:
                return canonical[key]
        return None

    link_code_col = first_col("link_code", "linkcode", "code")
    ref_col = first_col("road_refno", "road_ref", "ref", "route")
    name_col = first_col("link_name", "road_name", "name")
    class_col = first_col("road_class", "rd_class", "fclass")
    design_speed_col = first_col("design_speed_kmh", "design_speed", "Design Speed", "designspeed")
    aadt_col = first_col(cfg.road_aadt_field, "AADT_2023", "aadt_2023")
    future_aadt_col = first_col(cfg.road_future_aadt_field, "AADT_2024_25", "aadt_2024_25")

    if link_code_col:
        out["link_code"] = out[link_code_col]
        out["code"] = out[link_code_col]
        out["osm_id"] = out[link_code_col]
    if ref_col:
        out["road_refno"] = out[ref_col]
        out["ref"] = out[ref_col]
    if name_col:
        out["link_name"] = out[name_col]
        out["road_name"] = out[name_col]
        out["name"] = out[name_col]
    if class_col:
        out["fclass"] = out[class_col]
    else:
        out["fclass"] = "national_highway"
    if not design_speed_col:
        raise ValueError(
            "Nepal.gpkg layer NH is missing the required design-speed field "
            "needed to compute mean_speed_kmh."
        )
    out["design_speed_kmh"] = out[design_speed_col].map(_positive_number)
    out["design_speed_source_field"] = design_speed_col
    if out["design_speed_kmh"].notna().sum() == 0:
        raise ValueError(
            f"Nepal.gpkg layer NH field {design_speed_col!r} has no usable "
            "positive design-speed values."
        )
    if aadt_col:
        out["aadt_source_value"] = pd.to_numeric(out[aadt_col], errors="coerce")
        out["aadt_source_field"] = aadt_col
    else:
        raise ValueError(
            f"Nepal.gpkg layer NH does not include required AADT field "
            f"{cfg.road_aadt_field!r}."
        )
    if future_aadt_col:
        out["aadt_future_value"] = pd.to_numeric(out[future_aadt_col], errors="coerce")
        out["aadt_future_field"] = future_aadt_col
    if "oneway" not in out.columns:
        out["oneway"] = "no"
    return out


def _load_master_geodata(cfg: MLPIConfig) -> Dict[str, gpd.GeoDataFrame]:
    out: Dict[str, gpd.GeoDataFrame] = {}
    for key in MASTER_GEODATA_KEYS:
        if key in MASTER_GPKG_LAYERS:
            out[key] = _read_master_layer(cfg.master_gpkg_path, key)
            if key == "nh_roads":
                out[key] = _harmonize_master_roads(out[key], cfg)
                before = len(out[key])
                out[key] = out[key].loc[_road_ref_filter_mask(out[key], cfg)].copy()
                if before != len(out[key]):
                    print(
                        f"  road ref filter: {_road_ref_filter_description(cfg)} "
                        f"kept {len(out[key])}/{before} features"
                    )
        elif key in MASTER_OPTIONAL_KEYS:
            out[key] = _empty_gdf()
        else:
            raise KeyError(f"No master-layer mapping configured for {key}")
        print(f"  {key:16s}: {len(out[key]):5d} features")
    return out


def _road_ref_filter_description(cfg: Optional[MLPIConfig]) -> str:
    raw = str(cfg.road_ref_filter if cfg is not None else "NH%, SH%").strip()
    return raw or "ALL"


def _road_ref_filter_mask(gdf: gpd.GeoDataFrame, cfg: Optional[MLPIConfig]) -> pd.Series:
    if "ref" not in gdf.columns:
        return pd.Series(True, index=gdf.index)
    raw = _road_ref_filter_description(cfg)
    tokens = [token.strip().upper() for token in re.split(r"[,;|]+", raw) if token.strip()]
    if not tokens or any(token in {"ALL", "*", "ANY", "NO FILTER", "NONE"} for token in tokens):
        return pd.Series(True, index=gdf.index)
    ref = gdf["ref"].fillna("").astype(str).str.upper().str.strip()
    mask = pd.Series(False, index=gdf.index)
    for token in tokens:
        include_blank = token in {"", "BLANK", "EMPTY", "UNREFERENCED", "NO_REF", "NO REF"}
        if include_blank:
            mask |= ref.eq("")
            continue
        token = token.replace("%", "").replace("*", "").strip()
        if not token:
            continue
        mask |= ref.str.startswith(token)
    return mask


def load_geodata(gpkg_dir: Path, cfg: Optional[MLPIConfig] = None) -> Dict[str, gpd.GeoDataFrame]:
    if cfg is not None and cfg.master_gpkg_path.exists():
        print(f"  master GeoPackage: {cfg.master_gpkg_path}")
        return _load_master_geodata(cfg)

    raise FileNotFoundError(
        "Nepal.gpkg is the required current geodata source. "
        "Pass MLPIConfig with master_gpkg_path pointing to the master GeoPackage."
    )


def _iter_lines(gdf: gpd.GeoDataFrame) -> Iterable[Tuple[int, List[Tuple[float, float]], Dict[str, Any]]]:
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        attrs = {k: row[k] for k in row.index if k != "geometry"}
        geoms = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for line in geoms:
            if not isinstance(line, LineString):
                continue
            pts = [(float(x), float(y)) for x, y in line.coords]
            if len(pts) >= 2:
                yield int(idx), pts, attrs


def _centroids(gdf: gpd.GeoDataFrame) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if gdf.empty:
        return pts
    projected = gdf.to_crs(epsg=32645)
    for geom in projected.geometry:
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        w = gpd.GeoSeries([c], crs=projected.crs).to_crs(epsg=4326).iloc[0]
        pts.append((float(w.x), float(w.y)))
    return pts


def _km(pts: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        total += math.hypot((x2 - x1) * 111.0 * math.cos(math.radians((y1 + y2) / 2.0)), (y2 - y1) * 111.0)
    return total


def _dist_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot((a[0] - b[0]) * 111.0 * math.cos(math.radians((a[1] + b[1]) / 2.0)), (a[1] - b[1]) * 111.0)


def _positive_number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).lower().replace("km/h", "").replace("kph", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        out = float(match.group(0))
    except ValueError:
        return None
    return out if out > 0 else None


def _road_speed_from_attrs(attrs: Dict[str, Any], cfg: MLPIConfig, zone: str) -> Tuple[float, str, float]:
    design_speed = _positive_number(attrs.get("design_speed_kmh"))
    if design_speed is None:
        raise ValueError(
            "Nepal.gpkg NH road feature is missing a usable design speed. "
            "The current MLPI run computes mean_speed_kmh as "
            f"{cfg.mean_speed_factor:.2f} * design_speed_kmh."
        )
    mean_speed = design_speed * float(cfg.mean_speed_factor)
    return (
        mean_speed,
        f"mean_speed_{cfg.mean_speed_factor:.2f}_design_speed",
        design_speed,
    )


def _road_aadt_from_attrs(attrs: Dict[str, Any], cfg: MLPIConfig) -> Tuple[float, str]:
    for value_key, field_key in (
        ("aadt_source_value", "aadt_source_field"),
        (cfg.road_aadt_field, cfg.road_aadt_field),
        ("AADT_2023", "AADT_2023"),
        ("aadt_2023", "aadt_2023"),
    ):
        value = _positive_number(attrs.get(value_key))
        if value is not None:
            field_name = str(attrs.get(field_key, value_key)) if field_key in attrs else str(field_key)
            return value, f"Nepal.gpkg NH {field_name}"
    return cfg.aadt_default, "nepal_gpkg_missing_aadt_default"


DISTRICT_NAME_ALIASES = {
    "chitwan": "chitawan",
    "dangdeukhuri": "dang",
    "dhanusa": "dhanusha",
    "eastrukum": "rukume",
    "kanchanpurdodharachandani": "kanchanpur",
    "kapilvastu": "kapilbastu",
    "kavrepalanchok": "kabhrepalanchok",
    "makwanpur": "makawanpur",
    "nawalpurnawalparasieast": "nawalparasie",
    "nawalpurnawalparasie": "nawalparasie",
    "nawalpur": "nawalparasie",
    "parasinawalparasiwest": "nawalparasiw",
    "parasinawalparasiw": "nawalparasiw",
    "parasi": "nawalparasiw",
    "sindhupalchowk": "sindhupalchok",
    "tanahun": "tanahu",
    "tehrathum": "terhathum",
    "westrukum": "rukumw",
}

TERAI_DISTRICTS = {
    "jhapa", "morang", "sunsari", "saptari", "siraha", "dhanusha", "mahottari",
    "sarlahi", "rautahat", "bara", "parsa", "chitawan", "nawalparasie",
    "nawalparasiw", "rupandehi", "kapilbastu", "dang", "banke", "bardiya",
    "kailali", "kanchanpur",
}

MOUNTAIN_DISTRICTS = {
    "taplejung", "sankhuwasabha", "solukhumbu", "dolakha", "rasuwa",
    "sindhupalchok", "manang", "mustang", "dolpa", "mugu", "humla",
    "jumla", "kalikot", "bajura", "bajhang", "darchula",
}


def _district_key(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(value or "").lower().replace("district", ""))
    return DISTRICT_NAME_ALIASES.get(key, key)


def _terrain_from_district_name(value: Any) -> Optional[str]:
    key = _district_key(value)
    if key in TERAI_DISTRICTS:
        return "Terai"
    if key in MOUNTAIN_DISTRICTS:
        return "Mountains"
    return "Hills" if key else None


def _map_safe_text(value: Any) -> str:
    text = str(value or "").strip()
    if text and text.isascii() and any(ch.isalpha() for ch in text):
        return text
    return ""


def _midpoint(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    return pts[len(pts) // 2]


def _zone(lat: float) -> str:
    if lat < 27.5:
        return "Terai"
    if lat >= 29.0:
        return "Mountains"
    return "Hills"


def _district_terrain_lookup(districts: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
    if districts is None or districts.empty:
        return []
    name_col = next(
        (
            c for c in districts.columns
            if str(c).strip().lower() in {"district", "first_dist", "dist_name", "name"}
        ),
        None,
    )
    if name_col is None:
        return []
    lookup: List[Dict[str, Any]] = []
    for _, row in districts.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        zone = _terrain_from_district_name(row.get(name_col))
        if not zone:
            continue
        lookup.append(
            dict(
                bounds=geom.bounds,
                geometry=geom,
                district=str(row.get(name_col, "")).strip(),
                zone=zone,
            )
        )
    return lookup


def _terrain_zone_for_point(
    lon: float,
    lat: float,
    district_lookup: Sequence[Dict[str, Any]],
) -> Tuple[str, str, str]:
    pt = Point(float(lon), float(lat))
    for item in district_lookup:
        minx, miny, maxx, maxy = item["bounds"]
        if minx <= lon <= maxx and miny <= lat <= maxy and item["geometry"].covers(pt):
            return str(item["zone"]), "district_polygon", str(item["district"])
    return _zone(float(lat)), "latitude_zone", ""


def _first_text_attr(attrs: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(k).lower(): v for k, v in attrs.items()}
    for c in candidates:
        v = lower.get(c.lower())
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _surface_from_attrs(attrs: Dict[str, Any], zone: str, aadt: float) -> str:
    val = _first_text_attr(attrs, ["surface", "pavement", "pavement_t", "road_surf", "type"])
    if val:
        low = val.lower()
        if any(x in low for x in ["black", "paved", "bitumen", "asphalt", "sealed", "gravel"]):
            return "Black topped"
        if any(x in low for x in ["unpaved", "earthen", "fair weather", "dirt"]):
            return "Unpaved"
    # Fallback: when the source has no pavement attributes, keep the map
    # differentiated using a conservative low-volume/terrain proxy, while
    # storing a flag so the limitation is explicit in the ranked CSV.
    if zone in {"Mountains", "Himalaya"} and aadt < 2000:
        return "Unpaved"
    return "Black topped"


def nearest_place(cfg: MLPIConfig, pt: Tuple[float, float], exclude: Optional[str] = None) -> Tuple[str, float]:
    choices = []
    for lon, lat, name, _ in cfg.place_lookup:
        if exclude and name == exclude:
            continue
        choices.append((name, _dist_km(pt, (lon, lat))))
    if not choices:
        return "Junction", float("inf")
    return min(choices, key=lambda x: x[1])


def _three_char_code(raw: str, used: set[str]) -> str:
    chars = "".join(ch for ch in str(raw).upper() if ch.isalnum())
    if not chars:
        chars = "AIR"
    base = chars[:3].ljust(3, "X")
    if base not in used:
        used.add(base)
        return base
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        cand = (base[:2] + ch)[:3]
        if cand not in used:
            used.add(cand)
            return cand
    i = 1
    while True:
        cand = f"{base[0]}{i:02d}"[-3:]
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def airport_display_code(cfg: MLPIConfig, pt: Tuple[float, float], used: set[str]) -> str:
    if cfg.airport_registry:
        code, dist = min(
            ((code, _dist_km(pt, (lon, lat))) for lon, lat, code, _name in cfg.airport_registry),
            key=lambda x: x[1],
        )
        if dist <= AIRPORT_REGISTRY_MATCH_KM:
            return _three_char_code(code, used)
    place, _ = nearest_place(cfg, pt)
    return _three_char_code(place, used)


def airport_registry_match(cfg: MLPIConfig, pt: Tuple[float, float]) -> Tuple[str, str, float]:
    if cfg.airport_registry:
        lon, lat, code, name = min(
            cfg.airport_registry,
            key=lambda r: _dist_km(pt, (r[0], r[1])),
        )
        return str(code), str(name), _dist_km(pt, (lon, lat))
    place, dist = nearest_place(cfg, pt)
    return place[:3].upper(), place, dist


NON_OPERATIONAL_AIRPORT_KEYS = {
    "DARCHULA", "MAHENDRANAGAR", "TIKAPUR", "KALIKOT", "MASINECHAUR", "CHAURJAHARI",
    "ROLPA", "GULMI", "DHORPATAN", "MANANG", "LANGTANG", "JIRI", "SYANGBOCHE",
    "RUMJATAR", "LAMIDANDA", "KHANIDANDA", "KANGELDANDA", "MEGHAULI",
    "BGL", "BIT", "DAP", "GKH", "JIR", "SIH", "XMG", "NGX", "MEY",
    "KTL", "DHO", "MAN", "SYH", "LDN",
}


def airport_status_from_figure(code: str, name: str) -> str:
    raw = f"{code} {name}".upper().replace("/", " ")
    compact = re.sub(r"[^A-Z0-9]+", "", raw)
    for key in NON_OPERATIONAL_AIRPORT_KEYS:
        k = re.sub(r"[^A-Z0-9]+", "", key.upper())
        if k and (k in compact or compact in k):
            return "Not in operation"
    return "In operation"


def airport_status_from_registry(cfg: MLPIConfig, code: str, name: str) -> str:
    figure_status = airport_status_from_figure(code, name)
    if figure_status == "Not in operation":
        return figure_status
    for key in (code, name):
        k = _airport_lookup_key(key)
        if k in cfg.airport_status_lookup:
            return cfg.airport_status_lookup[k]
    return figure_status


def edge_place_name(cfg: MLPIConfig, G: nx.Graph, u: int, v: int) -> str:
    pu, pv = G.nodes[u].get("pos"), G.nodes[v].get("pos")
    nu = G.nodes[u].get("place_name") or (nearest_place(cfg, pu)[0] if pu else str(u))
    nv = G.nodes[v].get("place_name") or (nearest_place(cfg, pv, exclude=nu)[0] if pv else str(v))
    if nu == nv and pv:
        nv = nearest_place(cfg, pv, exclude=nu)[0]
    return f"{nu} - {nv}"


def _name_value_lookup(values: Dict[str, float], name: str, default: float) -> float:
    key = str(name).strip().lower()
    if key in values:
        return values[key]
    for k, v in values.items():
        if key and (key in k or k in key):
            return v
    return default


def node_population(cfg: MLPIConfig, node_data: Dict[str, Any]) -> float:
    if "population_weight" in node_data:
        return max(float(node_data.get("population_weight", 0.0)), 0.0)
    name = str(node_data.get("place_name", ""))
    pop = _name_value_lookup(cfg.population_by_name, name, float("nan"))
    if math.isfinite(pop):
        return max(pop, 1.0)
    return max(1000.0, 5000.0 * float(node_data.get("settlement_count", 0)))


def node_gdp_per_capita(cfg: MLPIConfig, node_data: Dict[str, Any]) -> float:
    pop = float(node_data.get("population_weight", 0.0))
    gdp = float(node_data.get("gdp_weight", 0.0))
    if pop > 0.0 and gdp > 0.0:
        return gdp / pop
    name = str(node_data.get("place_name", ""))
    return _name_value_lookup(cfg.gdp_by_name, name, cfg.gdp_default)


def _point_segment_distance_km(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat0 = math.radians((p[1] + a[1] + b[1]) / 3.0)
    px, py = p[0] * 111.0 * math.cos(lat0), p[1] * 111.0
    ax, ay = a[0] * 111.0 * math.cos(lat0), a[1] * 111.0
    bx, by = b[0] * 111.0 * math.cos(lat0), b[1] * 111.0
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = max(0.0, min(1.0, (wx * vx + wy * vy) / max(vx * vx + vy * vy, 1e-9)))
    qx, qy = ax + c * vx, ay + c * vy
    return math.hypot(px - qx, py - qy)


# Named-corridor context layer.
#
# A light geometry-based corridor context is kept only to label and group known
# nationally important corridors. It does not create stochastic failures; it
# stabilizes interpretation of fragmentary road polylines when ranking and plotting.
CRITICAL_CORRIDORS = [
    dict(
        route_id="NH17/NH44 - Narayanghat-Mugling-Naubise-Kathmandu",
        short_name="Bharatpur-Mugling-Naubise-Kathmandu",
        anchors=[(84.429, 27.702), (84.476, 27.874), (84.897, 27.910), (84.983, 27.681), (85.316, 27.698)],
        core_km=14.0,
        shoulder_km=55.0,
        pf_boost=0.28,
        repair_days=2.0,
        physical_prior=0.95,
        social_prior=0.45,
        economic_prior=1.00,
        interconnected_prior=0.55,
    ),
    dict(
        route_id="NH04 - Prithvi Highway",
        short_name="Kathmandu-Mugling-Pokhara",
        anchors=[(85.316, 27.698), (84.983, 27.681), (84.897, 27.910), (84.476, 27.874), (83.986, 28.210)],
        core_km=12.0,
        shoulder_km=42.0,
        pf_boost=0.15,
        repair_days=1.5,
        physical_prior=0.85,
        social_prior=0.35,
        economic_prior=0.85,
        interconnected_prior=0.55,
    ),
    dict(
        route_id="NH13 - BP Koirala Highway",
        short_name="BP Koirala Highway",
        anchors=[(85.545, 27.622), (85.830, 27.470), (85.985, 27.323), (85.912, 27.256), (85.890, 26.999)],
        core_km=16.0,
        shoulder_km=58.0,
        pf_boost=0.30,
        repair_days=2.0,
        physical_prior=0.90,
        social_prior=0.55,
        economic_prior=0.75,
        interconnected_prior=0.35,
    ),
    dict(
        route_id="NH03/NH08 - Dharan-Dhankuta-Hile",
        short_name="Dharan-Dhankuta-Hile",
        anchors=[(87.274, 26.661), (87.292, 26.812), (87.320, 27.050), (87.333, 27.188)],
        core_km=10.0,
        shoulder_km=34.0,
        pf_boost=0.18,
        repair_days=1.5,
        physical_prior=0.70,
        social_prior=0.35,
        economic_prior=0.50,
        interconnected_prior=0.35,
    ),
    dict(
        route_id="NH05 - Siddhartha Highway",
        short_name="Bhairahawa-Butwal-Tansen-Pokhara",
        anchors=[(83.449, 27.506), (83.449, 27.701), (83.550, 27.870), (83.100, 28.050), (83.986, 28.210)],
        core_km=14.0,
        shoulder_km=45.0,
        pf_boost=0.12,
        repair_days=1.5,
        physical_prior=0.70,
        social_prior=0.45,
        economic_prior=0.70,
        interconnected_prior=0.50,
    ),
    dict(
        route_id="NH01 - East-West Highway",
        short_name="East-West Highway",
        anchors=[(80.599, 28.682), (81.617, 28.050), (83.449, 27.701), (84.429, 27.702), (85.034, 27.429), (85.890, 26.999), (87.274, 26.661), (88.080, 26.571)],
        core_km=14.0,
        shoulder_km=48.0,
        pf_boost=0.10,
        repair_days=1.0,
        physical_prior=0.75,
        social_prior=0.30,
        economic_prior=0.95,
        interconnected_prior=0.45,
    ),
]


def critical_corridor_prior(lon: float, lat: float) -> Dict[str, Any]:
    p = (lon, lat)
    best: Optional[Dict[str, Any]] = None
    for corr in CRITICAL_CORRIDORS:
        anchors = corr["anchors"]
        d = min(_point_segment_distance_km(p, a, b) for a, b in zip(anchors, anchors[1:]))
        if d <= corr["core_km"]:
            score = 1.0
        elif d <= corr["shoulder_km"]:
            score = (corr["shoulder_km"] - d) / max(corr["shoulder_km"] - corr["core_km"], 1e-9)
        else:
            score = 0.0
        if score > 0 and (best is None or score > best["score"]):
            best = {**corr, "score": float(score), "distance_km": float(d)}
    if best is None:
        return {"score": 0.0, "route_id": None, "short_name": None, "pf_boost": 0.0, "repair_days": 0.0}
    return best


def central_blockage_corridor_score(lon: float, lat: float) -> float:
    """Compatibility wrapper for the central Narayanghat-Mugling prior."""
    p = (lon, lat)
    anchors = CRITICAL_CORRIDORS[0]["anchors"]
    d = min(_point_segment_distance_km(p, a, b) for a, b in zip(anchors, anchors[1:]))
    if d <= CRITICAL_CORRIDORS[0]["core_km"]:
        return 1.0
    if d <= CRITICAL_CORRIDORS[0]["shoulder_km"]:
        return (CRITICAL_CORRIDORS[0]["shoulder_km"] - d) / max(
            CRITICAL_CORRIDORS[0]["shoulder_km"] - CRITICAL_CORRIDORS[0]["core_km"],
            1e-9,
        )
    return 0.0


def build_contour_density(cfg: MLPIConfig, contours: gpd.GeoDataFrame) -> Dict[Tuple[float, float], float]:
    cell: Dict[Tuple[float, float], set] = defaultdict(set)
    for idx, pts, _ in _iter_lines(contours):
        for x, y in pts:
            key = (round(x / cfg.terr_grid_deg) * cfg.terr_grid_deg, round(y / cfg.terr_grid_deg) * cfg.terr_grid_deg)
            cell[key].add(idx)
    raw = {k: len(v) for k, v in cell.items()}
    if not raw:
        return {}
    vals = np.array(list(raw.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return {k: 0.0 for k in raw}
    return {k: (v - lo) / (hi - lo) for k, v in raw.items()}


def assign_display_nh_codes(G: nx.Graph) -> None:
    R = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("type") == "road":
            R.add_edge(u, v, length_km=d.get("length_km", 0.0), mean_lon=d.get("mean_lon", 0.0))
    comps = []
    for comp in nx.connected_components(R):
        edges = [
            (u, v, G[u][v])
            for u, v in G.edges(comp)
            if G[u][v].get("type") == "road" and u in comp and v in comp
        ]
        if not edges:
            continue
        total_len = sum(float(d.get("length_km", 0.0)) for _, _, d in edges)
        mean_lon = np.average(
            [float(d.get("mean_lon", 0.0)) for _, _, d in edges],
            weights=[max(float(d.get("length_km", 0.0)), 0.01) for _, _, d in edges],
        )
        comps.append((total_len, mean_lon, comp, edges))
    comps.sort(key=lambda x: (-x[0], x[1]))
    for i, (_total_len, _mean_lon, _comp, edges) in enumerate(comps, start=1):
        code = f"NH{i}"
        for u, v, d in edges:
            G[u][v]["nh_code"] = code
            if not d.get("route_id"):
                G[u][v]["route_id"] = code
            if not d.get("corridor_id"):
                G[u][v]["corridor_id"] = G[u][v].get("route_id", code)
            if not d.get("corridor_label"):
                G[u][v]["corridor_label"] = G[u][v].get("route_id", code)


def _route_prefix(raw: Any) -> str:
    m = re.search(r"\b(NH|SH)\s*0*(\d+)", str(raw).upper())
    return f"{m.group(1)}{int(m.group(2)):02d}" if m else ""


def _route_tokens(raw: Any) -> set[str]:
    text = str(raw or "").upper()
    toks = re.findall(r"\b(NH|SH)\s*0*(\d+)", text)
    return {f"{prefix}{int(num):02d}" for prefix, num in toks}


def _edge_route_tokens(d: Dict[str, Any]) -> set[str]:
    fields = [
        d.get("route_id"),
        d.get("nh_code"),
        d.get("osm_ref"),
        d.get("osm_name"),
        d.get("corridor_id"),
        d.get("corridor_label"),
    ]
    out: set[str] = set()
    for field in fields:
        out |= _route_tokens(field)
    return out


def _station_location_point(cfg: MLPIConfig, location: str) -> Optional[Tuple[float, float]]:
    loc = re.sub(r"\b(east|west|north|south|road|bypass|bridge)\b", " ", str(location).lower())
    loc = re.sub(r"[^a-z]+", " ", loc).strip()
    best: Optional[Tuple[float, float, float]] = None
    for lon, lat, name, _pop in cfg.place_lookup:
        n = str(name).lower()
        if not n:
            continue
        hit = n in loc or loc in n
        if not hit:
            words = [w for w in loc.split() if len(w) >= 4]
            hit = any(w in n or n in w for w in words)
        if hit:
            score = abs(len(n) - len(loc))
            if best is None or score < best[2]:
                best = (lon, lat, float(score))
    return None if best is None else (best[0], best[1])


def _point_to_polyline_distance_km(p: Tuple[float, float], pts: Sequence[Tuple[float, float]]) -> float:
    if len(pts) < 2:
        return float("inf")
    return min(_point_segment_distance_km(p, a, b) for a, b in zip(pts, pts[1:]))


def apply_aadt_station_matches(cfg: MLPIConfig, G: nx.Graph, sp: Dict[Tuple[int, int], List[Tuple[float, float]]]) -> None:
    if not cfg.allow_input_aadt_gap_fill:
        return

    route_vals: Dict[str, List[float]] = defaultdict(list)
    edge_vals: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    road_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("type") == "road"]
    for row in cfg.aadt_rows:
        aadt = float(row.get("aadt", float("nan")))
        if not math.isfinite(aadt):
            continue
        code = _route_prefix(row.get("road_link", ""))
        if code:
            route_vals[code].append(aadt)
        p = None
        if math.isfinite(float(row.get("lon", float("nan")))) and math.isfinite(float(row.get("lat", float("nan")))):
            p = (float(row["lon"]), float(row["lat"]))
        else:
            p = _station_location_point(cfg, row.get("location", ""))
        edge: Optional[Tuple[int, int]] = None
        if p is not None:
            candidates = []
            route_candidates = [
                (u, v, d)
                for u, v, d in road_edges
                if not code or code in _edge_route_tokens(d)
            ]
            for u, v, _d in (route_candidates or road_edges):
                key = (min(u, v), max(u, v))
                dist = _point_to_polyline_distance_km(p, sp.get(key, []))
                candidates.append((dist, key))
            if candidates:
                dist, key = min(candidates, key=lambda x: x[0])
                if dist <= (cfg.station_match_km if math.isfinite(float(row.get("lon", float("nan")))) else max(20.0, cfg.station_match_km)):
                    edge = key
                    edge_vals[key].append(aadt)
        if edge is not None:
            G[edge[0]][edge[1]]["aadt_station_count"] = G[edge[0]][edge[1]].get("aadt_station_count", 0) + 1
    median_aadt = cfg.aadt_default
    source_counts: Counter[str] = Counter()
    for u, v, d in road_edges:
        existing_source = str(d.get("aadt_source", ""))
        existing_aadt = _positive_number(d.get("aadt"))
        if existing_aadt is not None and existing_source.startswith("Nepal.gpkg"):
            d["capacity"] = max(100.0, min(600.0, existing_aadt / 20.0))
            d["base_cap"] = d["capacity"]
            source_counts[existing_source] += 1
            continue
        key = (min(u, v), max(u, v))
        vals = edge_vals.get(key)
        source = "input_station_location"
        if not vals:
            route_aadt: List[float] = []
            for token in _edge_route_tokens(d):
                route_aadt.extend(route_vals.get(token, []))
            vals = route_aadt
            source = "input_route_median"
        if vals:
            aadt = float(np.nanmedian(vals))
            d["aadt"] = aadt
            d["aadt_source"] = source
        else:
            d["aadt"] = median_aadt
            d["aadt_source"] = "input_network_median"
        d["capacity"] = max(100.0, min(600.0, d["aadt"] / 20.0))
        d["base_cap"] = d["capacity"]
        if d.get("surface_source") == "terrain_aadt_proxy":
            d["surface"] = _surface_from_attrs({}, d.get("zone", "Hills"), d["aadt"])
        source_counts[str(d["aadt_source"])] += 1
    if cfg.aadt_rows:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
        print(f"  road-edge AADT source summary: {summary}")


def add_topology_connectors(cfg: MLPIConfig, G: nx.Graph) -> int:
    """Connect tiny digitizing gaps for path analysis without plotting/ranking them."""
    road_nodes: List[int] = []
    route_labels: Dict[int, set[str]] = {}
    road_degree: Dict[int, int] = {}
    for n, d in G.nodes(data=True):
        if d.get("is_airport"):
            continue
        incident = [ed for _, _, ed in G.edges(n, data=True) if ed.get("type") == "road"]
        if not incident:
            continue
        road_nodes.append(n)
        road_degree[n] = len(incident)
        route_labels[n] = {
            str(ed.get("route_id") or ed.get("nh_code") or "")
            for ed in incident
            if str(ed.get("route_id") or ed.get("nh_code") or "")
        }
    if not road_nodes or cfg.topology_connector_max_km <= 0:
        return 0

    components = list(nx.connected_components(G.subgraph(road_nodes)))
    component_id = {node: idx for idx, comp in enumerate(components) for node in comp}
    parent = list(range(len(components)))

    def find(component: int) -> int:
        while parent[component] != component:
            parent[component] = parent[parent[component]]
            component = parent[component]
        return component

    def union(left: int, right: int) -> bool:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return False
        parent[root_right] = root_left
        return True

    cell_deg = max(cfg.topology_connector_max_km / 80.0, 1e-6)
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for n in road_nodes:
        lon, lat = G.nodes[n]["pos"]
        buckets[(math.floor(lon / cell_deg), math.floor(lat / cell_deg))].append(n)

    candidates: List[Tuple[float, int, int]] = []
    for n in road_nodes:
        lon, lat = G.nodes[n]["pos"]
        cell = (math.floor(lon / cell_deg), math.floor(lat / cell_deg))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in buckets.get((cell[0] + dx, cell[1] + dy), []):
                    if m <= n or G.has_edge(n, m):
                        continue
                    if road_degree.get(n, 0) > 1 and road_degree.get(m, 0) > 1:
                        continue
                    dist = _dist_km((lon, lat), G.nodes[m]["pos"])
                    if dist <= cfg.topology_connector_max_km:
                        candidates.append((dist, n, m))

    added = 0
    repaired_dangling: set[int] = set()
    for dist, n, m in sorted(candidates):
        dangling = {node for node in (n, m) if road_degree.get(node, 0) <= 1}
        if dangling and dangling.issubset(repaired_dangling):
            continue
        union(component_id[n], component_id[m])
        common_labels = route_labels[n] & route_labels[m]
        label = sorted(common_labels)[0] if common_labels else "JUNCTION_GAP"
        zone = _zone((G.nodes[n]["pos"][1] + G.nodes[m]["pos"][1]) / 2.0)
        G.add_edge(
            n,
            m,
            type="road_connector",
            road_class="Topology connector",
            surface="Connector",
            surface_source="topology_gap",
            length_km=dist,
            aadt=cfg.aadt_default,
            aadt_source="connector_default",
            zone=zone,
            terrain_source="connector_latitude_zone",
            terrain_district="",
            speed_kmh=max(float(cfg.speed_road), 1.0),
            speed_source="topology_connector_default",
            model_speed_source="topology_connector_default",
            capacity=max(100.0, cfg.aadt_default / 20.0),
            base_cap=max(100.0, cfg.aadt_default / 20.0),
            mean_lon=(G.nodes[n]["pos"][0] + G.nodes[m]["pos"][0]) / 2.0,
            mean_lat=(G.nodes[n]["pos"][1] + G.nodes[m]["pos"][1]) / 2.0,
            route_id=label,
            corridor_id=label,
            corridor_label=label,
            virtual_connector=True,
            blockages=[],
        )
        repaired_dangling.update(dangling)
        added += 1
    return added


def build_network(cfg: MLPIConfig, geo: Dict[str, gpd.GeoDataFrame]) -> Tuple[nx.Graph, Dict[Tuple[int, int], List[Tuple[float, float]]], List[Tuple[List[Tuple[float, float]], float, str, int]]]:
    print("[1] Building road graph from GeoPackages")
    terrain_lookup = _district_terrain_lookup(geo.get("districts", gpd.GeoDataFrame()))
    roads = list(_iter_lines(geo["nh_roads"]))
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]] = []
    score_roads: List[Tuple[int, List[Tuple[float, float]], Dict[str, Any], float]] = []

    for idx, pts, attrs in roads:
        km = _km(pts)
        if km >= cfg.road_min_km_plot:
            z, _terrain_source, _district = _terrain_zone_for_point(
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
                terrain_lookup,
            )
            surf = _surface_from_attrs(attrs, z, cfg.aadt_default)
            plot_roads.append((pts, km, surf, idx))
        if km >= cfg.road_min_km_score:
            score_roads.append((idx, pts, attrs, km))

    if not score_roads and roads:
        auto_min = max(1.0, np.percentile([_km(r[1]) for r in roads], 90))
        warnings.warn(f"No roads met score threshold; auto-lowering to {auto_min:.1f} km")
        for idx, pts, attrs in roads:
            km = _km(pts)
            if km >= auto_min:
                score_roads.append((idx, pts, attrs, km))

    coord_feature_count: Counter[Tuple[float, float]] = Counter()
    for _idx, pts, _attrs, _km_value in score_roads:
        coord_feature_count.update(
            set((round(p[0], cfg.endpoint_precision), round(p[1], cfg.endpoint_precision)) for p in pts)
        )
    shared_node_keys = {key for key, count in coord_feature_count.items() if count > 1}
    noded_score_roads: List[Tuple[int, int, List[Tuple[float, float]], Dict[str, Any], float]] = []
    for orig_idx, pts, attrs, _km_value in score_roads:
        split_indices = {0, len(pts) - 1}
        split_indices.update(
            i
            for i, p in enumerate(pts)
            if (round(p[0], cfg.endpoint_precision), round(p[1], cfg.endpoint_precision)) in shared_node_keys
        )
        ordered = sorted(split_indices)
        for segment_idx, (start, end) in enumerate(zip(ordered, ordered[1:])):
            segment_pts = pts[start : end + 1]
            segment_km = _km(segment_pts)
            if segment_km > 0:
                noded_score_roads.append((orig_idx, segment_idx, segment_pts, attrs, segment_km))

    G = nx.Graph()
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    key_to_node: Dict[Tuple[float, float], int] = {}
    node_pos: Dict[int, Tuple[float, float]] = {}

    def node_for(pt: Tuple[float, float]) -> int:
        key = (round(pt[0], cfg.endpoint_precision), round(pt[1], cfg.endpoint_precision))
        if key not in key_to_node:
            nid = len(key_to_node)
            key_to_node[key] = nid
            node_pos[nid] = pt
            G.add_node(nid, pos=pt, is_airport=False)
        return key_to_node[key]

    for orig_idx, segment_idx, pts, attrs, km in noded_score_roads:
        u, v = node_for(pts[0]), node_for(pts[-1])
        if u == v:
            continue
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        mean_lon, mean_lat = float(np.mean(lons)), float(np.mean(lats))
        zone, terrain_source, terrain_district = _terrain_zone_for_point(mean_lon, mean_lat, terrain_lookup)
        route_id = _first_text_attr(attrs, ["ref", "route", "route_id", "road_link", "nh", "name"])
        corridor_prior = critical_corridor_prior(mean_lon, mean_lat)
        central_route_score = central_blockage_corridor_score(mean_lon, mean_lat)
        if not route_id and corridor_prior["score"] > 0.08:
            route_id = corridor_prior["route_id"]
        aadt, aadt_source = _road_aadt_from_attrs(attrs, cfg)
        speed_kmh, speed_source, source_design_speed = _road_speed_from_attrs(attrs, cfg, zone)
        cap = max(100.0, min(600.0, aadt / 20.0))
        surface = _surface_from_attrs(attrs, zone, aadt)
        edge_key = (min(u, v), max(u, v))
        data = dict(
            type="road",
            road_class="Strategic Road Network" if str(route_id or "").upper().startswith(("NH", "SH")) else "Road",
            surface=surface,
            surface_source="attribute" if _first_text_attr(attrs, ["surface", "pavement", "pavement_t", "road_surf", "type"]) else "terrain_aadt_proxy",
            length_km=km,
            aadt=aadt,
            aadt_source=aadt_source,
            zone=zone,
            terrain_source=terrain_source,
            terrain_district=terrain_district,
            speed_kmh=speed_kmh,
            mean_speed_kmh=speed_kmh,
            mean_speed_factor=cfg.mean_speed_factor,
            speed_source=speed_source,
            model_speed_source=speed_source,
            source_design_speed_kmh=source_design_speed,
            design_speed_source=str(attrs.get("design_speed_source_field", "design_speed_kmh") or "design_speed_kmh"),
            capacity=cap,
            base_cap=cap,
            mean_lon=mean_lon,
            mean_lat=mean_lat,
            orig_idx=orig_idx,
            segment_idx=segment_idx,
            osm_id=str(attrs.get("osm_id", "") or ""),
            osm_code=str(attrs.get("code", "") or ""),
            osm_fclass=str(attrs.get("fclass", "") or ""),
            osm_name=str(attrs.get("name", "") or ""),
            osm_ref=str(attrs.get("ref", "") or route_id or ""),
            osm_oneway=str(attrs.get("oneway", "") or ""),
            osm_layer=str(attrs.get("layer", "") or ""),
            osm_bridge=str(attrs.get("bridge", "") or ""),
            osm_tunnel=str(attrs.get("tunnel", "") or ""),
            route_id=route_id,
            corridor_id=corridor_prior.get("route_id") if corridor_prior["score"] > 0.08 else route_id,
            corridor_label=corridor_prior.get("short_name") if corridor_prior["score"] > 0.08 else route_id,
            central_corridor=central_route_score,
            critical_corridor=corridor_prior.get("short_name"),
            corridor_score=corridor_prior["score"],
            physical_prior=float(corridor_prior.get("physical_prior", 0.0)) * float(corridor_prior["score"]),
            social_prior=float(corridor_prior.get("social_prior", 0.0)) * float(corridor_prior["score"]),
            economic_prior=float(corridor_prior.get("economic_prior", 0.0)) * float(corridor_prior["score"]),
            interconnected_prior=float(corridor_prior.get("interconnected_prior", 0.0)) * float(corridor_prior["score"]),
            blockages=[],
            phase1_done=False,
        )
        if G.has_edge(*edge_key):
            if km > G[edge_key[0]][edge_key[1]].get("length_km", 0):
                G[edge_key[0]][edge_key[1]].update(data)
                sp[edge_key] = pts
        else:
            G.add_edge(edge_key[0], edge_key[1], **data)
            sp[edge_key] = pts

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_edges = G.subgraph(comps[0]).number_of_edges() if comps else 0
    print(
        f"  raw road features: {len(roads)} | noded road segments: {len(noded_score_roads)} | plot lines: {len(plot_roads)} | "
        f"scored edges: {G.number_of_edges()} | components: {len(comps)} | "
        f"largest component edges: {largest_edges}"
    )
    # Keep all supplied NH components. The component-aware scoring routines below
    # evaluate each link only within its own pre-disruption component, avoiding
    # analysis connectors while preventing unrelated disconnected components from
    # being counted as newly isolated.
    assign_display_nh_codes(G)

    # Stable junction names for ranking. If text attributes are absent, these IDs
    # are still reproducible and tied to actual junction geometry.
    for i, (n, d) in enumerate(sorted(G.nodes(data=True), key=lambda x: (x[1]["pos"][0], x[1]["pos"][1])), start=1):
        d["junction_name"] = f"J{i:03d}"
        pname, pdist = nearest_place(cfg, d["pos"])
        d["place_name"] = pname if pdist <= 35.0 else d["junction_name"]
        d["place_distance_km"] = pdist
        d["abbrev"] = d["place_name"]

    apply_aadt_station_matches(cfg, G, sp)
    connectors = add_topology_connectors(cfg, G)
    if connectors:
        comps2 = sorted(nx.connected_components(_assignment_graph(cfg, G)), key=len, reverse=True)
        largest2 = _assignment_graph(cfg, G).subgraph(comps2[0]).number_of_edges() if comps2 else 0
        print(
            f"  topology gap connectors added for analysis only: {connectors} | "
            f"analysis components: {len(comps2)} | largest component edges: {largest2}"
        )

    # Settlement and healthcare proximity scores.
    settlements = _centroids(geo["settlement"])
    healthcare = [(float(g.x), float(g.y)) for g in geo["healthcare"].geometry if isinstance(g, Point)]
    for n, d in G.nodes(data=True):
        p = d["pos"]
        d["settlement_count"] = sum(1 for s in settlements if _dist_km(p, s) <= cfg.sett_window_km)
        d["healthcare_count"] = sum(1 for h in healthcare if _dist_km(p, h) <= cfg.sett_window_km)

    # Airports from Nepal.gpkg layer airport. The GeoPackage carries the geometry; the
    # CAAN/Wikipedia RTFD table provides display codes and operational status.
    airport_pts = _centroids(geo["airports"])
    linked = 0
    used_airport_codes: set[str] = set()
    used_registry_airports: set[str] = set()
    for i, ap in enumerate(sorted(airport_pts, key=lambda p: (p[0], p[1])), start=1):
        if not G.nodes:
            break
        reg_code, reg_name, reg_dist = airport_registry_match(cfg, ap)
        registry_key = _airport_lookup_key(reg_code)
        has_registry_match = bool(registry_key) and reg_dist <= AIRPORT_REGISTRY_MATCH_KM
        if has_registry_match and registry_key in used_registry_airports:
            continue
        if has_registry_match:
            used_registry_airports.add(registry_key)
            code = _three_char_code(reg_code, used_airport_codes)
            display_name = reg_name
        else:
            code = airport_display_code(cfg, ap, used_airport_codes)
            display_name = code
        near_n, near_d = min(((n, _dist_km(ap, d["pos"])) for n, d in G.nodes(data=True) if not d.get("is_airport")), key=lambda x: x[1])
        node_id = f"AP{i:03d}"
        status = airport_status_from_registry(cfg, reg_code if has_registry_match else code, display_name)
        G.add_node(
            node_id,
            pos=ap,
            is_airport=True,
            is_air_asset=True,
            airport_name=code,
            airport_full_name=display_name,
            airport_status=status,
            airport_registry_code=reg_code,
            airport_registry_distance_km=reg_dist,
            airport_type="Airport",
            abbrev=code,
        )
        G.add_edge(
            near_n,
            node_id,
            type="airport_access",
            road_class="Airport access",
            length_km=near_d,
            aadt=0,
            zone="Air",
            capacity=5,
            base_cap=5,
            mean_lon=ap[0],
            mean_lat=ap[1],
            blockages=[],
        )
        G.nodes[near_n]["has_airport_link"] = True
        linked += 1
    print(f"  airports linked from Nepal.gpkg layer airport: {linked}")

    helipads_linked = 0
    road_nodes = [n for n, d in G.nodes(data=True) if not d.get("is_air_asset")]
    for i, (lon, lat, name, notes) in enumerate(cfg.helipads, start=1):
        if not road_nodes:
            break
        hp = (float(lon), float(lat))
        near_n, near_d = min(((n, _dist_km(hp, G.nodes[n]["pos"])) for n in road_nodes), key=lambda x: x[1])
        node_id = f"HP{i:03d}"
        G.add_node(
            node_id,
            pos=hp,
            is_airport=False,
            is_air_asset=True,
            airport_name=f"HP{i:02d}",
            airport_full_name=name,
            airport_status="In operation",
            airport_registry_code="",
            airport_registry_distance_km=0.0,
            airport_type="Helipad",
            helipad_notes=notes,
            abbrev=f"HP{i:02d}",
        )
        G.add_edge(
            near_n,
            node_id,
            type="airport_access",
            road_class="Helipad access",
            length_km=near_d,
            aadt=0,
            zone="Air",
            capacity=5,
            base_cap=5,
            mean_lon=lon,
            mean_lat=lat,
            blockages=[],
        )
        G.nodes[near_n]["has_helipad_link"] = True
        helipads_linked += 1
    print(f"  verified helipads linked from Input.xlsx: {helipads_linked}")

    operational_air_assets = [
        n
        for n, d in G.nodes(data=True)
        if d.get("is_air_asset") and d.get("airport_status") != "Not in operation"
    ]
    air_routes = 0
    for i, a in enumerate(operational_air_assets):
        for b in operational_air_assets[i + 1 :]:
            dist = _dist_km(G.nodes[a]["pos"], G.nodes[b]["pos"])
            G.add_edge(
                a,
                b,
                type="air_route",
                road_class="Virtual operational air route",
                length_km=dist,
                aadt=0.0,
                capacity=10000.0,
                base_cap=10000.0,
                air_cost_multiplier=cfg.air_cost_multiplier,
                virtual_air_route=True,
            )
            air_routes += 1
    print(f"  multimodal operational airport/helipad air routes added: {air_routes}")

    return G, sp, plot_roads


def assign_socioeconomic_weights(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    """Distribute every district total over representative NH/SH road nodes."""
    table = cfg.population_gdp.copy()
    if table.empty:
        raise RuntimeError("15_Population_GDP is empty; district weights cannot be constructed.")

    normalized_columns = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in table.columns}

    def exact_column(*names: str) -> Any:
        for name in names:
            if name in normalized_columns:
                return normalized_columns[name]
        return None

    name_col = exact_column("district", "city / node name", "city", "node")
    pop_col = exact_column("population", "population total")
    province_col = exact_column("province")
    gdp_col = exact_column("gdp per capita npr", "provincial gdp per capita (npr)", "gdp per capita")
    lon_col = exact_column("longitude", "lon", "long")
    lat_col = exact_column("latitude", "lat")
    required = {
        "district": name_col,
        "population": pop_col,
        "GDP per capita": gdp_col,
        "longitude": lon_col,
        "latitude": lat_col,
    }
    missing = [label for label, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"15_Population_GDP lacks required columns: {', '.join(missing)}")

    records: List[Dict[str, Any]] = []
    for _, row in table.iterrows():
        population = _numeric_or_nan(row.get(pop_col))
        gdp_pc = _numeric_or_nan(row.get(gdp_col))
        lon = _numeric_or_nan(row.get(lon_col))
        lat = _numeric_or_nan(row.get(lat_col))
        name = str(row.get(name_col, "")).strip()
        if name and all(math.isfinite(v) for v in (population, gdp_pc, lon, lat)):
            records.append(
                dict(
                    district=name,
                    province=str(row.get(province_col, "")).strip() if province_col else "",
                    population=max(population, 0.0),
                    gdp_per_capita_npr=max(gdp_pc, 0.0),
                    point=(lon, lat),
                )
            )
    if len(records) != 77:
        raise RuntimeError(f"Expected 77 auditable district rows in 15_Population_GDP; found {len(records)}.")

    road_nodes = [n for n, data in G.nodes(data=True) if not data.get("is_air_asset")]
    if not road_nodes:
        raise RuntimeError("No road nodes are available for socioeconomic allocation.")
    node_pos = {n: tuple(G.nodes[n]["pos"]) for n in road_nodes}
    node_points = {n: Point(node_pos[n]) for n in road_nodes}
    node_xy = np.array([node_pos[n] for n in road_nodes], dtype=float)
    lon_scale = math.cos(math.radians(float(np.mean(node_xy[:, 1]))))
    allocation_cap = max(1, int(cfg.socioeconomic_nodes_per_district))

    def nearest_road_node(point: Tuple[float, float]) -> Tuple[Any, float]:
        dx = (node_xy[:, 0] - point[0]) * lon_scale
        dy = node_xy[:, 1] - point[1]
        index = int(np.argmin(dx * dx + dy * dy))
        node = road_nodes[index]
        return node, _dist_km(point, node_pos[node])

    districts = geo.get("districts", gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326"))
    district_polygons: Dict[str, Any] = {}
    if districts is not None and not districts.empty:
        dname_col = next(
            (
                c for c in districts.columns
                if str(c).strip().lower() in {"district", "first_dist", "dist_name", "name"}
            ),
            None,
        )
        if dname_col:
            for _, drow in districts.iterrows():
                geom = drow.geometry
                if geom is None or geom.is_empty:
                    continue
                district_polygons[_district_key(drow.get(dname_col))] = geom

    settlement_points = _centroids(geo.get("settlement", gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")))

    def _spread_sample(candidates: Sequence[Any], anchor: Tuple[float, float]) -> List[Any]:
        unique = sorted(set(candidates), key=lambda n: (_dist_km(anchor, node_pos[n]), str(n)))
        if len(unique) <= allocation_cap:
            return unique
        selected = [unique[0]]
        remaining = unique[1:]
        while remaining and len(selected) < allocation_cap:
            nxt = max(
                remaining,
                key=lambda n: (
                    min(_dist_km(node_pos[n], node_pos[s]) for s in selected),
                    -_dist_km(anchor, node_pos[n]),
                    str(n),
                ),
            )
            selected.append(nxt)
            remaining.remove(nxt)
        return sorted(selected, key=lambda n: (_dist_km(anchor, node_pos[n]), str(n)))

    def _nodes_inside(geom: Any) -> List[Any]:
        minx, miny, maxx, maxy = geom.bounds
        out: List[Any] = []
        for n in road_nodes:
            lon, lat = node_pos[n]
            if minx <= lon <= maxx and miny <= lat <= maxy and geom.covers(node_points[n]):
                out.append(n)
        return out

    def _settlements_inside(geom: Any) -> List[Tuple[float, float]]:
        if not settlement_points:
            return []
        minx, miny, maxx, maxy = geom.bounds
        out: List[Tuple[float, float]] = []
        for pt in settlement_points:
            if minx <= pt[0] <= maxx and miny <= pt[1] <= maxy and geom.covers(Point(pt)):
                out.append(pt)
        return out

    def representative_nodes(record: Dict[str, Any]) -> Tuple[List[Any], str, bool, int]:
        geom = district_polygons.get(_district_key(record["district"]))
        nodes = _nodes_inside(geom) if geom is not None else []
        if nodes:
            return (
                _spread_sample(nodes, record["point"]),
                "district polygon -> representative NH/SH nodes",
                True,
                len(_settlements_inside(geom)),
            )
        nearest = sorted(
            road_nodes,
            key=lambda n: (_dist_km(record["point"], node_pos[n]), str(n)),
        )[:allocation_cap]
        return nearest, "district anchor -> nearest representative NH/SH nodes", False, 0

    for node in road_nodes:
        G.nodes[node]["population_weight"] = 0.0
        G.nodes[node]["gdp_weight"] = 0.0
        G.nodes[node]["socioeconomic_contributions"] = defaultdict(float)

    audit_rows: List[Dict[str, Any]] = []
    for _district_index, record in enumerate(records):
        selected_nodes, method, used_polygon, settlement_count = representative_nodes(record)
        if not selected_nodes:
            node, _snap = nearest_road_node(record["point"])
            selected_nodes = [node]
            method = "district anchor -> nearest NH/SH node assignment"
        geom = district_polygons.get(_district_key(record["district"]))
        settlement_weights: Dict[Any, float] = defaultdict(float)
        if used_polygon and geom is not None:
            for pt in _settlements_inside(geom):
                nearest_selected = min(selected_nodes, key=lambda n: _dist_km(pt, node_pos[n]))
                settlement_weights[nearest_selected] += 1.0
        raw_weights = {
            node: (0.25 + settlement_weights.get(node, 0.0)) if settlement_count else 1.0
            for node in selected_nodes
        }
        total_weight = max(sum(raw_weights.values()), 1e-12)
        for allocation_index, node in enumerate(selected_nodes, start=1):
            share = raw_weights[node] / total_weight
            population_share = record["population"] * share
            node_data = G.nodes[node]
            node_data["population_weight"] += population_share
            node_data["gdp_weight"] += population_share * record["gdp_per_capita_npr"]
            node_data["socioeconomic_contributions"][record["district"]] += population_share
            snap_distance = _dist_km(record["point"], node_pos[node])
            audit_rows.append(
                dict(
                    district=record["district"],
                    province=record["province"],
                    allocation_point=allocation_index,
                    allocation_method=method,
                    district_polygon_used=used_polygon,
                    district_allocation_nodes=len(selected_nodes),
                    settlement_centroids_used=settlement_count,
                    road_node=node,
                    road_node_longitude=node_data["pos"][0],
                    road_node_latitude=node_data["pos"][1],
                    snap_distance_km=snap_distance,
                    allocation_share=share,
                    allocation_weight=raw_weights[node],
                    population_weight=population_share,
                    gdp_per_capita_npr=record["gdp_per_capita_npr"],
                    gdp_weight_npr=population_share * record["gdp_per_capita_npr"],
                )
            )

    for node in road_nodes:
        contributions = dict(G.nodes[node].pop("socioeconomic_contributions"))
        G.nodes[node]["district_population_contributions"] = contributions
        if contributions:
            names = sorted(contributions, key=contributions.get, reverse=True)
            G.nodes[node]["socioeconomic_name"] = " / ".join(names[:3])
            G.nodes[node]["socioeconomic_districts"] = " | ".join(sorted(names))
        else:
            G.nodes[node]["socioeconomic_name"] = ""
            G.nodes[node]["socioeconomic_districts"] = ""

    expected_population = sum(record["population"] for record in records)
    allocated_population = sum(float(G.nodes[node]["population_weight"]) for node in road_nodes)
    if not math.isclose(expected_population, allocated_population, rel_tol=1e-10, abs_tol=1e-5):
        raise RuntimeError(
            f"Population allocation failed conservation check: expected {expected_population}, got {allocated_population}."
        )
    cfg.population_total = expected_population
    print(
        f"  socioeconomic allocation: {len(records)} districts -> "
        f"{sum(node_population(cfg, G.nodes[n]) > 0 for n in road_nodes)} populated road nodes | "
        f"max {allocation_cap}/district | population conserved: {allocated_population:,.0f}"
    )
    return pd.DataFrame(audit_rows)


# ------------------- four-dimensional criticality helpers --------------------


def _road_graph(G: nx.Graph, weight: str = "length_km", active_only: bool = False) -> nx.Graph:
    R = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d.get("type") == "road":
            if active_only and (d.get("blockages") or float(d.get("capacity", 0.0)) <= 0.0):
                continue
            R.add_edge(u, v, weight=d.get(weight, d.get("length_km", 1.0)), length_km=d.get("length_km", 1.0))
    return R


def _main_iso_after_removal(R: nx.Graph, u: int, v: int, node_weight: Optional[Dict[int, float]] = None) -> Tuple[set, set]:
    if u not in R or v not in R:
        return set(), set()
    base_component = nx.node_connected_component(R, u)
    if v not in base_component:
        return set(base_component), set()
    R2 = R.subgraph(base_component).copy()
    if R2.has_edge(u, v):
        R2.remove_edge(u, v)
    comps = list(nx.connected_components(R2))
    if len(comps) <= 1:
        return set(comps[0]) if comps else set(), set()
    if node_weight:
        main = max(comps, key=lambda c: sum(node_weight.get(n, 0.0) for n in c))
    else:
        main = max(comps, key=len)
    iso = set().union(*[c for c in comps if c is not main])
    return set(main), iso


def _iso_for_edge_state(R_state: nx.Graph, u: int, v: int, node_weight: Optional[Dict[int, float]] = None) -> Tuple[set, set]:
    if u not in R_state and v not in R_state:
        return set(), set()
    if u not in R_state:
        return set(nx.node_connected_component(R_state, v)), {v}
    if v not in R_state:
        return set(nx.node_connected_component(R_state, u)), {u}
    if not nx.has_path(R_state, u, v):
        comps = [set(nx.node_connected_component(R_state, u)), set(nx.node_connected_component(R_state, v))]
        if node_weight:
            main = max(comps, key=lambda c: sum(node_weight.get(n, 0.0) for n in c))
        else:
            main = max(comps, key=len)
        iso = set().union(*[c for c in comps if c is not main])
        return set(main), iso
    return _main_iso_after_removal(R_state, u, v, node_weight)


def _merge_dimension_scores(
    marginal: Dict[Tuple[int, int], Dict[str, float]],
    joint: Dict[Tuple[int, int], Dict[str, float]],
    component_keys: Sequence[str],
    composite_key: str,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key in set(marginal) | set(joint):
        row: Dict[str, float] = {}
        for c in component_keys:
            m = float(marginal.get(key, {}).get(c, 0.0))
            j = float(joint.get(key, {}).get(c, 0.0))
            row[c] = 0.5 * (m + j)
            row[f"{c}_marginal"] = m
            row[f"{c}_joint"] = j
        mcomp = float(marginal.get(key, {}).get(composite_key, 0.0))
        jcomp = float(joint.get(key, {}).get(composite_key, 0.0))
        row[composite_key] = 0.5 * (mcomp + jcomp)
        row[f"{composite_key}_marginal"] = mcomp
        row[f"{composite_key}_joint"] = jcomp
        out[key] = row
    return out


def _minmax(values: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not math.isfinite(hi) or hi <= lo + 1e-12:
        return {k: 0.0 for k in values}
    return {k: float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)) for k, v in values.items()}


def _analysis_edge_key(u: Any, v: Any) -> Tuple[Any, Any]:
    return (u, v) if str(u) <= str(v) else (v, u)


def _assignment_graph(cfg: MLPIConfig, G: nx.Graph, include_air: bool = True) -> nx.Graph:
    """Generalized-cost graph shared by all four dimensions."""
    R = nx.Graph()
    for u, v, d in G.edges(data=True):
        edge_type = d.get("type")
        if edge_type not in {"road", "road_connector", "airport_access", "air_route"}:
            continue
        if edge_type in {"airport_access", "air_route"} and not include_air:
            continue
        if edge_type == "airport_access":
            air_asset = u if G.nodes[u].get("is_air_asset") else v
            if G.nodes[air_asset].get("airport_status") == "Not in operation":
                continue
        key = _analysis_edge_key(u, v)
        length = float(d.get("length_km", 1.0))
        if edge_type == "air_route":
            t0 = cfg.air_fixed_penalty_h + cfg.air_cost_multiplier * length / max(cfg.speed_air, 1.0)
            R.add_edge(
                u,
                v,
                key=key,
                edge_type=edge_type,
                weight=t0,
                t0=t0,
                length_km=length,
                capacity=max(float(d.get("capacity", 10000.0)), 10000.0),
                aadt=0.0,
            )
            continue
        speed = float(d.get("speed_kmh") or cfg.speed_road)
        t0 = length / max(speed, 1.0)
        R.add_edge(
            u,
            v,
            key=key,
            edge_type=edge_type,
            weight=t0,
            t0=t0,
            length_km=length,
            capacity=max(float(d.get("aadt", cfg.aadt_default)), 300.0) if edge_type in {"road", "road_connector"} else 10000.0,
            aadt=float(d.get("aadt", cfg.aadt_default)),
            speed_kmh=speed,
            zone=d.get("zone", ""),
        )
    return R


def _node_activity(cfg: MLPIConfig, G: nx.Graph, nodes: Optional[Iterable[int]] = None) -> Dict[int, float]:
    nodes = list(nodes) if nodes is not None else [n for n, d in G.nodes(data=True) if not d.get("is_air_asset")]
    raw: Dict[int, float] = {}
    for n in nodes:
        d = G.nodes[n]
        pop = node_population(cfg, d)
        if pop <= 0.0:
            continue
        gdp = pop * node_gdp_per_capita(cfg, d)
        aadt_km = sum(
            float(ed.get("aadt", 0.0)) * max(float(ed.get("length_km", 0.0)), 0.10)
            for _, _, ed in G.edges(n, data=True)
            if ed.get("type") == "road"
        )
        raw[n] = max(pop, 1.0) ** 0.45 + max(gdp, 1.0) ** 0.25 + 1.50 * max(aadt_km, 1.0) ** 0.55
    return raw


def _macro_region_from_lon(lon: float) -> str:
    if lon < 82.5:
        return "West"
    if lon < 85.7:
        return "Central"
    return "East"


ME2_ZONE_LABEL_RE = re.compile(
    r"^\s*(?P<zone_id>\d+)\s+"
    r"(?P<district>.*?)\s*\|\s*"
    r"(?P<hq>.*?)(?:\s*\((?P<hq_district>[^()]*)\))?\s*$"
)


def _place_match_key(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"n\s*\.?\s*p\s*\.?", "", text)
    text = re.sub(r"\b(metropolitan|submetropolitan|municipality|airport)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _parse_me2_zone_label(label: Any) -> Dict[str, str]:
    text = str(label or "").strip()
    match = ME2_ZONE_LABEL_RE.match(text)
    if match:
        return {
            "zone_id": match.group("zone_id").strip(),
            "district": match.group("district").strip(),
            "hq_label": match.group("hq").strip(),
            "hq_district": (match.group("hq_district") or "").strip(),
            "zone_label": text,
        }
    left, sep, right = text.partition("|")
    zone_match = re.match(r"^\s*(\d+)\s+(.*?)\s*$", left)
    if sep and zone_match:
        return {
            "zone_id": zone_match.group(1).strip(),
            "district": zone_match.group(2).strip(),
            "hq_label": re.sub(r"\([^)]*\)\s*$", "", right).strip(),
            "hq_district": (re.search(r"\(([^()]*)\)\s*$", right) or ["", ""])[1].strip(),
            "zone_label": text,
        }
    raise ValueError(
        f"Could not parse ME2 OD zone label '{text}'. Expected labels like "
        "'1 Achham | Mangalsen (Achham)'."
    )


def _read_me2_od_matrix(cfg: MLPIConfig) -> pd.DataFrame:
    path = cfg.od_matrix_path.expanduser()
    raw = pd.read_csv(path, index_col=0)
    raw.index = raw.index.map(lambda value: str(value).strip())
    raw.columns = raw.columns.map(lambda value: str(value).strip())
    if raw.index.duplicated().any():
        dupes = raw.index[raw.index.duplicated()].tolist()
        raise ValueError(f"ME2 OD matrix has duplicate row labels: {dupes[:5]}")
    if pd.Index(raw.columns).duplicated().any():
        dupes = pd.Index(raw.columns)[pd.Index(raw.columns).duplicated()].tolist()
        raise ValueError(f"ME2 OD matrix has duplicate column labels: {dupes[:5]}")
    if set(raw.index) != set(raw.columns):
        missing_cols = sorted(set(raw.index) - set(raw.columns))[:8]
        missing_rows = sorted(set(raw.columns) - set(raw.index))[:8]
        raise ValueError(
            "ME2 OD matrix row/column labels do not match. "
            f"Rows without columns: {missing_cols}; columns without rows: {missing_rows}"
        )
    raw = raw.loc[list(raw.index), list(raw.index)]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    bad = np.argwhere(numeric.isna().to_numpy())
    if len(bad):
        sample = [
            f"{numeric.index[i]} -> {numeric.columns[j]}"
            for i, j in bad[:8]
        ]
        raise ValueError(f"ME2 OD matrix has non-numeric cells at: {sample}")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ME2 OD matrix contains infinite values.")
    if (values < -1e-9).any():
        raise ValueError("ME2 OD matrix contains negative demand values.")
    np.fill_diagonal(values, 0.0)
    numeric.iloc[:, :] = values
    return numeric


def _district_anchor_lookup(cfg: MLPIConfig) -> Dict[str, Tuple[float, float]]:
    table = cfg.population_gdp.copy()
    if table.empty:
        return {}
    normalized = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in table.columns}
    name_col = normalized.get("district") or normalized.get("city") or normalized.get("node")
    lon_col = normalized.get("longitude") or normalized.get("lon") or normalized.get("long")
    lat_col = normalized.get("latitude") or normalized.get("lat")
    if not (name_col and lon_col and lat_col):
        return {}
    anchors: Dict[str, Tuple[float, float]] = {}
    for _, row in table.iterrows():
        name = str(row.get(name_col, "")).strip()
        lon = _numeric_or_nan(row.get(lon_col))
        lat = _numeric_or_nan(row.get(lat_col))
        if name and math.isfinite(lon) and math.isfinite(lat):
            anchors[_district_key(name)] = (float(lon), float(lat))
    return anchors


def _zone_anchor_point(
    cfg: MLPIConfig,
    parsed: Dict[str, str],
    district_anchors: Dict[str, Tuple[float, float]],
) -> Tuple[Optional[Tuple[float, float]], str]:
    hq_key = _place_match_key(parsed.get("hq_label", ""))
    if hq_key:
        candidates: List[Tuple[int, float, float, str]] = []
        for lon, lat, name, _pop in cfg.place_lookup:
            place_key = _place_match_key(name)
            if place_key and (hq_key == place_key or hq_key in place_key or place_key in hq_key):
                candidates.append((abs(len(place_key) - len(hq_key)), float(lon), float(lat), str(name)))
        if candidates:
            _score, lon, lat, name = min(candidates, key=lambda item: (item[0], item[3]))
            return (lon, lat), f"named place match: {name}"
    district_key = _district_key(parsed.get("district", ""))
    if district_key in district_anchors:
        return district_anchors[district_key], "district anchor from 15_Population_GDP"
    return None, "no coordinate anchor available"


def _build_me2_zone_mapping(
    cfg: MLPIConfig,
    G: nx.Graph,
    R: nx.Graph,
    labels: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    road_nodes = [n for n in R.nodes if n in G.nodes and not G.nodes[n].get("is_air_asset")]
    if not road_nodes:
        raise RuntimeError("No road nodes are available for ME2 OD zone mapping.")
    district_candidates: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
    for node in road_nodes:
        contributions = G.nodes[node].get("district_population_contributions", {})
        for district, population in dict(contributions).items():
            pop = float(population)
            if pop > 0.0:
                district_candidates[_district_key(district)].append((node, pop))

    district_anchors = _district_anchor_lookup(cfg)
    node_pos = {n: tuple(G.nodes[n]["pos"]) for n in road_nodes}
    mapping: Dict[str, Dict[str, Any]] = {}
    audit_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for label in labels:
        parsed = _parse_me2_zone_label(label)
        district_key = _district_key(parsed["district"])
        anchor, anchor_source = _zone_anchor_point(cfg, parsed, district_anchors)
        candidates = district_candidates.get(district_key, [])
        method = "district population allocation -> representative road node"
        if candidates:
            if anchor is not None:
                node, contribution = min(
                    candidates,
                    key=lambda item: (_dist_km(anchor, node_pos[item[0]]), -item[1], str(item[0])),
                )
                snap_distance = _dist_km(anchor, node_pos[node])
            else:
                node, contribution = max(candidates, key=lambda item: (item[1], str(item[0])))
                snap_distance = float("nan")
        elif anchor is not None:
            node, snap_distance = min(
                ((n, _dist_km(anchor, node_pos[n])) for n in road_nodes),
                key=lambda item: (item[1], str(item[0])),
            )
            contribution = 0.0
            method = "district/HQ anchor -> nearest road node assignment"
        else:
            missing.append(f"{label}: no socioeconomic node allocation and no anchor")
            continue

        row = {
            **parsed,
            "district_key": district_key,
            "road_node": node,
            "road_node_longitude": node_pos[node][0],
            "road_node_latitude": node_pos[node][1],
            "anchor_longitude": anchor[0] if anchor else np.nan,
            "anchor_latitude": anchor[1] if anchor else np.nan,
            "anchor_source": anchor_source,
            "snap_distance_km": snap_distance,
            "node_population_contribution": contribution,
            "mapping_method": method,
        }
        mapping[label] = row
        audit_rows.append(row)

    if missing:
        raise RuntimeError(
            "Could not map every ME2 OD zone to the MLPI road graph: "
            + "; ".join(missing[:12])
        )
    cfg.od_zone_mapping_audit = pd.DataFrame(audit_rows)
    return mapping


def load_me2_od_pairs(cfg: MLPIConfig, G: nx.Graph, R: nx.Graph) -> List[Dict[str, Any]]:
    matrix = _read_me2_od_matrix(cfg)
    labels = list(matrix.index)
    parsed_by_label = {label: _parse_me2_zone_label(label) for label in labels}
    zone_mapping = _build_me2_zone_mapping(cfg, G, R, labels)

    ods: List[Dict[str, Any]] = []
    total_positive_demand = 0.0
    same_node_pairs = 0
    same_node_demand = 0.0
    for i, origin_label in enumerate(labels):
        origin_map = zone_mapping[origin_label]
        origin_parsed = parsed_by_label[origin_label]
        for j, destination_label in enumerate(labels):
            if i == j:
                continue
            demand = float(matrix.iat[i, j])
            if demand <= 0.0:
                continue
            total_positive_demand += demand
            destination_map = zone_mapping[destination_label]
            destination_parsed = parsed_by_label[destination_label]
            if origin_map["road_node"] == destination_map["road_node"]:
                same_node_pairs += 1
                same_node_demand += demand
                continue
            ods.append(
                dict(
                    o=origin_map["road_node"],
                    d=destination_map["road_node"],
                    demand=demand,
                    origin_zone_label=origin_label,
                    destination_zone_label=destination_label,
                    origin_zone_id=origin_parsed["zone_id"],
                    destination_zone_id=destination_parsed["zone_id"],
                    origin_district=origin_parsed["district"],
                    destination_district=destination_parsed["district"],
                    origin_hq=origin_parsed["hq_label"],
                    destination_hq=destination_parsed["hq_label"],
                    origin_node_snap_km=origin_map["snap_distance_km"],
                    destination_node_snap_km=destination_map["snap_distance_km"],
                    od_model="external ME2 OD matrix",
                    od_source=str(cfg.od_matrix_path),
                    demand_units="ME2 output trips/day",
                    base_connected=False,
                    base_time_h=np.nan,
                    base_path_edge_count=0,
                )
            )

    cfg.od_matrix_total_demand = total_positive_demand
    cfg.od_pairs_loaded = len(ods)
    cfg.od_pairs_same_node_skipped = same_node_pairs
    cfg.od_same_node_demand_skipped = same_node_demand
    if not ods:
        raise RuntimeError(
            f"ME2 OD matrix {cfg.od_matrix_path} has no positive inter-node OD pairs after zone mapping."
        )
    print(
        f"  ME2 OD matrix loaded: {len(labels)} zones | {len(ods)} positive inter-node OD pairs | "
        f"matrix demand {total_positive_demand:,.3f}"
    )
    if same_node_pairs:
        print(
            f"  ME2 OD same-node pairs skipped: {same_node_pairs} "
            f"({same_node_demand:,.3f} demand)"
        )
    return ods


def build_gravity_od(cfg: MLPIConfig, G: nx.Graph, R: nx.Graph) -> List[Dict[str, float]]:
    raise RuntimeError(
        "MLPI requires load_me2_od_pairs() with the calibrated ME2 OD CSV."
    )


def _edge_time_attrs(cfg: MLPIConfig, R: nx.Graph, flows: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    times = {}
    for u, v, d in R.edges(data=True):
        key = d["key"]
        x = max(0.0, flows.get(key, 0.0))
        cap = max(float(d.get("capacity", 1.0)), 1.0)
        times[key] = float(d.get("t0", d.get("weight", 1.0))) * (1.0 + cfg.bpr_alpha * (x / cap) ** cfg.bpr_beta_power)
    return times


def _all_or_nothing(R: nx.Graph, ods: List[Dict[str, float]], times: Dict[Tuple[int, int], float]) -> Tuple[Dict[Tuple[int, int], float], float]:
    flows = {d["key"]: 0.0 for _, _, d in R.edges(data=True)}
    for u, v, d in R.edges(data=True):
        d["fw_time"] = times.get(d["key"], d.get("t0", 1.0))
    disconnected = 0.0
    ods_by_origin: Dict[Any, List[Dict[str, float]]] = defaultdict(list)
    for od in ods:
        ods_by_origin[od["o"]].append(od)
    for origin, origin_ods in ods_by_origin.items():
        if origin not in R:
            disconnected += sum(float(od["demand"]) for od in origin_ods)
            continue
        _lengths, paths = nx.single_source_dijkstra(R, origin, weight="fw_time")
        for od in origin_ods:
            destination, demand = od["d"], float(od["demand"])
            path = paths.get(destination)
            if not path:
                disconnected += demand
                continue
            for a, b in zip(path, path[1:]):
                flows[R[a][b]["key"]] += demand
    return flows, disconnected


def _edge_arrays(R: nx.Graph) -> Tuple[List[Tuple[Any, Any]], np.ndarray, np.ndarray]:
    keys = [d["key"] for _, _, d in R.edges(data=True)]
    t0 = np.array([float(d.get("t0", d.get("weight", 1.0))) for _, _, d in R.edges(data=True)], dtype=float)
    cap = np.array([max(float(d.get("capacity", 1.0)), 1.0) for _, _, d in R.edges(data=True)], dtype=float)
    return keys, t0, cap


def _flow_array(keys: Sequence[Tuple[Any, Any]], flows: Dict[Tuple[Any, Any], float]) -> np.ndarray:
    return np.array([float(flows.get(k, 0.0)) for k in keys], dtype=float)


def _array_to_flow(keys: Sequence[Tuple[Any, Any]], arr: np.ndarray) -> Dict[Tuple[Any, Any], float]:
    return {k: float(max(v, 0.0)) for k, v in zip(keys, arr)}


def _bpr_times_array(cfg: MLPIConfig, t0: np.ndarray, cap: np.ndarray, flows: np.ndarray) -> np.ndarray:
    x = np.maximum(flows, 0.0)
    return t0 * (1.0 + cfg.bpr_alpha * np.power(x / np.maximum(cap, 1.0), cfg.bpr_beta_power))


def _bpr_derivative_array(cfg: MLPIConfig, t0: np.ndarray, cap: np.ndarray, flows: np.ndarray) -> np.ndarray:
    if cfg.bpr_alpha == 0.0:
        return np.zeros_like(flows, dtype=float)
    x = np.maximum(flows, 0.0)
    beta = cfg.bpr_beta_power
    return t0 * cfg.bpr_alpha * beta * np.power(x, beta - 1.0) / np.power(np.maximum(cap, 1.0), beta)


def _relative_gap(flow: np.ndarray, aon: np.ndarray, times: np.ndarray) -> float:
    current = float(np.dot(flow, times))
    shortest = float(np.dot(aon, times))
    if current <= 1e-12:
        return 0.0
    return max(0.0, (current - shortest) / current)


def _line_search_bpr(cfg: MLPIConfig, t0: np.ndarray, cap: np.ndarray, flow: np.ndarray, target: np.ndarray) -> float:
    direction = target - flow
    if float(np.max(np.abs(direction))) <= 1e-12:
        return 0.0

    def derivative(alpha: float) -> float:
        return float(np.dot(_bpr_times_array(cfg, t0, cap, flow + alpha * direction), direction))

    d0 = derivative(0.0)
    d1 = derivative(1.0)
    if d0 >= -1e-10:
        return 0.0
    if d1 <= 0.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if derivative(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _conjugate_target(flow: np.ndarray, aon: np.ndarray, prev_sd: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    den = float(np.dot((prev_sd - flow) * (aon - prev_sd), derivative))
    if abs(den) <= 1e-12:
        return aon
    alpha = float(np.dot((prev_sd - flow) * (aon - flow), derivative) / den)
    if not math.isfinite(alpha):
        alpha = 0.0
    alpha = float(np.clip(alpha, 0.0, 0.99999))
    return (1.0 - alpha) * aon + alpha * prev_sd


def _biconjugate_target(
    flow: np.ndarray,
    aon: np.ndarray,
    prev_sd: np.ndarray,
    prev_prev_sd: np.ndarray,
    derivative: np.ndarray,
    prev_step: float,
) -> np.ndarray:
    step = float(np.clip(prev_step, 1e-6, 0.99999))
    x_vec = prev_sd * step + prev_prev_sd * (1.0 - step) - flow
    y_vec = aon - flow
    z_vec = prev_sd - flow
    w_vec = prev_prev_sd - prev_sd
    den_mu = float(np.dot(x_vec * w_vec, derivative))
    mu = -float(np.dot(x_vec * y_vec, derivative)) / den_mu if abs(den_mu) > 1e-12 else 0.0
    mu = max(mu, 0.0) if math.isfinite(mu) else 0.0
    den_nu = float(np.dot(z_vec * z_vec, derivative))
    nu = -float(np.dot(z_vec * y_vec, derivative)) / den_nu + mu * step / (1.0 - step) if abs(den_nu) > 1e-12 else 0.0
    nu = max(nu, 0.0) if math.isfinite(nu) else 0.0
    beta0 = 1.0 / max(1.0 + nu + mu, 1e-12)
    target = beta0 * aon + (nu * beta0) * prev_sd + (mu * beta0) * prev_prev_sd
    if not np.all(np.isfinite(target)):
        return _conjugate_target(flow, aon, prev_sd, derivative)
    return target


def frank_wolfe_user_equilibrium(cfg: MLPIConfig, R: nx.Graph, ods: List[Dict[str, float]]) -> Dict[str, Any]:
    if R.number_of_edges() == 0 or not ods:
        return dict(flows={}, times={}, tstt=0.0, disconnected_demand=0.0, penalty_tstt=0.0, rgap=0.0, iterations=0, algorithm="bfw")
    keys, t0_arr, cap_arr = _edge_arrays(R)
    free = {d["key"]: float(d.get("t0", d.get("weight", 1.0))) for _, _, d in R.edges(data=True)}
    flows0, disconnected = _all_or_nothing(R, ods, free)
    flow = _flow_array(keys, flows0)
    prev_sd: Optional[np.ndarray] = None
    prev_prev_sd: Optional[np.ndarray] = None
    prev_step = 1.0
    rgap = float("inf")
    iterations = 0
    history: List[Dict[str, float]] = []
    for it in range(1, cfg.fw_max_iter + 1):
        iterations = it
        time_arr = _bpr_times_array(cfg, t0_arr, cap_arr, flow)
        times_now = {k: float(v) for k, v in zip(keys, time_arr)}
        aux, disconnected = _all_or_nothing(R, ods, times_now)
        aon = _flow_array(keys, aux)
        rgap = _relative_gap(flow, aon, time_arr)
        history.append(dict(iteration=float(it), relative_gap=float(rgap)))
        if rgap <= cfg.assignment_rgap_target:
            break
        derivative = _bpr_derivative_array(cfg, t0_arr, cap_arr, flow)
        if prev_sd is None:
            target = aon
        elif prev_prev_sd is None:
            target = _conjugate_target(flow, aon, prev_sd, derivative)
        else:
            target = _biconjugate_target(flow, aon, prev_sd, prev_prev_sd, derivative, prev_step)
        step = _line_search_bpr(cfg, t0_arr, cap_arr, flow, target)
        if step <= 1e-10:
            target = aon
            step = _line_search_bpr(cfg, t0_arr, cap_arr, flow, target)
        flow = np.maximum(flow + step * (target - flow), 0.0)
        prev_prev_sd = prev_sd
        prev_sd = target.copy()
        prev_step = step
    final_time_arr = _bpr_times_array(cfg, t0_arr, cap_arr, flow)
    final_times = {k: float(v) for k, v in zip(keys, final_time_arr)}
    final_aux, disconnected = _all_or_nothing(R, ods, final_times)
    rgap = _relative_gap(flow, _flow_array(keys, final_aux), final_time_arr)
    flows = _array_to_flow(keys, flow)
    times = _edge_time_attrs(cfg, R, flows)
    tstt = sum(flows.get(d["key"], 0.0) * times.get(d["key"], d.get("t0", 1.0)) for _, _, d in R.edges(data=True))
    penalty = 0.0
    for od in ods:
        o, dst, demand = int(od["o"]), int(od["d"]), float(od["demand"])
        if o not in R or dst not in R or not nx.has_path(R, o, dst):
            penalty += demand * max(cfg.disconnected_penalty_h, float(od.get("base_time", 0.0)) * 5.0)
    return dict(
        flows=flows,
        times=times,
        tstt=tstt + penalty,
        disconnected_demand=disconnected,
        penalty_tstt=penalty,
        rgap=float(rgap if math.isfinite(rgap) else 0.0),
        iterations=iterations,
        algorithm="bfw",
        convergence_history=history,
    )


def compute_physical_fftdi(cfg: MLPIConfig, G: nx.Graph) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], pd.DataFrame, List[Dict[str, float]]]:
    """FFTDI-style physical importance from ME2 OD loading and link removal."""
    R = _assignment_graph(cfg, G)
    ods = load_me2_od_pairs(cfg, G, R)
    base = frank_wolfe_user_equilibrium(cfg, R, ods)
    base_tstt = max(float(base.get("tstt", 0.0)), 1e-9)
    raw_fftdi: Dict[Tuple[int, int], float] = {}
    disconn: Dict[Tuple[int, int], float] = {}
    flow_share: Dict[Tuple[int, int], float] = {}
    total_demand = max(sum(float(od["demand"]) for od in ods), 1.0)
    max_flow = max(base.get("flows", {}).values(), default=1.0)
    for u, v, d in R.edges(data=True):
        key = d["key"]
        R2 = R.copy()
        if R2.has_edge(u, v):
            R2.remove_edge(u, v)
        perf = frank_wolfe_user_equilibrium(cfg, R2, ods)
        raw_fftdi[key] = max(0.0, (float(perf.get("tstt", 0.0)) - base_tstt) / base_tstt)
        disconn[key] = float(perf.get("disconnected_demand", 0.0)) / total_demand
        flow_share[key] = float(base.get("flows", {}).get(key, 0.0)) / max(max_flow, 1e-9)
    fftdi_norm = _minmax(raw_fftdi)
    disc_norm = _minmax(disconn)
    flow_norm = _minmax(flow_share)
    out = {}
    for u, v, d in R.edges(data=True):
        key = d["key"]
        gd = G[key[0]][key[1]] if G.has_edge(key[0], key[1]) else {}
        prior = float(gd.get("physical_prior", 0.0))
        base_score = 0.65 * fftdi_norm.get(key, 0.0) + 0.20 * disc_norm.get(key, 0.0) + 0.10 * flow_norm.get(key, 0.0) + 0.05 * prior
        out[key] = dict(
            FFTDI_raw=raw_fftdi.get(key, 0.0),
            FFTDI=fftdi_norm.get(key, 0.0),
            P1_FFTDI=fftdi_norm.get(key, 0.0),
            P2_disconnected_demand=disc_norm.get(key, 0.0),
            P3_UE_flow=flow_norm.get(key, 0.0),
            P4_corridor_context=prior,
            physical=float(np.clip(base_score, 0.0, 1.0)),
            base_ue_tstt_h=base_tstt,
            ue_flow=base.get("flows", {}).get(key, 0.0),
        )
    od_df = pd.DataFrame(ods)
    return out, od_df, ods


def compute_social(cfg: MLPIConfig, G: nx.Graph, geo: Dict[str, gpd.GeoDataFrame], state_graph: Optional[nx.Graph] = None, active_only: bool = False) -> Dict[Tuple[int, int], Dict[str, float]]:
    R = _assignment_graph(cfg, state_graph or G)
    road_nodes = [n for n in R.nodes if n in G.nodes]
    node_pop = {n: node_population(cfg, G.nodes[n]) for n in road_nodes}
    total_pop = max(sum(node_pop.values()), 1.0)
    healthcare_pts = [(float(g.x), float(g.y)) for g in geo["healthcare"].geometry if isinstance(g, Point)]
    healthcare_nodes: set[int] = set()
    if healthcare_pts and road_nodes:
        for hp in healthcare_pts:
            n, dist = min(((rn, _dist_km(hp, G.nodes[rn]["pos"])) for rn in road_nodes), key=lambda x: x[1])
            if dist <= 35.0:
                healthcare_nodes.add(n)
    if not healthcare_nodes and road_nodes:
        warnings.warn("No healthcare facilities matched; using highest-density road node as proxy.")
        healthcare_nodes = {max(road_nodes, key=lambda n: G.nodes[n].get("healthcare_count", 0))}
    base_health = nx.multi_source_dijkstra_path_length(R, list(healthcare_nodes), weight="weight") if healthcare_nodes else {}
    raw_s1: Dict[Tuple[int, int], float] = {}
    raw_s2: Dict[Tuple[int, int], float] = {}
    for u, v, d in R.edges(data=True):
        key = d["key"]
        _, iso = _iso_for_edge_state(R, u, v, node_pop)
        iso_pop = sum(node_pop.get(n, 0.0) for n in iso)
        raw_s1[key] = iso_pop / total_pop
        R2 = R.copy()
        R2.remove_edge(u, v)
        sources = [n for n in healthcare_nodes if n in R2]
        after_health = nx.multi_source_dijkstra_path_length(R2, sources, weight="weight") if sources else {}
        inc = 0.0
        for n in road_nodes:
            base_t = float(base_health.get(n, cfg.disconnected_penalty_h))
            after_t = float(after_health.get(n, cfg.disconnected_penalty_h + base_t))
            inc += node_pop.get(n, 0.0) * max(0.0, after_t - base_t)
        raw_s2[key] = inc / total_pop
    n_s1 = _minmax(raw_s1)
    n_s2 = _minmax(raw_s2)
    total_w = max(cfg.w_s1 + cfg.w_s2, 1e-9)
    out = {}
    for key in raw_s1:
        prior = float(G[key[0]][key[1]].get("social_prior", 0.0)) if G.has_edge(key[0], key[1]) else 0.0
        base = (cfg.w_s1 * n_s1.get(key, 0.0) + cfg.w_s2 * n_s2.get(key, 0.0)) / total_w
        out[key] = dict(S1=n_s1.get(key, 0.0), S2=n_s2.get(key, 0.0), S3=prior, social=float(np.clip(0.90 * base + 0.10 * prior, 0.0, 1.0)), S1_raw=raw_s1.get(key, 0.0), S2_raw=raw_s2.get(key, 0.0))
    return out


def compute_economic(cfg: MLPIConfig, G: nx.Graph, state_graph: Optional[nx.Graph] = None, active_only: bool = False) -> Dict[Tuple[int, int], Dict[str, float]]:
    R = _assignment_graph(cfg, state_graph or G)
    nodes = list(R.nodes)
    node_gdp = {n: node_population(cfg, G.nodes[n]) * node_gdp_per_capita(cfg, G.nodes[n]) for n in nodes}
    total_gdp = max(sum(node_gdp.values()), 1.0)
    gateways: List[Tuple[int, float]] = []
    for lon, lat, volume, _name in cfg.border_trade:
        if nodes:
            n, _dist = min(((rn, _dist_km((lon, lat), G.nodes[rn]["pos"])) for rn in nodes), key=lambda x: x[1])
            gateways.append((n, max(float(volume), 1.0)))
    if not gateways and nodes:
        for n in sorted(nodes, key=lambda x: node_gdp.get(x, 0.0), reverse=True)[:5]:
            gateways.append((n, 1.0))

    def gateway_access(graph: nx.Graph) -> Dict[int, float]:
        acc = {n: 0.0 for n in graph.nodes}
        for g, volume in gateways:
            if g not in graph:
                continue
            lens = nx.single_source_dijkstra_path_length(graph, g, weight="weight")
            for n, t in lens.items():
                acc[n] += volume * math.exp(-cfg.beta * float(t) / max(cfg.od_deterrence_h, 1e-9))
        return acc

    base_access = gateway_access(R)
    raw_e1: Dict[Tuple[int, int], float] = {}
    raw_e2: Dict[Tuple[int, int], float] = {}
    raw_e3: Dict[Tuple[int, int], float] = {}
    for u, v, d in R.edges(data=True):
        key = d["key"]
        _, iso = _iso_for_edge_state(R, u, v, node_gdp)
        raw_e1[key] = sum(node_gdp.get(n, 0.0) for n in iso) / total_gdp
        prior = float(G[key[0]][key[1]].get("economic_prior", 0.0)) if G.has_edge(key[0], key[1]) else 0.0
        raw_e2[key] = float(d.get("aadt", cfg.aadt_default)) * float(d.get("length_km", 1.0)) * (1.0 + 1.25 * prior)
        R2 = R.copy()
        R2.remove_edge(u, v)
        after_access = gateway_access(R2)
        raw_e3[key] = sum(node_gdp.get(n, 0.0) * max(0.0, base_access.get(n, 0.0) - after_access.get(n, 0.0)) for n in nodes) / total_gdp
    n_e1, n_e2, n_e3 = _minmax(raw_e1), _minmax(raw_e2), _minmax(raw_e3)
    total_w = max(cfg.w_e1 + cfg.w_e2 + cfg.w_e3, 1e-9)
    out = {}
    for key in raw_e1:
        prior = float(G[key[0]][key[1]].get("economic_prior", 0.0)) if G.has_edge(key[0], key[1]) else 0.0
        base = (cfg.w_e1 * n_e1.get(key, 0.0) + cfg.w_e2 * n_e2.get(key, 0.0) + cfg.w_e3 * n_e3.get(key, 0.0)) / total_w
        out[key] = dict(E1=n_e1.get(key, 0.0), E2=n_e2.get(key, 0.0), E3=n_e3.get(key, 0.0), economic=float(np.clip(0.75 * base + 0.25 * prior, 0.0, 1.0)), E1_raw=raw_e1.get(key, 0.0), E2_raw=raw_e2.get(key, 0.0), E3_raw=raw_e3.get(key, 0.0))
    return out


def compute_interconnected(cfg: MLPIConfig, G: nx.Graph, state_graph: Optional[nx.Graph] = None, active_only: bool = False) -> Dict[Tuple[int, int], Dict[str, float]]:
    R = _assignment_graph(cfg, state_graph or G)
    nodes = list(R.nodes)
    activity = _node_activity(cfg, G, nodes)
    service_nodes: set[Any] = set()
    for u, v, d in G.edges(data=True):
        if d.get("type") != "airport_access":
            continue
        u_asset = bool(G.nodes[u].get("is_air_asset"))
        v_asset = bool(G.nodes[v].get("is_air_asset"))
        if u_asset == v_asset:
            continue
        asset = u if u_asset else v
        road_node = v if u_asset else u
        if road_node in nodes and G.nodes[asset].get("airport_status") != "Not in operation":
            service_nodes.add(road_node)
    if not service_nodes and nodes:
        service_nodes = {max(nodes, key=lambda n: activity.get(n, 0.0))}
    sources = sorted(nodes, key=lambda n: activity.get(n, 0.0), reverse=True)[: min(40, len(nodes))]
    targets = list(service_nodes)[: min(40, len(service_nodes))]
    try:
        subset_bc = nx.edge_betweenness_centrality_subset(R, sources=sources, targets=targets, normalized=True, weight="weight") if sources and targets else {}
    except Exception:
        subset_bc = {}
    base_service = nx.multi_source_dijkstra_path_length(R, list(service_nodes), weight="weight") if service_nodes else {}
    sample_nodes = sorted(nodes, key=lambda n: activity.get(n, 0.0), reverse=True)[: min(36, len(nodes))]
    sample_pairs = [(a, b) for a in sample_nodes for b in service_nodes if a != b and nx.has_path(R, a, b)]
    def efficiency(graph: nx.Graph) -> float:
        total = 0.0
        for a, b in sample_pairs:
            if a in graph and b in graph:
                try:
                    dist = nx.shortest_path_length(graph, a, b, weight="weight")
                    total += 1.0 / max(float(dist), 1e-6)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        return total
    base_eff = max(efficiency(R), 1e-9)
    raw_i1: Dict[Tuple[int, int], float] = {}
    raw_i2: Dict[Tuple[int, int], float] = {}
    raw_i3: Dict[Tuple[int, int], float] = {}
    total_activity = max(sum(activity.values()), 1.0)
    for u, v, d in R.edges(data=True):
        key = d["key"]
        R2 = R.copy()
        R2.remove_edge(u, v)
        after_service = nx.multi_source_dijkstra_path_length(R2, [n for n in service_nodes if n in R2], weight="weight") if service_nodes else {}
        raw_i1[key] = sum(activity.get(n, 0.0) * max(0.0, float(after_service.get(n, cfg.disconnected_penalty_h)) - float(base_service.get(n, cfg.disconnected_penalty_h))) for n in nodes) / total_activity
        raw_i2[key] = float(subset_bc.get((u, v), subset_bc.get((v, u), 0.0)))
        adjacent_air = 1.0 if (u in service_nodes or v in service_nodes) else 0.0
        prior = float(G[key[0]][key[1]].get("interconnected_prior", 0.0)) if G.has_edge(key[0], key[1]) else 0.0
        raw_i3[key] = max(0.0, (base_eff - efficiency(R2)) / base_eff) + 0.25 * adjacent_air + 0.15 * prior
    n_i1, n_i2, n_i3 = _minmax(raw_i1), _minmax(raw_i2), _minmax(raw_i3)
    total_w = max(cfg.w_i1 + cfg.w_i2 + cfg.w_i3, 1e-9)
    out = {}
    for key in raw_i1:
        prior = float(G[key[0]][key[1]].get("interconnected_prior", 0.0)) if G.has_edge(key[0], key[1]) else 0.0
        base = (cfg.w_i1 * n_i1.get(key, 0.0) + cfg.w_i2 * n_i2.get(key, 0.0) + cfg.w_i3 * n_i3.get(key, 0.0)) / total_w
        out[key] = dict(I1=n_i1.get(key, 0.0), I2=n_i2.get(key, 0.0), I3=n_i3.get(key, 0.0), interconnected=float(np.clip(0.85 * base + 0.15 * prior, 0.0, 1.0)), I1_raw=raw_i1.get(key, 0.0), I2_raw=raw_i2.get(key, 0.0), I3_raw=raw_i3.get(key, 0.0))
    return out


def build_named_sections(
    cfg: MLPIConfig,
    G: nx.Graph,
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
) -> List[Dict[str, Any]]:
    """Build place-to-place road sections when explicit section specs are supplied."""
    R = _assignment_graph(cfg, G, include_air=False)
    road_nodes = [n for n in R.nodes if not G.nodes[n].get("is_air_asset")]
    anchors: Dict[str, Dict[str, Any]] = {}
    used_nodes: set[Any] = set()
    for lon, lat, name, _pop in cfg.place_lookup:
        if name in anchors or not road_nodes:
            continue
        node, dist = min(((n, _dist_km((lon, lat), G.nodes[n]["pos"])) for n in road_nodes), key=lambda x: x[1])
        if dist <= cfg.section_anchor_max_km and node not in used_nodes:
            anchors[name] = dict(node=node, dist=dist, pos=(lon, lat))
            used_nodes.add(node)

    sections: List[Dict[str, Any]] = []
    seen_paths: set[frozenset] = set()
    skipped: List[str] = []
    for spec in cfg.section_specs:
        place_sequence = list(spec)
        origin, destination = place_sequence[0], place_sequence[-1]
        section_name = "-".join(place_sequence)
        if any(place not in anchors for place in place_sequence):
            skipped.append(f"{section_name}: missing anchor")
            continue
        u, v = anchors[origin]["node"], anchors[destination]["node"]
        path: List[Any] = []
        try:
            for a_name, b_name in zip(place_sequence, place_sequence[1:]):
                leg = nx.shortest_path(R, anchors[a_name]["node"], anchors[b_name]["node"], weight="length_km")
                path.extend(leg if not path else leg[1:])
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped.append(f"{section_name}: no path")
            continue
        analysis_edges = [_analysis_edge_key(a, b) for a, b in zip(path, path[1:])]
        road_edges = [
            e
            for e in analysis_edges
            if G.has_edge(e[0], e[1]) and G[e[0]][e[1]].get("type") == "road" and e in sp
        ]
        if not road_edges:
            skipped.append(f"{section_name}: no plotted road edges")
            continue
        length_km = sum(float(R[a][b].get("length_km", 0.0)) for a, b in zip(path, path[1:]))
        direct_km = _dist_km(anchors[origin]["pos"], anchors[destination]["pos"])
        detour = length_km / max(direct_km, 1.0)
        if length_km > cfg.section_max_length_km or detour > cfg.section_max_detour_ratio:
            skipped.append(f"{section_name}: implausible path {length_km:.1f} km / {detour:.2f}x")
            continue
        signature = frozenset(road_edges)
        if signature in seen_paths:
            continue
        seen_paths.add(signature)
        road_length = sum(float(G[a][b].get("length_km", 0.0)) for a, b in road_edges)
        aadt_num = sum(float(G[a][b].get("aadt", cfg.aadt_default)) * float(G[a][b].get("length_km", 0.0)) for a, b in road_edges)
        mean_aadt = aadt_num / max(road_length, 1e-9)
        surfaces = [str(G[a][b].get("surface", "")) for a, b in road_edges]
        surface_sources = [str(G[a][b].get("surface_source", "")) for a, b in road_edges]
        aadt_sources = [str(G[a][b].get("aadt_source", "")) for a, b in road_edges]
        speed_sources = [str(G[a][b].get("speed_source", "")) for a, b in road_edges]
        terrain_zones = [str(G[a][b].get("zone", "")) for a, b in road_edges]
        travel_speed_num = sum(float(G[a][b].get("speed_kmh", cfg.speed_road)) * float(G[a][b].get("length_km", 0.0)) for a, b in road_edges)
        section_id = f"SEC{len(sections) + 1:02d}"
        midpoint = G.nodes[path[len(path) // 2]]["pos"]
        sections.append(
            dict(
                section_id=section_id,
                section_name=section_name,
                origin_place=origin,
                destination_place=destination,
                origin_longitude=anchors[origin]["pos"][0],
                origin_latitude=anchors[origin]["pos"][1],
                destination_longitude=anchors[destination]["pos"][0],
                destination_latitude=anchors[destination]["pos"][1],
                origin_anchor_distance_km=anchors[origin]["dist"],
                destination_anchor_distance_km=anchors[destination]["dist"],
                origin_node=u,
                destination_node=v,
                path_nodes=path,
                analysis_edges=analysis_edges,
                road_edges=road_edges,
                length_km=length_km,
                direct_km=direct_km,
                detour_ratio=detour,
                edge_count=len(road_edges),
                connector_edge_count=max(0, len(analysis_edges) - len(road_edges)),
                mean_aadt=mean_aadt,
                aadt_source=sorted(set(aadt_sources), key=lambda value: (-aadt_sources.count(value), value))[0] if aadt_sources else "input_network_median",
                travel_speed_kmh=travel_speed_num / max(road_length, 1e-9),
                model_speed_kmh=travel_speed_num / max(road_length, 1e-9),
                model_speed_source=sorted(set(speed_sources), key=lambda value: (-speed_sources.count(value), value))[0] if speed_sources else f"mean_speed_{cfg.mean_speed_factor:.2f}_design_speed",
                terrain_zone=sorted(set(terrain_zones), key=lambda value: (-terrain_zones.count(value), value))[0] if terrain_zones else "",
                surface=sorted(set(surfaces), key=lambda value: (-surfaces.count(value), value))[0] if surfaces else "",
                surface_source=sorted(set(surface_sources), key=lambda value: (-surface_sources.count(value), value))[0] if surface_sources else "",
                midpoint=midpoint,
                macro_region=_macro_region_from_lon(midpoint[0]),
            )
        )
    print(f"  named sections built: {len(sections)}")
    if skipped:
        print(f"  named sections skipped: {len(skipped)}")
        for reason in skipped:
            print(f"    - {reason}")
    return sections


def build_junction_links(
    cfg: MLPIConfig,
    G: nx.Graph,
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
) -> List[Dict[str, Any]]:
    """Aggregate the road graph into non-overlapping road links."""
    R = nx.Graph()
    for u, v, data in G.edges(data=True):
        if data.get("type") in {"road", "road_connector"}:
            R.add_edge(u, v)
    if R.number_of_edges() == 0:
        return []

    def physical_incident(node: Any) -> List[Dict[str, Any]]:
        return [
            G[node][nbr]
            for nbr in R.neighbors(node)
            if G.has_edge(node, nbr) and G[node][nbr].get("type") == "road"
        ]

    endpoints: set[Any] = set()
    for node in R.nodes:
        degree = R.degree(node)
        if degree != 2:
            endpoints.add(node)

    # A pure ring can have no natural endpoint. One deterministic break keeps
    # every physical edge assigned to exactly one failure unit.
    for component in nx.connected_components(R):
        if not any(node in endpoints for node in component):
            endpoints.add(min(component, key=str))

    visited: set[Tuple[Any, Any]] = set()
    paths: List[List[Any]] = []
    for start in sorted(endpoints, key=str):
        for neighbor in sorted(R.neighbors(start), key=str):
            first_edge = _analysis_edge_key(start, neighbor)
            if first_edge in visited:
                continue
            path = [start, neighbor]
            visited.add(first_edge)
            previous, current = start, neighbor
            while current not in endpoints:
                candidates = [node for node in R.neighbors(current) if node != previous]
                if not candidates:
                    break
                nxt = sorted(candidates, key=str)[0]
                edge = _analysis_edge_key(current, nxt)
                if edge in visited:
                    break
                path.append(nxt)
                visited.add(edge)
                previous, current = current, nxt
            paths.append(path)

    # Defensive coverage for any edge left by unusual loop/parallel topology.
    for u, v in R.edges:
        edge = _analysis_edge_key(u, v)
        if edge not in visited:
            paths.append([u, v])
            visited.add(edge)

    links: List[Dict[str, Any]] = []
    assigned_physical: set[Tuple[Any, Any]] = set()
    for path in paths:
        analysis_edges = [_analysis_edge_key(a, b) for a, b in zip(path, path[1:])]
        road_edges = [
            edge
            for edge in analysis_edges
            if G.has_edge(*edge) and G[edge[0]][edge[1]].get("type") == "road"
        ]
        road_edges = [edge for edge in road_edges if edge not in assigned_physical]
        if not road_edges:
            continue
        assigned_physical.update(road_edges)
        edge_data = [G[u][v] for u, v in road_edges]
        road_length = sum(float(data.get("length_km", 0.0)) for data in edge_data)

        def unique_values(field: str) -> List[str]:
            return sorted(
                {
                    str(data.get(field, "") or "").strip()
                    for data in edge_data
                    if str(data.get(field, "") or "").strip()
                }
            )

        def dominant(field: str, default: str = "") -> str:
            values = unique_values(field)
            if not values:
                return default
            return max(
                values,
                key=lambda value: sum(
                    float(data.get("length_km", 0.0))
                    for data in edge_data
                    if str(data.get(field, "") or "").strip() == value
                ),
            )

        def dominant_source(field: str, default: str = "") -> str:
            values = unique_values(field)
            if not values:
                return default
            priority = {
                "input_station_location": 0,
                "input_route_median": 1,
                "input_network_median": 2,
                "default_median": 3,
                "connector_default": 4,
            }
            return min(values, key=lambda value: (priority.get(value, 99), value))

        start, end = path[0], path[-1]
        start_pos, end_pos = G.nodes[start]["pos"], G.nodes[end]["pos"]
        ref = dominant("osm_ref", dominant("route_id", "NH/SH"))
        osm_name = dominant("osm_name", ref)
        start_junction = str(G.nodes[start].get("junction_name", start))
        end_junction = str(G.nodes[end].get("junction_name", end))

        def endpoint_label(node: Any, junction: str, exclude: Optional[str] = None) -> str:
            node_place = str(G.nodes[node].get("place_name", "") or "").strip()
            if node_place and not re.fullmatch(r"J\d+", node_place):
                return node_place
            place, dist = nearest_place(cfg, G.nodes[node]["pos"], exclude=exclude)
            if place and place != "Junction" and math.isfinite(dist):
                return f"near {place}" if dist > 25.0 else place
            return junction

        start_place = endpoint_label(start, start_junction)
        end_place = endpoint_label(end, end_junction, exclude=start_place)
        link_id = f"LNK{len(links) + 1:04d}"
        if start_place == end_place:
            display_road_name = _map_safe_text(osm_name)
            if display_road_name and display_road_name != ref:
                place_part = f"{display_road_name} ({start_place}, {start_junction}-{end_junction})"
            else:
                place_part = f"{start_place} link {start_junction}-{end_junction}"
        else:
            place_part = f"{start_place}-{end_place}"
        link_name = f"{ref} | {place_part}"
        aadt_num = sum(
            float(data.get("aadt", cfg.aadt_default)) * float(data.get("length_km", 0.0))
            for data in edge_data
        )
        design_speeds = [
            float(data.get("source_design_speed_kmh", 0.0))
            for data in edge_data
            if float(data.get("source_design_speed_kmh", 0.0)) > 0
        ]
        travel_speeds = [
            float(data.get("speed_kmh", 0.0))
            for data in edge_data
            if float(data.get("speed_kmh", 0.0)) > 0
        ]
        model_speed_kmh = (
            float(
                np.average(
                    travel_speeds,
                    weights=[
                        max(float(data.get("length_km", 0.0)), 0.01)
                        for data in edge_data
                        if float(data.get("speed_kmh", 0.0)) > 0
                    ],
                )
            )
            if travel_speeds
            else cfg.speed_road
        )
        source_design_speed_kmh = float(np.median(design_speeds)) if design_speeds else np.nan
        midpoint = G.nodes[path[len(path) // 2]]["pos"]
        direct_km = _dist_km(start_pos, end_pos)
        links.append(
            dict(
                link_id=link_id,
                link_name=link_name,
                section_id=link_id,
                section_name=link_name,
                failure_unit="junction_to_junction_link",
                from_node=start,
                to_node=end,
                origin_node=start,
                destination_node=end,
                from_junction=start_junction,
                to_junction=end_junction,
                origin_place=start_place,
                destination_place=end_place,
                origin_longitude=start_pos[0],
                origin_latitude=start_pos[1],
                destination_longitude=end_pos[0],
                destination_latitude=end_pos[1],
                path_nodes=path,
                analysis_edges=analysis_edges,
                road_edges=road_edges,
                length_km=road_length,
                direct_km=direct_km,
                detour_ratio=road_length / max(direct_km, 0.01),
                edge_count=len(road_edges),
                connector_edge_count=max(0, len(analysis_edges) - len(road_edges)),
                ref=ref,
                name=osm_name,
                fclass=dominant("osm_fclass"),
                oneway=" | ".join(unique_values("osm_oneway")),
                source_design_speed_kmh=source_design_speed_kmh,
                mean_speed_kmh=model_speed_kmh,
                design_speed_source=dominant_source("design_speed_source", "Nepal.gpkg NH design_speed"),
                travel_speed_kmh=model_speed_kmh,
                model_speed_kmh=model_speed_kmh,
                model_speed_source=dominant_source("speed_source", f"mean_speed_{cfg.mean_speed_factor:.2f}_design_speed"),
                terrain_zone=dominant("zone"),
                terrain_source=dominant_source("terrain_source"),
                terrain_district=dominant("terrain_district"),
                bridge="YES" if any(str(data.get("osm_bridge", "")).upper() == "T" for data in edge_data) else "NO",
                tunnel="YES" if any(str(data.get("osm_tunnel", "")).upper() == "T" for data in edge_data) else "NO",
                layer=" | ".join(unique_values("osm_layer")),
                osm_ids=" | ".join(unique_values("osm_id")),
                mean_aadt=aadt_num / max(road_length, 1e-9),
                aadt_source=dominant_source("aadt_source", "input_network_median"),
                surface=dominant("surface"),
                surface_source=dominant("surface_source"),
                midpoint=midpoint,
                macro_region=_macro_region_from_lon(midpoint[0]),
            )
        )

    if len(assigned_physical) != sum(1 for _, _, data in G.edges(data=True) if data.get("type") == "road"):
        warnings.warn("Some physical road edges were not uniquely assigned to a road link.")
    print(f"  road links built between junctions/dead ends: {len(links)}")
    return links


def _remove_section(R: nx.Graph, section: Dict[str, Any]) -> nx.Graph:
    R2 = R.copy()
    for u, v in section["road_edges"]:
        if R2.has_edge(u, v):
            R2.remove_edge(u, v)
    return R2


def _section_removed_view(R: nx.Graph, section: Dict[str, Any]) -> nx.Graph:
    removed = {_analysis_edge_key(u, v) for u, v in section["road_edges"]}
    return nx.subgraph_view(R, filter_edge=lambda u, v: _analysis_edge_key(u, v) not in removed)


_BFW_WORKER_CFG: Optional[MLPIConfig] = None
_BFW_WORKER_GRAPH: Optional[nx.Graph] = None
_BFW_WORKER_ODS: Optional[List[Dict[str, float]]] = None


def _initialize_bfw_worker(cfg: MLPIConfig, graph: nx.Graph, ods: List[Dict[str, float]]) -> None:
    global _BFW_WORKER_CFG, _BFW_WORKER_GRAPH, _BFW_WORKER_ODS
    _BFW_WORKER_CFG = cfg
    _BFW_WORKER_GRAPH = graph
    _BFW_WORKER_ODS = ods


def _run_link_bfw_worker(section: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if _BFW_WORKER_CFG is None or _BFW_WORKER_GRAPH is None or _BFW_WORKER_ODS is None:
        raise RuntimeError("BFW worker was not initialized.")
    return (
        str(section["section_id"]),
        frank_wolfe_user_equilibrium(
            _BFW_WORKER_CFG,
            _remove_section(_BFW_WORKER_GRAPH, section),
            _BFW_WORKER_ODS,
        ),
    )


def _weighted_isolated_total(graph: nx.Graph, nodes: Sequence[Any], weights: Dict[Any, float]) -> float:
    if graph.number_of_nodes() == 0:
        return sum(weights.get(n, 0.0) for n in nodes)
    components = list(nx.connected_components(graph))
    if not components:
        return sum(weights.get(n, 0.0) for n in nodes)
    totals = [sum(weights.get(n, 0.0) for n in comp if n in weights) for comp in components]
    return max(0.0, sum(weights.get(n, 0.0) for n in nodes) - max(totals, default=0.0))


def compute_section_physical(
    cfg: MLPIConfig,
    G: nx.Graph,
    sections: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Road-only free-flow FFTDI for exhaustive, non-overlapping road links.

    With BPR alpha fixed at zero, the user-equilibrium solution is the
    all-or-nothing shortest-path loading of the OD matrix. P1 is therefore the
    increase in OD-weighted total system travel time after one road link is
    removed. P2 is the baseline OD exposure on that failed link, weighted by the
    same failure detour ratio. High flow alone is not enough.
    """
    R = _assignment_graph(cfg, G, include_air=False)
    ods = load_me2_od_pairs(cfg, G, R)
    raw_fftdi: Dict[str, float] = {}
    disconn: Dict[str, float] = {}
    flow: Dict[str, float] = {}
    raw_exposure_vht: Dict[str, float] = {}
    removed_tstt: Dict[str, float] = {}
    vehicle_hour_loss: Dict[str, float] = {}
    affected_od_base_vht: Dict[str, float] = {}
    affected_od_removed_vht: Dict[str, float] = {}
    affected_od_demand: Dict[str, float] = {}
    baseline_link_od_flow: Dict[str, float] = {}
    affected_od_pair_count: Dict[str, int] = {}
    me2_top_od_pair: Dict[str, str] = {}
    me2_top_od_origin: Dict[str, str] = {}
    me2_top_od_destination: Dict[str, str] = {}
    me2_top_od_origin_district: Dict[str, str] = {}
    me2_top_od_destination_district: Dict[str, str] = {}
    me2_top_od_demand: Dict[str, float] = {}
    me2_top_od_base_time: Dict[str, float] = {}
    me2_top_od_removed_time: Dict[str, float] = {}
    me2_top_od_delta_time: Dict[str, float] = {}
    me2_top_od_vehicle_hour_loss: Dict[str, float] = {}
    me2_top5_od_pairs: Dict[str, str] = {}
    redundancy_factor: Dict[str, float] = {}
    detour_ratio_after_failure: Dict[str, float] = {}
    removed_rgap: Dict[str, float] = {}
    removed_iterations: Dict[str, int] = {}
    print(f"  physical score uses OD-weighted free-flow TSTT for {len(sections)} road links")

    base_od_records: List[Dict[str, Any]] = []
    base_edge_flow: Dict[Tuple[Any, Any], float] = defaultdict(float)
    edge_to_od_indices: Dict[Tuple[Any, Any], List[int]] = defaultdict(list)
    base_tstt = 0.0
    total_demand = 0.0
    for od in ods:
        origin = od.get("o")
        destination = od.get("d")
        demand = max(float(od.get("demand", 0.0)), 0.0)
        if demand <= 0.0 or origin not in R or destination not in R:
            od["base_connected"] = False
            od["base_skip_reason"] = "nonpositive_demand_or_missing_node"
            continue
        try:
            path = nx.dijkstra_path(R, origin, destination, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            od["base_connected"] = False
            od["base_skip_reason"] = "no_baseline_road_path"
            continue
        path_edges = [_analysis_edge_key(a, b) for a, b in zip(path, path[1:])]
        base_time = sum(float(R[a][b].get("weight", R[a][b].get("t0", 0.0))) for a, b in zip(path, path[1:]))
        if base_time <= 0.0:
            od["base_connected"] = False
            od["base_skip_reason"] = "zero_baseline_time"
            continue
        od["base_connected"] = True
        od["base_skip_reason"] = ""
        od["base_time_h"] = base_time
        od["base_path_edge_count"] = len(path_edges)
        record_index = len(base_od_records)
        base_od_records.append(
            dict(
                o=origin,
                d=destination,
                demand=demand,
                base_time=base_time,
                path_edges=set(path_edges),
                origin_zone_label=od.get("origin_zone_label", ""),
                destination_zone_label=od.get("destination_zone_label", ""),
                origin_district=od.get("origin_district", ""),
                destination_district=od.get("destination_district", ""),
                origin_hq=od.get("origin_hq", ""),
                destination_hq=od.get("destination_hq", ""),
            )
        )
        for key in path_edges:
            base_edge_flow[key] += demand
            edge_to_od_indices[key].append(record_index)
        base_tstt += demand * base_time
        total_demand += demand
    base_tstt = max(base_tstt, 1e-9)
    total_demand = max(total_demand, 1e-9)

    cfg.od_baseline_connected_pairs = len(base_od_records)
    cfg.od_baseline_connected_demand = sum(float(od["demand"]) for od in base_od_records)
    cfg.od_baseline_disconnected_pairs = sum(1 for od in ods if not od.get("base_connected"))
    cfg.od_baseline_disconnected_demand = sum(float(od["demand"]) for od in ods if not od.get("base_connected"))
    if cfg.od_baseline_disconnected_pairs:
        disconnected_share = cfg.od_baseline_disconnected_demand / max(
            cfg.od_baseline_connected_demand + cfg.od_baseline_disconnected_demand,
            1e-9,
        )
        warnings.warn(
            f"ME2 OD baseline has {cfg.od_baseline_disconnected_pairs} disconnected OD pairs "
            f"({disconnected_share:.1%} of mapped demand). They are excluded from link-removal loss; "
            "check road-layer filtering/connectivity if this is unexpected."
        )
    if not base_od_records:
        raise RuntimeError(
            "ME2 OD matrix loaded, but no positive OD pair is connected on the MLPI road graph. "
            "Check the OD zone mapping, road layer filter, and topology connectors."
        )

    for section in sections:
        sid = section["section_id"]
        removed_edges = {_analysis_edge_key(u, v) for u, v in section.get("road_edges", [])}
        baseline_link_vht = 0.0
        baseline_link_flow = 0.0
        for u, v in section.get("road_edges", []):
            key = _analysis_edge_key(u, v)
            if R.has_edge(u, v):
                edge_flow = base_edge_flow.get(key, 0.0)
                baseline_link_flow += edge_flow
                baseline_link_vht += edge_flow * float(R[u][v].get("t0", R[u][v].get("weight", 0.0)))

        loss = 0.0
        disconnected_demand = 0.0
        affected_base = 0.0
        affected_after = 0.0
        affected_demand_total = 0.0
        top_od_rows: List[Dict[str, Any]] = []
        affected_indices: set[int] = set()
        for edge in removed_edges:
            affected_indices.update(edge_to_od_indices.get(edge, []))
        if affected_indices:
            R2 = _remove_section(R, section)
            affected_by_origin: Dict[Any, List[int]] = defaultdict(list)
            for index in affected_indices:
                affected_by_origin[base_od_records[index]["o"]].append(index)
            for origin, indices in affected_by_origin.items():
                if origin not in R2:
                    lengths: Dict[Any, float] = {}
                else:
                    try:
                        lengths = nx.single_source_dijkstra_path_length(R2, origin, weight="weight")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        lengths = {}
                for index in indices:
                    od = base_od_records[index]
                    demand = float(od["demand"])
                    base_time = float(od["base_time"])
                    destination = od["d"]
                    after_time = float(lengths[destination]) if destination in lengths else None
                    if after_time is None:
                        disconnected_demand += demand
                        after_time = base_time + max(cfg.disconnected_penalty_h, base_time * 5.0)
                    delta = max(0.0, after_time - base_time)
                    top_od_rows.append(
                        dict(
                            origin_zone_label=od.get("origin_zone_label", ""),
                            destination_zone_label=od.get("destination_zone_label", ""),
                            origin_district=od.get("origin_district", ""),
                            destination_district=od.get("destination_district", ""),
                            demand=demand,
                            base_time=base_time,
                            removed_time=after_time,
                            delta_time=delta,
                            vehicle_hour_loss=demand * delta,
                        )
                    )
                    affected_demand_total += demand
                    affected_base += demand * base_time
                    affected_after += demand * after_time
                    loss += demand * delta

        if affected_base > 0.0:
            section_detour_ratio = affected_after / max(affected_base, 1e-9)
            factor = min(1.0, max(0.0, (section_detour_ratio - 1.0) / max(cfg.detour_full_credit_ratio, 1e-9)))
        else:
            section_detour_ratio = 1.0
            factor = 0.0

        flow[sid] = baseline_link_vht * factor
        raw_exposure_vht[sid] = baseline_link_vht
        raw_fftdi[sid] = loss / base_tstt
        disconn[sid] = disconnected_demand / total_demand
        removed_tstt[sid] = base_tstt + loss
        vehicle_hour_loss[sid] = loss
        affected_od_base_vht[sid] = affected_base
        affected_od_removed_vht[sid] = affected_after
        affected_od_demand[sid] = affected_demand_total
        baseline_link_od_flow[sid] = baseline_link_flow
        affected_od_pair_count[sid] = len(affected_indices)
        if top_od_rows:
            top_od_rows.sort(
                key=lambda row: (
                    float(row.get("vehicle_hour_loss", 0.0)),
                    float(row.get("demand", 0.0)),
                ),
                reverse=True,
            )
            top = top_od_rows[0]
            me2_top_od_origin[sid] = str(top.get("origin_zone_label", ""))
            me2_top_od_destination[sid] = str(top.get("destination_zone_label", ""))
            me2_top_od_origin_district[sid] = str(top.get("origin_district", ""))
            me2_top_od_destination_district[sid] = str(top.get("destination_district", ""))
            me2_top_od_demand[sid] = float(top.get("demand", 0.0))
            me2_top_od_base_time[sid] = float(top.get("base_time", 0.0))
            me2_top_od_removed_time[sid] = float(top.get("removed_time", 0.0))
            me2_top_od_delta_time[sid] = float(top.get("delta_time", 0.0))
            me2_top_od_vehicle_hour_loss[sid] = float(top.get("vehicle_hour_loss", 0.0))
            me2_top_od_pair[sid] = (
                f"{top.get('origin_zone_label', '')} -> "
                f"{top.get('destination_zone_label', '')}"
            )
            me2_top5_od_pairs[sid] = " ; ".join(
                (
                    f"{row.get('origin_district', '')}->{row.get('destination_district', '')} "
                    f"q={float(row.get('demand', 0.0)):.3f}, "
                    f"dt={float(row.get('delta_time', 0.0)):.3f}h"
                )
                for row in top_od_rows[:5]
            )
        else:
            me2_top_od_pair[sid] = ""
            me2_top_od_origin[sid] = ""
            me2_top_od_destination[sid] = ""
            me2_top_od_origin_district[sid] = ""
            me2_top_od_destination_district[sid] = ""
            me2_top_od_demand[sid] = 0.0
            me2_top_od_base_time[sid] = 0.0
            me2_top_od_removed_time[sid] = 0.0
            me2_top_od_delta_time[sid] = 0.0
            me2_top_od_vehicle_hour_loss[sid] = 0.0
            me2_top5_od_pairs[sid] = ""
        redundancy_factor[sid] = factor
        detour_ratio_after_failure[sid] = section_detour_ratio
        removed_rgap[sid] = 0.0
        removed_iterations[sid] = 1
    fftdi_norm = _minmax(raw_fftdi)
    flow_norm = _minmax(flow)
    p_total_w = max(cfg.w_p_fftdi + cfg.w_p_flow, 1e-9)
    out = {
        s["section_id"]: dict(
            physical=(
                cfg.w_p_fftdi * fftdi_norm.get(s["section_id"], 0.0)
                + cfg.w_p_flow * flow_norm.get(s["section_id"], 0.0)
            )
            / p_total_w,
            FFTDI=fftdi_norm.get(s["section_id"], 0.0),
            FFTDI_raw=raw_fftdi.get(s["section_id"], 0.0),
            P1_FFTDI=fftdi_norm.get(s["section_id"], 0.0),
            P2_detour_weighted_exposure=flow_norm.get(s["section_id"], 0.0),
            P2_AADT_exposure=flow_norm.get(s["section_id"], 0.0),
            P2_UE_flow=flow_norm.get(s["section_id"], 0.0),
            disconnected_demand_raw=disconn.get(s["section_id"], 0.0),
            base_ue_tstt_h=base_tstt,
            removed_ue_tstt_h=removed_tstt.get(s["section_id"], base_tstt),
            vehicle_hour_loss=vehicle_hour_loss.get(s["section_id"], 0.0),
            od_vehicle_hour_loss=vehicle_hour_loss.get(s["section_id"], 0.0),
            affected_od_base_vht=affected_od_base_vht.get(s["section_id"], 0.0),
            affected_od_removed_vht=affected_od_removed_vht.get(s["section_id"], 0.0),
            affected_od_demand=affected_od_demand.get(s["section_id"], 0.0),
            affected_od_pair_count=affected_od_pair_count.get(s["section_id"], 0),
            baseline_link_od_flow=baseline_link_od_flow.get(s["section_id"], 0.0),
            me2_top_od_pair=me2_top_od_pair.get(s["section_id"], ""),
            me2_top_od_origin=me2_top_od_origin.get(s["section_id"], ""),
            me2_top_od_destination=me2_top_od_destination.get(s["section_id"], ""),
            me2_top_od_origin_district=me2_top_od_origin_district.get(s["section_id"], ""),
            me2_top_od_destination_district=me2_top_od_destination_district.get(s["section_id"], ""),
            me2_top_od_demand=me2_top_od_demand.get(s["section_id"], 0.0),
            me2_top_od_base_time_h=me2_top_od_base_time.get(s["section_id"], 0.0),
            me2_top_od_removed_time_h=me2_top_od_removed_time.get(s["section_id"], 0.0),
            me2_top_od_delta_time_h=me2_top_od_delta_time.get(s["section_id"], 0.0),
            me2_top_od_vehicle_hour_loss=me2_top_od_vehicle_hour_loss.get(s["section_id"], 0.0),
            me2_top5_od_pairs=me2_top5_od_pairs.get(s["section_id"], ""),
            detour_ratio_after_failure=detour_ratio_after_failure.get(s["section_id"], 1.0),
            physical_redundancy_factor=redundancy_factor.get(s["section_id"], 0.0),
            raw_aadt_exposure_vht=raw_exposure_vht.get(s["section_id"], 0.0),
            aadt_exposure_vht=flow.get(s["section_id"], 0.0),
            detour_weighted_exposure_vht=flow.get(s["section_id"], 0.0),
            base_assignment_rgap=0.0,
            base_assignment_iterations=1,
            removed_assignment_rgap=removed_rgap.get(s["section_id"], np.nan),
            removed_assignment_iterations=removed_iterations.get(s["section_id"], 0),
            ue_flow=flow.get(s["section_id"], 0.0),
            assignment_rerun=True,
        )
        for s in sections
    }
    return out, pd.DataFrame(ods)


def compute_section_social(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sections: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    R = _assignment_graph(cfg, G, include_air=False)
    road_nodes = [n for n in R.nodes if node_population(cfg, G.nodes[n]) > 0.0]
    node_pop = {n: node_population(cfg, G.nodes[n]) for n in road_nodes}
    total_pop = max(sum(node_pop.values()), 1.0)

    def newly_isolated_population(after_graph: nx.Graph) -> float:
        isolated = 0.0
        for baseline_component in nx.connected_components(R):
            baseline_population = sum(node_pop.get(node, 0.0) for node in baseline_component)
            if baseline_population <= 0.0:
                continue
            after_components = list(nx.connected_components(after_graph.subgraph(baseline_component)))
            retained_population = max(
                (sum(node_pop.get(node, 0.0) for node in component) for component in after_components),
                default=0.0,
            )
            isolated += max(0.0, baseline_population - retained_population)
        return isolated

    healthcare = geo["healthcare"]
    accepted_types = {
        "hospital", "clinic", "doctors", "doctor", "health_post", "health post",
        "health_centre", "health centre", "health_center", "health center",
        "medical_centre", "medical centre", "medical_center", "medical center",
        "birthing_centre", "birthing centre", "community_health_centre",
    }
    healthcare_pts: List[Tuple[float, float]] = []
    for _, row in healthcare.iterrows():
        geom = row.geometry
        if not isinstance(geom, Point):
            continue
        descriptors = {
            str(row.get(column, "")).strip().lower()
            for column in ("amenity", "healthcare", "healthca_1")
            if str(row.get(column, "")).strip()
        }
        if descriptors & accepted_types:
            healthcare_pts.append((float(geom.x), float(geom.y)))
    healthcare_nodes: set[Any] = set()
    for hp in healthcare_pts:
        if R.number_of_nodes() == 0:
            break
        n = min(R.nodes, key=lambda rn: _dist_km(hp, G.nodes[rn]["pos"]))
        healthcare_nodes.add(n)
    healthcare_proxy_used = False
    healthcare_proxy_node = ""
    if not healthcare_nodes:
        if R.number_of_nodes():
            warnings.warn("No healthcare facilities matched; using highest-density road node as proxy.")
            proxy = max(R.nodes, key=lambda n: G.nodes[n].get("healthcare_count", 0))
            healthcare_nodes = {proxy}
            healthcare_proxy_used = True
            healthcare_proxy_node = str(proxy)
        else:
            warnings.warn("No road nodes are available for healthcare proxy assignment; S2 will be zero.")
    base_health = nx.multi_source_dijkstra_path_length(R, list(healthcare_nodes), weight="weight") if healthcare_nodes else {}
    raw_s1: Dict[str, float] = {}
    raw_s1_population: Dict[str, float] = {}
    raw_s2: Dict[str, float] = {}
    for section in sections:
        sid = section["section_id"]
        R2 = _section_removed_view(R, section)
        isolated_population = newly_isolated_population(R2)
        raw_s1_population[sid] = isolated_population
        raw_s1[sid] = isolated_population / total_pop
        sources = [n for n in healthcare_nodes if n in R2]
        after_health = nx.multi_source_dijkstra_path_length(R2, sources, weight="weight") if sources else {}
        access_loss = 0.0
        for n in road_nodes:
            if n not in base_health:
                continue
            base_t = float(base_health[n])
            after_t = float(after_health.get(n, base_t + cfg.disconnected_penalty_h))
            delta = max(0.0, after_t - base_t)
            access_loss += node_pop.get(n, 0.0) * delta
        raw_s2[sid] = access_loss / total_pop
    n_s1, n_s2 = _minmax(raw_s1), _minmax(raw_s2)
    total_w = max(cfg.w_s1 + cfg.w_s2, 1e-9)
    return {
        s["section_id"]: dict(
            S1=n_s1.get(s["section_id"], 0.0),
            S2=n_s2.get(s["section_id"], 0.0),
            S1_raw=raw_s1.get(s["section_id"], 0.0),
            S1_raw_population=raw_s1_population.get(s["section_id"], 0.0),
            S2_raw=raw_s2.get(s["section_id"], 0.0),
            healthcare_facilities_used=len(healthcare_pts),
            healthcare_proxy_used=healthcare_proxy_used,
            healthcare_proxy_node=healthcare_proxy_node,
            social=(
                cfg.w_s1 * n_s1.get(s["section_id"], 0.0)
                + cfg.w_s2 * n_s2.get(s["section_id"], 0.0)
            )
            / total_w,
        )
        for s in sections
    }


def compute_section_economic(
    cfg: MLPIConfig,
    G: nx.Graph,
    sections: Sequence[Dict[str, Any]],
    physical: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    R = _assignment_graph(cfg, G, include_air=False)
    road_nodes = [n for n in R.nodes if node_population(cfg, G.nodes[n]) > 0.0]
    node_gdp_raw = {n: float(G.nodes[n].get("gdp_weight", 0.0)) for n in road_nodes}
    total_raw_gdp = max(sum(max(value, 0.0) for value in node_gdp_raw.values()), 1e-12)
    if cfg.e2_regional_balance:
        region_totals: Dict[str, float] = defaultdict(float)
        for n, value in node_gdp_raw.items():
            region_totals[_macro_region_from_lon(G.nodes[n]["pos"][0])] += max(value, 0.0)
        active_regions = [region for region, total in region_totals.items() if total > 0.0]
        # E2 remains GDP-weighted, but each macro-region contributes equally
        # before aggregation. This avoids Kathmandu's absolute GDP mass making
        # every economic-access result look like a Kathmandu result.
        node_gdp = {
            n: (
                max(node_gdp_raw.get(n, 0.0), 0.0)
                / max(region_totals.get(_macro_region_from_lon(G.nodes[n]["pos"][0]), 0.0), 1e-12)
                / max(len(active_regions), 1)
            )
            for n in road_nodes
        }
    else:
        node_gdp = {
            n: max(node_gdp_raw.get(n, 0.0), 0.0) / total_raw_gdp
            for n in road_nodes
        }
    total_gdp = max(sum(node_gdp.values()), 1e-12)
    gateways: List[Tuple[Any, float, str]] = []
    for lon, lat, volume, name in cfg.border_trade:
        if R.number_of_nodes():
            n, _dist = min(((rn, _dist_km((lon, lat), G.nodes[rn]["pos"])) for rn in R.nodes), key=lambda x: x[1])
            gateways.append((n, max(float(volume), 0.01), str(name)))

    def gateway_access(graph: nx.Graph) -> Dict[Any, float]:
        if not gateways:
            return {n: 0.0 for n in road_nodes}
        max_volume = max(volume for _gateway, volume, _name in gateways)
        access = {n: 0.0 for n in road_nodes}
        for gateway, volume, _name in gateways:
            if gateway not in graph:
                continue
            lengths = nx.single_source_dijkstra_path_length(graph, gateway, weight="weight")
            volume_weight = volume / max(max_volume, 1e-12)
            for n in road_nodes:
                if n in lengths:
                    access[n] += volume_weight * math.exp(-cfg.beta * float(lengths[n]) / max(cfg.od_deterrence_h, 1e-9))
        return access

    base_access = gateway_access(R)
    max_access = max(base_access.values(), default=0.0)
    road_aadt_values = [
        max(float(data.get("aadt", cfg.aadt_default)), 0.0)
        for _, _, data in G.edges(data=True)
        if data.get("type") == "road"
    ]
    median_aadt = max(float(np.nanmedian(road_aadt_values)) if road_aadt_values else cfg.aadt_default, 1.0)

    def section_trade_exposure(section: Dict[str, Any]) -> float:
        numerator = 0.0
        denominator = 0.0
        for u, v in section.get("road_edges", []):
            if not R.has_edge(u, v):
                continue
            data = R[u][v]
            length = max(float(data.get("length_km", 0.0)), 0.0)
            if length <= 0.0:
                continue
            aadt_factor = max(float(data.get("aadt", cfg.aadt_default)), 0.0) / median_aadt
            if max_access > 0.0:
                access_factor = 0.5 * (base_access.get(u, 0.0) + base_access.get(v, 0.0)) / max_access
            else:
                access_factor = 1.0
            numerator += length * aadt_factor * access_factor
            denominator += length
        return numerator / max(denominator, 1e-9)

    raw_e1: Dict[str, float] = {}
    raw_e2: Dict[str, float] = {}
    raw_trade_exposure: Dict[str, float] = {}
    for section in sections:
        sid = section["section_id"]
        R2 = _section_removed_view(R, section)
        trade_exposure = section_trade_exposure(section)
        raw_trade_exposure[sid] = trade_exposure
        raw_e1[sid] = float(physical.get(sid, {}).get("vehicle_hour_loss", 0.0)) * trade_exposure
        after_access = gateway_access(R2)
        raw_e2[sid] = sum(
            node_gdp.get(n, 0.0) * max(0.0, base_access.get(n, 0.0) - after_access.get(n, 0.0))
            for n in road_nodes
        ) / total_gdp
    n_e1, n_e2 = _minmax(raw_e1), _minmax(raw_e2)
    total_w = max(cfg.w_e1 + cfg.w_e2, 1e-9)
    return {
        s["section_id"]: dict(
            E1=n_e1.get(s["section_id"], 0.0),
            E2=n_e2.get(s["section_id"], 0.0),
            E1_raw=raw_e1.get(s["section_id"], 0.0),
            E2_raw=raw_e2.get(s["section_id"], 0.0),
            E1_trade_exposure_raw=raw_trade_exposure.get(s["section_id"], 0.0),
            E2_regional_balance=cfg.e2_regional_balance,
            economic=(
                cfg.w_e1 * n_e1.get(s["section_id"], 0.0)
                + cfg.w_e2 * n_e2.get(s["section_id"], 0.0)
            )
            / total_w,
        )
        for s in sections
    }


def compute_section_interconnected(
    cfg: MLPIConfig,
    G: nx.Graph,
    sections: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    R = _assignment_graph(cfg, G, include_air=False)
    road_nodes = [n for n in R.nodes if node_population(cfg, G.nodes[n]) > 0.0]
    node_pop = {n: node_population(cfg, G.nodes[n]) for n in road_nodes}
    airports = [
        n for n, data in G.nodes(data=True)
        if data.get("is_airport") and data.get("airport_status") != "Not in operation"
    ]
    if not airports:
        return {s["section_id"]: dict(I1=0.0, I2=0.0, interconnected=0.0, target_airport="") for s in sections}
    airport_sources: Dict[Any, str] = {}
    for airport in airports:
        for neighbor in G.neighbors(airport):
            if neighbor in R and G[airport][neighbor].get("type") == "airport_access":
                airport_sources.setdefault(neighbor, str(G.nodes[airport].get("airport_name", airport)))
    if not airport_sources:
        return {s["section_id"]: dict(I1=0.0, I2=0.0, interconnected=0.0, target_airport="") for s in sections}
    base_dist, base_paths = nx.multi_source_dijkstra(R, list(airport_sources), weight="weight")
    accessible_population = max(sum(node_pop[n] for n in road_nodes if n in base_dist), 1.0)
    base_target = {
        n: airport_sources.get(base_paths[n][0], "")
        for n in road_nodes
        if n in base_paths and base_paths[n]
    }
    raw_i1: Dict[str, float] = {}
    raw_i2: Dict[str, float] = {}
    target_airport: Dict[str, str] = {}
    for section in sections:
        sid = section["section_id"]
        R2 = _section_removed_view(R, section)
        available_sources = [source for source in airport_sources if source in R2]
        after_dist = nx.multi_source_dijkstra_path_length(R2, available_sources, weight="weight") if available_sources else {}
        access_loss = 0.0
        cutoff_population = 0.0
        target_counts: Dict[str, float] = defaultdict(float)
        for n in road_nodes:
            if n not in base_dist:
                continue
            base_t = float(base_dist[n])
            if n not in after_dist:
                cutoff_population += node_pop[n]
                after_t = cfg.disconnected_penalty_h + base_t
            else:
                after_t = float(after_dist[n])
            delta = max(0.0, after_t - base_t)
            access_loss += node_pop[n] * delta
            if delta > 1e-9 and base_target.get(n):
                target_counts[base_target[n]] += node_pop[n] * delta
        raw_i1[sid] = access_loss / accessible_population
        raw_i2[sid] = cutoff_population / accessible_population
        target_airport[sid] = max(target_counts, key=target_counts.get) if target_counts else ""
    n_i1, n_i2 = _minmax(raw_i1), _minmax(raw_i2)
    total_w = max(cfg.w_i1 + cfg.w_i2, 1e-9)
    return {
        s["section_id"]: dict(
            I1=n_i1.get(s["section_id"], 0.0),
            I2=n_i2.get(s["section_id"], 0.0),
            I1_raw=raw_i1.get(s["section_id"], 0.0),
            I2_raw=raw_i2.get(s["section_id"], 0.0),
            target_airport=target_airport.get(s["section_id"], ""),
            interconnected=(
                cfg.w_i1 * n_i1.get(s["section_id"], 0.0)
                + cfg.w_i2 * n_i2.get(s["section_id"], 0.0)
            )
            / total_w,
        )
        for s in sections
    }


def compute_section_mlpi(
    cfg: MLPIConfig,
    sections: Sequence[Dict[str, Any]],
    physical: Dict[str, Dict[str, float]],
    social: Dict[str, Dict[str, float]],
    economic: Dict[str, Dict[str, float]],
    interconnected: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for section in sections:
        sid = section["section_id"]
        P = float(physical.get(sid, {}).get("physical", 0.0))
        S = float(social.get(sid, {}).get("social", 0.0))
        E = float(economic.get(sid, {}).get("economic", 0.0))
        I = float(interconnected.get(sid, {}).get("interconnected", 0.0))
        row = {
            k: v
            for k, v in section.items()
            if k not in {"path_nodes", "analysis_edges", "road_edges", "midpoint"}
        }
        row.update(physical.get(sid, {}))
        row.update(social.get(sid, {}))
        row.update(economic.get(sid, {}))
        row.update(interconnected.get(sid, {}))
        row["mlpi"] = (
            cfg.weights["physical"] * P
            + cfg.weights["social"] * S
            + cfg.weights["economic"] * E
            + cfg.weights["interconnected"] * I
        )
        rows.append(row)
    ranked = pd.DataFrame(rows).sort_values("mlpi", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def compute_mlpi(
    cfg: MLPIConfig,
    G: nx.Graph,
    physical: Dict[Tuple[int, int], Dict[str, float]],
    social: Dict[Tuple[int, int], Dict[str, float]],
    economic: Dict[Tuple[int, int], Dict[str, float]],
    interconnected: Dict[Tuple[int, int], Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        if d.get("type") != "road":
            continue
        key = (min(u, v), max(u, v))
        P = physical.get(key, {}).get("physical", 0.0)
        S = social.get(key, {}).get("social", 0.0)
        E = economic.get(key, {}).get("economic", 0.0)
        I = interconnected.get(key, {}).get("interconnected", 0.0)
        score = cfg.weights["physical"] * P + cfg.weights["social"] * S + cfg.weights["economic"] * E + cfg.weights["interconnected"] * I
        rows.append(
            dict(
                u=u,
                v=v,
                NH_or_route=d.get("route_id") or "National Highway",
                corridor_id=d.get("corridor_id", d.get("route_id", "")),
                corridor_label=d.get("corridor_label", d.get("route_id", "")),
                corridor_score=d.get("corridor_score", 0.0),
                edge_name=edge_place_name(cfg, G, u, v),
                origin_junction=G.nodes[u].get("junction_name", u),
                origin_place=G.nodes[u].get("place_name", G.nodes[u].get("abbrev", u)),
                destination_junction=G.nodes[v].get("junction_name", v),
                destination_place=G.nodes[v].get("place_name", G.nodes[v].get("abbrev", v)),
                mlpi=score,
                physical=P,
                FFTDI=physical.get(key, {}).get("FFTDI", 0.0),
                FFTDI_raw=physical.get(key, {}).get("FFTDI_raw", 0.0),
                ue_flow=physical.get(key, {}).get("ue_flow", 0.0),
                P1_FFTDI=physical.get(key, {}).get("P1_FFTDI", physical.get(key, {}).get("FFTDI", 0.0)),
                P2_disconnected_demand=physical.get(key, {}).get("P2_disconnected_demand", 0.0),
                P3_UE_flow=physical.get(key, {}).get("P3_UE_flow", 0.0),
                P4_corridor_context=physical.get(key, {}).get("P4_corridor_context", 0.0),
                social=S,
                economic=E,
                interconnected=I,
                social_marginal=social.get(key, {}).get("social_marginal", S),
                social_joint=social.get(key, {}).get("social_joint", S),
                economic_marginal=economic.get(key, {}).get("economic_marginal", E),
                economic_joint=economic.get(key, {}).get("economic_joint", E),
                interconnected_marginal=interconnected.get(key, {}).get("interconnected_marginal", I),
                interconnected_joint=interconnected.get(key, {}).get("interconnected_joint", I),
                S1=social.get(key, {}).get("S1", 0.0),
                S2=social.get(key, {}).get("S2", 0.0),
                S3=social.get(key, {}).get("S3", 0.0),
                S1_marginal=social.get(key, {}).get("S1_marginal", social.get(key, {}).get("S1", 0.0)),
                S1_joint=social.get(key, {}).get("S1_joint", social.get(key, {}).get("S1", 0.0)),
                S2_marginal=social.get(key, {}).get("S2_marginal", social.get(key, {}).get("S2", 0.0)),
                S2_joint=social.get(key, {}).get("S2_joint", social.get(key, {}).get("S2", 0.0)),
                S3_marginal=social.get(key, {}).get("S3_marginal", social.get(key, {}).get("S3", 0.0)),
                S3_joint=social.get(key, {}).get("S3_joint", social.get(key, {}).get("S3", 0.0)),
                E1=economic.get(key, {}).get("E1", 0.0),
                E2=economic.get(key, {}).get("E2", 0.0),
                E3=economic.get(key, {}).get("E3", 0.0),
                E1_marginal=economic.get(key, {}).get("E1_marginal", economic.get(key, {}).get("E1", 0.0)),
                E1_joint=economic.get(key, {}).get("E1_joint", economic.get(key, {}).get("E1", 0.0)),
                E2_marginal=economic.get(key, {}).get("E2_marginal", economic.get(key, {}).get("E2", 0.0)),
                E2_joint=economic.get(key, {}).get("E2_joint", economic.get(key, {}).get("E2", 0.0)),
                E3_marginal=economic.get(key, {}).get("E3_marginal", economic.get(key, {}).get("E3", 0.0)),
                E3_joint=economic.get(key, {}).get("E3_joint", economic.get(key, {}).get("E3", 0.0)),
                I1=interconnected.get(key, {}).get("I1", 0.0),
                I2=interconnected.get(key, {}).get("I2", 0.0),
                I3=interconnected.get(key, {}).get("I3", 0.0),
                I1_marginal=interconnected.get(key, {}).get("I1_marginal", interconnected.get(key, {}).get("I1", 0.0)),
                I1_joint=interconnected.get(key, {}).get("I1_joint", interconnected.get(key, {}).get("I1", 0.0)),
                I2_marginal=interconnected.get(key, {}).get("I2_marginal", interconnected.get(key, {}).get("I2", 0.0)),
                I2_joint=interconnected.get(key, {}).get("I2_joint", interconnected.get(key, {}).get("I2", 0.0)),
                I3_marginal=interconnected.get(key, {}).get("I3_marginal", interconnected.get(key, {}).get("I3", 0.0)),
                I3_joint=interconnected.get(key, {}).get("I3_joint", interconnected.get(key, {}).get("I3", 0.0)),
                length_km=d.get("length_km", 0.0),
                AADT=d.get("aadt", 0.0),
                aadt_source=d.get("aadt_source", "input_network_median"),
                zone=d.get("zone", ""),
                speed_kmh=d.get("speed_kmh", np.nan),
                terrain_source=d.get("terrain_source", ""),
                terrain_district=d.get("terrain_district", ""),
                surface=d.get("surface", ""),
                surface_source=d.get("surface_source", ""),
                route_id=d.get("route_id", ""),
            )
        )
    df = pd.DataFrame(rows).sort_values("mlpi", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def airport_assets_dataframe(G: nx.Graph) -> pd.DataFrame:
    rows = []
    for n, d in G.nodes(data=True):
        if not d.get("is_air_asset"):
            continue
        lon, lat = d.get("pos", (np.nan, np.nan))
        rows.append(
            dict(
                node_id=n,
                map_label=d.get("airport_name", ""),
                registry_code=d.get("airport_registry_code", ""),
                airport_name=d.get("airport_full_name", ""),
                asset_type=d.get("airport_type", "Airport"),
                operational_status=d.get("airport_status", ""),
                registry_match_distance_km=d.get("airport_registry_distance_km", np.nan),
                lon=lon,
                lat=lat,
            )
        )
    return pd.DataFrame(rows).sort_values(["operational_status", "map_label"]).reset_index(drop=True)


# --------------------------------- plotting ----------------------------------


def _plot_safe_line(ax: plt.Axes, pts: Sequence[Tuple[float, float]], **kwargs: Any) -> None:
    if len(pts) < 2:
        return
    xs, ys = zip(*pts)
    kwargs.setdefault("antialiased", True)
    kwargs.setdefault("solid_capstyle", "round")
    kwargs.setdefault("solid_joinstyle", "round")
    ax.plot(xs, ys, **kwargs)


def _plot_line_gdf(ax: plt.Axes, gdf: gpd.GeoDataFrame, color: str, lw: float, alpha: float, zorder: int, thin: int = 1) -> None:
    for i, pts, _ in _iter_lines(gdf):
        if thin > 1 and i % thin:
            continue
        _plot_safe_line(ax, pts, color=color, lw=lw, alpha=alpha, zorder=zorder)


def _plot_polygon_boundaries(ax: plt.Axes, gdf: gpd.GeoDataFrame, color: str, lw: float, alpha: float, zorder: int) -> None:
    if gdf.empty:
        return
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polygons = [geom] if isinstance(geom, Polygon) else [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
        for polygon in polygons:
            xs, ys = polygon.exterior.xy
            ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=zorder)


def _country_polygons(country: gpd.GeoDataFrame) -> List[Polygon]:
    lines = []
    for _, pts, _ in _iter_lines(country):
        if len(pts) >= 2:
            lines.append(LineString(pts))
    try:
        polys = list(polygonize(unary_union(lines)))
        return [p for p in polys if p.area > 0.05]
    except Exception:
        return []


def _setup_map(ax: plt.Axes, geo: Dict[str, gpd.GeoDataFrame], title: str) -> None:
    ax.set_facecolor("white")
    polys = _country_polygons(geo["country"])
    if polys:
        for p in polys:
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color="#F8F7F2", alpha=1.0, zorder=0)
    _plot_line_gdf(ax, geo["country"], color="#222222", lw=0.7, alpha=0.9, zorder=4, thin=1)
    _plot_line_gdf(ax, geo["contours"], color="#B9C6B2", lw=0.18, alpha=0.28, zorder=1, thin=10)
    bounds = np.array([g.total_bounds for g in [geo["country"], geo["districts"], geo["nh_roads"], geo["district_hq"]] if not g.empty])
    minx, miny = np.nanmin(bounds[:, [0, 1]], axis=0)
    maxx, maxy = np.nanmax(bounds[:, [2, 3]], axis=0)
    ax.set_xlim(max(79.7, minx - 0.35), min(89.0, maxx + 0.35))
    ax.set_ylim(max(25.8, miny - 0.35), min(31.1, maxy + 0.65))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", lw=0.35, alpha=0.25)
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude (N)")
    ax.set_title(title, fontsize=11, fontweight="bold")


def _label_box(text: str, x: float, y: float, fs: float, dx: float, dy: float) -> Tuple[float, float, float, float]:
    w = max(0.080, len(text) * fs * 0.0038)
    h = fs * 0.0100
    return (x + dx, y + dy, x + dx + w, y + dy + h)


def _overlaps(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _add_label(ax: plt.Axes, placed: List[Tuple[float, float, float, float]], text: str, xy: Tuple[float, float], color: str = "#222", fs: float = 6.0, weight: str = "bold", z: int = 20) -> bool:
    offsets = [(0.025, 0.015), (0.025, -0.035), (-0.08, 0.015), (-0.08, -0.035), (0.0, 0.05), (0.0, -0.06)]
    for dx, dy in offsets:
        box = _label_box(text, xy[0], xy[1], fs, dx, dy)
        if not any(_overlaps(box, old) for old in placed):
            ax.text(xy[0] + dx, xy[1] + dy, text, fontsize=fs, color=color, fontweight=weight, zorder=z, bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.8))
            placed.append(box)
            return True
    return False


def _add_forced_callout_label(ax: plt.Axes, text: str, xy: Tuple[float, float], idx: int, color: str = "#B71C1C") -> None:
    ring = 1 + (idx // 12) % 3
    offsets = [
        (8, 6), (12, 0), (8, -7), (0, -10), (-9, -7), (-13, 0),
        (-9, 7), (0, 10), (16, 9), (18, -6), (-17, -8), (-18, 8),
    ]
    dx, dy = offsets[idx % len(offsets)]
    ax.annotate(
        text,
        xy=xy,
        xytext=(dx * ring, dy * ring),
        textcoords="offset points",
        fontsize=4.7,
        color=color,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=18,
        bbox=dict(fc="white", ec=color, lw=0.18, alpha=0.82, boxstyle="round,pad=0.10"),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.22, alpha=0.55, shrinkA=0, shrinkB=1.5),
    )


def _draw_roads(ax: plt.Axes, plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]], alpha: float = 0.85, thin: bool = False) -> None:
    styles = {
        "Black topped": dict(color="#202020", lw=0.38 if thin else 0.75, ls="-", alpha=alpha),
        "Unpaved": dict(color="#8A6A3D", lw=0.32 if thin else 0.65, ls=(0, (3, 2)), alpha=alpha),
    }
    for pts, km, surface, _ in plot_roads:
        st = styles.get(surface, styles["Black topped"])
        _plot_safe_line(ax, pts, zorder=6, **st)


def route_display_label(d: Dict[str, Any]) -> str:
    rid = str(d.get("route_id") or d.get("nh_code") or "NH").strip()
    if " - " in rid and rid.startswith("NH"):
        return rid.split(" - ", 1)[0]
    return rid


def map1(cfg: MLPIConfig, G: nx.Graph, geo: Dict[str, gpd.GeoDataFrame], plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]]) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, "Map 1 - NH/SH Roads and District Headquarters")
    _plot_polygon_boundaries(ax, geo.get("districts", gpd.GeoDataFrame()), color="#D4D0C6", lw=0.28, alpha=0.75, zorder=2)
    _draw_roads(ax, plot_roads, alpha=0.56)
    _plot_line_gdf(ax, geo["nh_name"], color="#111111", lw=0.18, alpha=0.86, zorder=12, thin=1)
    _plot_line_gdf(ax, geo["district_hq_name"], color="#111111", lw=0.13, alpha=0.82, zorder=13, thin=1)
    for lon, lat in _centroids(geo["district_hq"]):
        ax.scatter(lon, lat, s=11, c="#C62828", marker="o", edgecolors="white", lw=0.35, zorder=12)
    placed: List[Tuple[float, float, float, float]] = []
    for lon, lat, name, _pop in cfg.place_lookup:
        if name not in MAP1_KEY_PLACES:
            continue
        ax.scatter(lon, lat, s=7, c="#303030", marker=".", zorder=14)
        _add_label(ax, placed, name, (lon, lat), color="#303030", fs=4.9, weight="normal", z=15)
    legend = [
        plt.Line2D([0], [0], color="#202020", lw=1.2, label="Black topped / paved NH"),
        plt.Line2D([0], [0], color="#8A6A3D", lw=1.2, ls=(0, (3, 2)), label="Unpaved / low-volume proxy"),
        plt.Line2D([0], [0], color="#D4D0C6", lw=0.8, label="District boundary"),
        plt.Line2D([0], [0], color="#111111", lw=0.35, label="Existing NH/DHQ annotation layers"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C62828", ms=5, label="District HQ"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=7, framealpha=0.92)
    fig.savefig(cfg.out_dir / "map1_roads_districts.png", dpi=700, bbox_inches="tight")
    fig.savefig(cfg.out_dir / "map1_roads_districts.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panel(
    ax: plt.Axes,
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    values: Dict[Tuple[int, int], float],
    title: str,
    cmap_name: str,
) -> None:
    _setup_map(ax, geo, title)
    vals = np.array(list(values.values()), dtype=float) if values else np.array([0.0])
    vmax = max(float(np.nanpercentile(vals, 95)), 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    for u, v, d in G.edges(data=True):
        if d.get("type") == "road":
            pts = sp.get((min(u, v), max(u, v)), [])
            if pts:
                _plot_safe_line(ax, pts, color="#D8D8D8", lw=0.35, alpha=0.65, zorder=5)
    if float(np.nanmax(vals)) <= 1e-9:
        ax.set_title(f"{title} (all zero)", fontsize=11, fontweight="bold")
    for u, v, d in G.edges(data=True):
        if d.get("type") != "road":
            continue
        key = (min(u, v), max(u, v))
        pts = sp.get(key, [])
        val = values.get(key, 0.0)
        if pts:
            _plot_safe_line(ax, pts, color=cmap(norm(val)), lw=0.35 + 1.7 * norm(val), alpha=0.86, zorder=7)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.62, pad=0.01)


def fig_dimension_maps(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    score_dict: Dict[Tuple[int, int], Dict[str, float]],
    component_keys: Sequence[str],
    composite_key: str,
    title: str,
    outname: str,
    cmap_name: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    panels = [(composite_key, title)] + [(k, k) for k in component_keys]
    for ax, (key, panel_title) in zip(axes.ravel(), panels):
        vals = {e: d.get(key, 0.0) for e, d in score_dict.items()}
        _plot_metric_panel(ax, cfg, G, geo, sp, vals, panel_title, cmap_name)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(cfg.out_dir / outname, dpi=220, bbox_inches="tight")
    plt.close(fig)


def map_top_ranked_links(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    ranked: pd.DataFrame,
    top_n: int,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, f"Map 6 - Top {top_n} Combined MLPI Links (P, S, E and I)")
    _draw_roads(ax, plot_roads, alpha=0.23, thin=True)
    df = ranked.copy()
    df["map_group"] = df.apply(lambda r: str(r.get("corridor_id") or r.get("NH_or_route") or f"{int(r.u)}-{int(r.v)}"), axis=1)
    df["map_label"] = df.apply(lambda r: str(r.get("corridor_label") or r.get("NH_or_route") or r.get("edge_name") or r["map_group"]), axis=1)
    top = (
        df.groupby("map_group")
        .agg(mlpi=("mlpi", "max"), mlpi_mean=("mlpi", "mean"), corridor_score=("corridor_score", "max"), label=("map_label", "first"))
        .reset_index()
    )
    top["map_score"] = 0.75 * top["mlpi"] + 0.17 * top["mlpi_mean"] + 0.08 * top["corridor_score"]
    top = top.sort_values("map_score", ascending=False).head(top_n)
    norm = Normalize(vmin=float(top["map_score"].min()), vmax=float(top["map_score"].max()) if len(top) else 1.0)
    cmap = plt.get_cmap("magma_r")
    placed: List[Tuple[float, float, float, float]] = []
    for map_rank, row in enumerate(top.itertuples(index=False), start=1):
        color = cmap(norm(float(row.map_score)))
        mids: List[Tuple[float, float]] = []
        for u, v, d in G.edges(data=True):
            if d.get("type") != "road" or str(d.get("corridor_id") or d.get("route_id") or "") != str(row.map_group):
                continue
            pts = sp.get((min(u, v), max(u, v)), [])
            if pts:
                _plot_safe_line(ax, pts, color=color, lw=1.35, alpha=0.92, zorder=15)
                mids.append(_midpoint(pts))
        if mids:
            label = f"{map_rank}. {str(row.label).split(' - ', 1)[0]}"
            _add_label(ax, placed, label, mids[len(mids) // 2], color="#7A0019", fs=5.9, z=17)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.48, pad=0.01, label="MLPI")
    fig.savefig(cfg.out_dir / "map6_combined_mlpi_top10.png", dpi=300, bbox_inches="tight")
    fig.savefig(cfg.out_dir / "map6_combined_mlpi_top10.pdf", bbox_inches="tight")
    plt.close(fig)


def map_dimension_top_links(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    ranked: pd.DataFrame,
    score_col: str,
    title: str,
    outname: str,
    color: str,
    top_n: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, title)
    _draw_roads(ax, plot_roads, alpha=0.24, thin=True)
    df = ranked.copy()
    df["map_group"] = df.apply(
        lambda r: str(r.get("corridor_id") or r.get("NH_or_route") or f"{int(r.u)}-{int(r.v)}"),
        axis=1,
    )
    df["map_label"] = df.apply(
        lambda r: str(r.get("corridor_label") or r.get("NH_or_route") or r.get("edge_name") or r["map_group"]),
        axis=1,
    )
    groups = (
        df.groupby("map_group")
        .agg(
            score_max=(score_col, "max"),
            score_mean=(score_col, "mean"),
            mlpi_max=("mlpi", "max"),
            corridor_score=("corridor_score", "max"),
            label=("map_label", "first"),
        )
        .reset_index()
    )
    groups["map_score"] = 0.72 * groups["score_max"] + 0.20 * groups["score_mean"] + 0.08 * groups["corridor_score"]
    top = groups.sort_values("map_score", ascending=False).head(top_n).copy()
    placed: List[Tuple[float, float, float, float]] = []
    for map_rank, row in enumerate(top.itertuples(index=False), start=1):
        group = str(row.map_group)
        group_pts: List[List[Tuple[float, float]]] = []
        mids: List[Tuple[float, float]] = []
        for u, v, d in G.edges(data=True):
            if d.get("type") != "road":
                continue
            if str(d.get("corridor_id") or d.get("route_id") or "") != group:
                continue
            key = (min(u, v), max(u, v))
            pts = sp.get(key, [])
            if pts:
                group_pts.append(pts)
                mids.append(_midpoint(pts))
        if not group_pts:
            continue
        for pts in group_pts:
            _plot_safe_line(ax, pts, color=color, lw=1.25, alpha=0.88, zorder=15)
        if mids:
            mid = mids[len(mids) // 2]
            label = f"{map_rank}. {str(row.label).split(' - ', 1)[0]}"
            _add_label(ax, placed, label, mid, color=color, fs=5.9, z=17)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#202020", lw=0.9, label="Other NH/SH roads"),
            plt.Line2D([0], [0], color=color, lw=1.4, label=f"Top {top_n} corridors by {score_col}"),
        ],
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
    )
    fig.savefig(cfg.out_dir / outname, dpi=320, bbox_inches="tight")
    plt.close(fig)


def map_airport_interconnected_top_links(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    ranked: pd.DataFrame,
    top_n: int = 10,
) -> None:
    """Map nonredundant road approaches to operational airports."""
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, f"Map 5 - Top {top_n} Nonredundant Airport-Access Links")
    _draw_roads(ax, plot_roads, alpha=0.24, thin=True)

    airports: List[Tuple[str, Tuple[float, float], str]] = []
    access_nodes: Dict[str, List[int]] = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d.get("type") != "air":
            continue
        if G.nodes[u].get("is_airport"):
            ap, road = u, v
        elif G.nodes[v].get("is_airport"):
            ap, road = v, u
        else:
            continue
        status = str(G.nodes[ap].get("airport_status", "In operation"))
        if status == "Not in operation":
            continue
        code = str(G.nodes[ap].get("airport_name", ap))
        pos = G.nodes[ap].get("pos")
        if pos:
            airports.append((code, pos, status))
            access_nodes[code].append(road)

    if not airports:
        fig.savefig(cfg.out_dir / "map5_interconnected_top10.png", dpi=320, bbox_inches="tight")
        plt.close(fig)
        return

    rows: List[Dict[str, Any]] = []
    for _, row in ranked.iterrows():
        key = (min(int(row.u), int(row.v)), max(int(row.u), int(row.v)))
        pts = sp.get(key, [])
        if not pts:
            continue
        mid = _midpoint(pts)
        code, ap_pos, _status = min(airports, key=lambda rec: _dist_km(mid, rec[1]))
        dist = _dist_km(mid, ap_pos)
        if dist > max(cfg.airport_access_km * 2.2, 85.0):
            continue
        proximity = max(0.0, 1.0 - dist / max(cfg.airport_access_km * 2.2, 85.0))
        airport_access_score = (
            0.48 * float(row.get("interconnected", 0.0))
            + 0.22 * float(row.get("I1", 0.0))
            + 0.22 * float(row.get("I2", 0.0))
            + 0.08 * proximity
        )
        rows.append(
            dict(
                score=airport_access_score,
                airport=code,
                airport_pos=ap_pos,
                u=int(row.u),
                v=int(row.v),
                corridor_id=str(row.get("corridor_id") or row.get("NH_or_route") or f"{int(row.u)}-{int(row.v)}"),
                label=str(row.get("corridor_label") or row.get("NH_or_route") or row.get("edge_name") or ""),
                pts=pts,
                mid=mid,
                dist=dist,
            )
        )

    if not rows:
        rows = []
        for _, row in ranked.head(top_n).iterrows():
            key = (min(int(row.u), int(row.v)), max(int(row.u), int(row.v)))
            pts = sp.get(key, [])
            if not pts:
                continue
            mid = _midpoint(pts)
            code, ap_pos, _status = min(airports, key=lambda rec: _dist_km(mid, rec[1]))
            rows.append(
                dict(
                    score=float(row.get("interconnected", 0.0)),
                    airport=code,
                    airport_pos=ap_pos,
                    u=int(row.u),
                    v=int(row.v),
                    corridor_id=str(row.get("corridor_id") or row.get("NH_or_route") or f"{int(row.u)}-{int(row.v)}"),
                    label=str(row.get("corridor_label") or row.get("NH_or_route") or row.get("edge_name") or ""),
                    pts=pts,
                    mid=mid,
                    dist=_dist_km(mid, ap_pos),
                )
            )

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[(item["corridor_id"], item["airport"])].append(item)

    group_rows = []
    for (corridor, airport), items in grouped.items():
        best = max(items, key=lambda x: x["score"])
        group_rows.append(
            dict(
                corridor=corridor,
                airport=airport,
                label=best["label"],
                airport_pos=best["airport_pos"],
                score=0.70 * max(x["score"] for x in items) + 0.30 * float(np.mean([x["score"] for x in items])),
                items=items,
            )
        )
    group_rows = sorted(group_rows, key=lambda x: x["score"], reverse=True)[:top_n]

    placed: List[Tuple[float, float, float, float]] = []
    for code, pos, _status in airports:
        ax.scatter(pos[0], pos[1], s=30, c="#1565C0", marker="o", edgecolors="white", lw=0.45, zorder=13)
        _add_label(ax, placed, code, pos, color="#1A237E", fs=5.0, weight="bold", z=14)

    cmap = plt.get_cmap("Purples")
    vals = [g["score"] for g in group_rows]
    norm = Normalize(vmin=min(vals) if vals else 0.0, vmax=max(vals) if vals else 1.0)
    for map_rank, group in enumerate(group_rows, start=1):
        items = sorted(group["items"], key=lambda x: x["score"], reverse=True)
        color = cmap(0.45 + 0.50 * norm(float(group["score"])))
        top_items = items[: max(2, min(8, int(math.ceil(len(items) * 0.28))))]
        mids = []
        for item in top_items:
            _plot_safe_line(ax, item["pts"], color=color, lw=1.45, alpha=0.95, zorder=16)
            mids.append(item["mid"])
        if mids:
            mid = mids[len(mids) // 2]
            ap_pos = group["airport_pos"]
            ax.plot(
                [mid[0], ap_pos[0]],
                [mid[1], ap_pos[1]],
                color="#6A1B9A",
                lw=0.32,
                ls=(0, (3, 3)),
                alpha=0.42,
                zorder=12,
            )
            label_base = str(group["label"]).split(" - ", 1)[0] or str(group["corridor"])
            _add_label(ax, placed, f"{map_rank}. {label_base} -> {group['airport']}", mid, color="#4A148C", fs=5.8, z=17)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#202020", lw=0.9, label="Other NH/SH roads"),
            plt.Line2D([0], [0], color="#6A1B9A", lw=1.5, label="Airport-access bottleneck links"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565C0", markeredgecolor="white", markersize=6, label="Operational airport"),
        ],
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
    )
    fig.savefig(cfg.out_dir / "map5_interconnected_top10.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def zoom_ranked_links(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    ranked: pd.DataFrame,
    top_n: int = 20,
) -> None:
    regions = [
        ("West Nepal", 80.0, 83.2, 27.4, 30.5, "zoom_west_top_links.png"),
        ("Mid Nepal", 83.0, 86.0, 26.7, 29.4, "zoom_mid_top_links.png"),
        ("East Nepal", 86.0, 88.5, 26.2, 28.2, "zoom_east_top_links.png"),
    ]
    top = ranked.head(top_n)
    top_keys = {(int(r.u), int(r.v)): int(r["rank"]) for _, r in top.iterrows()}
    top_keys |= {(b, a): rk for (a, b), rk in list(top_keys.items())}
    for title, x0, x1, y0, y1, outname in regions:
        fig, ax = plt.subplots(figsize=(11, 7))
        _setup_map(ax, geo, f"{title} - Top {top_n} Ranked Links")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        _draw_roads(ax, plot_roads, alpha=0.28, thin=True)
        placed: List[Tuple[float, float, float, float]] = []
        for _, row in top.iterrows():
            u, v, rk = int(row.u), int(row.v), int(row["rank"])
            key = (min(u, v), max(u, v))
            pts = sp.get(key, [])
            if not pts:
                continue
            mid = _midpoint(pts)
            if not (x0 <= mid[0] <= x1 and y0 <= mid[1] <= y1):
                continue
            _plot_safe_line(ax, pts, color="#C62828", lw=2.2, alpha=0.95, zorder=12)
            _add_label(ax, placed, f"{rk}. {row.edge_name}", mid, color="#9A0007", fs=6.2, z=15)
        fig.savefig(cfg.out_dir / outname, dpi=220, bbox_inches="tight")
        plt.close(fig)


def _section_lookup(sections: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(s["section_id"]): s for s in sections}


def _distinct_top_sections(
    ranked: pd.DataFrame,
    sections: Sequence[Dict[str, Any]],
    score_col: str,
    top_n: int,
    *,
    region_balanced: bool = False,
    min_per_region: int = 1,
) -> pd.DataFrame:
    lookup = _section_lookup(sections)
    selected: List[pd.Series] = []
    selected_edges: List[set] = []

    def add_if_distinct(row: pd.Series, *, overlap_limit: float = 0.60) -> bool:
        section = lookup.get(str(row.section_id))
        if not section:
            return False
        edges = set(section["road_edges"])
        overlap = max(
            (len(edges & old) / max(min(len(edges), len(old)), 1) for old in selected_edges),
            default=0.0,
        )
        if overlap <= overlap_limit:
            selected.append(row)
            selected_edges.append(edges)
            return True
        return False

    tie_cols = [score_col]
    tie_ascending = [False]
    if "mlpi" in ranked.columns and score_col != "mlpi":
        tie_cols.append("mlpi")
        tie_ascending.append(False)
    if "rank" in ranked.columns:
        tie_cols.append("rank")
        tie_ascending.append(True)
    sorted_ranked = ranked.sort_values(tie_cols, ascending=tie_ascending, kind="mergesort")
    if region_balanced and "macro_region" in ranked.columns:
        region_order = ["West", "Central", "East"]
        present = [r for r in region_order if r in set(ranked["macro_region"].astype(str))]
        present += sorted(set(ranked["macro_region"].astype(str)) - set(present))
        for _ in range(max(1, min_per_region)):
            for region in present:
                if len(selected) >= top_n:
                    break
                region_rows = sorted_ranked[sorted_ranked["macro_region"].astype(str) == region]
                for _, row in region_rows.iterrows():
                    if str(row.section_id) in {str(r.section_id) for r in selected}:
                        continue
                    if add_if_distinct(row):
                        break
            if len(selected) >= top_n:
                break

    for _, row in sorted_ranked.iterrows():
        if str(row.section_id) in {str(r.section_id) for r in selected}:
            continue
        add_if_distinct(row)
        if len(selected) >= top_n:
            break
    if len(selected) < top_n:
        used = {str(r.section_id) for r in selected}
        for _, row in sorted_ranked.iterrows():
            if str(row.section_id) not in used:
                selected.append(row)
                used.add(str(row.section_id))
            if len(selected) >= top_n:
                break
    return pd.DataFrame(selected)


def _draw_exact_section(
    ax: plt.Axes,
    section: Dict[str, Any],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    color: Any,
    lw: float = 1.55,
    zorder: int = 16,
) -> None:
    for key in section["road_edges"]:
        pts = sp.get(key, [])
        if pts:
            _plot_safe_line(ax, pts, color=color, lw=lw, alpha=0.96, zorder=zorder)


def _add_rank_marker(ax: plt.Axes, rank: int, pos: Tuple[float, float], color: str) -> None:
    ax.scatter(pos[0], pos[1], s=34, facecolors="white", edgecolors=color, linewidths=0.9, zorder=24)
    ax.text(
        pos[0],
        pos[1],
        str(rank),
        ha="center",
        va="center",
        color=color,
        fontsize=5.2,
        fontweight="bold",
        zorder=25,
    )


def _add_ranked_section_list(ax: plt.Axes, rows: Sequence[Tuple[int, str]], color: str) -> None:
    lines = ["Top road links"]
    for rank, name in rows:
        lines.extend(textwrap.wrap(f"{rank}. {name}", width=43, subsequent_indent="   "))
    ax.text(
        0.985,
        0.975,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=color,
        fontsize=6.0,
        linespacing=1.22,
        fontweight="bold",
        zorder=30,
    )


def map_exact_dimension_sections(
    cfg: MLPIConfig,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    sections: Sequence[Dict[str, Any]],
    ranked: pd.DataFrame,
    score_col: str,
    title: str,
    outname: str,
    color: str,
    top_n: int = 10,
    region_balanced: bool = False,
    min_per_region: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, title)
    _draw_roads(ax, plot_roads, alpha=0.20, thin=True)
    lookup = _section_lookup(sections)
    if region_balanced:
        top = _distinct_top_sections(
            ranked,
            sections,
            score_col,
            top_n,
            region_balanced=True,
            min_per_region=min_per_region,
        )
    else:
        top = ranked.sort_values(score_col, ascending=False, kind="mergesort").head(top_n)
    top_rows = list(top.itertuples(index=False))
    for row in reversed(top_rows):
        section = lookup[str(row.section_id)]
        _draw_exact_section(ax, section, sp, color=color, lw=1.55)
    ranked_names = []
    for map_rank, row in enumerate(top_rows, start=1):
        section = lookup[str(row.section_id)]
        _add_rank_marker(ax, map_rank, section["midpoint"], color)
        ranked_names.append((map_rank, section["section_name"]))
    _add_ranked_section_list(ax, ranked_names, color)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#202020", lw=0.8, label="Other NH/SH roads"),
            plt.Line2D(
                [0],
                [0],
                color=color,
                lw=1.6,
                label=(f"Top {top_n} road links, balanced by region" if region_balanced else f"Top {top_n} road links"),
            ),
        ],
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
    )
    fig.savefig(cfg.out_dir / outname, dpi=340, bbox_inches="tight")
    plt.close(fig)


def map_exact_interconnected_sections(
    cfg: MLPIConfig,
    G: nx.Graph,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    sections: Sequence[Dict[str, Any]],
    ranked: pd.DataFrame,
    top_n: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, f"Map 5 - Top {top_n} Roads Affecting Airport Access")
    _draw_roads(ax, plot_roads, alpha=0.20, thin=True)
    lookup = _section_lookup(sections)
    top = ranked.sort_values("interconnected", ascending=False, kind="mergesort").head(top_n)
    airport_pos = {
        str(d.get("airport_name", n)): d["pos"]
        for n, d in G.nodes(data=True)
        if d.get("is_airport") and d.get("airport_status") != "Not in operation"
    }
    placed: List[Tuple[float, float, float, float]] = []
    for code, pos in airport_pos.items():
        ax.scatter(pos[0], pos[1], s=24, c="#1565C0", marker="o", edgecolors="white", lw=0.4, zorder=13)
        _add_label(ax, placed, code, pos, color="#1A237E", fs=4.7, weight="bold", z=14)
    top_rows = list(top.itertuples(index=False))
    for row in reversed(top_rows):
        section = lookup[str(row.section_id)]
        _draw_exact_section(ax, section, sp, color="#6A1B9A", lw=1.60)
        target = str(getattr(row, "target_airport", "") or "")
        if target:
            if target in airport_pos:
                mid = section["midpoint"]
                ap = airport_pos[target]
                ax.plot([mid[0], ap[0]], [mid[1], ap[1]], color="#6A1B9A", lw=0.32, ls=(0, (3, 3)), alpha=0.45, zorder=12)
    ranked_names = []
    for map_rank, row in enumerate(top_rows, start=1):
        section = lookup[str(row.section_id)]
        target = str(getattr(row, "target_airport", "") or "")
        _add_rank_marker(ax, map_rank, section["midpoint"], "#4A148C")
        ranked_names.append((map_rank, f"{section['section_name']} -> {target}" if target else section["section_name"]))
    _add_ranked_section_list(ax, ranked_names, "#4A148C")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#202020", lw=0.8, label="Other NH/SH roads"),
            plt.Line2D([0], [0], color="#6A1B9A", lw=1.6, label="Road links affecting airport access"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565C0", markersize=5, label="Operational airport"),
        ],
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
    )
    fig.savefig(cfg.out_dir / "map5_interconnected_top10.png", dpi=340, bbox_inches="tight")
    plt.close(fig)


def map_exact_combined_sections(
    cfg: MLPIConfig,
    geo: Dict[str, gpd.GeoDataFrame],
    sp: Dict[Tuple[int, int], List[Tuple[float, float]]],
    plot_roads: List[Tuple[List[Tuple[float, float]], float, str, int]],
    sections: Sequence[Dict[str, Any]],
    ranked: pd.DataFrame,
    top_n: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _setup_map(ax, geo, f"Map 6 - Top {top_n} Overall Important Road Links")
    _draw_roads(ax, plot_roads, alpha=0.20, thin=True)
    lookup = _section_lookup(sections)
    top = ranked.sort_values("mlpi", ascending=False, kind="mergesort").head(top_n)
    vals = top["mlpi"].astype(float).to_numpy() if len(top) else np.array([0.0])
    norm = Normalize(vmin=float(vals.min()), vmax=max(float(vals.max()), float(vals.min()) + 1e-9))
    cmap = plt.get_cmap("magma_r")
    top_rows = list(top.itertuples(index=False))
    for row in reversed(top_rows):
        section = lookup[str(row.section_id)]
        color = cmap(norm(float(row.mlpi)))
        _draw_exact_section(ax, section, sp, color=color, lw=1.65)
    ranked_names = []
    for map_rank, row in enumerate(top_rows, start=1):
        section = lookup[str(row.section_id)]
        _add_rank_marker(ax, map_rank, section["midpoint"], "#7A0019")
        ranked_names.append((map_rank, section["section_name"]))
    _add_ranked_section_list(ax, ranked_names, "#7A0019")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.48, pad=0.01, label="MLPI")
    fig.savefig(cfg.out_dir / "map6_combined_mlpi_top10.png", dpi=340, bbox_inches="tight")
    fig.savefig(cfg.out_dir / "map6_combined_mlpi_top10.pdf", bbox_inches="tight")
    plt.close(fig)


def write_methodology_workbook(
    cfg: MLPIConfig,
    ranked: pd.DataFrame,
) -> Optional[Path]:
    """Create an auditable Excel workbook listing the formulas used."""
    out = cfg.out_dir / "mlpi_methodology_formulas.xlsx"
    default_formula_rows = [
        ("Final MLPI", "MLPI(link)", "wP*P + wS*S + wE*E + wI*I", "Overall score for each non-overlapping road link."),
        ("Failure unit", "Road link", "continuous road path between junctions or dead ends", "Every physical Nepal.gpkg NH edge is assigned to one and only one scored link."),
        ("Physical", "P", f"({cfg.w_p_fftdi:g}*OD_TSTT_FFTDI_norm + {cfg.w_p_flow:g}*detour_weighted_OD_exposure_norm)/({cfg.w_p_fftdi:g}+{cfg.w_p_flow:g})", "Road-only free-flow link-removal consequence. High flow matters only when the failed link creates OD travel-time loss."),
        ("Physical", "P1 / OD_TSTT_FFTDI", "SUM(q_ij*max(0,t_ij_removed-t_ij_base))/SUM(q_ij*t_ij_base)", "Each failed road link is removed and OD shortest paths are recomputed under free-flow costs."),
        ("Physical", "P2 / detour-weighted OD exposure", "affected_OD_base_vht * min(1,max(0,(affected_OD_removed_vht/affected_OD_base_vht - 1)/detour_full_credit_ratio))", "Baseline OD exposure on the failed link is credited only when failure produces a meaningful detour."),
        ("Assignment", "Dijkstra free-flow shortest path", "min_path SUM(edge_travel_time_h)", "Each edge travel time is computed before routing; Dijkstra selects the route with the minimum total travel time. The demand surface is the externally generated ME2 OD matrix."),
        ("ME2 OD", "External calibrated matrix", "q_ij is read from OD/outputs/od_matrix_names.csv", "The CSV is treated as the authoritative calibrated district/HQ OD matrix for physical link-removal scoring."),
        ("ME2 OD", "Zone-node mapping", "district/HQ OD zones -> representative NH/SH road nodes", "Mapping first uses the conserved district socioeconomic allocation; district/HQ anchor nearest-node assignment is audited separately."),
        ("ME2 OD", "Baseline path filter", "Only OD pairs connected on the baseline MLPI road graph contribute to TSTT loss", "Disconnected baseline OD pairs are written to me2_od_pairs_used.csv and should be checked against road filtering/connectivity."),
        ("Physical", "FFTDI_raw", "(TSTT_removed(link)-TSTT_base)/TSTT_base", "TSTT is OD-weighted free-flow vehicle-hours; topology connectors remain analysis-only."),
        ("Population allocation", "pop_n / GDP_n", f"district total split over up to {cfg.socioeconomic_nodes_per_district} representative NH/SH nodes", "District polygons are used when supplied; otherwise the district/HQ anchor is allocated to the closest representative Nepal.gpkg NH nodes."),
        ("Social", "S", "(wS1*S1 + wS2*S2)/(wS1+wS2)", "Only S1 and S2 are active; both are min-max normalized before aggregation."),
        ("Social", "S1", "newly_isolated_population_after_link_removal / total_conserved_population", "New isolation is evaluated within each baseline road component using representative district road-node population."),
        ("Social", "S2", "SUM(pop_n * max(0, t_healthcare_removed_n - t_healthcare_base_n))/SUM(pop_n)", "Road travel-time increase to eligible hospital/clinic/health-post facilities."),
        ("Economic", "E", "(wE1*E1 + wE2*E2)/(wE1+wE2)", "Economic terms are min-max normalized before aggregation."),
        ("Economic", "E1", "OD_vehicle_hour_loss * section_trade_exposure", "Travel-time loss weighted by the failed link's AADT and border-gateway accessibility, so urban circulation links do not dominate trade criticality."),
        ("Economic", "E2", "SUM(GDP_weight_n * max(0, border_access_base_n - border_access_removed_n))/SUM(GDP_weight_n)", "GDP-weighted Hansen-style border-trade gateway access loss. GDP weights are region-balanced when e2_regional_balance is TRUE."),
        ("Interconnected", "I", "(wI1*I1 + wI2*I2)/(wI1+wI2)", "Only I1 and I2 are active; both are min-max normalized before aggregation."),
        ("Interconnected", "I1", "SUM(pop_n*max(0,t_airport_removed_n-t_airport_base_n))/SUM(pop with baseline access)", "Population-weighted road travel-time increase to the nearest operational airport."),
        ("Interconnected", "I2", "population losing all operational-airport road access / population with baseline access", "Complete road-access loss; alternatives naturally reduce the consequence."),
    ]
    formula_df = pd.DataFrame(default_formula_rows, columns=["Dimension", "Term", "Formula", "Implementation_note"])
    params = [
        ("Input workbook", str(cfg.input_xlsx)),
        ("Master GeoPackage", str(cfg.master_gpkg_path)),
        ("Road layer source", f"{cfg.master_gpkg_path}:NH"),
        ("District-HQ source", f"{cfg.master_gpkg_path}:district_hq"),
        ("Road layer filter", cfg.road_ref_filter),
        ("Road layer filter source", cfg.road_ref_filter_source),
        ("Road AADT field", cfg.road_aadt_field),
        ("Future road AADT field", cfg.road_future_aadt_field),
        ("Allow Input.xlsx AADT gap fill", cfg.allow_input_aadt_gap_fill),
        ("Mean speed factor", cfg.mean_speed_factor),
        ("Formula index source", "Implemented formulas in mlpi_robust.py"),
        ("Named-place source", cfg.named_place_source),
        ("Airport table source", cfg.airport_registry_source or "Not loaded"),
        ("Verified helipads from Input.xlsx", len(cfg.helipads)),
        ("Road speed source", f"mean_speed_kmh = {cfg.mean_speed_factor:.2f} * Nepal.gpkg layer NH design_speed_kmh."),
        ("Assignment algorithm", "Free-flow shortest-path OD loading and road-link removal"),
        ("Physical detour factor full credit", f"{1.0 + cfg.detour_full_credit_ratio:g}x route-time increase or disconnection"),
        ("ME2 OD matrix", str(cfg.od_matrix_path)),
        ("ME2 OD total positive demand", cfg.od_matrix_total_demand),
        ("ME2 OD inter-node pairs loaded", cfg.od_pairs_loaded),
        ("ME2 OD baseline connected pairs", cfg.od_baseline_connected_pairs),
        ("ME2 OD baseline disconnected pairs", cfg.od_baseline_disconnected_pairs),
        ("Hansen access decay time h", cfg.od_deterrence_h),
        ("Detour full-credit ratio", cfg.detour_full_credit_ratio),
        ("E2 regional balance", cfg.e2_regional_balance),
        ("BPR congestion alpha", cfg.bpr_alpha),
        ("BPR congestion beta power", cfg.bpr_beta_power),
        ("Socioeconomic allocation nodes per district", cfg.socioeconomic_nodes_per_district),
        ("Disconnected OD penalty h", cfg.disconnected_penalty_h),
        ("Airport ground-access radius km", cfg.airport_access_km),
        ("Air cost multiplier", cfg.air_cost_multiplier),
        ("Air fixed penalty h", cfg.air_fixed_penalty_h),
        ("Air speed km/h", cfg.speed_air),
        ("Road links scored", len(ranked)),
        ("Failure-unit source", "Generated from Nepal.gpkg layer NH topology"),
        ("AADT station-match radius km", f"{cfg.station_match_km}"),
    ]
    weights = [
        ("MLPI physical", cfg.weights.get("physical")),
        ("MLPI social", cfg.weights.get("social")),
        ("MLPI economic", cfg.weights.get("economic")),
        ("MLPI interconnected", cfg.weights.get("interconnected")),
        ("Physical P1 FFTDI", cfg.w_p_fftdi),
        ("Physical P2 detour-weighted OD exposure", cfg.w_p_flow),
        ("Social S1", cfg.w_s1),
        ("Social S2", cfg.w_s2),
        ("Economic E1", cfg.w_e1),
        ("Economic E2", cfg.w_e2),
        ("Interconnected I1", cfg.w_i1),
        ("Interconnected I2", cfg.w_i2),
        ("Hansen beta", cfg.beta),
    ]
    references = [
        ("Physical FFTDI", "Almotahari & Yazici", "A link criticality index concept adapted here as a Free-Flow Travel Disruption Index for road-link removal", "Transportation Research Part A, 2019", "10.1016/j.tra.2019.06.005"),
        ("ME2 OD", "Van Zuylen & Willumsen", "The most likely trip matrix estimated from traffic counts", "Transportation Research B, 1980", ""),
        ("ME2 OD", "Spiess", "A maximum likelihood model for estimating origin-destination matrices", "Transportation Research B, 1987", ""),
        ("Air generalized cost", "Airport choice/access-mode literature", "Airport and route choice models include access time, access cost/fare, flying time, waiting/transfer/terminal components", "Transportation airport choice literature", "Modeling joint airport and route choice behavior, 2014"),
        ("Airport accessibility", "Kouwenhoven", "The Role of Accessibility in Passengers' Choice of Airports: access time, access cost, waiting time and generalized travel cost motivate high-impedance air alternatives", "OECD/ITF Discussion Paper, 2008", "10.1787/235278552305"),
        ("Airport access reliability", "Koster, Kroes & Verhoef", "Travel time variability and airport accessibility: airport access has high schedule/reliability costs", "Transportation Research Part B, 2011", "10.1016/j.trb.2011.05.027"),
        ("Air speed calibration", "General helicopter/fixed-wing continuity assumption", "Air legs use 400 km/h with a 10x perceived-cost multiplier and 4 h fixed mobilization/access penalty.", "Model calibration", "Workbook-overridable assumption"),
        ("Economic criticality", "Colon, Hallegatte & Rozenberg", "Criticality analysis of a country's transport network via an agent-based supply chain model", "Nature Sustainability, 2021", "10.1038/s41893-020-00649-4"),
        ("Interconnectedness", "Thacker, Pant & Hall", "System-of-systems formulation and disruption analysis for multi-scale critical national infrastructures", "Reliability Engineering & System Safety, 2017", "10.1016/j.ress.2016.08.012"),
        ("Interconnected multimodal resilience", "Xu & Chopra", "Interconnectedness enhances network resilience of multimodal public transportation systems for Safe-to-Fail urban mobility", "Nature Communications, 2023", "10.1038/s41467-023-39999-w"),
    ]
    joint_rows = [
        ("Road-link removal", "Every scored unit is one continuous road link between junctions or dead ends; every physical Nepal.gpkg NH edge belongs to one unit."),
        ("Dimension graphs", "Physical, Social and Economic scores use the road graph only. Airport access measures road access to operational airports; no air substitution is allowed."),
        ("Air assumptions", "The 400 km/h, 10x multiplier and 4 h fixed penalty are explicit workbook scenario values."),
        ("ME2 OD input", f"OD demand is read from {cfg.od_matrix_path}."),
        ("Economic E2 regional balance", "When TRUE, each West/Central/East macro-region contributes equal total GDP weight before border-access loss is aggregated. When FALSE, raw national GDP weights are used."),
        ("ME2 OD limitation", "The OD matrix is accepted as the calibrated demand input. Baseline-disconnected mapped pairs are excluded from link-removal TSTT and are audited for network-connectivity review."),
        ("Physical speeds", f"Road mean speed is computed as {cfg.mean_speed_factor:.2f} times Nepal.gpkg layer NH design speed."),
        ("AADT use", f"Road AADT is read from Nepal.gpkg layer NH field {cfg.road_aadt_field}."),
        ("Physical score guardrail", "Raw flow alone is not a criticality measure. A failed link must increase OD total system travel time before its exposure strongly affects P."),
        ("Score basis", "Scores arise from modeled consequences rather than manually boosted highway or regional priors."),
        ("OD demand", "The ME2 OD matrix is the demand input for link-removal scoring."),
        ("Topology gaps", "Short analysis-only connectors close small source-geometry endpoint gaps and are not counted as ranked physical road edges."),
    ]
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            normalized_cols = [
                "rank", "section_id", "section_name", "macro_region", "mlpi",
                "physical", "social", "economic", "interconnected",
                "me2_top_od_pair", "me2_top_od_origin", "me2_top_od_destination",
                "me2_top_od_demand", "me2_top_od_delta_time_h", "affected_od_pair_count",
                "affected_od_demand", "length_km", "mean_aadt", "aadt_source",
                "terrain_zone", "model_speed_kmh", "travel_speed_kmh", "source_design_speed_kmh",
                "mean_speed_kmh", "design_speed_source", "surface", "surface_source", "target_airport",
            ]
            component_cols = [
                "rank", "section_id", "section_name", "macro_region", "length_km", "mean_aadt",
                "physical", "P1_FFTDI", "P2_detour_weighted_exposure", "P2_AADT_exposure", "P2_UE_flow", "disconnected_demand_raw",
                "FFTDI_raw", "vehicle_hour_loss", "od_vehicle_hour_loss",
                "affected_od_base_vht", "affected_od_removed_vht", "affected_od_demand", "affected_od_pair_count", "baseline_link_od_flow",
                "me2_top_od_pair", "me2_top_od_origin", "me2_top_od_destination",
                "me2_top_od_origin_district", "me2_top_od_destination_district",
                "me2_top_od_demand", "me2_top_od_base_time_h", "me2_top_od_removed_time_h",
                "me2_top_od_delta_time_h", "me2_top_od_vehicle_hour_loss", "me2_top5_od_pairs",
                "detour_ratio_after_failure", "physical_redundancy_factor",
                "raw_aadt_exposure_vht", "detour_weighted_exposure_vht", "aadt_exposure_vht", "ue_flow", "base_ue_tstt_h", "base_assignment_rgap",
                "base_assignment_iterations", "removed_assignment_rgap", "removed_assignment_iterations",
                "social", "S1", "S2", "S1_raw", "S2_raw", "healthcare_facilities_used", "healthcare_proxy_used", "healthcare_proxy_node",
                "economic", "E1", "E2", "E1_raw", "E2_raw", "E1_trade_exposure_raw", "E2_regional_balance",
                "interconnected", "I1", "I2", "I1_raw", "I2_raw", "target_airport",
                "direct_km", "detour_ratio", "edge_count", "aadt_source", "terrain_zone", "terrain_source",
                "model_speed_kmh", "travel_speed_kmh", "source_design_speed_kmh", "mean_speed_kmh", "design_speed_source", "model_speed_source", "surface_source",
            ]
            ranked[[c for c in normalized_cols if c in ranked.columns]].to_excel(writer, sheet_name="Scores_Normalized", index=False)
            ranked[[c for c in component_cols if c in ranked.columns]].to_excel(writer, sheet_name="Components_Raw", index=False)
            pd.DataFrame(params, columns=["Parameter", "Value"]).to_excel(writer, sheet_name="Inputs_Used", index=False)
            pd.DataFrame(weights, columns=["Weight_or_factor", "Value"]).to_excel(writer, sheet_name="Weights", index=False)
            formula_df.to_excel(writer, sheet_name="Formula_Index", index=False)
            pd.DataFrame(joint_rows, columns=["Principle", "Definition"]).to_excel(writer, sheet_name="Model_Assumptions", index=False)
            pd.DataFrame(references, columns=["Dimension", "Authors", "Paper", "Venue", "DOI"]).to_excel(writer, sheet_name="References", index=False)
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                for col in ws.columns:
                    header = str(col[0].value) if col and col[0].value is not None else ""
                    max_len = max([len(header)] + [len(str(c.value)) for c in col[1:40] if c.value is not None])
                    ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 70)
        return out
    except Exception as exc:
        warnings.warn(f"Could not write methodology workbook {out}: {exc}")
        return None


def methodology_figure(cfg: MLPIConfig, title: str, steps: Sequence[str], outname: str, accent: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    n = len(steps)
    xs = np.linspace(0.07, 0.93, n)
    y = 0.50
    for i, (x, txt) in enumerate(zip(xs, steps)):
        rect = mpatches.FancyBboxPatch(
            (x - 0.075, y - 0.16),
            0.15,
            0.32,
            boxstyle="round,pad=0.018,rounding_size=0.012",
            fc="#FFFFFF",
            ec=accent,
            lw=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.2, color="#202020", transform=ax.transAxes, wrap=True)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.085, y),
                xytext=(x + 0.085, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.1, color="#555555"),
            )
    fig.savefig(cfg.out_dir / outname, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_methodology_figures(cfg: MLPIConfig) -> None:
    methodology_figure(
        cfg,
        "Fig. 2 - Physical: Travel Delay After a Road Link Fails",
        [
            f"Nepal.gpkg NH graph + AADT_2023 + {cfg.mean_speed_factor:.2f} design speed",
            "External ME2 OD matrix",
            "Baseline OD shortest paths",
            "Remove one road link",
            "Measure added system travel time",
        ],
        "fig2_methodology_fftdi.png",
        "#1565C0",
    )
    methodology_figure(
        cfg,
        "Fig. 3 - Social: People and Healthcare Access",
        [
            "Road graph + conserved population",
            "Healthcare access baseline",
            "Remove one road link",
            "Measure isolation and longer healthcare trips",
            "Combine S1 and S2",
        ],
        "fig3_methodology_social.png",
        "#C62828",
    )
    methodology_figure(
        cfg,
        "Fig. 4 - Economic: Travel Time and Border Access",
        [
            "Road graph + GDP/AADT",
            "Border-trade access baseline",
            "Remove one road link",
            "E1 trade-exposed vehicle-hour loss",
            "E2 border-access loss",
        ],
        "fig4_methodology_economic.png",
        "#8A6A00",
    )
    methodology_figure(
        cfg,
        "Fig. 5 - Airport Access",
        [
            "Population origins",
            "Operational airport destinations",
            "Baseline road paths",
            "Remove one road link",
            "Measure longer or lost airport access",
        ],
        "fig5_methodology_interconnectedness.png",
        "#6A1B9A",
    )


# ------------------------------- orchestrator --------------------------------


def run_mlpi(
    input_path: str | Path = DEFAULT_INPUT,
    gpkg_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    od_matrix_path: str | Path | None = None,
    gen_maps: Optional[bool] = None,
) -> Dict[str, Any]:
    print("=" * 72)
    print("MLPI Robust - Nepal National Highway Resilience")
    print("=" * 72)
    cfg = load_config(input_path, gpkg_dir=gpkg_dir, out_dir=out_dir, od_matrix_path=od_matrix_path)
    if gen_maps is not None:
        cfg.gen_maps = bool(gen_maps)
    np.random.seed(cfg.seed)
    for stale in [
        "fig_social_vulnerability.png",
        "fig_economic_vulnerability.png",
        "fig_interconnectedness.png",
        "map2_hazard_eva.png",
        "map3_multimodal_airbridge.png",
        "fig4_aupc_strategies.png",
        "recovery_metrics.csv",
        "investment_effectiveness.csv",
        "zoom_west_top_links.png",
        "zoom_mid_top_links.png",
        "zoom_east_top_links.png",
        "fig2_methodology_fftdi.png",
        "fig3_methodology_social.png",
        "fig4_methodology_economic.png",
        "fig5_methodology_interconnectedness.png",
    ]:
        try:
            (cfg.out_dir / stale).unlink()
        except FileNotFoundError:
            pass
    print(f"Input workbook : {cfg.input_xlsx}")
    print(f"GeoPackage dir : {cfg.gpkg_dir}")
    print(f"ME2 OD matrix  : {cfg.od_matrix_path}")
    print(f"Output dir     : {cfg.out_dir}")

    print("[0] Loading GeoPackages")
    geo = load_geodata(cfg.gpkg_dir, cfg=cfg)
    G, sp, plot_roads = build_network(cfg, geo)
    population_audit = assign_socioeconomic_weights(cfg, G, geo)
    population_audit_path = cfg.out_dir / "population_node_allocation.csv"
    population_audit.to_csv(population_audit_path, index=False)

    print("[2] Building road links to test one at a time")
    sections = build_junction_links(cfg, G, sp)
    if len(sections) < 10:
        raise RuntimeError(f"Only {len(sections)} road links were built; at least 10 are required.")
    section_candidates_path = cfg.out_dir / "junction_links_used.csv"
    section_candidates_df = pd.DataFrame(sections)
    section_candidate_cols = [
        "link_id", "link_name", "from_node", "to_node", "from_junction", "to_junction",
        "origin_place", "destination_place", "macro_region", "ref", "name", "fclass",
        "oneway", "source_design_speed_kmh", "mean_speed_kmh", "design_speed_source", "model_speed_kmh", "model_speed_source", "bridge", "tunnel", "layer", "osm_ids",
        "origin_longitude", "origin_latitude", "destination_longitude", "destination_latitude",
        "origin_anchor_distance_km", "destination_anchor_distance_km",
        "length_km", "direct_km", "detour_ratio", "edge_count", "connector_edge_count", "mean_aadt",
        "aadt_source", "terrain_zone", "terrain_source", "terrain_district", "travel_speed_kmh",
        "surface", "surface_source",
    ]
    section_candidates_df[[c for c in section_candidate_cols if c in section_candidates_df.columns]].to_csv(section_candidates_path, index=False)
    print("  failure-unit source: topology derived from Nepal.gpkg layer NH")
    print(f"  junction-link audit written: {section_candidates_path}")

    print("[3] Computing four junction-link removal dimensions")
    physical, od_table = compute_section_physical(cfg, G, sections)
    od_path = cfg.out_dir / "me2_od_pairs_used.csv"
    od_table.to_csv(od_path, index=False)
    od_zone_mapping_path = cfg.out_dir / "me2_od_zone_mapping.csv"
    cfg.od_zone_mapping_audit.to_csv(od_zone_mapping_path, index=False)
    print(
        f"  physical OD-TSTT FFTDI complete | ME2 OD pairs: {len(od_table)} | "
        f"baseline connected: {cfg.od_baseline_connected_pairs} | written: {od_path}"
    )
    print(f"  ME2 OD zone mapping written: {od_zone_mapping_path}")
    social = compute_section_social(cfg, G, geo, sections)
    economic = compute_section_economic(cfg, G, sections, physical)
    interconnected = compute_section_interconnected(cfg, G, sections)
    ranked = compute_section_mlpi(cfg, sections, physical, social, economic, interconnected)
    ranked_path = cfg.out_dir / "mlpi_ranked_links.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"  ranked road links written: {ranked_path}")
    airport_assets = airport_assets_dataframe(G)
    airport_assets_path = cfg.out_dir / "airport_assets_used.csv"
    airport_assets.to_csv(airport_assets_path, index=False)
    print(f"  airport assets written: {airport_assets_path}")
    methodology_path = write_methodology_workbook(cfg, ranked)
    if methodology_path:
        print(f"  methodology formulas workbook written: {methodology_path}")

    if cfg.gen_maps:
        print("[5] Rendering maps")
        map1(cfg, G, geo, plot_roads)
        map_exact_dimension_sections(
            cfg,
            geo,
            sp,
            plot_roads,
            sections,
            ranked,
            "physical",
            "Map 2 - Top 10 Roads Causing Travel Delay",
            "map2_physical_fftdi_top10.png",
            "#1565C0",
        )
        map_exact_dimension_sections(cfg, geo, sp, plot_roads, sections, ranked, "social", "Map 3 - Top 10 Roads Affecting People", "map3_social_top10.png", "#C62828")
        map_exact_dimension_sections(cfg, geo, sp, plot_roads, sections, ranked, "economic", "Map 4 - Top 10 Roads Affecting Trade and Access", "map4_economic_top10.png", "#8A6A00")
        map_exact_interconnected_sections(cfg, G, geo, sp, plot_roads, sections, ranked, top_n=10)
        map_exact_combined_sections(cfg, geo, sp, plot_roads, sections, ranked, top_n=min(10, len(ranked)))
    print("\nTop ranked road links")
    cols = ["rank", "section_id", "section_name", "mlpi", "physical", "social", "economic", "interconnected", "length_km", "mean_aadt", "target_airport"]
    print(ranked[cols].head(cfg.top_n).to_string(index=False))
    print("\nOutputs:")
    output_names = [
        "mlpi_ranked_links.csv", "junction_links_used.csv", "population_node_allocation.csv", "me2_od_pairs_used.csv", "me2_od_zone_mapping.csv", "airport_assets_used.csv", "mlpi_methodology_formulas.xlsx",
    ]
    if cfg.gen_maps:
        output_names = [
            "map1_roads_districts.png", "map1_roads_districts.pdf", "map2_physical_fftdi_top10.png", "map3_social_top10.png",
            "map4_economic_top10.png", "map5_interconnected_top10.png", "map6_combined_mlpi_top10.png",
            "map6_combined_mlpi_top10.pdf",
        ] + output_names
    for name in output_names:
        print(f"  {cfg.out_dir / name}")

    return dict(
        config=cfg,
        geodata=geo,
        G=G,
        sp=sp,
        plot_roads=plot_roads,
        sections=sections,
        physical=physical,
        social=social,
        economic=economic,
        interconnected=interconnected,
        ranked=ranked,
        airport_assets=airport_assets,
        methodology_path=methodology_path,
        section_candidates_path=section_candidates_path,
        population_audit_path=population_audit_path,
        od_zone_mapping_path=od_zone_mapping_path,
        od_table=od_table,
    )


if __name__ == "__main__":
    RESULTS = run_mlpi()

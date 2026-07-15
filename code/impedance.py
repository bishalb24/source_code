#!/usr/bin/env python3
"""
impedance.py
============================

Constructs a district-to-district travel-time impedance matrix for Nepal
from the master Nepal GeoPackage, using its official-link national-highway
layer and district-headquarter area layer.

This script is designed for academic (thesis-grade) research use. It
emphasizes correct network topology, reproducibility, computational
efficiency, and transparent methodology. All configurable parameters are
centralized in the ``CONFIG`` dictionary below; no other constants are
scattered through the code.

Run with:

    python impedance.py

Only the file paths in the CONFIG section should require editing.

Author: Generated for master's thesis research use.
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import (
    LineString,
    MultiLineString,
    Point,
    base,
)
from shapely.ops import split as shapely_split
from shapely.ops import nearest_points, substring, unary_union
from shapely.validation import make_valid
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================
# CONFIGURATION
# ======================================================================

CONFIG: Dict = {
    # ---- master input ----
    "master_gpkg_path": "/Users/bishalbhurtel/Desktop/Project/input/Nepal.gpkg",
    "road_layer": "NH",
    "district_hq_layer": "district_hq",

    # ---- output directory ----
    "output_dir": "/Users/bishalbhurtel/Desktop/Project/output/impedance",

    # ---- coordinate reference system ----
    "projection": "EPSG:32645",

    # ---- topology parameters ----
    "snap_tolerance": 20.0,          # metres, endpoint snapping
    "coord_precision": 3,            # decimal places for node-key rounding

    # ---- HQ insertion parameters ----
    "hq_warning_distance": 100.0,    # metres, snap-distance QA threshold

    # ---- network attributes ----
    "aadt_field": "AADT_2023",
    "aadt_units": "vehicles/day",
    "design_speed_fields": ["design_speed_kmh", "design_speed", "Design Speed", "designspeed"],
    "mean_speed_factor": 0.40,
    "write_speed_audit": False,
    "write_processed_network": False,
    "write_graph_nodes": False,
    "write_methodology_md": False,

    # ---- reproducibility ----
    "random_seed": 42,
}

# ======================================================================
# LOGGING
# ======================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to both console and RunLog.txt.

    Parameters
    ----------
    output_dir : Path
        Directory in which RunLog.txt will be written.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "RunLog.txt"

    logger = logging.getLogger("travel_time_matrix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


# ======================================================================
# DATA CLASSES
# ======================================================================


@dataclass
class TopologyReport:
    """Container for topology-construction diagnostics."""

    n_nodes: int = 0
    n_edges: int = 0
    n_components: int = 0
    largest_component_size: int = 0
    n_isolated_nodes: int = 0
    n_removed_edges: int = 0
    n_split_edges: int = 0


# ======================================================================
# CRS HANDLING
# ======================================================================


def load_and_reproject(
    path: str,
    target_crs: str,
    logger: logging.Logger,
    label: str,
    layer: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load a vector layer and reproject it to the target CRS.

    Parameters
    ----------
    path : str
        Path to the input vector file (any format supported by pyogrio).
    target_crs : str
        Target CRS as an EPSG string (e.g. "EPSG:32645").
    logger : logging.Logger
        Logger for reporting original/projected CRS.
    label : str
        Human-readable label for logging purposes.

    Returns
    -------
    geopandas.GeoDataFrame
        The reprojected GeoDataFrame.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the layer has no CRS and cannot be safely assumed.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label}: input file not found at '{path}'")

    gdf = gpd.read_file(file_path, layer=layer, engine="pyogrio")
    original_crs = gdf.crs

    if original_crs is None:
        raise ValueError(
            f"{label}: layer has no CRS defined. Please assign a CRS "
            f"before running this script (cannot safely assume one)."
        )

    layer_note = f", layer={layer}" if layer else ""
    logger.info(f"[{label}] Source: {file_path}{layer_note}")
    logger.info(f"[{label}] Original CRS: {original_crs}")

    if str(original_crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
        logger.info(f"[{label}] Reprojected to: {target_crs}")
    else:
        logger.info(f"[{label}] Already in target CRS: {target_crs}")

    return gdf


# ======================================================================
# GEOMETRY VALIDATION
# ======================================================================


def validate_and_clean_geometries(
    gdf: gpd.GeoDataFrame, logger: logging.Logger, label: str
) -> gpd.GeoDataFrame:
    """Repair invalid geometries, remove empties/duplicates, explode multiparts.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input layer.
    logger : logging.Logger
        Logger for reporting removed/repaired feature counts.
    label : str
        Human-readable label for logging.

    Returns
    -------
    geopandas.GeoDataFrame
        Cleaned, exploded, deduplicated GeoDataFrame with a fresh index.
    """
    n_start = len(gdf)

    # Remove empty / missing geometries
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    n_after_empty = len(gdf)

    # Repair invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid > 0:
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(
            make_valid
        )
        logger.info(f"[{label}] Repaired {n_invalid} invalid geometries.")

    # Explode multipart geometries
    n_before_explode = len(gdf)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    n_after_explode = len(gdf)
    if n_after_explode != n_before_explode:
        logger.info(
            f"[{label}] Exploded multipart geometries: "
            f"{n_before_explode} -> {n_after_explode} features."
        )

    # Remove duplicate geometries (by WKB)
    n_before_dedupe = len(gdf)
    gdf["_wkb"] = gdf.geometry.apply(lambda g: g.wkb)
    gdf = gdf.drop_duplicates(subset="_wkb").drop(columns="_wkb").reset_index(
        drop=True
    )
    n_after_dedupe = len(gdf)
    if n_after_dedupe != n_before_dedupe:
        logger.info(
            f"[{label}] Removed {n_before_dedupe - n_after_dedupe} duplicate "
            f"geometries."
        )

    n_removed_total = n_start - len(gdf)
    logger.info(
        f"[{label}] Geometry validation complete. "
        f"Start: {n_start}, Final: {len(gdf)}, Removed: {n_removed_total} "
        f"(empty: {n_start - n_after_empty}, invalid repaired: {n_invalid})."
    )

    return gdf


def attach_hq_area_labels(
    hq_areas: gpd.GeoDataFrame,
    label_points: gpd.GeoDataFrame,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Attach district-HQ labels to area polygons using nearest label points."""
    required = {"VDC_NAME", "DIST_NAME", "REGION"}
    if required.issubset(hq_areas.columns):
        return hq_areas

    missing = required - set(label_points.columns)
    if missing:
        raise ValueError(
            "HQ area layer has no labels, and the label-point layer is missing "
            f"required fields: {sorted(missing)}"
        )
    if hq_areas.crs != label_points.crs:
        label_points = label_points.to_crs(hq_areas.crs)

    areas = hq_areas.reset_index(drop=True).copy()
    points = label_points.reset_index(drop=True).copy()
    candidate_lists: Dict[int, List[Tuple[float, int]]] = {}
    for point_idx, point_row in points.iterrows():
        distances = areas.geometry.distance(point_row.geometry)
        candidate_lists[int(point_idx)] = sorted(
            (float(distance), int(area_idx))
            for area_idx, distance in distances.items()
        )

    assigned_area_ids: Set[int] = set()
    assigned_rows: List[pd.Series] = []
    point_order = sorted(
        range(len(points)),
        key=lambda idx: candidate_lists[idx][0][0]
        if candidate_lists[idx]
        else float("inf"),
    )
    for point_idx in point_order:
        candidates = candidate_lists[point_idx]
        if not candidates:
            continue
        distance, area_idx = next(
            (
                (distance, area_idx)
                for distance, area_idx in candidates
                if area_idx not in assigned_area_ids
            ),
            candidates[0],
        )
        assigned_area_ids.add(area_idx)
        area_row = areas.loc[area_idx].copy()
        point_row = points.loc[point_idx]
        for col in ["VDC_NAME", "ZONE_NAME", "REGION", "DIST_NAME"]:
            if col in point_row:
                area_row[col] = point_row[col]
        area_row["hq_area_id"] = area_idx
        area_row["label_point_distance_m"] = distance
        area_row["label_source"] = "nearest District_hq.gpkg point"
        assigned_rows.append(area_row)

    labelled = gpd.GeoDataFrame(
        assigned_rows, geometry="geometry", crs=hq_areas.crs
    )
    if len(labelled) != len(label_points):
        raise ValueError(
            f"Expected {len(label_points)} labelled HQ areas, built {len(labelled)}."
        )
    logger.info(
        "Attached labels to %d HQ area polygons from District_hq.gpkg points. "
        "Median label-to-area distance: %.2f m; max: %.2f m.",
        len(labelled),
        float(labelled["label_point_distance_m"].median()),
        float(labelled["label_point_distance_m"].max()),
    )
    return labelled.reset_index(drop=True)


def _canonical_column_map(gdf: gpd.GeoDataFrame) -> Dict[str, str]:
    return {
        re.sub(r"[^a-z0-9]+", "", str(col).lower()): col
        for col in gdf.columns
        if col != "geometry"
    }


def _first_existing_column(gdf: gpd.GeoDataFrame, candidates: Sequence[str]) -> Optional[str]:
    canon = _canonical_column_map(gdf)
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in canon:
            return canon[key]
    return None


def _first_value(row: Any, candidates: Sequence[str], default: Any = None) -> Any:
    for candidate in candidates:
        if candidate in row.index:
            value = row.get(candidate)
            if pd.notna(value) and str(value).strip() != "":
                return value
    return default


def _positive_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).lower()
    text = text.replace("km/h", "").replace("kph", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        out = float(match.group(0))
    except ValueError:
        return None
    return out if out > 0 else None


def _first_positive_numeric(row: Any, candidates: Sequence[str]) -> Tuple[Optional[float], str]:
    for candidate in candidates:
        if candidate not in row.index:
            continue
        value = _positive_float(row.get(candidate))
        if value is not None:
            return value, candidate
    return None, ""


def standardize_nepal_nh_attributes(
    roads: gpd.GeoDataFrame,
    config: Dict,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Normalize the Nepal.gpkg NH schema to the fields used downstream."""
    roads = roads.copy()
    link_code_col = _first_existing_column(roads, ["link_code", "code", "linkcode"])
    road_ref_col = _first_existing_column(roads, ["road_refno", "road_ref", "ref", "route"])
    link_name_col = _first_existing_column(roads, ["link_name", "road_name", "name"])
    road_name_col = _first_existing_column(roads, ["road_name", "name", "link_name"])
    road_class_col = _first_existing_column(roads, ["road_class", "rd_class", "fclass"])

    if link_code_col and "link_code" not in roads.columns:
        roads["link_code"] = roads[link_code_col]
    if link_code_col and "code" not in roads.columns:
        roads["code"] = roads[link_code_col]
    if road_ref_col and "road_refno" not in roads.columns:
        roads["road_refno"] = roads[road_ref_col]
    if road_ref_col and "ref" not in roads.columns:
        roads["ref"] = roads[road_ref_col]
    if link_name_col and "link_name" not in roads.columns:
        roads["link_name"] = roads[link_name_col]
    if road_name_col and "road_name" not in roads.columns:
        roads["road_name"] = roads[road_name_col]
    if link_name_col and "name" not in roads.columns:
        roads["name"] = roads[link_name_col]
    if road_class_col and "road_class" not in roads.columns:
        roads["road_class"] = roads[road_class_col]

    aadt_field = str(config["aadt_field"])
    if aadt_field not in roads.columns:
        raise ValueError(
            f"Nepal.gpkg layer {config['road_layer']} is missing required "
            f"AADT field '{aadt_field}'. Available fields: {sorted(roads.columns)}"
        )

    design_speed_col = _first_existing_column(roads, config["design_speed_fields"])
    if design_speed_col is None:
        raise ValueError(
            "Nepal.gpkg NH is missing the required design-speed field. "
            f"Tried {config['design_speed_fields']}. Available fields: {sorted(roads.columns)}"
        )
    roads["design_speed_kmh"] = roads[design_speed_col].map(_positive_float)
    roads["design_speed_source_field"] = design_speed_col

    if roads["design_speed_kmh"].notna().sum() == 0:
        raise ValueError(
            f"Nepal.gpkg NH field '{design_speed_col}' has no usable positive "
            "design-speed values."
        )

    logger.info(
        "[Roads] Using Nepal.gpkg NH with AADT field '%s' and mean speed = %.2f x design speed from '%s'.",
        aadt_field,
        float(config["mean_speed_factor"]),
        design_speed_col,
    )
    return roads


def standardize_hq_area_labels(
    hqs: gpd.GeoDataFrame,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    """Make district-HQ area labels explicit without treating areas as points."""
    hqs = hqs.copy()
    district_col = _first_existing_column(
        hqs,
        ["DIST_NAME", "district_name", "district", "dist_name", "DNAME"],
    )
    hq_col = _first_existing_column(
        hqs,
        ["VDC_NAME", "hq_name", "district_hq", "headquarter", "name", "municipality"],
    )
    region_col = _first_existing_column(hqs, ["REGION", "province", "prov_name", "zone_name"])

    if district_col is None:
        raise ValueError(
            "Nepal.gpkg layer district_hq must contain a district-name field. "
            f"Available fields: {sorted(hqs.columns)}"
        )

    hqs["DIST_NAME"] = hqs[district_col].astype(str).str.strip()
    if hq_col is not None:
        hqs["VDC_NAME"] = hqs[hq_col].astype(str).str.strip()
    else:
        hqs["VDC_NAME"] = hqs["DIST_NAME"]
    hqs["REGION"] = hqs[region_col].astype(str).str.strip() if region_col else ""
    hqs = hqs[["VDC_NAME", "DIST_NAME", "REGION", "geometry"]].copy()

    before = len(hqs)
    hqs = hqs[hqs["DIST_NAME"].astype(str).str.len() > 0].copy()
    hqs["_geom_wkb"] = hqs.geometry.apply(lambda geom: geom.wkb if geom is not None else b"")
    hqs = (
        hqs.drop_duplicates(subset=["DIST_NAME", "VDC_NAME", "_geom_wkb"])
        .drop(columns="_geom_wkb")
        .reset_index(drop=True)
    )
    if len(hqs) != before:
        logger.info("[District HQs] Removed %d blank/duplicate HQ areas.", before - len(hqs))
    logger.info("[District HQs] Prepared %d labelled HQ area geometries.", len(hqs))
    return hqs


# ======================================================================
# TOPOLOGY CONSTRUCTION
# ======================================================================


def _round_coord(
    coord: Tuple[float, float], precision: int
) -> Tuple[float, float]:
    """Round a coordinate tuple for use as a graph node key."""
    return (round(coord[0], precision), round(coord[1], precision))


def snap_endpoints(
    lines: List[LineString], tolerance: float, precision: int
) -> List[LineString]:
    """Snap nearby line endpoints together using a KDTree.

    Endpoints within ``tolerance`` metres of each other are merged to the
    same coordinate (the coordinate of the first point in each cluster),
    reducing spurious near-miss disconnections in the network.

    Parameters
    ----------
    lines : list of shapely.geometry.LineString
        Line geometries to process.
    tolerance : float
        Snapping distance in metres.
    precision : int
        Decimal precision for coordinate rounding after snapping.

    Returns
    -------
    list of shapely.geometry.LineString
        Lines with snapped endpoints.
    """
    endpoints = []
    endpoint_index = []  # (line_idx, 'start'/'end')

    for i, line in enumerate(lines):
        coords = list(line.coords)
        endpoints.append(coords[0])
        endpoint_index.append((i, "start"))
        endpoints.append(coords[-1])
        endpoint_index.append((i, "end"))

    if not endpoints:
        return lines

    pts = np.array(endpoints)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=tolerance)

    # Union-find to cluster nearby endpoints
    parent = list(range(len(pts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    cluster_repr: Dict[int, Tuple[float, float]] = {}
    for idx in range(len(pts)):
        root = find(idx)
        if root not in cluster_repr:
            cluster_repr[root] = tuple(pts[root])

    new_lines = list(lines)
    for idx, (line_idx, which) in enumerate(endpoint_index):
        root = find(idx)
        snapped_coord = _round_coord(cluster_repr[root], precision)
        coords = list(new_lines[line_idx].coords)
        if which == "start":
            coords[0] = snapped_coord
        else:
            coords[-1] = snapped_coord
        if len(coords) >= 2:
            new_lines[line_idx] = LineString(coords)

    return new_lines


def split_lines_at_intersections(
    gdf: gpd.GeoDataFrame, logger: logging.Logger
) -> Tuple[gpd.GeoDataFrame, int]:
    """Split road segments at true geometric intersections.

    Uses an STRtree for candidate-pair generation, then computes exact
    intersections between candidate line pairs. Intersections that occur
    only at pre-existing shared endpoints are ignored (they already form a
    valid junction with no splitting required). Crossings caused only by
    bridges/tunnels passing over/under another road (i.e. no true junction)
    are approximated by requiring the intersection geometry to be a single
    point strictly interior to at least one of the two lines.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Road network with LineString geometries (already exploded/cleaned).
    logger : logging.Logger
        Logger for progress/diagnostics.

    Returns
    -------
    tuple(geopandas.GeoDataFrame, int)
        Road network with edges split at all valid intersection points,
        with attributes preserved on both resulting halves, and the count
        of original edges that were split.
    """
    lines = list(gdf.geometry.values)
    attrs = gdf.drop(columns="geometry").to_dict("records")

    tree = gpd.GeoSeries(lines).sindex

    # Determine, for each line, the set of interior split points.
    split_points: List[List[Point]] = [[] for _ in lines]
    n_split_events = 0

    for i, line in enumerate(tqdm(lines, desc="Detecting intersections")):
        candidate_idx = list(tree.query(line, predicate="intersects"))
        endpoints_i = {line.coords[0], line.coords[-1]}

        for j in candidate_idx:
            if j <= i:
                continue
            other = lines[j]
            inter = line.intersection(other)

            if inter.is_empty:
                continue

            endpoints_j = {other.coords[0], other.coords[-1]}

            candidate_points = []
            if inter.geom_type == "Point":
                candidate_points = [inter]
            elif inter.geom_type == "MultiPoint":
                candidate_points = list(inter.geoms)
            else:
                # Overlapping / collinear segments: skip, not a simple
                # junction that should trigger a split.
                continue

            for pt in candidate_points:
                coord = (pt.x, pt.y)
                # Skip if this intersection point is merely a pre-existing
                # shared endpoint of both lines (already a valid junction).
                if coord in endpoints_i and coord in endpoints_j:
                    continue
                split_points[i].append(pt)
                split_points[j].append(pt)
                n_split_events += 1

    # Perform the actual splitting
    new_geoms: List[LineString] = []
    new_attrs: List[Dict] = []
    n_split_edges = 0

    for i, line in enumerate(tqdm(lines, desc="Splitting edges")):
        pts = split_points[i]
        if not pts:
            new_geoms.append(line)
            new_attrs.append(attrs[i])
            continue

        # Filter out points that coincide with the line's own endpoints
        interior_pts = [
            p
            for p in pts
            if Point(line.coords[0]).distance(p) > 1e-6
            and Point(line.coords[-1]).distance(p) > 1e-6
        ]
        if not interior_pts:
            new_geoms.append(line)
            new_attrs.append(attrs[i])
            continue

        splitter = unary_union(interior_pts)
        try:
            result = shapely_split(line, splitter)
            pieces = [g for g in result.geoms if g.geom_type == "LineString"]
        except Exception:
            pieces = [line]

        if len(pieces) <= 1:
            new_geoms.append(line)
            new_attrs.append(attrs[i])
        else:
            n_split_edges += 1
            for piece in pieces:
                if piece.length > 0:
                    new_geoms.append(piece)
                    new_attrs.append(attrs[i])

    result_gdf = gpd.GeoDataFrame(new_attrs, geometry=new_geoms, crs=gdf.crs)
    logger.info(
        f"Intersection splitting complete: {n_split_edges} original edges "
        f"split into {len(result_gdf)} total edges "
        f"({n_split_events} intersection events detected)."
    )
    return result_gdf, n_split_edges


def build_road_graph(
    gdf: gpd.GeoDataFrame,
    config: Dict,
    logger: logging.Logger,
) -> Tuple[nx.DiGraph, TopologyReport]:
    """Construct a directed, topologically-correct road graph.

    Applies endpoint snapping, zero-length/duplicate edge removal, oneway
    semantics, and largest-connected-component filtering.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Split, cleaned road network.
    config : dict
        Configuration dictionary (uses "snap_tolerance", "coord_precision").
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    tuple(networkx.DiGraph, TopologyReport)
        The constructed graph (largest component only) and a diagnostics
        report.
    """
    tolerance = config["snap_tolerance"]
    precision = config["coord_precision"]

    lines = list(gdf.geometry.values)
    snapped_lines = snap_endpoints(lines, tolerance, precision)

    gdf = gdf.copy()
    gdf["geometry"] = snapped_lines

    # Remove zero-length edges
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.length > 1e-6].reset_index(drop=True)
    n_zero_length_removed = n_before - len(gdf)

    # Remove duplicate edges (same endpoints + near-identical length)
    def edge_key(line: LineString) -> Tuple:
        c0 = _round_coord(line.coords[0], precision)
        c1 = _round_coord(line.coords[-1], precision)
        key = tuple(sorted([c0, c1]))
        return (key, round(line.length, 1))

    gdf["_edge_key"] = gdf.geometry.apply(edge_key)
    n_before_dupe = len(gdf)
    gdf = gdf.drop_duplicates(subset="_edge_key").drop(columns="_edge_key")
    gdf = gdf.reset_index(drop=True)
    n_duplicate_edges_removed = n_before_dupe - len(gdf)

    n_removed_edges = n_zero_length_removed + n_duplicate_edges_removed

    # Build directed graph respecting oneway
    G = nx.DiGraph()
    oneway_col = "oneway" if "oneway" in gdf.columns else None

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Building graph"):
        line = row.geometry
        c_start = _round_coord(line.coords[0], precision)
        c_end = _round_coord(line.coords[-1], precision)

        if c_start == c_end:
            continue  # degenerate loop edge, skip

        length_m = line.length
        design_speed_value, design_speed_field = _first_positive_numeric(
            row, ["design_speed_kmh"]
        )
        aadt_value, aadt_field = _first_positive_numeric(row, [str(config["aadt_field"])])

        edge_attrs = {
            "length": length_m,
            "geometry": line,
            "osm_id": row.get("osm_id", row.get("link_code", None)),
            "name": row.get("name", row.get("link_name", row.get("road_name", None))),
            "ref": row.get("ref", row.get("road_refno", None)),
            "road_class": row.get("road_class", row.get("fclass", row.get("rd_class", None))),
            "code": row.get("code", row.get("link_code", None)),
            "bridge": row.get("bridge", None),
            "tunnel": row.get("tunnel", None),
            "design_speed_kmh": design_speed_value,
            "design_speed_source_field": row.get("design_speed_source_field", design_speed_field),
            "source_terrain_code": row.get("terrain", None),
            "road_name": row.get("road_name", None),
            "link_code": row.get("link_code", None),
            "link_name": row.get("link_name", None),
            "link_from": row.get("link_from", None),
            "link_to": row.get("link_to", None),
            "link_len": row.get("link_len", None),
            "div_name": row.get("div_name", None),
            "dist_name": row.get("dist_name", None),
            "aadt": aadt_value,
            "aadt_pcu_per_day": aadt_value,
            "aadt_source_field": aadt_field,
            "aadt_source": f"Nepal.gpkg:{config['road_layer']}:{config['aadt_field']}",
            "aadt_units": str(config["aadt_units"]),
            "source_gpkg": Path(config["master_gpkg_path"]).name,
            "source_layer": config["road_layer"],
        }

        oneway_val = str(row.get(oneway_col)).strip(
        ).lower() if oneway_col else "no"

        G.add_node(c_start, x=c_start[0], y=c_start[1])
        G.add_node(c_end, x=c_end[0], y=c_end[1])

        if oneway_val in ("yes", "true", "1", "t", "f"):
            # "F" in OSM oneway sometimes denotes forward-only; treat as forward
            G.add_edge(c_start, c_end, **edge_attrs)
        elif oneway_val in ("-1", "reverse", "r"):
            reversed_line = LineString(list(line.coords)[::-1])
            rev_attrs = dict(edge_attrs)
            rev_attrs["geometry"] = reversed_line
            G.add_edge(c_end, c_start, **rev_attrs)
        else:
            # Missing or "no": assume bidirectional
            G.add_edge(c_start, c_end, **edge_attrs)
            reversed_line = LineString(list(line.coords)[::-1])
            rev_attrs = dict(edge_attrs)
            rev_attrs["geometry"] = reversed_line
            G.add_edge(c_end, c_start, **rev_attrs)

    n_nodes_all = G.number_of_nodes()
    n_edges_all = G.number_of_edges()

    # Connected components (weakly connected, since directionality applies)
    components = list(nx.weakly_connected_components(G))
    n_components = len(components)
    largest_cc = max(components, key=len) if components else set()
    isolated_nodes = sum(1 for c in components if len(c) == 1)

    G_largest = G.subgraph(largest_cc).copy()

    report = TopologyReport(
        n_nodes=G_largest.number_of_nodes(),
        n_edges=G_largest.number_of_edges(),
        n_components=n_components,
        largest_component_size=len(largest_cc),
        n_isolated_nodes=isolated_nodes,
        n_removed_edges=n_removed_edges,
        n_split_edges=0,  # filled in by caller after split step
    )

    logger.info(
        f"Graph built (pre-filter): {n_nodes_all} nodes, {n_edges_all} edges."
    )
    logger.info(
        f"Connected components: {n_components}. Largest component: "
        f"{len(largest_cc)} nodes retained. Isolated nodes: {isolated_nodes}."
    )
    logger.info(
        f"Final graph (largest component only): {report.n_nodes} nodes, "
        f"{report.n_edges} edges. Removed edges (zero-length/duplicate): "
        f"{n_removed_edges}."
    )

    return G_largest, report


# ======================================================================
# DISTRICT HQ INSERTION
# ======================================================================


def insert_hqs_into_graph(
    G: nx.DiGraph,
    hq_gdf: gpd.GeoDataFrame,
    config: Dict,
    logger: logging.Logger,
) -> Tuple[nx.DiGraph, gpd.GeoDataFrame, pd.DataFrame]:
    """Insert district HQ connectors into the graph by orthogonal projection.

    Each HQ geometry may be a point or an area. The connector point is the
    nearest point on the traversable road edge to that geometry. If an HQ area
    touches or intersects the road graph, the snap distance is zero. A new node
    is created at the projection point and the original edge is split into two
    edges preserving all attributes.

    Parameters
    ----------
    G : networkx.DiGraph
        Road network graph (largest connected component).
    hq_gdf : geopandas.GeoDataFrame
        District headquarters geometries with VDC_NAME, DIST_NAME, REGION.
    config : dict
        Configuration dictionary (uses "hq_warning_distance",
        "coord_precision").
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    tuple(networkx.DiGraph, geopandas.GeoDataFrame, pandas.DataFrame)
        Updated graph with HQ nodes inserted, a GeoDataFrame of projected
        HQ points (with snap distances and node keys), and a snap-quality
        report DataFrame.
    """
    precision = config["coord_precision"]
    warn_dist = config["hq_warning_distance"]

    projected_points = []
    snap_distances = []
    hq_labels = []
    node_keys = []

    for idx, row in tqdm(
        hq_gdf.iterrows(), total=len(hq_gdf), desc="Projecting HQs onto network"
    ):
        hq_geom = row.geometry
        # Rebuild from the current graph because an earlier HQ may have split
        # the same long official highway link.
        edge_records = [
            (u, v, data) for u, v, data in G.edges(data=True)
        ]
        edge_series = gpd.GeoSeries(
            [data["geometry"] for _, _, data in edge_records],
            crs=hq_gdf.crs,
        )
        edge_sindex = edge_series.sindex
        nearest_idx = list(edge_sindex.nearest(hq_geom, return_all=False))
        # sindex.nearest returns array-like; normalize to a single int index
        if isinstance(nearest_idx, tuple):
            nearest_idx = nearest_idx[1]
        edge_idx = int(np.ravel(nearest_idx)[-1])

        u, v, data = edge_records[edge_idx]
        line = data["geometry"]

        _, road_nearest_point = nearest_points(hq_geom, line)
        proj_dist_along = line.project(road_nearest_point)
        proj_point = line.interpolate(proj_dist_along)
        snap_dist = hq_geom.distance(proj_point)

        new_node = _round_coord((proj_point.x, proj_point.y), precision)

        if new_node != u and new_node != v and 0 < proj_dist_along < line.length:
            part_a = substring(line, 0, proj_dist_along)
            part_b = substring(line, proj_dist_along, line.length)

            if part_a.length > 0 and part_b.length > 0:
                G.add_node(new_node, x=new_node[0], y=new_node[1])

                attrs_a = dict(data)
                attrs_a["geometry"] = part_a
                attrs_a["length"] = part_a.length

                attrs_b = dict(data)
                attrs_b["geometry"] = part_b
                attrs_b["length"] = part_b.length

                G.add_edge(u, new_node, **attrs_a)
                G.add_edge(new_node, v, **attrs_b)

                # If the reverse edge exists (bidirectional road), split it too
                if G.has_edge(v, u):
                    rev_data = G.get_edge_data(v, u)
                    rev_line = rev_data["geometry"]
                    rev_proj_dist = rev_line.project(proj_point)
                    rpart_a = substring(rev_line, 0, rev_proj_dist)
                    rpart_b = substring(
                        rev_line, rev_proj_dist, rev_line.length)
                    if rpart_a.length > 0 and rpart_b.length > 0:
                        rattrs_a = dict(rev_data)
                        rattrs_a["geometry"] = rpart_a
                        rattrs_a["length"] = rpart_a.length
                        rattrs_b = dict(rev_data)
                        rattrs_b["geometry"] = rpart_b
                        rattrs_b["length"] = rpart_b.length
                        G.remove_edge(v, u)
                        G.add_edge(v, new_node, **rattrs_a)
                        G.add_edge(new_node, u, **rattrs_b)

                G.remove_edge(u, v)
        else:
            # Projection coincides with an existing node
            new_node = new_node if new_node in G.nodes else (
                u if hq_geom.distance(Point(u)) < hq_geom.distance(Point(v)) else v
            )

        projected_points.append(proj_point)
        snap_distances.append(snap_dist)
        hq_labels.append(f"{row['DIST_NAME']} | {row['VDC_NAME']}")
        node_keys.append(new_node)

    hq_result = hq_gdf.copy()
    hq_result["projected_geometry"] = projected_points
    hq_result["snap_distance_m"] = snap_distances
    hq_result["hq_label"] = hq_labels
    hq_result["node_key"] = node_keys
    hq_result["hq_source_geom_type"] = hq_gdf.geometry.geom_type.astype(str)

    projected_gdf = gpd.GeoDataFrame(
        hq_result.drop(columns="geometry"),
        geometry="projected_geometry",
        crs=hq_gdf.crs,
    ).rename_geometry("geometry")

    max_snap = float(np.max(snap_distances)) if snap_distances else 0.0
    avg_snap = float(np.mean(snap_distances)) if snap_distances else 0.0
    n_exceeding = int(sum(d > warn_dist for d in snap_distances))

    logger.info(
        f"HQ insertion complete. Max snap distance: {max_snap:.2f} m, "
        f"Average snap distance: {avg_snap:.2f} m. "
        f"HQs exceeding {warn_dist} m: {n_exceeding}."
    )
    if n_exceeding > 0:
        bad = hq_result.loc[
            pd.Series(snap_distances) > warn_dist, "hq_label"
        ].tolist()
        logger.warning(
            f"The following HQs exceed the snap-distance warning threshold: "
            f"{bad}"
        )

    snap_report = pd.DataFrame(
        {
            "hq_label": hq_labels,
            "snap_distance_m": snap_distances,
            "exceeds_threshold": [d > warn_dist for d in snap_distances],
        }
    )

    return G, projected_gdf, snap_report


# ======================================================================
# TERRAIN CLASSIFICATION
# ======================================================================


def classify_terrain(elevation: float, breaks: Dict[str, Tuple[float, float]]) -> str:
    """Classify a single elevation value into a terrain category.

    Parameters
    ----------
    elevation : float
        Elevation in metres.
    breaks : dict
        Mapping of terrain name -> (lower_bound, upper_bound) in metres.

    Returns
    -------
    str
        The matching terrain category name.
    """
    for terrain, (lo, hi) in breaks.items():
        if lo <= elevation < hi:
            return terrain
    return list(breaks.keys())[-1]


def assign_terrain_to_edges(
    G: nx.DiGraph,
    elevation_gdf: gpd.GeoDataFrame,
    config: Dict,
    logger: logging.Logger,
) -> nx.DiGraph:
    """Assign terrain classification to every graph edge using elevation data.

    Methodology: for each road edge, contour lines within a search buffer
    are identified via spatial index; the mean of their CEL attribute is
    used as the estimated elevation for that edge. If no contours are
    found within the buffer, nearest-neighbour interpolation (nearest
    contour centroid) is used instead. If elevation data is entirely
    unavailable for an edge, a configurable default terrain is applied
    and logged.

    Parameters
    ----------
    G : networkx.DiGraph
        Road graph with HQ nodes already inserted.
    elevation_gdf : geopandas.GeoDataFrame
        Contour lines (MultiLineString/LineString) with a "CEL" attribute.
    config : dict
        Configuration dictionary (uses "elevation_search_buffer",
        "terrain_elevation_breaks", "default_terrain").
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    networkx.DiGraph
        Graph with "elevation_m" and "terrain" edge attributes populated.
    """
    buffer_dist = config["elevation_search_buffer"]
    breaks = config["terrain_elevation_breaks"]
    default_terrain = config["default_terrain"]

    elevation_gdf = elevation_gdf.explode(
        index_parts=False).reset_index(drop=True)
    elev_sindex = elevation_gdf.sindex
    elev_centroids = elevation_gdf.geometry.centroid
    centroid_coords = np.array(
        [[p.x, p.y] for p in elev_centroids]
    ) if len(elevation_gdf) > 0 else np.empty((0, 2))
    centroid_tree = cKDTree(centroid_coords) if len(
        centroid_coords) > 0 else None

    terrain_code_map = {
        "P": "Plain",
        "R": "Rolling",
        "M": "Mountainous",
        "S": "Steep",
    }
    n_source = 0
    n_direct = 0
    n_nearest = 0
    n_default = 0

    for u, v, data in tqdm(G.edges(data=True), desc="Assigning terrain"):
        line = data["geometry"]
        source_code = str(data.get("source_terrain_code", "")).strip().upper()

        if source_code in terrain_code_map:
            data["elevation_m"] = np.nan
            data["terrain"] = terrain_code_map[source_code]
            data["terrain_source"] = "source_NH_layer"
            n_source += 1
            continue

        if len(elevation_gdf) == 0:
            elevation_val = None
        else:
            buffered = line.buffer(buffer_dist)
            candidate_idx = list(elev_sindex.query(
                buffered, predicate="intersects"))

            if candidate_idx:
                cel_values = elevation_gdf.iloc[candidate_idx]["CEL"].dropna()
                if len(cel_values) > 0:
                    elevation_val = float(cel_values.mean())
                    n_direct += 1
                else:
                    elevation_val = None
            else:
                elevation_val = None

            if elevation_val is None and centroid_tree is not None:
                mid = line.interpolate(0.5, normalized=True)
                dist, nearest_i = centroid_tree.query([mid.x, mid.y])
                elevation_val = float(
                    elevation_gdf.iloc[int(nearest_i)]["CEL"])
                n_nearest += 1

        if elevation_val is None:
            terrain = default_terrain
            elevation_val = np.nan
            n_default += 1
        else:
            terrain = classify_terrain(elevation_val, breaks)

        data["elevation_m"] = elevation_val
        data["terrain"] = terrain
        data["terrain_source"] = "contour_default"

    logger.info(
        f"Terrain assignment complete. Source terrain codes: {n_source}, "
        f"direct contour estimates: {n_direct}, "
        f"nearest-neighbour interpolations: {n_nearest}, "
        f"default terrain applied: {n_default}."
    )
    if n_default > 0:
        logger.warning(
            f"{n_default} edges had no usable elevation data and were "
            f"assigned the default terrain '{default_terrain}'. See "
            f"Methodology.md for limitations."
        )

    return G


# ======================================================================
# OPERATING SPEED & EDGE WEIGHTS
# ======================================================================


def assign_speeds_and_weights(
    G: nx.DiGraph, config: Dict, logger: logging.Logger
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """Assign operating speeds and compute travel-time edge weights.

    Rule: use network-wide mean operating speed, defined as 40 percent of
    the Nepal.gpkg ``NH`` design-speed attribute. This represents
    mixed-traffic average operating speed for impedance calculation.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph with terrain already assigned.
    config : dict
        Configuration dictionary.
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    tuple(networkx.DiGraph, pandas.DataFrame)
        Graph with speed/travel-time edge attributes populated, and a
        DataFrame summarizing every speed source applied.
    """
    rules = []
    factor = float(config.get("mean_speed_factor", 0.40))

    for u, v, data in tqdm(G.edges(data=True), desc="Assigning speeds"):
        valid_design_speed = _positive_float(data.get("design_speed_kmh"))

        if valid_design_speed is None:
            raise ValueError(
                "A graph edge has no usable design-speed value. Nepal.gpkg NH "
                "must carry design speed so mean_speed_kmh can be computed as "
                f"{factor:.2f} * design_speed_kmh."
            )
        speed = valid_design_speed * factor
        rule = f"mean_speed_{factor:.2f}_design_speed"
        source_field = str(data.get("design_speed_source_field", "design_speed_kmh"))

        length_km = data["length"] / 1000.0
        travel_time_hours = length_km / speed if speed > 0 else float("inf")
        travel_time_minutes = travel_time_hours * 60.0

        data["speed_kmh"] = speed
        data["mean_speed_kmh"] = speed
        data["mean_speed_factor"] = factor
        data["speed_source"] = rule
        data["speed_source_field"] = source_field
        data["road_length_km"] = length_km
        data["travel_time_hours"] = travel_time_hours
        data["travel_time_minutes"] = travel_time_minutes
        data["speed_rule"] = rule

        rules.append(
            {
                "osm_id": data.get("osm_id"),
                "link_code": data.get("link_code"),
                "road_refno": data.get("ref"),
                "design_speed_raw": data.get("design_speed_kmh"),
                "design_speed_valid": valid_design_speed,
                "mean_speed_factor": factor,
                "mean_speed_kmh": speed,
                "assigned_speed_kmh": speed,
                "rule_applied": rule,
                "source_field": source_field,
            }
        )

    logger.info(
        "Mean-speed travel-time weights assigned to all edges using %.2f x design speed.",
        factor,
    )
    return G, pd.DataFrame(rules)


# ======================================================================
# SHORTEST PATHS
# ======================================================================


def compute_od_matrices(
    G: nx.DiGraph,
    hq_nodes: Dict[str, Tuple[float, float]],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute travel-time and distance OD matrices via one Dijkstra run per HQ.

    For each HQ, ``networkx.single_source_dijkstra`` is called exactly
    once (weighted by ``travel_time_hours``), producing both the shortest
    travel time and the corresponding shortest path to every other node
    in a single pass. The distance matrix is derived by summing the
    ``length`` attribute along the same travel-time-optimal paths, thereby
    avoiding a second independent all-pairs computation.

    Parameters
    ----------
    G : networkx.DiGraph
        Fully-weighted road graph with HQ nodes inserted.
    hq_nodes : dict
        Mapping of HQ label -> graph node key.
    logger : logging.Logger
        Logger for diagnostics.

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        (travel_time_matrix_hours, distance_matrix_km), both indexed and
        columned by HQ label, using ``numpy.inf`` for unreachable pairs.
    """
    labels = sorted(
        hq_nodes.keys(),
        key=lambda label: re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip(),
    )
    n = len(labels)
    time_matrix = np.full((n, n), np.inf)
    dist_matrix = np.full((n, n), np.inf)

    for i, label_i in enumerate(tqdm(labels, desc="Running shortest paths")):
        source = hq_nodes[label_i]
        if source not in G.nodes:
            logger.warning(
                f"HQ '{label_i}' node not found in graph; row set to infinity."
            )
            continue

        distances, paths = nx.single_source_dijkstra(
            G, source=source, weight="travel_time_hours"
        )

        for j, label_j in enumerate(labels):
            target = hq_nodes[label_j]
            if i == j:
                time_matrix[i, j] = 0.0
                dist_matrix[i, j] = 0.0
                continue
            if target in distances:
                time_matrix[i, j] = distances[target]
                path = paths[target]
                path_length_m = sum(
                    G[path[k]][path[k + 1]]["length"] for k in range(len(path) - 1)
                )
                dist_matrix[i, j] = path_length_m / 1000.0

    time_df = pd.DataFrame(time_matrix, index=labels, columns=labels)
    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return time_df, dist_df


# ======================================================================
# QUALITY ASSURANCE
# ======================================================================


def run_quality_checks(
    time_df: pd.DataFrame,
    hq_gdf: gpd.GeoDataFrame,
    G: nx.DiGraph,
    logger: logging.Logger,
) -> None:
    """Run and log QA checks on the OD matrix and network.

    Checks performed: zero diagonal, matrix (a)symmetry given directed
    routing, finiteness for reachable pairs, duplicate HQ labels, and
    duplicate graph nodes. Infinite values are never replaced with
    arbitrary numbers.

    Parameters
    ----------
    time_df : pandas.DataFrame
        Travel-time matrix (hours).
    hq_gdf : geopandas.GeoDataFrame
        Projected HQ layer with an "hq_label" column.
    G : networkx.DiGraph
        The road graph (post HQ-insertion).
    logger : logging.Logger
        Logger for reporting warnings.
    """
    diag = np.diag(time_df.values)
    if not np.allclose(diag, 0.0):
        logger.warning("QA: Non-zero values found on the matrix diagonal.")
    else:
        logger.info("QA: Diagonal is zero for all HQs. OK.")

    values = time_df.values
    finite_mask = np.isfinite(values)
    asym = np.abs(
        np.where(finite_mask & finite_mask.T, values - values.T, 0.0)
    )
    max_asym = float(np.nanmax(asym)) if asym.size else 0.0
    if max_asym > 1e-6:
        logger.info(
            f"QA: Matrix is not perfectly symmetric (max diff: "
            f"{max_asym:.4f} hours). This is expected under directed "
            f"(oneway-aware) routing."
        )
    else:
        logger.info("QA: Matrix is symmetric.")

    n_infinite_offdiag = int(np.sum(~finite_mask) - np.sum(~np.isfinite(diag)))
    logger.info(
        f"QA: {n_infinite_offdiag} unreachable (infinite) OD pairs preserved "
        f"as infinity (not replaced with arbitrary values)."
    )

    labels = hq_gdf["hq_label"]
    dup_labels = labels[labels.duplicated()].tolist()
    if dup_labels:
        logger.warning(f"QA: Duplicate HQ names detected: {dup_labels}")
    else:
        logger.info("QA: No duplicate HQ names. OK.")

    nodes = list(G.nodes)
    if len(nodes) != len(set(nodes)):
        logger.warning("QA: Duplicate graph nodes detected.")
    else:
        logger.info("QA: No duplicate graph nodes. OK.")


def build_disconnected_report(
    time_df: pd.DataFrame,
    G: nx.DiGraph,
    hq_nodes: Dict[str, Tuple[float, float]],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Identify and document HQ pairs with no feasible route.

    Parameters
    ----------
    time_df : pandas.DataFrame
        Travel-time matrix (hours), with numpy.inf for unreachable pairs.
    G : networkx.DiGraph
        The road graph.
    hq_nodes : dict
        Mapping of HQ label -> graph node key.
    logger : logging.Logger
        Logger for terminal reporting.

    Returns
    -------
    pandas.DataFrame
        One row per HQ with at least one unreachable destination,
        containing columns: HQ, Connected_Component, Nearest_Reachable_HQ,
        Reason.
    """
    components = list(nx.weakly_connected_components(G))
    node_to_component = {}
    for comp_id, comp in enumerate(components):
        for node in comp:
            node_to_component[node] = comp_id

    records = []
    for label in time_df.index:
        row = time_df.loc[label]
        unreachable = row[~np.isfinite(row)].index.tolist()
        unreachable = [u for u in unreachable if u != label]
        if not unreachable:
            continue

        node = hq_nodes.get(label)
        comp_id = node_to_component.get(node, -1)

        reachable = row[np.isfinite(row) & (row.index != label)]
        nearest_reachable = reachable.idxmin() if len(reachable) > 0 else "None"

        records.append(
            {
                "HQ": label,
                "Connected_Component": comp_id,
                "Nearest_Reachable_HQ": nearest_reachable,
                "Unreachable_Count": len(unreachable),
                "Reason": (
                    "The projected HQ-area connector lies on the available "
                    "NH graph, but the graph itself still has no continuous "
                    "route to one or more other HQ connectors. This is a "
                    "network-topology gap, not a disconnected-HQ-point issue."
                ),
            }
        )

    report_df = pd.DataFrame(records)
    if len(report_df) > 0:
        logger.warning(
            f"{len(report_df)} HQs have at least one unreachable destination. "
            f"See DisconnectedHQs.csv."
        )
    else:
        logger.info(
            "No disconnected HQ pairs detected. All HQs are mutually "
            "reachable within the largest network component."
        )
    return report_df


# ======================================================================
# METHODOLOGY REPORT
# ======================================================================


def write_methodology(
    output_dir: Path,
    config: Dict,
    topology_report: TopologyReport,
    n_disconnected: int,
    n_hqs: int,
) -> None:
    """Generate the Methodology.md report documenting the full pipeline.

    Parameters
    ----------
    output_dir : Path
        Directory to write Methodology.md into.
    config : dict
        The configuration dictionary used for this run.
    topology_report : TopologyReport
        Diagnostics from graph construction.
    n_disconnected : int
        Number of HQs with at least one unreachable destination.
    n_hqs : int
        Total number of district HQs processed in this run.
    """
    content = f"""# Methodology

## 1. Overview

This document describes the methodology used to construct a
district-to-district travel-time impedance matrix for {n_hqs} Nepali
district headquarters, derived from `Nepal.gpkg` layer `NH` and the
`district_hq` area layer. The pipeline was implemented in
Python 3.13 using `geopandas`, `networkx`, `shapely`, `scipy`, `numpy`,
`pandas`, `rtree`, `pyogrio`, and `fiona`.

## 2. Coordinate Reference System

All input layers were reprojected (where necessary) to `{config['projection']}`
(UTM Zone 45N) prior to any distance or topology computation, ensuring
metric accuracy for a Nepal-extent study area.

## 3. Geometry Validation

Prior to topology construction, all layers were checked for invalid,
empty, duplicate, and multipart geometries. Invalid geometries were
repaired using Shapely's `make_valid`; empty and duplicate geometries were
removed; multipart features were exploded into single-part geometries.
Counts of removed/repaired features are recorded in `RunLog.txt`.

## 4. Network Topology Construction

True geometric intersections between road segments were detected using an
STRtree spatial index. Intersections coincident only with pre-existing
shared endpoints were treated as already-valid junctions and were not
re-split. Roads were split into topologically distinct edges at all
remaining interior intersection points using Shapely's `split` operation,
with all original attributes preserved on the resulting edge fragments.

Endpoint coordinates within `{config['snap_tolerance']} m` of one another
were snapped together using a KD-tree-based union-find clustering
procedure, resolving small digitisation gaps that would otherwise
fragment the network. Zero-length and duplicate edges are excluded from
the analysis graph.

The resulting graph was reduced to its largest weakly-connected component
to ensure a single, fully analysable network for shortest-path
computation.

**Topology diagnostics for this run:**

- Nodes (largest component): {topology_report.n_nodes}
- Edges (largest component): {topology_report.n_edges}
- Connected components detected: {topology_report.n_components}
- Largest component size: {topology_report.largest_component_size}
- Isolated nodes: {topology_report.n_isolated_nodes}
- Edges removed (zero-length/duplicate): {topology_report.n_removed_edges}
- Original edges split at true intersections: {topology_report.n_split_edges}

## 5. Directionality (Oneway Handling)

The road graph is a directed graph (`networkx.DiGraph`). Where the
`oneway` attribute indicated a one-directional restriction, only the
forward (or reverse, where explicitly tagged `-1`/`reverse`) edge was
added. Where `oneway` was missing or explicitly `no`, bidirectional edges
were added, following standard OSM tagging conventions (see
OpenStreetMap Wiki, n.d.).

## 6. District HQ Insertion

Rather than snapping district headquarters directly to the nearest graph
node (which can introduce large, unrepresentative positional error),
each HQ was orthogonally projected onto its nearest traversable road edge
using Shapely's `project`/`interpolate` methods. A new graph node was
inserted at the projection point, and the underlying edge (and its
reverse counterpart, where present) was split into two edges with
attributes preserved on both halves. Snap-distance diagnostics
(maximum, average, and count exceeding
`{config['hq_warning_distance']} m`) were logged and are not silently
discarded; see `RunLog.txt` and the QA checks below.

## 7. Terrain Attribution

Terrain is retained as a descriptive source attribute where present. Speed
is assigned from the Nepal.gpkg design-speed field through the mean-speed
factor.

## 8. Operating Speed Model

The populated design-speed field from `Nepal.gpkg`, layer `NH`, is the
source speed standard. The network-wide mean operating speed is computed as
`mean_speed_kmh = {config.get('mean_speed_factor', 0.40)} * design_speed_kmh`.

The resulting travel times represent uncongested planning impedance and do
not include congestion, pavement condition, weather, or seasonal closure.

## 9. Edge Weights

For every edge: `road_length_km = length_m / 1000`,
`travel_time_hours = road_length_km / mean_speed_kmh`, and
`travel_time_minutes = travel_time_hours * 60`. `travel_time_hours` was
used exclusively as the NetworkX routing weight; raw geometric distance
was never used for path-finding, consistent with standard
impedance-matrix construction practice in transportation geography
(Rodrigue, 2020).

## 10. Shortest-Path Computation

Shortest paths were computed using exactly one call to
`networkx.single_source_dijkstra` per district HQ (source), rather than
independent pairwise Dijkstra runs for every OD pair, following the
standard efficient formulation of the single-source shortest path problem
(Dijkstra, 1959). Each call simultaneously returns the shortest
travel-time and shortest path to every other reachable node. The
distance matrix was derived by summing edge lengths along these same
travel-time-optimal paths, avoiding a duplicate all-pairs computation.

**Computational complexity:** with Dijkstra's algorithm implemented via a
binary heap, each single-source run costs O((V + E) log V). Across all
{n_hqs} HQs this yields O({n_hqs} * (V + E) log V), which is substantially
more efficient than a naive O({n_hqs}^2) independent-pair formulation for
networks of this size.

## 11. Quality Assurance

Prior to writing outputs, the following checks were performed and logged:
matrix diagonal equal to zero, matrix symmetry (informational only, as
directed/oneway routing can legitimately produce asymmetric results),
duplicate HQ names, and duplicate graph nodes. Infinite (unreachable)
values were never replaced with arbitrary finite numbers.

## 12. Disconnected Districts

HQ areas are not classified as disconnected merely because they are polygons
rather than graph nodes. Each HQ area is connected to the closest point on
the NH graph, with zero connector distance when the area touches the graph.
If any unreachable pair remains, it is a road-network continuity issue after
projection, not a detached-HQ-point issue. In this run, {n_disconnected}
HQ(s) had at least one unreachable destination.

## 13. Assumptions and Limitations

- Where `oneway` was missing, bidirectional travel was assumed.
- Mean operating speed is computed as `{config.get('mean_speed_factor', 0.40)} *
  design_speed_kmh`; this still does not explicitly account for pavement
  condition, traffic volume, seasonal road closures, or weather-related
  impedance.
- HQ-to-network snap distances above the configured threshold
  ({config['hq_warning_distance']} m) may indicate digitisation
  mismatches between the HQ point layer and the road network and should
  be reviewed before final thesis submission (see `RunLog.txt`).

## References (APA)

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
*Numerische Mathematik, 1*(1), 269-271.

OpenStreetMap Wiki. (n.d.). *Key:oneway*. OpenStreetMap Foundation.

Rodrigue, J.-P. (2020). *The geography of transport systems* (5th ed.).
Routledge.

Transportation Research Board. (2016). *Highway capacity manual: A guide
for multimodal mobility analysis* (6th ed.). National Academies Press.

Wilson, J. P., & Gallant, J. C. (Eds.). (2000). *Terrain analysis:
Principles and applications*. John Wiley & Sons.
"""

    (output_dir / "Methodology.md").write_text(content, encoding="utf-8")


# ======================================================================
# OUTPUT WRITING
# ======================================================================


def write_network_statistics(
    output_dir: Path,
    topology_report: TopologyReport,
    hq_snap_report: pd.DataFrame,
    n_disconnected: int,
) -> None:
    """Write NetworkStatistics.txt summarizing the final network.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    topology_report : TopologyReport
        Graph diagnostics.
    hq_snap_report : pandas.DataFrame
        Per-HQ snap-distance report.
    n_disconnected : int
        Number of HQs with unreachable destinations.
    """
    lines = [
        "NETWORK STATISTICS",
        "===================",
        f"Nodes (largest component): {topology_report.n_nodes}",
        f"Edges (largest component): {topology_report.n_edges}",
        f"Connected components detected: {topology_report.n_components}",
        f"Largest component size: {topology_report.largest_component_size}",
        f"Isolated nodes: {topology_report.n_isolated_nodes}",
        f"Edges removed (zero-length/duplicate): {topology_report.n_removed_edges}",
        f"Original edges split at true intersections: {topology_report.n_split_edges}",
        "",
        "HQ SNAP QUALITY",
        "================",
        f"Max snap distance (m): {hq_snap_report['snap_distance_m'].max():.2f}",
        f"Average snap distance (m): {hq_snap_report['snap_distance_m'].mean():.2f}",
        f"HQs exceeding threshold: {int(hq_snap_report['exceeds_threshold'].sum())}",
        "",
        "CONNECTIVITY",
        "=============",
        f"HQs with at least one unreachable destination: {n_disconnected}",
    ]
    (output_dir / "NetworkStatistics.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _gpkg_safe_value(value: Any) -> Any:
    """Convert values that GeoPackage writers cannot store directly."""
    if value is None:
        return None
    if isinstance(value, (tuple, list, dict, set)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _prepare_gpkg_export(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a write-safe copy with unique, case-insensitive field names."""
    out = gdf.copy()
    geometry_name = out.geometry.name
    rename_map: Dict[str, str] = {}
    seen: set[str] = set()

    for col in out.columns:
        if col == geometry_name:
            continue
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") or "field"
        if safe.lower() == "geometry":
            safe = "attr_geometry"
        base = safe[:55]
        candidate = base
        suffix = 1
        while candidate.lower() in seen:
            suffix_text = f"_{suffix}"
            candidate = f"{base[:55 - len(suffix_text)]}{suffix_text}"
            suffix += 1
        rename_map[col] = candidate
        seen.add(candidate.lower())

    if rename_map:
        out = out.rename(columns=rename_map)

    for col in out.columns:
        if col == out.geometry.name:
            continue
        if out[col].dtype == object:
            out[col] = out[col].map(_gpkg_safe_value)
    return out


def _write_gpkg_clean(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a single-layer GeoPackage with a fresh schema."""
    if path.exists():
        path.unlink()
    _prepare_gpkg_export(gdf).to_file(path, driver="GPKG", engine="pyogrio")


def write_gpkg_outputs(
    output_dir: Path,
    processed_roads: gpd.GeoDataFrame,
    G: nx.DiGraph,
    projected_hqs: gpd.GeoDataFrame,
    crs: str,
    config: Dict,
) -> None:
    """Write the graph products needed by OD calibration.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    processed_roads : geopandas.GeoDataFrame
        Cleaned and split road network prior to graph construction.
    G : networkx.DiGraph
        Final road graph with all attributes populated.
    projected_hqs : geopandas.GeoDataFrame
        HQ points projected onto the network.
    crs : str
        CRS to assign to all written layers.
    """
    if config.get("write_processed_network", False):
        _write_gpkg_clean(processed_roads, output_dir / "ProcessedRoadNetwork.gpkg")

    if config.get("write_graph_nodes", False):
        node_records = [
            {"node_id": f"{x}_{y}", "x": x, "y": y, "geometry": Point(x, y)}
            for (x, y) in G.nodes
        ]
        nodes_gdf = gpd.GeoDataFrame(node_records, geometry="geometry", crs=crs)
        _write_gpkg_clean(nodes_gdf, output_dir / "GraphNodes.gpkg")

    edge_records = []
    for u, v, data in G.edges(data=True):
        rec = {k: v_ for k, v_ in data.items() if k != "geometry"}
        rec["geometry"] = data["geometry"]
        rec["u"] = f"{u[0]}_{u[1]}"
        rec["v"] = f"{v[0]}_{v[1]}"
        edge_records.append(rec)
    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=crs)
    _write_gpkg_clean(edges_gdf, output_dir / "GraphEdges.gpkg")

    _write_gpkg_clean(projected_hqs, output_dir / "ProjectedHQs.gpkg")


# ======================================================================
# MAIN PIPELINE
# ======================================================================


def main() -> None:
    """Execute the full travel-time impedance matrix construction pipeline."""
    np.random.seed(CONFIG["random_seed"])
    output_dir = Path(CONFIG["output_dir"])
    logger = setup_logging(output_dir)

    logger.info("Loading data")
    roads = load_and_reproject(
        CONFIG["master_gpkg_path"],
        CONFIG["projection"],
        logger,
        "Roads",
        layer=CONFIG["road_layer"],
    )
    hqs = load_and_reproject(
        CONFIG["master_gpkg_path"],
        CONFIG["projection"],
        logger,
        "District HQs",
        layer=CONFIG["district_hq_layer"],
    )

    logger.info("Validating geometries")
    roads = validate_and_clean_geometries(roads, logger, "Roads")
    roads = standardize_nepal_nh_attributes(roads, CONFIG, logger)
    hqs = validate_and_clean_geometries(hqs, logger, "District HQs")
    hqs = standardize_hq_area_labels(hqs, logger)

    dup_hq_labels = (
        hqs["VDC_NAME"].astype(str) + " (" + hqs["DIST_NAME"].astype(str) + ")"
    )
    if dup_hq_labels.duplicated().any():
        logger.warning(
            f"Duplicate HQ labels detected in source data: "
            f"{dup_hq_labels[dup_hq_labels.duplicated()].tolist()}"
        )

    logger.info("Cleaning topology / Splitting intersections")
    roads_split, n_split_edges = split_lines_at_intersections(roads, logger)

    logger.info("Building graph")
    G, topology_report = build_road_graph(roads_split, CONFIG, logger)
    topology_report.n_split_edges = n_split_edges

    logger.info("Removing disconnected components")
    # (already applied inside build_road_graph via largest-component filter)

    logger.info("Projecting HQs")
    G, projected_hqs, hq_snap_report = insert_hqs_into_graph(
        G, hqs, CONFIG, logger)

    logger.info("Assigning mean-speed travel-time weights")
    G, speed_audit_df = assign_speeds_and_weights(G, CONFIG, logger)

    hq_nodes = dict(zip(projected_hqs["hq_label"], projected_hqs["node_key"]))

    logger.info("Running shortest paths")
    time_df, dist_df = compute_od_matrices(G, hq_nodes, logger)

    logger.info("Running quality checks")
    run_quality_checks(time_df, projected_hqs, G, logger)
    disconnected_df = build_disconnected_report(time_df, G, hq_nodes, logger)

    logger.info("Writing outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    time_df.to_csv(output_dir / "impedance.csv")
    dist_df.to_csv(output_dir / "DistanceMatrix.csv")
    if CONFIG.get("write_speed_audit", False):
        speed_audit_df.to_csv(output_dir / "SpeedAudit.csv", index=False)
    if not disconnected_df.empty:
        disconnected_df.to_csv(output_dir / "DisconnectedHQs.csv", index=False)

    write_network_statistics(
        output_dir, topology_report, hq_snap_report, len(disconnected_df)
    )
    write_gpkg_outputs(
        output_dir, roads_split, G, projected_hqs, CONFIG["projection"], CONFIG
    )
    if CONFIG.get("write_methodology_md", False):
        write_methodology(
            output_dir, CONFIG, topology_report, len(
                disconnected_df), len(hq_nodes)
        )

    logger.info("Finished")
    logger.info(f"All outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

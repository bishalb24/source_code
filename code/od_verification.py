#!/usr/bin/env python3
"""
Generate only the essential OD-calibration verification figures.

This script is for visual QA after odme.py has produced the final daily OD matrix
and compact calibration audit. It writes only two PDF figures to output/OD:

1. aadt_nh_pcu_overview_regions.pdf
   Shows the positive AADT_2023 sections from Nepal.gpkg, layer NH.
2. major_od_routed_paths_overview_regions.pdf
   Shows the largest final OD cells routed on the same GraphEdges travel-time
   network used by the OD calibration.

No SVG, PNG, HTML, CSV, or note files are written here.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import box
from shapely.ops import unary_union


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


PROJECT = Path("/Users/bishalbhurtel/Desktop/Project")
INPUTS = PROJECT / "input"
OUTPUT_ROOT = PROJECT / "output"
OUTPUTS = OUTPUT_ROOT / "OD"
FIGURES = OUTPUTS

MASTER_GPKG = INPUTS / "Nepal.gpkg"
ROAD_LAYER = "NH"
GRAPH_EDGES = OUTPUT_ROOT / "impedance" / "GraphEdges.gpkg"
PROJECTED_HQS = OUTPUT_ROOT / "impedance" / "ProjectedHQs.gpkg"
OD_MATRIX = OUTPUTS / "od_matrix.csv"
AADT_TARGET_FIT = OUTPUTS / "aadt_target_fit.csv"
CALIBRATION_LOG = OUTPUTS / "calibration_log.csv"

TOP_OD_ROUTES = 50
MAX_AADT_LABELS_PER_REGION = 70
MAX_OD_LABELS_PER_REGION = 18
ASSIGNMENT_WEIGHT = "travel_time_hours"
AADT_CMAP = "cividis"
OD_CMAP = "plasma"
BACKGROUND_ROAD_COLOR = "#C7CBD1"
ZONE_POINT_COLOR = "#1F2937"
REGION_PALETTE = {
    "Western": "#0072B2",
    "Central": "#E69F00",
    "Eastern": "#CC79A7",
}

REGIONS = {
    "Western": (80.0, 82.8),
    "Central": (82.8, 85.6),
    "Eastern": (85.6, 88.35),
}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    check_calibration_selection()
    aadt_metadata = load_active_aadt_metadata()

    edges = load_graph_edges()
    plot_edges = edges.copy()
    plot_edges["geometry"] = plot_edges.geometry.simplify(
        100.0, preserve_topology=True
    )

    aadt_links = build_aadt_layer(edges.crs, aadt_metadata["field"])
    write_aadt_figure(plot_edges, aadt_links, aadt_metadata)

    od_matrix, zone_table = read_final_od_matrix()
    zones = load_zone_points(zone_table, edges.crs)
    major_od = build_major_od_layer(edges, zones, od_matrix)
    write_od_figure(plot_edges, zones, major_od)

    logger.info("Wrote %s", FIGURES / "aadt_nh_pcu_overview_regions.pdf")
    logger.info("Wrote %s", FIGURES / "major_od_routed_paths_overview_regions.pdf")


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def load_graph_edges() -> gpd.GeoDataFrame:
    require_file(GRAPH_EDGES, "GraphEdges.gpkg")
    edges = gpd.read_file(GRAPH_EDGES, engine="pyogrio").copy()
    required = {"u", "v", ASSIGNMENT_WEIGHT, "geometry"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"GraphEdges.gpkg is missing required fields: {sorted(missing)}")
    if edges.crs is None:
        raise ValueError(f"{GRAPH_EDGES} has no CRS.")

    edges[ASSIGNMENT_WEIGHT] = pd.to_numeric(
        edges[ASSIGNMENT_WEIGHT], errors="coerce"
    )
    edges = edges[edges[ASSIGNMENT_WEIGHT].notna() & (edges[ASSIGNMENT_WEIGHT] > 0)].copy()
    edges["u"] = edges["u"].astype(str)
    edges["v"] = edges["v"].astype(str)
    edges["edge_key"] = edges["u"] + "->" + edges["v"]
    return edges


def load_active_aadt_metadata() -> Dict[str, str]:
    """Use the selected ODME run's AADT metadata in every map label."""
    metadata = {
        "field": "AADT_2023",
        "period": "",
        "units": "vehicles/day",
    }
    audit = pd.read_csv(CALIBRATION_LOG)
    if audit.empty:
        return metadata
    if "selected_final" in audit.columns:
        selected = audit[
            audit["selected_final"].astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes"})
        ]
        row = selected.iloc[-1] if not selected.empty else audit.iloc[-1]
    else:
        row = audit.iloc[-1]
    for source, destination in (
        ("aadt_value_field", "field"),
        ("aadt_source_period", "period"),
        ("aadt_units", "units"),
    ):
        value = row.get(source)
        if pd.notna(value) and str(value).strip():
            metadata[destination] = str(value).strip()
    return metadata


def build_aadt_layer(target_crs, aadt_field: str) -> gpd.GeoDataFrame:
    require_file(MASTER_GPKG, "Nepal.gpkg")
    links = gpd.read_file(MASTER_GPKG, layer=ROAD_LAYER, engine="pyogrio").copy()
    required = {"link_code", aadt_field, "geometry"}
    missing = required - set(links.columns)
    if missing:
        raise ValueError(f"Nepal.gpkg layer NH is missing required fields: {sorted(missing)}")

    links["observed_count"] = pd.to_numeric(links[aadt_field], errors="coerce")
    links = links[links["observed_count"].notna() & (links["observed_count"] > 0)].copy()
    if links.empty:
        raise ValueError(f"Nepal.gpkg layer NH has no positive {aadt_field} values.")
    if links.crs != target_crs:
        links = links.to_crs(target_crs)

    links["link_code_norm"] = links["link_code"].map(normalize_link_code)
    links["display_link_code"] = links["link_code"].astype(str)
    links["display_name"] = links.get("link_name", links["link_code"]).fillna("").astype(str)
    links["used_for_calibration"] = False

    if AADT_TARGET_FIT.exists():
        fit = pd.read_csv(AADT_TARGET_FIT)
        if not fit.empty:
            fit["link_code_norm"] = fit.apply(fit_link_code_norm, axis=1)
            if "used_for_calibration" in fit.columns:
                fit["objective_target"] = (
                    fit["used_for_calibration"].fillna(False).astype(bool)
                )
            fit_cols = [
                "link_code_norm",
                "target_key",
                "objective_target",
                "calibration_status",
                "modelled_final",
                "pct_error_final",
                "graph_edge_count",
            ]
            fit_cols = [col for col in fit_cols if col in fit.columns]
            fit = fit[fit_cols].drop_duplicates("link_code_norm")
            links = links.merge(fit, on="link_code_norm", how="left")
            if "objective_target" in links.columns:
                links["used_for_calibration"] = (
                    links["objective_target"].fillna(False).astype(bool)
                )
    else:
        logger.warning(
            "Missing %s. AADT figure will show source AADT only, without fit overlay.",
            AADT_TARGET_FIT,
        )

    return gpd.GeoDataFrame(links, geometry="geometry", crs=target_crs)


def check_calibration_selection() -> None:
    require_file(CALIBRATION_LOG, "direct district-level ODME calibration log")
    log = pd.read_csv(CALIBRATION_LOG)
    if log.empty:
        raise ValueError(f"{CALIBRATION_LOG} is empty.")
    if "selected_final" not in log.columns:
        raise ValueError(
            "calibration_log.csv has no selected_final column. Rerun odme.py "
            "so the OD matrix is generated by the direct district-level ODME estimator."
        )

    selected = log[log["selected_final"].fillna(False).astype(bool)].copy()
    if selected.empty:
        raise ValueError(
            "calibration_log.csv does not mark any row as selected_final. "
            "Rerun odme.py before producing verification figures."
        )

    selected_row = selected.iloc[-1]
    converged = selected_row.get("converged")
    if isinstance(converged, str):
        converged = converged.strip().lower() in {"1", "true", "yes", "y", "on"}
    if converged is False:
        logger.warning(
            "The selected OD estimate was not marked converged (%s). The map is "
            "still produced, but the calibration log should be reviewed before reporting.",
            selected_row.get("stop_reason", "no stop reason"),
        )

    logger.info(
        "Calibration log selected iteration %s for verification figures.",
        selected_row.get("iteration", "unknown"),
    )


def read_final_od_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_file(OD_MATRIX, "final named OD matrix")
    matrix = pd.read_csv(OD_MATRIX, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    zone_rows = [parse_zone_label(label) for label in matrix.index]
    zone_table = pd.DataFrame(zone_rows)
    zone_table["matrix_label"] = matrix.index.tolist()
    return matrix, zone_table


def load_zone_points(zone_table: pd.DataFrame, target_crs) -> gpd.GeoDataFrame:
    require_file(PROJECTED_HQS, "ProjectedHQs.gpkg")
    hqs = gpd.read_file(PROJECTED_HQS, engine="pyogrio").to_crs(target_crs)
    required = {"hq_label", "node_key", "geometry"}
    missing = required - set(hqs.columns)
    if missing:
        raise ValueError(f"ProjectedHQs.gpkg is missing required fields: {sorted(missing)}")

    hqs = hqs[["hq_label", "node_key", "geometry"]].drop_duplicates("hq_label")
    hqs["hq_display"] = hqs["hq_label"].map(hq_display_label)
    hqs["graph_node"] = hqs["node_key"].map(node_key_to_graph_node)

    zones = zone_table.merge(
        hqs[["hq_display", "hq_label", "node_key", "geometry", "graph_node"]],
        on="hq_display",
        how="left",
        suffixes=("_matrix", "_projected"),
    )
    zones["hq_label"] = zones["hq_label_projected"].fillna(zones["hq_label_matrix"])
    missing_zones = zones[zones["graph_node"].isna()]
    if not missing_zones.empty:
        labels = "; ".join(missing_zones["matrix_label"].astype(str).tolist())
        raise ValueError(f"Could not map OD zones to ProjectedHQs.gpkg: {labels}")

    return gpd.GeoDataFrame(zones, geometry="geometry", crs=target_crs)


def build_major_od_layer(
    edges: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    od_matrix: pd.DataFrame,
) -> gpd.GeoDataFrame:
    graph, edge_lookup = build_assignment_graph(edges)
    zone_lookup = zones.set_index("matrix_label")

    pairs = od_matrix.stack().rename("final_odme_pcu_per_day").reset_index()
    pairs.columns = ["origin_label", "destination_label", "final_odme_pcu_per_day"]
    pairs["final_odme_pcu_per_day"] = pd.to_numeric(pairs["final_odme_pcu_per_day"], errors="coerce")
    pairs = pairs[pairs["final_odme_pcu_per_day"].notna() & (pairs["final_odme_pcu_per_day"] > 0)].copy()
    pairs = pairs[pairs["origin_label"] != pairs["destination_label"]]
    pairs = pairs.nlargest(TOP_OD_ROUTES, "final_odme_pcu_per_day").copy()

    records: List[Dict[str, object]] = []
    for origin_label, group in pairs.groupby("origin_label", sort=False):
        origin_zone = zone_lookup.loc[origin_label]
        source = str(origin_zone.graph_node)
        try:
            distances, paths = nx.single_source_dijkstra(
                graph, source, weight=ASSIGNMENT_WEIGHT
            )
        except nx.NetworkXNoPath:
            distances, paths = {}, {}

        for row in group.itertuples(index=False):
            destination_zone = zone_lookup.loc[row.destination_label]
            target = str(destination_zone.graph_node)
            if target not in paths:
                logger.warning(
                    "No routed path for %s -> %s",
                    origin_zone.district_name,
                    destination_zone.district_name,
                )
                continue

            path_nodes = paths[target]
            edge_keys = [
                graph[u][v]["edge_key"]
                for u, v in zip(path_nodes[:-1], path_nodes[1:])
                if graph.has_edge(u, v)
            ]
            route_edges = [edge_lookup[key] for key in edge_keys if key in edge_lookup]
            if not route_edges:
                continue

            records.append(
                {
                    "origin_zone_id": origin_zone.zone_id,
                    "origin_district": origin_zone.district_name,
                    "origin_hq_label": origin_zone.hq_label,
                    "destination_zone_id": destination_zone.zone_id,
                    "destination_district": destination_zone.district_name,
                    "destination_hq_label": destination_zone.hq_label,
                    "final_odme_pcu_per_day": float(row.final_odme_pcu_per_day),
                    "path_cost_hours": float(distances[target]),
                    "path_edge_count": len(edge_keys),
                    "geometry": unary_union(route_edges),
                }
            )

    if not records:
        raise ValueError("Could not reconstruct any routed top-OD geometries.")
    return gpd.GeoDataFrame(records, geometry="geometry", crs=edges.crs)


def build_assignment_graph(edges: gpd.GeoDataFrame) -> Tuple[nx.DiGraph, Dict[str, object]]:
    graph = nx.DiGraph()
    edge_lookup: Dict[str, object] = {}
    work = edges.sort_values(ASSIGNMENT_WEIGHT, na_position="last")
    for row in work.itertuples(index=False):
        u = str(row.u)
        v = str(row.v)
        weight = float(getattr(row, ASSIGNMENT_WEIGHT))
        edge_key = str(row.edge_key)
        if edge_key not in edge_lookup:
            edge_lookup[edge_key] = row.geometry
        existing = graph.get_edge_data(u, v)
        if existing is None or weight < existing[ASSIGNMENT_WEIGHT]:
            graph.add_edge(
                u,
                v,
                edge_key=edge_key,
                weight=weight,
                **{ASSIGNMENT_WEIGHT: weight},
            )
    return graph, edge_lookup


def write_aadt_figure(
    edges: gpd.GeoDataFrame,
    aadt_links: gpd.GeoDataFrame,
    aadt_metadata: Dict[str, str],
) -> None:
    figure_data = aadt_links.copy()
    figure_data["geometry"] = figure_data.geometry.simplify(
        30.0, preserve_topology=True
    )
    vmax = max(float(figure_data["observed_count"].max()), 1.0)
    figure_data["plot_width"] = 0.3 + 2.2 * np.sqrt(
        figure_data["observed_count"] / vmax
    )
    figure_data["label_priority"] = (
        figure_data["used_for_calibration"].fillna(False).astype(int) * 1.0e9
        + figure_data["observed_count"]
    )

    fig, axes = make_overview_region_figure(
        f"{aadt_metadata['period'] + ' ' if aadt_metadata['period'] else ''}"
        "AADT on Nepal.gpkg national-highway sections",
        f"Labels show link code and two-way {aadt_metadata['field']}.",
    )
    region_boxes = make_region_boxes(edges.crs)
    plot_aadt_panel(
        axes["Overview"], edges, figure_data, region_boxes, "National overview", labels=False
    )
    for region_name in REGIONS:
        plot_aadt_panel(
            axes[region_name], edges, figure_data, region_boxes, region_name, labels=True
        )

    norm = Normalize(vmin=0.0, vmax=vmax)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=AADT_CMAP),
        ax=list(axes.values()),
        orientation="horizontal",
        fraction=0.025,
        pad=0.035,
        aspect=45,
    )
    colorbar.set_label(f"Two-way AADT ({aadt_metadata['units']})", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    save_pdf(fig, "aadt_nh_pcu_overview_regions")
    plt.close(fig)


def write_od_figure(
    edges: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    major_od: gpd.GeoDataFrame,
) -> None:
    figure_data = major_od.copy()
    vmax = max(float(figure_data["final_odme_pcu_per_day"].max()), 1.0)
    figure_data["geometry"] = figure_data.geometry.simplify(
        75.0, preserve_topology=True
    )
    figure_data["plot_width"] = 0.45 + 3.1 * np.sqrt(
        figure_data["final_odme_pcu_per_day"] / vmax
    )

    fig, axes = make_overview_region_figure(
        f"Top {len(figure_data)} final OD movements on modelled shortest paths",
        "Labels show symmetric OD pair and final mirrored matrix value.",
    )
    region_boxes = make_region_boxes(edges.crs)
    plot_od_panel(
        axes["Overview"],
        edges,
        zones,
        figure_data,
        region_boxes,
        "National overview",
        labels=False,
    )
    for region_name in REGIONS:
        plot_od_panel(
            axes[region_name],
            edges,
            zones,
            figure_data,
            region_boxes,
            region_name,
            labels=True,
        )

    norm = Normalize(
        vmin=float(figure_data["final_odme_pcu_per_day"].min()),
        vmax=vmax,
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=OD_CMAP),
        ax=list(axes.values()),
        orientation="horizontal",
        fraction=0.025,
        pad=0.035,
        aspect=45,
    )
    colorbar.set_label("Final symmetric OD matrix value", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    save_pdf(fig, "major_od_routed_paths_overview_regions")
    plt.close(fig)


def plot_aadt_panel(
    ax: plt.Axes,
    edges: gpd.GeoDataFrame,
    aadt_links: gpd.GeoDataFrame,
    region_boxes: Dict[str, object],
    panel_name: str,
    labels: bool,
) -> None:
    edges.plot(
        ax=ax,
        color=BACKGROUND_ROAD_COLOR,
        linewidth=0.18 if labels else 0.12,
        alpha=0.55,
    )
    ordered = aadt_links.sort_values("observed_count")
    ordered.plot(
        ax=ax,
        column="observed_count",
        cmap=AADT_CMAP,
        linewidth=ordered["plot_width"],
        alpha=0.9,
        vmin=0.0,
        vmax=float(aadt_links["observed_count"].max()),
    )

    if labels:
        region_geometry = region_boxes[panel_name]
        region_edges = edges[edges.intersects(region_geometry)]
        set_feature_extent(ax, region_edges)
        candidates = aadt_links[aadt_links.intersects(region_geometry)].copy()
        candidates["geometry"] = candidates.geometry.intersection(region_geometry)
        add_collision_filtered_labels(
            ax,
            candidates,
            lambda row: f"{row.display_link_code}  {row.observed_count:,.0f}",
            value_column="label_priority",
            max_labels=MAX_AADT_LABELS_PER_REGION,
            fontsize=4.3,
        )
    else:
        plot_region_frames(ax, region_boxes, edges.crs)
    style_map_axis(ax, panel_name)


def plot_od_panel(
    ax: plt.Axes,
    edges: gpd.GeoDataFrame,
    zones: gpd.GeoDataFrame,
    major_od: gpd.GeoDataFrame,
    region_boxes: Dict[str, object],
    panel_name: str,
    labels: bool,
) -> None:
    edges.plot(
        ax=ax,
        color=BACKGROUND_ROAD_COLOR,
        linewidth=0.2 if labels else 0.12,
        alpha=0.55,
    )
    ordered = major_od.sort_values("final_odme_pcu_per_day")
    ordered.plot(
        ax=ax,
        column="final_odme_pcu_per_day",
        cmap=OD_CMAP,
        linewidth=ordered["plot_width"],
        alpha=0.72,
        vmin=float(major_od["final_odme_pcu_per_day"].min()),
        vmax=float(major_od["final_odme_pcu_per_day"].max()),
    )
    zones.plot(ax=ax, markersize=2.2 if labels else 1.5, color=ZONE_POINT_COLOR, alpha=0.8)

    if labels:
        region_geometry = region_boxes[panel_name]
        region_edges = edges[edges.intersects(region_geometry)]
        set_feature_extent(ax, region_edges)
        candidates = major_od[major_od.intersects(region_geometry)].copy()
        candidates["geometry"] = candidates.geometry.intersection(region_geometry)
        add_collision_filtered_labels(
            ax,
            candidates,
            lambda row: (
                f"{row.origin_district}->{row.destination_district}  "
                f"{row.final_odme_pcu_per_day:,.0f}"
            ),
            value_column="final_odme_pcu_per_day",
            max_labels=MAX_OD_LABELS_PER_REGION,
            fontsize=4.15,
        )
    else:
        plot_region_frames(ax, region_boxes, edges.crs)
    style_map_axis(ax, panel_name)


def make_overview_region_figure(
    title: str,
    subtitle: str,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.18])
    axes = {
        "Overview": fig.add_subplot(grid[0, :]),
        "Western": fig.add_subplot(grid[1, 0]),
        "Central": fig.add_subplot(grid[1, 1]),
        "Eastern": fig.add_subplot(grid[1, 2]),
    }
    fig.suptitle(title, fontsize=14, fontweight="semibold")
    fig.text(0.5, 0.956, subtitle, ha="center", va="top", fontsize=8.5, color="#374151")
    return fig, axes


def make_region_boxes(target_crs) -> Dict[str, object]:
    regions = gpd.GeoDataFrame(
        {"region": list(REGIONS)},
        geometry=[box(west, 26.0, east, 30.7) for west, east in REGIONS.values()],
        crs="EPSG:4326",
    ).to_crs(target_crs)
    return dict(zip(regions["region"], regions.geometry))


def plot_region_frames(
    ax: plt.Axes,
    region_boxes: Dict[str, object],
    target_crs,
) -> None:
    for name, geometry in region_boxes.items():
        outline = gpd.GeoSeries([geometry.boundary], crs=target_crs)
        outline.plot(ax=ax, color=REGION_PALETTE[name], linewidth=0.65, linestyle="--", alpha=0.85)
        point = geometry.representative_point()
        ax.text(
            point.x,
            geometry.bounds[3],
            name,
            fontsize=5.5,
            color=REGION_PALETTE[name],
            ha="center",
            va="bottom",
        )


def style_map_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")


def set_feature_extent(
    ax: plt.Axes,
    features: gpd.GeoDataFrame,
    pad_fraction: float = 0.04,
) -> None:
    if features.empty:
        return
    minx, miny, maxx, maxy = features.total_bounds
    padx = max((maxx - minx) * pad_fraction, 1000.0)
    pady = max((maxy - miny) * pad_fraction, 1000.0)
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)


def add_collision_filtered_labels(
    ax: plt.Axes,
    rows: gpd.GeoDataFrame,
    label_builder,
    value_column: str,
    max_labels: int,
    fontsize: float,
) -> int:
    if rows.empty:
        return 0

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    accepted_boxes = []
    offsets = [
        (0, 5), (0, -6), (7, 5), (-7, 5), (8, -6), (-8, -6),
        (13, 0), (-13, 0), (13, 8), (-13, 8), (13, -9), (-13, -9),
    ]
    placed = 0
    for row in rows.sort_values(value_column, ascending=False).itertuples(index=False):
        if placed >= max_labels:
            break
        point = row.geometry.representative_point()
        for dx, dy in offsets:
            annotation = ax.annotate(
                label_builder(row),
                xy=(point.x, point.y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                ha="center",
                va="center",
                color="#111827",
                zorder=20,
                bbox={
                    "boxstyle": "square,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "#9ca3af",
                    "linewidth": 0.18,
                    "alpha": 0.82,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#6b7280",
                    "linewidth": 0.2,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
            bbox_display = annotation.get_window_extent(renderer=renderer).expanded(1.04, 1.12)
            if not any(bbox_display.overlaps(existing) for existing in accepted_boxes):
                accepted_boxes.append(bbox_display)
                placed += 1
                break
            annotation.remove()
    return placed


def save_pdf(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")


def parse_zone_label(label: str) -> Dict[str, str]:
    match = re.match(r"^\s*(?:(\d+)\s+)?(.+?)\s+\|\s+(.+?)\s*$", str(label))
    if not match:
        raise ValueError(f"Could not parse OD matrix zone label: {label!r}")
    hq_label = match.group(3).strip()
    return {
        "zone_id": match.group(1) or "",
        "district_name": match.group(2).strip(),
        "hq_label": hq_label,
        "hq_display": hq_display_label(hq_label),
    }


def hq_display_label(label: object) -> str:
    text = str(label or "").strip()
    if "|" in text:
        text = text.split("|", 1)[1].strip()
    return re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()


def node_key_to_graph_node(node_key: object) -> str:
    value = ast.literal_eval(str(node_key))
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"Unexpected ProjectedHQ node_key: {node_key}")
    return f"{value[0]}_{value[1]}"


def fit_link_code_norm(row: pd.Series) -> str:
    link_code = row.get("link_code", "")
    if pd.notna(link_code) and str(link_code).strip():
        return normalize_link_code(link_code)
    target_key = str(row.get("target_key", ""))
    if "::" in target_key:
        return normalize_link_code(target_key.split("::")[-1])
    return ""


def normalize_link_code(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper().strip())


if __name__ == "__main__":
    main()

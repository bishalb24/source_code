#!/usr/bin/env python3
"""
odme.py

District-level origin-destination matrix estimation (ODME) against observed
two-way AADT targets.

Implemented estimator
---------------------
The calibrated object is a symmetric 77 by 77 inter-district matrix x,
excluding district self-cells. The assignment/incidence matrix P has entries
p[a, r], where r is an unordered district-pair variable and a is an observed
two-way count target:

    v_a = sum_r p[a, r] * x_r

The estimator is a positive, prior-constrained district inverse problem solved
as a weighted generalized least-squares ODME problem:

    minimize sum_r x_r * (log(x_r / x0_r) - 1) + x0_r
             + 0.5 * sum_a w_a * (sum_r p[a, r] * x_r - y_a)^2

where w_a = 1 / sigma_a^2.  The count standard deviation combines count
uncertainty, path dilution, target category variance and seed-implausibility
diagnostics, so ambiguous AADT rows pull softly instead of forcing exact
station equality.

The gravity seed carries the prior district distribution implied by activity
and travel time. Its daily magnitude is estimated from the selected observed
AADT screenlines. The activity proxy is district population multiplied by the
province-level GDP-per-capita multiplier, so it is an economic-mass prior
rather than observed trips. Missing AADT sections are absent observations, not
zero-flow constraints.

Main references:
- Van Zuylen and Willumsen (1980), "The most likely trip matrix estimated from
  traffic counts", Transportation Research B.
- Spiess (1987), "A maximum likelihood model for estimating origin-destination
  matrices", Transportation Research B.

Current data note
-----------------
Nepal.gpkg is the authoritative road-network source for impedance, assignment
and count calibration. AADT calibration evidence is loaded from layer NH,
using AADT_2023 as two-way vehicles per day.
"""

from __future__ import annotations

import ast
import itertools
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    project_dir: Path = Path("/Users/bishalbhurtel/Desktop/Project")
    data_dir: Path = Path("/Users/bishalbhurtel/Desktop/Project/input")
    input_excel: str = "Input.xlsx"
    output_dir: Path = Path("/Users/bishalbhurtel/Desktop/Project/output/OD")
    master_gpkg: Path = Path("/Users/bishalbhurtel/Desktop/Project/input/Nepal.gpkg")
    road_layer: str = "NH"
    district_hq_layer: str = "district_hq"
    district_boundary_layer: str = "district_boundary"
    spatial_projection: str = "EPSG:32645"

    impedance_csv: Path = Path(
        "/Users/bishalbhurtel/Desktop/Project/output/impedance/impedance.csv")
    prefer_impedance_csv: bool = True
    graph_edges_gpkg: Path = Path(
        "/Users/bishalbhurtel/Desktop/Project/output/impedance/GraphEdges.gpkg")
    projected_hqs_gpkg: Path = Path(
        "/Users/bishalbhurtel/Desktop/Project/output/impedance/ProjectedHQs.gpkg")
    aadt_value_field: str = "AADT_2023"
    aadt_source_period: str = "2023"
    aadt_units: str = "vehicles/day"

    beta: float = 0.20
    max_iter_gravity: int = 200
    tol_gravity: float = 1e-8

    max_iter_odme: int = 200
    tol_odme: float = 1e-8
    # Count uncertainty is max(Poisson standard deviation, CV * AADT).
    aadt_count_cv: float = 0.20
    # Standard deviation of a district OD log multiplier around its gravity prior.
    district_prior_log_std: float = 1.50
    odme_update_damping: float = 0.40
    odme_multiplier_min: float = 0.10
    odme_multiplier_max: float = 10.0
    gls_category_variance_primary: float = 1.0
    gls_category_variance_low_volume: float = 100.0
    gls_category_variance_hq_adjacent: float = 16.0
    gls_category_variance_all_links: float = 100.0
    gls_path_dilution_power: float = 1.0
    gls_seed_implausibility_floor: float = 1.0
    gls_od_multiplier_min: float = 1e-4
    gls_od_multiplier_max: float = 1e4
    screenline_grouping_mode: str = "incidence_similarity"
    screenline_group_min_jaccard: float = 0.98
    screenline_group_min_cosine: float = 0.98
    screenline_group_observed_method: str = "median"
    zero_seed_ratio_threshold: float = 0.01
    zero_seed_abs_flow_threshold: float = 1.0
    beta_selection_mode: str = "fixed"
    beta_candidates: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    route_choice_mode: str = "logit_k_shortest"
    route_choice_k: int = 3
    route_choice_theta_per_hour: float = 2.0
    route_choice_max_cost_factor: float = 1.25
    use_district_screenline_targets: bool = True
    screenline_buffer_m: float = 500.0
    screenline_min_hq_distance_m: float = 3000.0
    min_calibration_aadt: float = 1000.0
    screenline_target_weight: float = 1.0
    context_target_weight: float = 0.0

    assignment_weight_field: str = "travel_time_hours"
    min_model_flow: float = 1e-9
    # OD demand is an expected daily flow; retain fractional expectations on export.
    human_od_decimals: int = 2
    matrix_output_float_format: str = "%.2f"
    write_rounded_matrix_outputs: bool = False
    write_id_matrix: bool = False
    write_calibration_audit: bool = True
    write_diagnostics: bool = False
    write_seed_outputs: bool = False
    write_pair_table: bool = False
    write_summary: bool = False
    observed_aadt_weight: float = 1.0
    allow_duplicate_impedance_labels: bool = True


class ODSynthesizer:
    DISTRICT_ALIASES: Dict[str, str] = {
        "dang deukhuri": "dang",
        "east rukum": "rukum east",
        "rukum east": "rukum east",
        "west rukum": "rukum west",
        "rukum west": "rukum west",
        "nawalpur nawalparasi e": "nawalparasi east",
        "nawalpur nawalparasi east": "nawalparasi east",
        "nawalparasi east": "nawalparasi east",
        "parasi nawalparasi w": "nawalparasi west",
        "parasi nawalparasi west": "nawalparasi west",
        "nawalparasi west": "nawalparasi west",
        "chitawan": "chitwan",
        "dhanusha": "dhanusa",
        "kabhrepalanchok": "kavrepalanchok",
        "kavre": "kavrepalanchok",
        "makawanpur": "makwanpur",
        "tanahu": "tanahun",
        "kapilvastu": "kapilbastu",
        "kavrepalanchok": "kavrepalanchok",
        "sindhupalchowk": "sindhupalchok",
        "terhathum": "tehrathum",
        "udaypur": "udayapur",
    }

    PREFERRED_DUPLICATE_LABELS: Dict[str, str] = {
        "dang deukhuri": "TribhuwanNagar",
    }

    def __init__(self, config: Config):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def workbook_path(self) -> Path:
        return self.config.data_dir / self.config.input_excel

    def _sheet_name(self, *aliases: str) -> str:
        available = pd.ExcelFile(self.workbook_path).sheet_names
        for alias in aliases:
            if alias in available:
                return alias
        raise ValueError(
            f"Input workbook lacks required sheet. Tried: {', '.join(aliases)}"
        )

    def run(self) -> None:
        params = self.load_gravity_params()
        zones = self.load_zones()
        impedance, alignment_report = self.load_impedance_matrix(zones)
        aadt = self.load_aadt_targets()

        if self.config.write_diagnostics:
            alignment_path = self.config.output_dir / "impedance_alignment_report.csv"
            alignment_report.to_csv(alignment_path, index=False)
            logger.info("Wrote impedance alignment report: %s", alignment_path)

        activity_weights = self.district_activity_vector(zones)
        production_weights = activity_weights.copy()
        attraction_weights = activity_weights.copy()
        production_weights, attraction_weights = self.balance_totals(
            production_weights, attraction_weights)

        beta = float(params.get("beta", self.config.beta))
        max_iter = int(params.get("max_iter_gravity",
                       self.config.max_iter_gravity))
        tol = float(params.get("tol_gravity", self.config.tol_gravity))

        logger.info(
            "Gravity inputs: zones=%d, impedance_shape=%s, configured_beta=%s, relative O_total=%.2f, relative D_total=%.2f",
            len(zones),
            impedance.shape,
            beta,
            production_weights.sum(),
            attraction_weights.sum(),
        )
        logger.info(
            "Impedance range: min=%.4f, max=%.4f. Source values appear to be hours from impedance.py.",
            float(np.nanmin(impedance)),
            float(np.nanmax(impedance)),
        )

        seed_od = GravityModel.furness_balance(
            production_weights,
            attraction_weights,
            impedance,
            beta,
            max_iter=max_iter,
            tol=tol,
        )
        seed_od = self.symmetrize_od_matrix(seed_od)
        self.write_named_matrix_output(
            zones=zones,
            alignment_report=alignment_report,
            matrix=seed_od,
            filename="gravity_seed_matrix.csv",
            log_label="pre-calibration symmetric gravity seed",
        )

        zone_ids = zones["zone_id"].tolist()
        final_od, seed_od, calib_log, target_fit, mapping_report = self.run_district_od_estimation(
            seed_od=seed_od,
            zones=zones,
            aadt=aadt,
            params=params,
            alignment_report=alignment_report,
            production_weights=production_weights,
            attraction_weights=attraction_weights,
            impedance=impedance,
            configured_beta=beta,
        )
        final_df = pd.DataFrame(final_od, index=zone_ids, columns=zone_ids)
        if self.config.write_seed_outputs:
            seed_df = pd.DataFrame(seed_od, index=zone_ids, columns=zone_ids)
            seed_path = self.config.output_dir / "seed_od_matrix_ids.csv"
            seed_df.to_csv(
                seed_path,
                index_label="zone_id",
                float_format=self.config.matrix_output_float_format,
            )
            logger.info("Wrote selected count-scaled gravity prior: %s", seed_path)
        if self.config.write_id_matrix or self.config.write_diagnostics:
            final_path = self.config.output_dir / "od_matrix_ids.csv"
            final_df.to_csv(
                final_path,
                index_label="zone_id",
                float_format=self.config.matrix_output_float_format,
            )
            logger.info("Wrote final OD matrix using zone IDs: %s", final_path)

        self.write_named_od_outputs(
            zones=zones,
            alignment_report=alignment_report,
            seed_od=seed_od,
            final_od=final_od,
        )
        if self.config.write_diagnostics and not target_fit.empty and not mapping_report.empty:
            self.write_station_fit_report(mapping_report, target_fit)

        if self.config.write_calibration_audit or self.config.write_diagnostics:
            calib_path = self.config.output_dir / "calibration_log.csv"
            calib_log.to_csv(calib_path, index=False)
            logger.info("Wrote calibration log: %s", calib_path)
            if not target_fit.empty:
                target_fit_path = self.config.output_dir / "aadt_target_fit.csv"
                target_fit.to_csv(target_fit_path, index=False)
                logger.info(
                    "Wrote compact AADT target fit audit: %s", target_fit_path)
        if self.config.write_summary:
            self.write_run_summary_report(
                zones=zones,
                params=params,
                calib_log=calib_log,
                target_fit=target_fit,
                mapping_report=mapping_report,
                final_od=final_od,
            )

        logger.info("Pipeline finished.")

    def load_zones(self) -> pd.DataFrame:
        if not self.workbook_path.exists():
            raise FileNotFoundError(
                f"OD input workbook not found: {self.workbook_path}. "
                "Restore Input.xlsx or update Config.input_excel."
            )
        zones_sheet = self._sheet_name("07_Population_GDP")
        zones = pd.read_excel(
            self.workbook_path, sheet_name=zones_sheet, header=3)
        if "zone_id" not in zones.columns:
            raise ValueError(
                f"{zones_sheet} must have a 'zone_id' column on Excel row 4.")
        if "district_name" not in zones.columns:
            raise ValueError(
                f"{zones_sheet} must have a 'district_name' column on Excel row 4.")

        zone_id_num = pd.to_numeric(zones["zone_id"], errors="coerce")
        zones = zones[zone_id_num.notna()].copy()
        zones["zone_id"] = zone_id_num[zone_id_num.notna()].astype(
            int).astype(str)
        zones["district_name"] = zones["district_name"].astype(str).str.strip()

        for col in ["Oi_trips", "Di_trips", "activity_proxy", "productions", "attractions"]:
            if col in zones.columns:
                zones[col] = pd.to_numeric(zones[col], errors="coerce")

        if zones["zone_id"].duplicated().any():
            dupes = zones.loc[zones["zone_id"].duplicated(),
                              "zone_id"].tolist()
            raise ValueError(
                f"Duplicate zone_id values in {zones_sheet} sheet: {dupes}")

        if len(zones) != 77:
            logger.warning("Expected 77 numeric zones, found %d.", len(zones))
        else:
            logger.info("Loaded 77 zones from the workbook.")
        # Every matrix, impedance and activity vector uses numeric zone-ID order.
        # Enforcing it here prevents a silently reordered workbook table from
        # misaligning a district's activity weight with its impedance row.
        return zones.sort_values(
            "zone_id", key=lambda values: values.astype(int), kind="mergesort"
        ).reset_index(drop=True)

    def load_gravity_params(self) -> Dict[str, float]:
        params: Dict[str, float] = {
            "beta": self.config.beta,
            "max_iter_gravity": self.config.max_iter_gravity,
            "tol_gravity": self.config.tol_gravity,
            "max_iter_odme": self.config.max_iter_odme,
            "tol_odme": self.config.tol_odme,
        }

        try:
            params_sheet = self._sheet_name("05_OD_Params")
            table = pd.read_excel(self.workbook_path,
                                  sheet_name=params_sheet, header=2)
        except Exception as exc:
            logger.warning(
                "Could not load OD parameter sheet: %s. Using config defaults.", exc)
            return params

        key_col = "purpose_code"
        value_col = "purpose_label"
        if key_col not in table.columns:
            logger.warning(
                "OD parameter sheet has no purpose_code column. Using config defaults.")
            return params

        keys = table[key_col].astype(str).str.strip()

        def setting_value(*names: str) -> Optional[str]:
            if value_col not in table.columns:
                return None
            normalized = keys.str.lower()
            for name in names:
                row = table[normalized == name.lower()]
                if row.empty:
                    continue
                value = row.iloc[0][value_col]
                if pd.notna(value) and str(value).strip():
                    return str(value).strip()
            return None

        def setting_bool(*names: str) -> Optional[bool]:
            value = setting_value(*names)
            if value is None:
                return None
            text = value.strip().lower()
            if text in {"1", "true", "yes", "y", "on"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
            number = pd.to_numeric(text, errors="coerce")
            if pd.notna(number):
                if float(number) == 1.0:
                    return True
                if float(number) == 0.0:
                    return False
            return None

        master_gpkg = setting_value("master_gpkg", "master_geopackage", "nepal_gpkg")
        if master_gpkg:
            self.config.master_gpkg = Path(master_gpkg).expanduser()
        road_layer = setting_value("road_layer", "road_network_layer")
        if road_layer:
            self.config.road_layer = road_layer
        district_hq_layer = setting_value("district_hq_layer", "hq_layer")
        if district_hq_layer:
            self.config.district_hq_layer = district_hq_layer
        district_boundary_layer = setting_value(
            "district_boundary_layer", "district_layer", "screenline_boundary_layer")
        if district_boundary_layer:
            self.config.district_boundary_layer = district_boundary_layer
        spatial_projection = setting_value("spatial_projection", "projected_crs")
        if spatial_projection:
            self.config.spatial_projection = spatial_projection
        aadt_field = setting_value("aadt_value_field", "road_aadt_field", "od_aadt_field")
        if aadt_field:
            self.config.aadt_value_field = aadt_field
        aadt_source_period = setting_value("aadt_source_period")
        if aadt_source_period:
            self.config.aadt_source_period = aadt_source_period
        aadt_units = setting_value("aadt_units")
        if aadt_units:
            self.config.aadt_units = aadt_units
        for workbook_key, attr in {
            "write_id_matrix": "write_id_matrix",
            "write_calibration_audit": "write_calibration_audit",
            "write_diagnostics": "write_diagnostics",
            "write_seed_outputs": "write_seed_outputs",
            "write_pair_table": "write_pair_table",
            "write_summary": "write_summary",
            "write_rounded_matrix_outputs": "write_rounded_matrix_outputs",
            "use_district_screenline_targets": "use_district_screenline_targets",
        }.items():
            parsed = setting_bool(workbook_key)
            if parsed is not None:
                setattr(self.config, attr, parsed)

        route_choice_mode = setting_value("route_choice_mode")
        if route_choice_mode:
            self.config.route_choice_mode = route_choice_mode
        matrix_output_float_format = setting_value("matrix_output_float_format")
        if matrix_output_float_format:
            self.config.matrix_output_float_format = matrix_output_float_format
        screenline_grouping_mode = setting_value("screenline_grouping_mode")
        if screenline_grouping_mode:
            self.config.screenline_grouping_mode = screenline_grouping_mode
        screenline_group_observed_method = setting_value(
            "screenline_group_observed_method"
        )
        if screenline_group_observed_method:
            self.config.screenline_group_observed_method = (
                screenline_group_observed_method
            )

        all_row = table[keys.str.upper() == "ALL"]
        if not all_row.empty and "param_beta" in table.columns:
            beta = pd.to_numeric(
                all_row.iloc[0]["param_beta"], errors="coerce")
            if pd.notna(beta):
                params["beta"] = float(beta)

        setting_map = {
            "max_iter_gravity": "max_iter_gravity",
            "tol_gravity": "tol_gravity",
            "max_iter_odme": "max_iter_odme",
            "tol_odme": "tol_odme",
            "aadt_count_cv": "aadt_count_cv",
            "district_prior_log_std": "district_prior_log_std",
            "route_choice_k": "route_choice_k",
            "route_choice_theta_per_hour": "route_choice_theta_per_hour",
            "route_choice_max_cost_factor": "route_choice_max_cost_factor",
            "human_od_decimals": "human_od_decimals",
            "screenline_buffer_m": "screenline_buffer_m",
            "screenline_min_hq_distance_m": "screenline_min_hq_distance_m",
            "min_calibration_aadt": "min_calibration_aadt",
            "screenline_target_weight": "screenline_target_weight",
            "context_target_weight": "context_target_weight",
            "odme_update_damping": "odme_update_damping",
            "odme_multiplier_min": "odme_multiplier_min",
            "odme_multiplier_max": "odme_multiplier_max",
            "gls_category_variance_primary": "gls_category_variance_primary",
            "gls_category_variance_low_volume": "gls_category_variance_low_volume",
            "gls_category_variance_hq_adjacent": "gls_category_variance_hq_adjacent",
            "gls_category_variance_all_links": "gls_category_variance_all_links",
            "gls_path_dilution_power": "gls_path_dilution_power",
            "gls_seed_implausibility_floor": "gls_seed_implausibility_floor",
            "gls_od_multiplier_min": "gls_od_multiplier_min",
            "gls_od_multiplier_max": "gls_od_multiplier_max",
            "screenline_group_min_jaccard": "screenline_group_min_jaccard",
            "screenline_group_min_cosine": "screenline_group_min_cosine",
            "zero_seed_ratio_threshold": "zero_seed_ratio_threshold",
            "zero_seed_abs_flow_threshold": "zero_seed_abs_flow_threshold",
        }
        for workbook_key, param_key in setting_map.items():
            row = table[keys == workbook_key]
            if not row.empty and value_col in table.columns:
                value = pd.to_numeric(row.iloc[0][value_col], errors="coerce")
                if pd.notna(value):
                    if hasattr(self.config, param_key):
                        current = getattr(self.config, param_key)
                        if isinstance(current, int) and not isinstance(current, bool):
                            setattr(self.config, param_key, int(value))
                        else:
                            setattr(self.config, param_key, float(value))
                    params[param_key] = float(value)

        beta_selection_mode = setting_value("beta_selection_mode")
        if beta_selection_mode:
            self.config.beta_selection_mode = beta_selection_mode.strip().lower()
        beta_candidates = setting_value("beta_candidates")
        if beta_candidates:
            parsed_candidates = tuple(
                float(value.strip())
                for value in beta_candidates.split(",")
                if value.strip()
            )
            if parsed_candidates and all(value > 0 for value in parsed_candidates):
                self.config.beta_candidates = parsed_candidates
            else:
                raise ValueError(
                    "beta_candidates must be a comma-separated list of positive values."
                )
        params["max_iter_odme"] = int(self.config.max_iter_odme)
        params["tol_odme"] = float(self.config.tol_odme)

        return params

    def load_aadt_targets(self) -> pd.DataFrame:
        if self.config.master_gpkg.exists():
            links = gpd.read_file(
                self.config.master_gpkg,
                layer=self.config.road_layer,
                engine="pyogrio",
            )
            required = {"link_code", self.config.aadt_value_field}
            missing = required - set(links.columns)
            if missing:
                raise ValueError(
                    f"{self.config.master_gpkg} layer {self.config.road_layer} "
                    f"is missing required OD calibration fields: {sorted(missing)}"
                )

            if "road_refno" not in links.columns:
                links["road_refno"] = links["link_code"].astype(
                    str).str.split("-").str[0]
            if "link_name" not in links.columns:
                links["link_name"] = links["link_code"].astype(str)

            links["aadt_pcu"] = pd.to_numeric(
                links[self.config.aadt_value_field], errors="coerce"
            )
            links = links[links["aadt_pcu"].notna()
                          & (links["aadt_pcu"] > 0)].copy()
            links = self.annotate_aadt_target_selection(links)

            aadt = pd.DataFrame(links.drop(columns="geometry"))
            aadt["aadt_period"] = self.config.aadt_source_period
            aadt["aadt_source"] = (
                f"{self.config.master_gpkg.name}:{self.config.road_layer}:"
                f"{self.config.aadt_value_field}"
            )
            aadt["aadt_units"] = self.config.aadt_units
            aadt["station_no"] = np.arange(1, len(aadt) + 1)
            aadt["road_link_id"] = aadt["link_code"].astype(str)
            aadt["location"] = aadt["link_name"].fillna(aadt["link_code"]).astype(str)
            aadt["link_ref"] = aadt["road_refno"].astype(str)
            aadt["route_ref"] = aadt["road_refno"].astype(str)
            aadt["route_ref_source"] = "Nepal.gpkg:NH:road_refno"
            logger.info(
                "Loaded %d positive %s rows from %s layer %s field %s; %d are objective ODME targets.",
                len(aadt),
                self.config.aadt_units,
                self.config.master_gpkg,
                self.config.road_layer,
                self.config.aadt_value_field,
                int(aadt["use_for_calibration"].sum()),
            )
            return aadt.reset_index(drop=True)

        raise FileNotFoundError(
            f"Master GeoPackage not found: {self.config.master_gpkg}. "
            "OD calibration requires Nepal.gpkg layer NH."
        )

    def annotate_aadt_target_selection(self, links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        links = links.copy()
        if not self.config.use_district_screenline_targets:
            links["is_district_screenline"] = True
            links["screenline_distance_m"] = 0.0
            links["screenline_buffer_m"] = np.nan
            links["hq_distance_m"] = np.nan
            links["screenline_min_hq_distance_m"] = np.nan
            links["target_class"] = "all_observed_links"
            links["aadt_status"] = f"observed_{self.config.aadt_source_period}"
            links["aadt_status_weight"] = self.config.observed_aadt_weight
            links["screenline_category_variance_factor"] = (
                self.config.gls_category_variance_all_links
            )
            links["use_for_calibration"] = True
            links["target_selection_note"] = (
                "District-screenline filtering is disabled; every positive "
                "AADT row is used as an objective ODME target."
            )
            return links

        try:
            boundaries = gpd.read_file(
                self.config.master_gpkg,
                layer=self.config.district_boundary_layer,
                engine="pyogrio",
            )
        except Exception as exc:
            raise ValueError(
                f"District-screenline target selection requires layer "
                f"{self.config.district_boundary_layer!r} in {self.config.master_gpkg}."
            ) from exc

        boundaries = boundaries[boundaries.geometry.notna()
                                & ~boundaries.geometry.is_empty].copy()
        if boundaries.empty:
            raise ValueError(
                f"{self.config.master_gpkg} layer {self.config.district_boundary_layer} "
                "has no usable geometry for OD target selection."
            )
        if links.crs is None or boundaries.crs is None:
            raise ValueError(
                "Road and district-boundary layers must both have CRS metadata "
                "for district-screenline target selection."
            )

        roads_proj = links.to_crs(self.config.spatial_projection)
        boundary_proj = boundaries.to_crs(self.config.spatial_projection)
        boundary_union = boundary_proj.geometry.unary_union
        distance_m = roads_proj.geometry.distance(boundary_union)
        hq_distance_m = pd.Series(
            np.inf, index=links.index, dtype=float)
        min_hq_distance_m = float(self.config.screenline_min_hq_distance_m)
        if min_hq_distance_m > 0:
            try:
                hqs = gpd.read_file(
                    self.config.master_gpkg,
                    layer=self.config.district_hq_layer,
                    engine="pyogrio",
                )
                hqs = hqs[hqs.geometry.notna()
                          & ~hqs.geometry.is_empty].copy()
                if not hqs.empty and hqs.crs is not None:
                    hq_union = hqs.to_crs(
                        self.config.spatial_projection).geometry.unary_union
                    hq_distance_m = roads_proj.geometry.distance(hq_union)
                else:
                    logger.warning(
                        "District-HQ layer %s is empty or lacks CRS; HQ-distance filter is skipped.",
                        self.config.district_hq_layer,
                    )
            except Exception as exc:
                logger.warning(
                    "Could not apply HQ-distance screenline filter from layer %s: %s",
                    self.config.district_hq_layer,
                    exc,
                )
        buffer_m = float(self.config.screenline_buffer_m)
        is_screenline = distance_m <= buffer_m
        is_large_enough = links["aadt_pcu"] >= float(
            self.config.min_calibration_aadt)
        is_far_from_hq = hq_distance_m >= min_hq_distance_m
        primary_screenline = is_screenline & is_large_enough & is_far_from_hq
        hq_adjacent_screenline = is_screenline & is_large_enough & ~is_far_from_hq
        low_volume_screenline = is_screenline & ~is_large_enough
        included_soft_screenline = (
            primary_screenline | hq_adjacent_screenline | low_volume_screenline
        )

        links["screenline_distance_m"] = distance_m.astype(float)
        links["screenline_buffer_m"] = buffer_m
        links["hq_distance_m"] = hq_distance_m.astype(float)
        links["screenline_min_hq_distance_m"] = min_hq_distance_m
        links["is_district_screenline"] = is_screenline.astype(bool)
        links["target_class"] = np.select(
            [
                primary_screenline,
                hq_adjacent_screenline,
                low_volume_screenline,
                ~is_screenline,
            ],
            [
                "district_screenline_primary",
                "hq_adjacent_screenline_context",
                "district_screenline_low_volume_context",
                "interior_or_local_context",
            ],
            default="context",
        )
        links["aadt_status"] = np.select(
            [
                primary_screenline,
                hq_adjacent_screenline,
                low_volume_screenline,
                ~is_screenline,
            ],
            [
                f"screenline_observed_{self.config.aadt_source_period}",
                f"hq_adjacent_screenline_context_{self.config.aadt_source_period}",
                f"screenline_low_volume_context_{self.config.aadt_source_period}",
                f"interior_context_{self.config.aadt_source_period}",
            ],
            default=f"context_{self.config.aadt_source_period}",
        )
        links["aadt_status_weight"] = np.where(
            included_soft_screenline,
            float(self.config.screenline_target_weight),
            float(self.config.context_target_weight),
        )
        links["screenline_category_variance_factor"] = np.select(
            [
                primary_screenline,
                hq_adjacent_screenline,
                low_volume_screenline,
                ~is_screenline,
            ],
            [
                float(self.config.gls_category_variance_primary),
                float(self.config.gls_category_variance_hq_adjacent),
                float(self.config.gls_category_variance_low_volume),
                float(self.config.gls_category_variance_all_links),
            ],
            default=float(self.config.gls_category_variance_all_links),
        )
        links["use_for_calibration"] = links["aadt_status_weight"] > 0
        links["target_selection_note"] = np.where(
            primary_screenline,
            (
                "Primary district-screenline AADT target: link is within "
                f"{buffer_m:g} m of a district boundary and AADT is at least "
                f"{self.config.min_calibration_aadt:g}, while remaining at least "
                f"{min_hq_distance_m:g} m from a district-HQ area."
            ),
            np.where(
                included_soft_screenline,
                (
                    "Soft district-screenline AADT target: link is near a "
                    "district boundary but receives higher GLS variance because "
                    "it is low-volume or close to a district-HQ area."
                ),
                (
                    "Context AADT only: link is interior/local to a district and "
                    "is hard-excluded from district-level ODME calibration."
                ),
            ),
        )
        return links

    def load_impedance_matrix(self, zones: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        if self.config.prefer_impedance_csv:
            if not self.config.impedance_csv.exists():
                raise FileNotFoundError(
                    f"Configured impedance CSV is missing: "
                    f"{self.config.impedance_csv}"
                )
            matrix = pd.read_csv(self.config.impedance_csv, index_col=0)
            source = str(self.config.impedance_csv)
            logger.info("Using impedance matrix: %s", source)
            matrix = self.clean_impedance_matrix(matrix, source)
            aligned, report = self.align_impedance_to_zones(matrix, zones)
            return aligned.values.astype(float), report

        try:
            matrix = self.read_impedance_from_workbook()
            source = "workbook:ImpedanceMatrix"
        except Exception as workbook_exc:
            if not self.config.impedance_csv.exists():
                raise ValueError(
                    f"Could not read workbook ImpedanceMatrix ({workbook_exc}) and "
                    f"configured CSV does not exist: {self.config.impedance_csv}"
                ) from workbook_exc
            logger.warning(
                "Could not read workbook ImpedanceMatrix: %s. Reading configured CSV %s",
                workbook_exc,
                self.config.impedance_csv,
            )
            matrix = pd.read_csv(self.config.impedance_csv, index_col=0)
            source = str(self.config.impedance_csv)

        matrix = self.clean_impedance_matrix(matrix, source)
        aligned, report = self.align_impedance_to_zones(matrix, zones)
        return aligned.values.astype(float), report

    def read_impedance_from_workbook(self) -> pd.DataFrame:
        return pd.read_excel(
            self.workbook_path,
            sheet_name="ImpedanceMatrix",
            header=2,
            index_col=0,
        )

    def clean_impedance_matrix(self, matrix: pd.DataFrame, source: str) -> pd.DataFrame:
        matrix = matrix.dropna(how="all").dropna(axis=1, how="all").copy()
        matrix.index = matrix.index.map(lambda x: str(x).strip())
        matrix.columns = matrix.columns.map(lambda x: str(x).strip())

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Impedance matrix from {source} is not square: {matrix.shape}")

        if set(matrix.index) == set(matrix.columns) and list(matrix.index) != list(matrix.columns):
            matrix = matrix.loc[matrix.index, matrix.index]

        numeric = matrix.apply(pd.to_numeric, errors="coerce")
        bad_cells = int(numeric.isna().sum().sum())
        if bad_cells:
            raise ValueError(
                f"Impedance matrix from {source} has {bad_cells} non-numeric cells after header parsing."
            )
        if not np.isfinite(numeric.values).all():
            raise ValueError(
                f"Impedance matrix from {source} contains infinite values.")

        np.fill_diagonal(numeric.values, 0.0)
        logger.info("Loaded impedance matrix from %s with shape %s.",
                    source, numeric.shape)
        return numeric

    def align_impedance_to_zones(
        self,
        matrix: pd.DataFrame,
        zones: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        label_to_district = {label: self.extract_matrix_district(
            label) for label in matrix.index}
        district_to_labels: Dict[str, List[str]] = {}
        for label, district in label_to_district.items():
            district_to_labels.setdefault(
                self.canonical_district(district), []).append(label)

        matched_labels: List[str] = []
        records: List[Dict[str, str]] = []
        hard_failures: List[str] = []

        for _, zone in zones.iterrows():
            zone_id = str(zone["zone_id"])
            district_name = str(zone["district_name"]).strip()
            canonical_zone = self.canonical_name(district_name)
            target_key = self.canonical_district(district_name)
            candidates = district_to_labels.get(target_key, [])

            chosen = ""
            status = "matched"
            note = ""
            if len(candidates) == 1:
                chosen = candidates[0]
                if target_key != canonical_zone:
                    status = "alias_match"
                    note = f"Used matrix district '{label_to_district[chosen]}' for zone district '{district_name}'."
            elif len(candidates) > 1:
                preferred_text = self.PREFERRED_DUPLICATE_LABELS.get(
                    canonical_zone)
                preferred = [
                    label for label in candidates
                    if preferred_text and preferred_text.lower() in label.lower()
                ]
                if len(preferred) == 1:
                    chosen = preferred[0]
                    status = "preferred_duplicate"
                    note = (
                        f"Multiple matrix labels for '{label_to_district[chosen]}'; "
                        f"selected '{chosen}'."
                    )
                else:
                    status = "ambiguous"
                    note = f"Candidates: {candidates}"
                    hard_failures.append(f"{zone_id} {district_name}: {note}")
            else:
                status = "missing"
                note = f"No matrix label found for canonical district '{target_key}'."
                hard_failures.append(f"{zone_id} {district_name}: {note}")

            matched_labels.append(chosen)
            records.append(
                {
                    "zone_id": zone_id,
                    "district_name": district_name,
                    "matched_matrix_label": chosen,
                    "matched_matrix_district": label_to_district.get(chosen, ""),
                    "status": status,
                    "note": note,
                }
            )

        report = pd.DataFrame(records)
        reused = report["matched_matrix_label"][report["matched_matrix_label"] != ""]
        reused_counts = Counter(reused)
        for label, count in reused_counts.items():
            if count > 1:
                report.loc[report["matched_matrix_label"] == label, "status"] = (
                    report.loc[report["matched_matrix_label"]
                               == label, "status"]
                    .replace("alias_match", "shared_label_reused")
                )
                logger.warning(
                    "Matrix label '%s' is reused for %d zones. Duplicate-label handling is applied.",
                    label,
                    count,
                )

        unused_labels = sorted(set(matrix.index) - set(reused))
        if unused_labels:
            logger.warning("Unused impedance matrix labels: %s", unused_labels)

        if hard_failures:
            details = "; ".join(hard_failures)
            raise ValueError(
                f"Could not align impedance matrix to zones: {details}")

        problematic = report[report["status"].isin(
            ["shared_label_reused", "preferred_duplicate"])]
        if not problematic.empty and not self.config.allow_duplicate_impedance_labels:
            raise ValueError(
                "Impedance alignment requires duplicate-label handling. See impedance_alignment_report.csv."
            )

        aligned = matrix.loc[matched_labels, matched_labels].copy()
        zone_ids = zones["zone_id"].tolist()
        aligned.index = zone_ids
        aligned.columns = zone_ids
        np.fill_diagonal(aligned.values, 0.0)

        return aligned, report

    @staticmethod
    def extract_matrix_district(label: str) -> str:
        text = str(label).strip()
        if "|" in text:
            return text.split("|", 1)[0].strip()
        match = re.search(r"\(([^()]*)\)\s*$", text)
        if match:
            return match.group(1).strip()
        return text

    @staticmethod
    def canonical_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

    @classmethod
    def canonical_district(cls, value: str) -> str:
        canonical = cls.canonical_name(value)
        return cls.DISTRICT_ALIASES.get(canonical, canonical)

    @staticmethod
    def trip_vector(zones: pd.DataFrame, candidates: List[str], label: str) -> np.ndarray:
        for col in candidates:
            if col in zones.columns:
                values = pd.to_numeric(zones[col], errors="coerce")
                if values.notna().all() and float(values.sum()) > 0:
                    logger.info("Using '%s' column for %s.", col, label)
                    return values.to_numpy(dtype=float)
        logger.warning(
            "No usable %s column found. Using 1000 trips per zone.", label)
        return np.ones(len(zones), dtype=float) * 1000.0

    def district_activity_vector(self, zones: pd.DataFrame) -> np.ndarray:
        """Return the symmetric economic activity prior A_i = P_i * gp(i)."""
        required = {"Population", "GDP per capita NPR"}
        if required.issubset(zones.columns):
            population = pd.to_numeric(zones["Population"], errors="coerce")
            gdp_per_capita = pd.to_numeric(
                zones["GDP per capita NPR"], errors="coerce")
            activity = population * gdp_per_capita
            valid = activity.notna() & np.isfinite(activity) & (activity > 0)
            if valid.all() and float(activity.sum()) > 0:
                if "activity_proxy" in zones.columns:
                    recorded = pd.to_numeric(
                        zones["activity_proxy"], errors="coerce")
                    if recorded.notna().all():
                        mismatch = np.nanmax(
                            np.abs(recorded.to_numpy(dtype=float) - activity.to_numpy(dtype=float))
                            / np.maximum(activity.to_numpy(dtype=float), 1.0)
                        )
                        if mismatch > 1e-6:
                            logger.warning(
                                "Workbook activity_proxy differs from Population * GDP per capita NPR "
                                "(max relative difference %.6g). Using the explicit GDP activity formula.",
                                float(mismatch),
                            )
                logger.info(
                    "Using symmetric GDP activity prior: A_i = Population_i * provincial GDP per capita_i."
                )
                return activity.to_numpy(dtype=float)

        logger.warning(
            "Population and GDP per capita NPR were not both usable. Falling back to workbook activity columns."
        )
        return self.trip_vector(
            zones,
            ["activity_proxy", "Oi_trips", "Di_trips", "productions", "attractions"],
            "relative symmetric activity weights",
        )

    @staticmethod
    def symmetrize_od_matrix(matrix: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        np.fill_diagonal(symmetric, 0.0)
        return symmetric

    @staticmethod
    def balance_totals(productions: np.ndarray, attractions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prod_total = float(productions.sum())
        attr_total = float(attractions.sum())
        if prod_total <= 0 or attr_total <= 0:
            raise ValueError(
                "Production and attraction totals must both be positive.")
        if not np.isclose(prod_total, attr_total, rtol=1e-6, atol=1e-6):
            factor = prod_total / attr_total
            logger.warning(
                "Production total %.2f differs from attraction total %.2f. Scaling attractions by %.8f.",
                prod_total,
                attr_total,
                factor,
            )
            attractions = attractions * factor
        return productions, attractions

    def run_district_od_estimation(
        self,
        seed_od: np.ndarray,
        zones: pd.DataFrame,
        aadt: pd.DataFrame,
        params: Dict[str, float],
        alignment_report: pd.DataFrame,
        production_weights: np.ndarray,
        attraction_weights: np.ndarray,
        impedance: np.ndarray,
        configured_beta: float,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if aadt.empty:
            note = "AADT source has no usable numeric count rows; final matrix equals gravity seed."
            logger.warning(note)
            return seed_od.copy(), seed_od.copy(), self.single_log_row(seed_od, note), pd.DataFrame(), pd.DataFrame()

        calibrator = DistrictODMECalibrator(self.config)
        graph, edges = calibrator.load_network()
        zone_nodes = calibrator.zone_nodes_from_alignment(
            zones, alignment_report)
        od_paths = calibrator.compute_shortest_path_features(
            graph, zones["zone_id"].tolist(), zone_nodes, seed_od)
        if self.config.write_diagnostics:
            od_paths_path = self.config.output_dir / "od_pair_path_features.csv"
            od_paths.to_csv(od_paths_path, index=False)
            logger.info("Wrote OD path feature report: %s", od_paths_path)

        targets, mapping_report = calibrator.build_calibration_targets(
            aadt, edges)
        if self.config.write_diagnostics:
            mapping_path = self.config.output_dir / "aadt_mapping_report.csv"
            mapping_report.to_csv(mapping_path, index=False)
            logger.info("Wrote AADT mapping report: %s", mapping_path)

        if targets.empty:
            note = "No AADT target could be mapped to the graph; final matrix equals gravity seed."
            logger.warning(note)
            return seed_od.copy(), seed_od.copy(), self.single_log_row(seed_od, note), pd.DataFrame(), mapping_report

        final_od, selected_seed_od, log_df, target_fit = calibrator.calibrate(
            seed_od=seed_od,
            od_paths=od_paths,
            targets=targets,
            max_iter=max(1000, int(params.get("max_iter_odme",
                         self.config.max_iter_odme))),
            tol=float(params.get("tol_odme", self.config.tol_odme)),
            production_weights=production_weights,
            attraction_weights=attraction_weights,
            impedance=impedance,
            configured_beta=configured_beta,
        )

        if self.config.write_diagnostics:
            targets_path = self.config.output_dir / "assignment_targets.csv"
            targets.to_csv(targets_path, index=False)
            logger.info("Wrote assignment target definitions: %s", targets_path)

        return final_od, selected_seed_od, log_df, target_fit, mapping_report

    def write_named_matrix_output(
        self,
        zones: pd.DataFrame,
        alignment_report: pd.DataFrame,
        matrix: np.ndarray,
        filename: str,
        log_label: str,
    ) -> None:
        labels = self.matrix_output_labels(zones, alignment_report).reset_index(
            drop=True
        )
        output_order = labels.sort_values(
            ["sort_key", "human_label"], kind="mergesort"
        ).index.to_numpy()
        ordered_labels = labels.iloc[output_order].reset_index(drop=True)
        human_labels = ordered_labels["human_label"].tolist()
        decimals = self.config.human_od_decimals
        ordered_matrix = matrix[np.ix_(output_order, output_order)]
        named = pd.DataFrame(
            ordered_matrix,
            index=human_labels,
            columns=human_labels,
        )
        output_path = self.config.output_dir / filename
        named.to_csv(
            output_path,
            index_label="origin_zone",
            float_format=self.config.matrix_output_float_format,
        )
        logger.info("Wrote %s: %s", log_label, output_path)
        if self.config.write_rounded_matrix_outputs:
            rounded_path = output_path.with_name(
                f"{output_path.stem}_rounded{output_path.suffix}"
            )
            pd.DataFrame(
                np.round(ordered_matrix, decimals),
                index=human_labels,
                columns=human_labels,
            ).to_csv(
                rounded_path,
                index_label="origin_zone",
                float_format=f"%.{decimals}f",
            )
            logger.info("Wrote rounded %s: %s", log_label, rounded_path)

    def matrix_output_labels(
        self,
        zones: pd.DataFrame,
        alignment_report: pd.DataFrame,
    ) -> pd.DataFrame:
        return (
            zones[["zone_id", "district_name"]]
            .merge(
                alignment_report[["zone_id", "matched_matrix_label"]],
                on="zone_id",
                how="left",
            )
            .assign(
                hq_display=lambda df: df["matched_matrix_label"].map(
                    self.hq_display_label
                ),
                human_label=lambda df: (
                    df["district_name"].astype(str)
                    + " | "
                    + df["hq_display"].astype(str)
                ),
                sort_key=lambda df: (
                    df["district_name"]
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[^a-z0-9]+", " ", regex=True)
                    .str.strip()
                    + " | "
                    + df["hq_display"]
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[^a-z0-9]+", " ", regex=True)
                    .str.strip()
                ),
            )
        )

    def write_named_od_outputs(
        self,
        zones: pd.DataFrame,
        alignment_report: pd.DataFrame,
        seed_od: np.ndarray,
        final_od: np.ndarray,
    ) -> None:
        labels = self.matrix_output_labels(zones, alignment_report).reset_index(
            drop=True
        )
        output_order = labels.sort_values(
            ["sort_key", "human_label"], kind="mergesort"
        ).index.to_numpy()
        ordered_labels = labels.iloc[output_order].reset_index(drop=True)
        human_labels = ordered_labels["human_label"].tolist()
        zone_ids = labels["zone_id"].astype(str).tolist()

        decimals = self.config.human_od_decimals
        precise_float_format = self.config.matrix_output_float_format
        rounded_float_format = f"%.{decimals}f"
        ordered_final = final_od[np.ix_(output_order, output_order)]
        ordered_seed = seed_od[np.ix_(output_order, output_order)]

        final_named = pd.DataFrame(
            ordered_final,
            index=human_labels,
            columns=human_labels,
        )
        final_named_path = self.config.output_dir / "od_matrix.csv"
        final_named.to_csv(
            final_named_path,
            index_label="origin_zone",
            float_format=precise_float_format,
        )
        logger.info("Wrote named final OD matrix: %s", final_named_path)

        seed_named = pd.DataFrame(
            ordered_seed,
            index=human_labels,
            columns=human_labels,
        )
        seed_named_path = self.config.output_dir / "seed_od_matrix.csv"
        seed_named.to_csv(
            seed_named_path,
            index_label="origin_zone",
            float_format=precise_float_format,
        )
        logger.info("Wrote named count-scaled seed OD matrix: %s", seed_named_path)

        if self.config.write_rounded_matrix_outputs:
            final_rounded_path = self.config.output_dir / "od_matrix_rounded.csv"
            pd.DataFrame(
                np.round(ordered_final, decimals),
                index=human_labels,
                columns=human_labels,
            ).to_csv(
                final_rounded_path,
                index_label="origin_zone",
                float_format=rounded_float_format,
            )
            logger.info("Wrote rounded final OD matrix: %s", final_rounded_path)

            seed_rounded_path = self.config.output_dir / "seed_od_matrix_rounded.csv"
            pd.DataFrame(
                np.round(ordered_seed, decimals),
                index=human_labels,
                columns=human_labels,
            ).to_csv(
                seed_rounded_path,
                index_label="origin_zone",
                float_format=rounded_float_format,
            )
            logger.info(
                "Wrote rounded count-scaled seed OD matrix: %s", seed_rounded_path
            )

        if self.config.write_pair_table:
            pair_rows: List[Dict[str, object]] = []
            for i, origin_zone in enumerate(zone_ids):
                for j in range(i + 1, len(zone_ids)):
                    destination_zone = zone_ids[j]
                    pair_rows.append(
                        {
                            "zone_a_id": origin_zone,
                            "zone_a_district": labels.iloc[i]["district_name"],
                            "zone_a_hq_label": labels.iloc[i]["matched_matrix_label"],
                            "zone_b_id": destination_zone,
                            "zone_b_district": labels.iloc[j]["district_name"],
                            "zone_b_hq_label": labels.iloc[j]["matched_matrix_label"],
                            "gravity_seed_per_day": float(seed_od[i, j]),
                            "final_symmetric_odme_per_day": float(final_od[i, j]),
                            "delta_per_day": float(final_od[i, j] - seed_od[i, j]),
                        }
                    )
            pairs_path = self.config.output_dir / "od_pairs_names.csv"
            pd.DataFrame(pair_rows).to_csv(
                pairs_path,
                index=False,
                float_format=precise_float_format,
            )
            logger.info("Wrote named OD pair table: %s", pairs_path)

    @staticmethod
    def hq_display_label(matrix_label: object) -> str:
        text = str(matrix_label or "").strip()
        if "|" in text:
            text = text.split("|", 1)[1].strip()
        return re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()

    def write_run_summary_report(
        self,
        zones: pd.DataFrame,
        params: Dict[str, object],
        calib_log: pd.DataFrame,
        target_fit: pd.DataFrame,
        mapping_report: pd.DataFrame,
        final_od: np.ndarray,
    ) -> None:
        summary_path = self.config.output_dir / "od_synthesis_summary.txt"
        first = calib_log.iloc[0] if not calib_log.empty else pd.Series(
            dtype=object)
        last = calib_log.iloc[-1] if not calib_log.empty else pd.Series(
            dtype=object)
        target_stats = self.fit_summary_stats(
            target_fit, "observed_count", "modelled_final")
        objective_target_count = 0
        if "used_for_calibration" in target_fit.columns:
            objective_target_count = int(
                target_fit["used_for_calibration"].fillna(False).astype(bool).sum()
            )

        matched_count = 0
        unmatched_count = 0
        active_aadt_count = 0
        link_status_lines: List[str] = []
        if not mapping_report.empty:
            matched_count = int(
                mapping_report["matched"].fillna(False).astype(bool).sum())
            unmatched_count = int(len(mapping_report) - matched_count)
            if "use_for_calibration" in mapping_report.columns:
                active_aadt_count = int(
                    mapping_report["use_for_calibration"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )
            if "link_code_match_status" in mapping_report.columns:
                for status, count in mapping_report["link_code_match_status"].fillna("blank").value_counts().items():
                    link_status_lines.append(f"  - {status}: {count}")

        target_mode = str(last.get("target_mode", "unknown"))
        scale_factor = float(last.get("prior_scale_factor", np.nan))
        lines = [
            "OD synthesis methodology summary",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Methodology",
            f"- Zones: {len(zones)}",
            f"- Gravity model beta: {params.get('beta', self.config.beta)} per hour because impedance is measured in hours.",
            "- Seed OD: doubly constrained gravity model with exponential impedance deterrence, balanced by Furness / iterative proportional fitting (IPF).",
            f"- AADT-derived gravity-prior scale: {scale_factor:.8g}",
            f"- Calibration: symmetric district-level weighted GLS ODME with observed two-way {self.config.aadt_units} AADT and an entropy prior around the gravity matrix.",
            f"- Calibration target mode: {target_mode}",
            f"- AADT coefficient of variation: {self.config.aadt_count_cv}",
            f"- District OD log-prior standard deviation: {self.config.district_prior_log_std}",
            f"- Selected gravity beta: {last.get('selected_beta', np.nan)} per hour.",
            f"- Route choice: {self.config.route_choice_mode}, K={self.config.route_choice_k}",
            "",
            "District ODME Formula Notes",
            f"- Count equation: v_a = sum_r p[a,r] * x_r, using only observed two-way {self.config.aadt_units} AADT targets.",
            "- Calibration object: unordered district-pair variables mirrored into a symmetric OD matrix; district self-cells are excluded.",
            "- Each district-pair variable starts from its positive count-scaled gravity-prior value and is optimized as a positive log-multiplier.",
            "- AADT residuals are divided by GLS sigma, combining count CV, path dilution, target class variance and seed-implausibility factors.",
            "- Missing AADT sections are absent constraints; they are not zero-flow targets.",
            (
                f"- {self.config.aadt_source_period} {self.config.aadt_units} AADT from "
                f"{self.config.master_gpkg.name} layer {self.config.road_layer} "
                f"field {self.config.aadt_value_field} is matched to "
                "Nepal.gpkg-derived graph fragments by link_code."
            ),
            "",
            "References",
            "- Van Zuylen and Willumsen (1980), The most likely trip matrix estimated from traffic counts, Transportation Research B.",
            "- Spiess (1987), A maximum likelihood model for estimating origin-destination matrices, Transportation Research B.",
            "- Sherali, Narayanan and Sivanandan (2003), Estimation of origin-destination trip-tables based on a partial set of traffic link volumes, Transportation Research B.",
            "",
            "AADT Mapping",
            f"- AADT station rows: {len(mapping_report)}",
            f"- Rows matched to graph geometry: {matched_count}",
            f"- District-screenline AADT candidates: {active_aadt_count}",
            f"- Path-supported objective targets: {objective_target_count}",
            f"- Unmatched rows: {unmatched_count}",
        ]
        if link_status_lines:
            lines += ["- Link-code match status counts:", *link_status_lines]

        lines += [
            "",
            "Calibration Fit",
            f"- District ODME solver iterations: {int(last.get('solver_nit', last.get('iteration', 0)))}",
            f"- District ODME convergence flag: {bool(last.get('converged', False))}",
            f"- Initial RMSE: {float(first.get('rmse_counts', np.nan)):.2f}",
            f"- Final RMSE: {float(last.get('rmse_counts', np.nan)):.2f}",
            f"- Final weighted RMSE: {float(last.get('weighted_rmse_counts', np.nan)):.2f}",
            f"- Initial total {self.config.aadt_units}: {float(first.get('total_pcu_per_day', np.nan)):.3f}",
            f"- Final total {self.config.aadt_units}: {float(last.get('total_pcu_per_day', np.nan)):.3f}",
            f"- Final target RMSE: {target_stats['rmse']:.2f}",
            f"- Final target MAE: {target_stats['mae']:.2f}",
            f"- Final target MAPE: {target_stats['mape_pct']:.2f}%",
            f"- Targets within 10%: {target_stats['within_10pct']} of {target_stats['rows']}",
            f"- Targets within 25%: {target_stats['within_25pct']} of {target_stats['rows']}",
            f"- Targets within 50%: {target_stats['within_50pct']} of {target_stats['rows']}",
            "",
            "Important Limitation",
            (
                f"- {self.config.master_gpkg.name} {self.config.aadt_value_field} is non-directional. Each observed link_code "
                "is matched to all GraphEdges fragments derived from "
                "Nepal.gpkg layer NH, so calibration uses a two-way section count and a symmetric OD matrix."
            ),
            "",
            "Main Outputs",
            "- od_matrix.csv: final OD matrix using district/HQ names; this is the MLPI input.",
            "- calibration_log.csv: compact optimizer, objective-component and prior-scale audit.",
            "- aadt_target_fit.csv: final fit audit for every objective screenline candidate, including path-support status.",
            "- od_matrix_ids.csv: optional raw ID-labelled matrix when write_id_matrix or write_diagnostics is enabled.",
            "- Detailed seed, pair, path-incidence, mapping, station-fit and assignment-target tables are written only when their workbook switches are enabled.",
            "",
            f"Final OD total {self.config.aadt_units}: {float(np.sum(final_od)):.3f}",
            f"Final OD cells above 1 {self.config.aadt_units}: {int(np.sum(final_od > 1.0))}",
            f"Final OD cells above 10 {self.config.aadt_units}: {int(np.sum(final_od > 10.0))}",
            f"Origin rows at or below 0.001 {self.config.aadt_units}: {int(np.sum(final_od.sum(axis=1) <= 0.001))}",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote OD synthesis summary: %s", summary_path)

    @staticmethod
    def fit_summary_stats(df: pd.DataFrame, observed_col: str, modelled_col: str) -> Dict[str, float]:
        empty_stats = {
            "rows": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape_pct": float("nan"),
            "within_10pct": 0,
            "within_25pct": 0,
            "within_50pct": 0,
        }
        if df.empty or observed_col not in df.columns or modelled_col not in df.columns:
            return empty_stats

        if "used_for_calibration" in df.columns:
            df = df[df["used_for_calibration"].fillna(False).astype(bool)].copy()
        if df.empty:
            return empty_stats

        observed = pd.to_numeric(df[observed_col], errors="coerce")
        modelled = pd.to_numeric(df[modelled_col], errors="coerce")
        valid = observed.notna() & modelled.notna() & (observed > 0)
        if not valid.any():
            return empty_stats

        errors = modelled[valid] - observed[valid]
        abs_pct = (errors.abs() / observed[valid]).astype(float)
        return {
            "rows": int(valid.sum()),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(errors.abs().mean()),
            "mape_pct": float(abs_pct.mean() * 100),
            "within_10pct": int((abs_pct <= 0.10).sum()),
            "within_25pct": int((abs_pct <= 0.25).sum()),
            "within_50pct": int((abs_pct <= 0.50).sum()),
        }

    def write_station_fit_report(self, mapping_report: pd.DataFrame, target_fit: pd.DataFrame) -> None:
        fit_cols = [
            "target_key",
            "observed_count",
            "used_for_calibration",
            "calibration_status",
            "constraint_drop_reason",
            "path_pair_count",
            "seed_to_observed_ratio",
            "modelled_raw_gravity_seed",
            "modelled_count_scaled_prior",
            "modelled_final",
            "pct_error_final",
            "target_mode",
        ]
        available_fit_cols = [
            col for col in fit_cols if col in target_fit.columns]
        station_fit = mapping_report.merge(
            target_fit[available_fit_cols],
            on="target_key",
            how="left",
            suffixes=("", "_target"),
        )
        station_fit["station_error_final"] = station_fit["modelled_final"] - \
            station_fit["aadt_pcu"]
        station_fit["station_pct_error_final"] = np.divide(
            station_fit["station_error_final"],
            np.maximum(pd.to_numeric(
                station_fit["aadt_pcu"], errors="coerce"), self.config.min_model_flow),
        )
        drop_reason = (
            station_fit.get("constraint_drop_reason", pd.Series("", index=station_fit.index))
            .fillna("")
            .astype(str)
        )
        station_fit["fit_warning"] = np.where(
            station_fit["modelled_final"].isna(),
            np.where(
                station_fit.get("use_for_calibration", False),
                "Observed link had no routed HQ OD path.",
                "Context-only AADT: reported as supporting evidence and excluded from ODME constraints.",
            ),
            np.where(
                station_fit.get("used_for_calibration", False),
                "",
                np.where(
                    drop_reason.str.strip() != "",
                    drop_reason,
                    "No routed HQ-to-HQ path crosses this otherwise objective screenline target.",
                ),
            ),
        )
        station_fit_path = self.config.output_dir / "aadt_station_fit.csv"
        station_fit.to_csv(station_fit_path, index=False)
        logger.info("Wrote station-level AADT fit report: %s",
                    station_fit_path)

    @staticmethod
    def single_log_row(seed_od: np.ndarray, note: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "iteration": 0,
                    "rmse_counts": np.nan,
                    "mae_counts": np.nan,
                    "r2_counts": np.nan,
                    "relative_rmse": np.nan,
                    "mean_geh": np.nan,
                    "share_geh_under_5": np.nan,
                    "total_pcu_per_day": float(seed_od.sum()),
                    "demand_units": "AADT/day",
                    "converged": False,
                    "target_mode": "none",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "notes": note,
                }
            ]
        )


class DistrictODMECalibrator:
    def __init__(self, config: Config):
        self.config = config

    def load_network(self) -> Tuple[nx.DiGraph, gpd.GeoDataFrame]:
        if not self.config.graph_edges_gpkg.exists():
            raise FileNotFoundError(
                f"Missing graph edges file: {self.config.graph_edges_gpkg}")

        edges = gpd.read_file(self.config.graph_edges_gpkg,
                              engine="pyogrio").copy()
        required = {"u", "v", self.config.assignment_weight_field}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(
                f"GraphEdges.gpkg is missing required columns: {sorted(missing)}")

        edges["edge_id"] = np.arange(1, len(edges) + 1, dtype=np.int64)
        edges["edge_key"] = edges["u"].astype(
            str) + "->" + edges["v"].astype(str)
        edges["ref"] = edges.get("ref", "").astype(str).str.strip()
        edges[self.config.assignment_weight_field] = pd.to_numeric(
            edges[self.config.assignment_weight_field], errors="coerce"
        )
        edges = edges[edges[self.config.assignment_weight_field].notna()
                      ].copy()
        edges = edges[edges[self.config.assignment_weight_field] > 0].copy()

        graph = nx.DiGraph()
        for row in edges.itertuples(index=False):
            u = str(row.u)
            v = str(row.v)
            weight = float(getattr(row, self.config.assignment_weight_field))
            existing = graph.get_edge_data(u, v)
            if existing is None or weight < existing[self.config.assignment_weight_field]:
                graph.add_edge(
                    u,
                    v,
                    edge_id=int(row.edge_id),
                    edge_key=str(row.edge_key),
                    ref=str(row.ref).strip(),
                    weight=weight,
                    **{self.config.assignment_weight_field: weight},
                )

        logger.info(
            "Loaded assignment graph: nodes=%d, directed_edges=%d, source_edges=%d.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(edges),
        )
        return graph, edges

    def zone_nodes_from_alignment(self, zones: pd.DataFrame, alignment_report: pd.DataFrame) -> Dict[str, str]:
        if not self.config.projected_hqs_gpkg.exists():
            raise FileNotFoundError(
                f"Missing projected HQs file: {self.config.projected_hqs_gpkg}")

        hqs = gpd.read_file(self.config.projected_hqs_gpkg, engine="pyogrio")
        hq_to_node = {
            str(row.hq_label).strip(): self.node_key_to_graph_node(str(row.node_key))
            for row in hqs.itertuples(index=False)
        }

        merged = zones[["zone_id", "district_name"]].merge(
            alignment_report[["zone_id", "matched_matrix_label"]],
            on="zone_id",
            how="left",
        )
        zone_nodes: Dict[str, str] = {}
        missing: List[str] = []
        for row in merged.itertuples(index=False):
            label = str(row.matched_matrix_label).strip()
            node = hq_to_node.get(label)
            if node:
                zone_nodes[str(row.zone_id)] = node
            else:
                missing.append(f"{row.zone_id}:{row.district_name}->{label}")

        if missing:
            raise ValueError(
                "Could not map these zones to projected HQ graph nodes: " + "; ".join(missing))

        return zone_nodes

    @staticmethod
    def node_key_to_graph_node(node_key: str) -> str:
        value = ast.literal_eval(str(node_key))
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(f"Unexpected ProjectedHQ node_key: {node_key}")
        return f"{value[0]}_{value[1]}"

    def compute_shortest_path_features(
        self,
        graph: nx.DiGraph,
        zone_ids: Sequence[str],
        zone_nodes: Dict[str, str],
        seed_od: np.ndarray,
    ) -> pd.DataFrame:
        """Build one fractional link-incidence representation per OD pair.

        The default uses a small logit set of time-based alternatives.  It is
        deliberately kept separate from demand estimation: route shares are
        fixed during a static daily assignment, while OD demand is estimated
        from the count evidence.
        """
        records: List[Dict[str, object]] = []
        mode = str(self.config.route_choice_mode).strip().lower()
        if mode not in {"single_shortest_path", "logit_k_shortest"}:
            raise ValueError(
                "route_choice_mode must be 'single_shortest_path' or "
                "'logit_k_shortest'."
            )
        alternative_count = max(1, int(self.config.route_choice_k))
        max_cost_factor = max(1.0, float(self.config.route_choice_max_cost_factor))

        def route_record(path_nodes: Sequence[str]) -> Dict[str, object]:
            edge_keys: List[str] = []
            refs: Set[str] = set()
            cost = 0.0
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = graph.get_edge_data(u, v)
                if not edge_data:
                    raise ValueError(f"Assignment graph lacks edge {u}->{v}.")
                edge_keys.append(str(edge_data.get("edge_key", f"{u}->{v}")))
                ref = str(edge_data.get("ref", "")).strip()
                if ref:
                    refs.add(ref)
                cost += float(edge_data[self.config.assignment_weight_field])
            return {
                "edge_keys": edge_keys,
                "refs": sorted(refs),
                "cost_hours": cost,
            }

        for i, origin_zone in enumerate(zone_ids):
            source = zone_nodes[str(origin_zone)]
            distances: Dict[str, float] = {}
            paths: Dict[str, List[str]] = {}
            if mode == "single_shortest_path":
                try:
                    distances, paths = nx.single_source_dijkstra(
                        graph,
                        source,
                        weight=self.config.assignment_weight_field,
                    )
                except nx.NetworkXNoPath:
                    distances, paths = {}, {}

            for j, destination_zone in enumerate(zone_ids):
                if i == j:
                    continue
                demand = float(seed_od[i, j])
                if demand <= 0:
                    continue

                target = zone_nodes[str(destination_zone)]
                alternatives: List[Dict[str, object]] = []
                if mode == "single_shortest_path":
                    path_nodes = paths.get(target)
                    if path_nodes:
                        alternatives = [route_record(path_nodes)]
                else:
                    try:
                        candidates = nx.shortest_simple_paths(
                            graph,
                            source,
                            target,
                            weight=self.config.assignment_weight_field,
                        )
                        for path_nodes in itertools.islice(candidates, alternative_count):
                            candidate = route_record(path_nodes)
                            if alternatives and candidate["cost_hours"] > (
                                float(alternatives[0]["cost_hours"]) * max_cost_factor
                            ):
                                break
                            alternatives.append(candidate)
                    except nx.NetworkXNoPath:
                        alternatives = []

                if not alternatives:
                    records.append(
                        {
                            "origin": str(origin_zone),
                            "destination": str(destination_zone),
                            "seed_demand": demand,
                            "path_cost": np.nan,
                            "path_found": False,
                            "path_edge_keys": "",
                            "path_refs": "",
                            "route_count": 0,
                            "route_alternatives": "[]",
                        }
                    )
                    continue

                costs = np.asarray(
                    [float(route["cost_hours"]) for route in alternatives], dtype=float)
                utilities = -float(self.config.route_choice_theta_per_hour) * (
                    costs - float(costs.min()))
                shares = np.exp(np.clip(utilities, -700.0, 0.0))
                shares /= shares.sum()
                for route, share in zip(alternatives, shares):
                    route["share"] = float(share)

                records.append(
                    {
                        "origin": str(origin_zone),
                        "destination": str(destination_zone),
                        "seed_demand": demand,
                        "path_cost": float(costs.min()),
                        "path_found": True,
                        "path_edge_keys": "|".join(alternatives[0]["edge_keys"]),
                        "path_refs": "|".join(alternatives[0]["refs"]),
                        "route_count": len(alternatives),
                        "route_alternatives": json.dumps(alternatives),
                    }
                )

        od_paths = pd.DataFrame(records)
        found = int(od_paths["path_found"].sum()) if not od_paths.empty else 0
        mean_alternatives = (
            float(od_paths.loc[od_paths["path_found"], "route_count"].mean())
            if found else 0.0
        )
        logger.info(
            "Computed OD route sets: %d found, %d missing, mean alternatives %.2f (%s).",
            found,
            len(od_paths) - found,
            mean_alternatives,
            mode,
        )
        return od_paths

    def build_calibration_targets(
        self,
        aadt: pd.DataFrame,
        edges: gpd.GeoDataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        required_aadt = {"link_code", "aadt_status", "aadt_status_weight"}
        missing_aadt = required_aadt - set(aadt.columns)
        if missing_aadt:
            raise ValueError(
                "Nepal.gpkg NH AADT target table is missing required fields: "
                f"{sorted(missing_aadt)}"
            )

        if "code" not in edges.columns or not edges["code"].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError(
                "GraphEdges.gpkg must contain a populated 'code' field copied "
                "from Nepal.gpkg NH link_code. Rebuild impedance from Nepal.gpkg."
            )

        targets, mapping = self.build_nepal_nh_link_targets(aadt, edges)
        if not targets.empty:
            return targets, mapping

        raise ValueError(
            "Nepal.gpkg NH AADT rows loaded, but none matched GraphEdges by "
            "link_code. Rebuild impedance from Nepal.gpkg, or inspect "
            "GraphEdges.gpkg/code against Nepal.gpkg NH/link_code."
        )

    def build_nepal_nh_link_targets(
        self,
        aadt: pd.DataFrame,
        edges: gpd.GeoDataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Match Nepal.gpkg NH AADT directly to graph fragments by link_code."""
        work_edges = edges.copy()
        work_edges["link_code_norm"] = (
            work_edges["code"].fillna("").map(self.normalize_link_code)
        )
        edge_keys_by_code = (
            work_edges[work_edges["link_code_norm"] != ""]
            .groupby("link_code_norm")["edge_key"]
            .apply(lambda values: sorted(set(values.astype(str))))
            .to_dict()
        )

        target_rows: List[Dict[str, object]] = []
        mapping_rows: List[Dict[str, object]] = []
        for row in aadt.itertuples(index=False):
            link_code = str(getattr(row, "link_code", "")).strip()
            code_norm = self.normalize_link_code(link_code)
            edge_keys = edge_keys_by_code.get(code_norm, [])
            observed = float(getattr(row, "aadt_pcu", np.nan))
            status = str(getattr(row, "aadt_status", "")).strip().lower()
            status_weight = float(
                getattr(row, "aadt_status_weight", 0.0)
            )
            target_class = str(getattr(row, "target_class", "")).strip()
            is_screenline = bool(getattr(row, "is_district_screenline", False))
            screenline_distance_m = float(
                getattr(row, "screenline_distance_m", np.nan))
            hq_distance_m = float(getattr(row, "hq_distance_m", np.nan))
            geometry_matched = bool(edge_keys) and math.isfinite(observed)
            active = geometry_matched and observed > 0 and status_weight > 0
            target_key = f"nepal_nh_link::{code_norm}" if active else ""

            if active:
                note = (
                    "Observed Nepal.gpkg NH AADT matched directly to all "
                    "GraphEdges fragments carrying this unique link_code."
                )
            elif geometry_matched:
                note = (
                    f"Nepal.gpkg NH link matched, but AADT_status={status!r} is "
                    "contextual evidence for reporting and is excluded from "
                    "the objective ODME count term."
                )
            else:
                note = "Nepal.gpkg NH link_code did not match any GraphEdges fragment."

            mapping_rows.append(
                {
                    "station_no": getattr(row, "station_no", ""),
                    "road_link_id": link_code,
                    "location": getattr(row, "location", ""),
                    "aadt_pcu": observed,
                    "aadt_status": status,
                    "aadt_period": getattr(row, "aadt_period", ""),
                    "aadt_units": getattr(row, "aadt_units", ""),
                    "aadt_status_weight": status_weight,
                    "use_for_calibration": active,
                    "target_class": target_class,
                    "screenline_category_variance_factor": getattr(
                        row, "screenline_category_variance_factor", np.nan
                    ),
                    "is_district_screenline": is_screenline,
                    "screenline_distance_m": screenline_distance_m,
                    "screenline_buffer_m": getattr(row, "screenline_buffer_m", np.nan),
                    "hq_distance_m": hq_distance_m,
                    "screenline_min_hq_distance_m": getattr(row, "screenline_min_hq_distance_m", np.nan),
                    "target_selection_note": getattr(row, "target_selection_note", ""),
                    "route_ref": getattr(row, "road_refno", ""),
                    "route_ref_source": "Nepal.gpkg:NH:road_refno",
                    "link_code": link_code,
                    "target_key": target_key,
                    "target_mode": "nepal_nh_link",
                    "matched": geometry_matched,
                    "note": note,
                    "graph_edge_count": len(edge_keys),
                }
            )

            if not active:
                continue
            target_rows.append(
                {
                    "target_key": target_key,
                    "selector": "|".join(edge_keys),
                    "target_mode": "nepal_nh_link",
                    "observed_count": observed,
                    "observed_mean": observed,
                    "observed_median": observed,
                    "observed_min": observed,
                    "observed_max": observed,
                    "target_weight": status_weight,
                    "station_count": 1,
                    "source_station_ids": str(getattr(row, "station_no", "")),
                    "source_road_links": link_code,
                    "link_code": link_code,
                    "aadt_status": status,
                    "aadt_period": getattr(row, "aadt_period", ""),
                    "aadt_units": getattr(row, "aadt_units", ""),
                    "target_class": target_class,
                    "screenline_category_variance_factor": getattr(
                        row, "screenline_category_variance_factor", np.nan
                    ),
                    "is_district_screenline": is_screenline,
                    "screenline_distance_m": screenline_distance_m,
                    "screenline_buffer_m": getattr(row, "screenline_buffer_m", np.nan),
                    "hq_distance_m": hq_distance_m,
                    "screenline_min_hq_distance_m": getattr(row, "screenline_min_hq_distance_m", np.nan),
                    "graph_edge_count": len(edge_keys),
                    "notes": (
                        "Exact Nepal.gpkg NH link-code target. Selector contains both "
                        "directions and all graph fragments derived from the source "
                        "section, so the target is a two-way section count. "
                        f"Target class: {target_class}."
                    ),
                }
            )

        targets = pd.DataFrame(target_rows)
        mapping = pd.DataFrame(mapping_rows)
        logger.info(
            "Built %d exact Nepal.gpkg NH observed-link targets; %d of %d source "
            "links matched graph geometry.",
            len(targets),
            int(mapping["matched"].sum()) if not mapping.empty else 0,
            len(mapping),
        )
        return targets, mapping

    @staticmethod
    def normalize_link_code(value: object) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(value).upper().strip())

    def group_screenline_targets_by_incidence(
        self,
        targets: pd.DataFrame,
        p_variables: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Collapse highly overlapping AADT targets into one corridor constraint."""
        mode = str(self.config.screenline_grouping_mode).strip().lower()
        targets = targets.reset_index(drop=True).copy()
        if len(targets) != p_variables.shape[0]:
            raise ValueError(
                "Target count does not match incidence row count before grouping."
            )

        n_targets = len(targets)
        supports = p_variables > self.config.min_model_flow
        support_counts = supports.sum(axis=1)
        norms = np.linalg.norm(p_variables, axis=1)
        observed = pd.to_numeric(
            targets["observed_count"], errors="coerce"
        ).to_numpy(dtype=float)
        target_weights = pd.to_numeric(
            targets.get("target_weight", pd.Series(1.0, index=targets.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        eligible = (
            np.isfinite(observed)
            & (observed > 0)
            & (target_weights > 0)
            & (support_counts > 0)
        )

        parent = list(range(n_targets))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        def pair_metrics(a: int, b: int) -> Tuple[float, float]:
            union_count = int(np.logical_or(supports[a], supports[b]).sum())
            if union_count == 0:
                return 0.0, 0.0
            intersection_count = int(np.logical_and(supports[a], supports[b]).sum())
            jaccard = float(intersection_count / union_count)
            denominator = float(norms[a] * norms[b])
            cosine = (
                float(np.dot(p_variables[a], p_variables[b]) / denominator)
                if denominator > 0
                else 0.0
            )
            return jaccard, cosine

        if mode not in {"none", "off", "false", "disabled"}:
            if mode != "incidence_similarity":
                raise ValueError(
                    "screenline_grouping_mode must be 'incidence_similarity' or 'none'."
                )
            min_jaccard = float(self.config.screenline_group_min_jaccard)
            min_cosine = float(self.config.screenline_group_min_cosine)
            if not (0.0 <= min_jaccard <= 1.0 and 0.0 <= min_cosine <= 1.0):
                raise ValueError(
                    "screenline_group_min_jaccard and screenline_group_min_cosine "
                    "must both be in [0, 1]."
                )
            eligible_indices = np.flatnonzero(eligible)
            for position, a in enumerate(eligible_indices):
                for b in eligible_indices[position + 1:]:
                    jaccard, cosine = pair_metrics(int(a), int(b))
                    if jaccard >= min_jaccard and cosine >= min_cosine:
                        union(int(a), int(b))

        groups_by_root: Dict[int, List[int]] = {}
        for idx in range(n_targets):
            root = find(idx) if eligible[idx] else idx
            groups_by_root.setdefault(root, []).append(idx)
        groups = sorted(groups_by_root.values(), key=lambda values: min(values))

        grouped_rows: List[Dict[str, object]] = []
        grouped_incidence: List[np.ndarray] = []
        report_rows: List[Dict[str, object]] = []
        observed_method = str(self.config.screenline_group_observed_method).strip().lower()
        if observed_method not in {"median", "mean", "max", "min"}:
            raise ValueError(
                "screenline_group_observed_method must be one of: median, mean, max, min."
            )

        for group_number, members in enumerate(groups, start=1):
            rows = targets.iloc[members].copy()
            group_size = len(members)
            group_id = f"corridor_group_{group_number:03d}"
            member_observed = pd.to_numeric(
                rows["observed_count"], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if len(member_observed) == 0:
                representative_observed = np.nan
            elif observed_method == "mean":
                representative_observed = float(np.mean(member_observed))
            elif observed_method == "max":
                representative_observed = float(np.max(member_observed))
            elif observed_method == "min":
                representative_observed = float(np.min(member_observed))
            else:
                representative_observed = float(np.median(member_observed))
            member_category_variance = pd.to_numeric(
                rows.get(
                    "screenline_category_variance_factor",
                    pd.Series(
                        self.config.gls_category_variance_all_links,
                        index=rows.index,
                    ),
                ),
                errors="coerce",
            ).fillna(float(self.config.gls_category_variance_all_links))
            group_category_variance = float(member_category_variance.max())

            pair_values = [
                pair_metrics(a, b)
                for i, a in enumerate(members)
                for b in members[i + 1:]
            ]
            min_group_jaccard = (
                float(min(value[0] for value in pair_values)) if pair_values else 1.0
            )
            min_group_cosine = (
                float(min(value[1] for value in pair_values)) if pair_values else 1.0
            )

            def joined(column: str) -> str:
                if column not in rows.columns:
                    return ""
                values = [
                    str(value).strip()
                    for value in rows[column].tolist()
                    if str(value).strip() and str(value).strip().lower() != "nan"
                ]
                return "|".join(dict.fromkeys(values))

            selector_edges: List[str] = []
            for selector in rows.get("selector", pd.Series(dtype=object)).astype(str):
                selector_edges.extend(
                    edge for edge in selector.split("|") if edge.strip()
                )
            selector = "|".join(sorted(set(selector_edges)))

            group_row = rows.iloc[0].to_dict()
            group_row.update(
                {
                    "target_key": (
                        f"corridor_group::{group_id}"
                        if group_size > 1
                        else str(rows.iloc[0]["target_key"])
                    ),
                    "selector": selector,
                    "target_mode": (
                        "nepal_nh_corridor_group"
                        if group_size > 1
                        else str(rows.iloc[0]["target_mode"])
                    ),
                    "observed_count": representative_observed,
                    "observed_mean": float(np.mean(member_observed))
                    if len(member_observed)
                    else np.nan,
                    "observed_median": float(np.median(member_observed))
                    if len(member_observed)
                    else np.nan,
                    "observed_min": float(np.min(member_observed))
                    if len(member_observed)
                    else np.nan,
                    "observed_max": float(np.max(member_observed))
                    if len(member_observed)
                    else np.nan,
                    "target_weight": float(np.mean(target_weights[members]))
                    if len(members)
                    else 1.0,
                    "station_count": int(
                        pd.to_numeric(
                            rows.get("station_count", pd.Series(1, index=rows.index)),
                            errors="coerce",
                        ).fillna(1).sum()
                    ),
                    "source_station_ids": joined("source_station_ids"),
                    "source_road_links": joined("source_road_links"),
                    "link_code": joined("link_code"),
                    "target_class": (
                        "incidence_similarity_corridor_group"
                        if group_size > 1
                        else str(rows.iloc[0].get("target_class", ""))
                    ),
                    "screenline_category_variance_factor": group_category_variance,
                    "is_district_screenline": bool(
                        rows.get(
                            "is_district_screenline",
                            pd.Series(False, index=rows.index),
                        )
                        .fillna(False)
                        .astype(bool)
                        .any()
                    ),
                    "screenline_distance_m": float(
                        pd.to_numeric(
                            rows.get(
                                "screenline_distance_m",
                                pd.Series(np.nan, index=rows.index),
                            ),
                            errors="coerce",
                        ).min()
                    ),
                    "hq_distance_m": float(
                        pd.to_numeric(
                            rows.get(
                                "hq_distance_m",
                                pd.Series(np.nan, index=rows.index),
                            ),
                            errors="coerce",
                        ).min()
                    ),
                    "graph_edge_count": int(
                        pd.to_numeric(
                            rows.get("graph_edge_count", pd.Series(0, index=rows.index)),
                            errors="coerce",
                        ).fillna(0).sum()
                    ),
                    "notes": (
                        f"Incidence-similarity corridor group ({group_size} target rows); "
                        f"observed_count uses {observed_method}; incidence is max/union. "
                        f"Members: {joined('link_code')}."
                        if group_size > 1
                        else str(rows.iloc[0].get("notes", ""))
                    ),
                    "screenline_group_id": group_id,
                    "screenline_group_size": group_size,
                    "screenline_group_member_target_keys": joined("target_key"),
                    "screenline_group_member_link_codes": joined("link_code"),
                    "screenline_group_observed_method": observed_method,
                    "screenline_group_incidence_method": "max_union",
                    "screenline_group_min_jaccard": min_group_jaccard,
                    "screenline_group_min_cosine": min_group_cosine,
                    "screenline_group_threshold_jaccard": float(
                        self.config.screenline_group_min_jaccard
                    ),
                    "screenline_group_threshold_cosine": float(
                        self.config.screenline_group_min_cosine
                    ),
                }
            )
            grouped_rows.append(group_row)
            grouped_incidence.append(np.max(p_variables[members, :], axis=0))

            for original_idx in members:
                original = targets.iloc[original_idx]
                jaccard_to_group = [
                    pair_metrics(original_idx, other_idx)[0]
                    for other_idx in members
                    if other_idx != original_idx
                ]
                cosine_to_group = [
                    pair_metrics(original_idx, other_idx)[1]
                    for other_idx in members
                    if other_idx != original_idx
                ]
                report_rows.append(
                    {
                        "screenline_group_id": group_id,
                        "screenline_group_size": group_size,
                        "grouped_target_key": group_row["target_key"],
                        "group_observed_count": representative_observed,
                        "group_member_link_codes": group_row[
                            "screenline_group_member_link_codes"
                        ],
                        "original_target_index": int(original_idx),
                        "original_target_key": original.get("target_key", ""),
                        "original_link_code": original.get("link_code", ""),
                        "original_observed_count": original.get("observed_count", np.nan),
                        "original_path_pair_count": int(support_counts[original_idx]),
                        "screenline_category_variance_factor": float(
                            pd.to_numeric(
                                pd.Series(
                                    [
                                        original.get(
                                            "screenline_category_variance_factor",
                                            self.config.gls_category_variance_all_links,
                                        )
                                    ]
                                ),
                                errors="coerce",
                            )
                            .fillna(float(self.config.gls_category_variance_all_links))
                            .iloc[0]
                        ),
                        "member_min_jaccard": float(min(jaccard_to_group))
                        if jaccard_to_group
                        else 1.0,
                        "member_min_cosine": float(min(cosine_to_group))
                        if cosine_to_group
                        else 1.0,
                        "group_min_jaccard": min_group_jaccard,
                        "group_min_cosine": min_group_cosine,
                        "screenline_grouping_mode": mode,
                        "screenline_group_observed_method": observed_method,
                        "screenline_group_incidence_method": "max_union",
                    }
                )

        grouped_targets = pd.DataFrame(grouped_rows)
        grouped_p = np.vstack(grouped_incidence) if grouped_incidence else p_variables
        grouping_report = pd.DataFrame(report_rows)
        multi_groups = int((grouping_report["screenline_group_size"] > 1).sum())
        multi_group_count = int(
            grouping_report.loc[
                grouping_report["screenline_group_size"] > 1,
                "screenline_group_id",
            ].nunique()
        )
        logger.info(
            "Incidence-similarity grouping reduced %d AADT target rows to %d constraints "
            "(%d multi-station groups, %d grouped member rows).",
            n_targets,
            len(grouped_targets),
            multi_group_count,
            multi_groups,
        )
        return grouped_targets, grouped_p, grouping_report

    @staticmethod
    def weighted_scale(
        raw_modelled: np.ndarray,
        observed: np.ndarray,
        count_stddev: np.ndarray,
        target_weights: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Estimate one daily-trip scale by weighted least squares."""
        g = raw_modelled[mask]
        y = observed[mask]
        sigma = count_stddev[mask]
        weights = target_weights[mask]
        denominator = float(np.sum(weights * g ** 2 / sigma ** 2))
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError("Could not estimate a positive daily-trip scale from AADT.")
        scale = float(np.sum(weights * g * y / sigma ** 2) / denominator)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("The AADT-derived daily-trip scale is invalid.")
        return scale

    def select_beta(
        self,
        p_variables: np.ndarray,
        targets: pd.DataFrame,
        observed: np.ndarray,
        count_stddev: np.ndarray,
        target_weights: np.ndarray,
        valid_targets: np.ndarray,
        symmetric_pairs: Sequence[Tuple[str, str]],
        zone_index: Dict[str, int],
        production_weights: np.ndarray,
        attraction_weights: np.ndarray,
        impedance: np.ndarray,
        configured_beta: float,
    ) -> Tuple[float, np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """Choose beta with leave-one-highway-out prior validation.

        This step only selects the gravity deterrence parameter. The final ODME
        still uses all objectively selected, path-supported AADT targets.
        """
        mode = self.config.beta_selection_mode
        if mode not in {"fixed", "corridor_cross_validation"}:
            raise ValueError(
                "beta_selection_mode must be 'fixed' or 'corridor_cross_validation'."
            )
        candidates = (configured_beta,) if mode == "fixed" else tuple(
            sorted(set(self.config.beta_candidates))
        )
        groups = (
            targets.get("source_road_links", targets["target_key"])
            .astype(str)
            .str.split("-", n=1)
            .str[0]
            .to_numpy()
        )
        usable_groups = sorted(set(groups[valid_targets]))
        use_cross_validation = mode == "corridor_cross_validation" and len(usable_groups) >= 2
        beta_audit: List[Dict[str, float]] = []
        best: Optional[Tuple[float, np.ndarray, np.ndarray, float]] = None

        for beta in candidates:
            raw_seed = GravityModel.furness_balance(
                production_weights,
                attraction_weights,
                impedance,
                beta,
                max_iter=self.config.max_iter_gravity,
                tol=self.config.tol_gravity,
            )
            raw_seed = ODSynthesizer.symmetrize_od_matrix(raw_seed)
            raw_vector = self.matrix_to_symmetric_vector(
                raw_seed, symmetric_pairs, zone_index
            )
            raw_modelled = p_variables @ raw_vector
            if use_cross_validation:
                squared_error = 0.0
                total_weight = 0.0
                absolute_error = 0.0
                observed_total = 0.0
                for group in usable_groups:
                    test = valid_targets & (groups == group)
                    train = valid_targets & ~test
                    scale = self.weighted_scale(
                        raw_modelled, observed, count_stddev, target_weights, train
                    )
                    residual = scale * raw_modelled[test] - observed[test]
                    weights = target_weights[test]
                    squared_error += float(np.sum(weights * residual ** 2))
                    total_weight += float(np.sum(weights))
                    absolute_error += float(np.sum(weights * np.abs(residual)))
                    observed_total += float(np.sum(weights * observed[test]))
                validation_rmse = math.sqrt(squared_error / max(total_weight, 1e-12))
                validation_wape = absolute_error / max(observed_total, 1e-12)
            else:
                scale = self.weighted_scale(
                    raw_modelled, observed, count_stddev, target_weights, valid_targets
                )
                residual = scale * raw_modelled[valid_targets] - observed[valid_targets]
                weights = target_weights[valid_targets]
                validation_rmse = math.sqrt(
                    float(np.sum(weights * residual ** 2) / np.sum(weights))
                )
                validation_wape = float(
                    np.sum(weights * np.abs(residual)) / np.sum(weights * observed[valid_targets])
                )
            beta_audit.append(
                {
                    "beta": float(beta),
                    "validation_rmse": float(validation_rmse),
                    "validation_wape": float(validation_wape),
                }
            )
            if best is None or validation_rmse < best[3]:
                best = (float(beta), raw_seed, raw_modelled, float(validation_rmse))

        if best is None:
            raise ValueError("No valid beta candidate was available for ODME.")
        return best[0], best[1], best[2], beta_audit

    def calibrate(
        self,
        seed_od: np.ndarray,
        od_paths: pd.DataFrame,
        targets: pd.DataFrame,
        max_iter: int,
        tol: float,
        production_weights: np.ndarray,
        attraction_weights: np.ndarray,
        impedance: np.ndarray,
        configured_beta: float,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Calibrate symmetric district-pair variables against observed counts."""
        if od_paths.empty or targets.empty:
            raise ValueError("OD paths and targets are required for district ODME.")

        pair_origins = od_paths["origin"].astype(str).tolist()
        pair_destinations = od_paths["destination"].astype(str).tolist()
        matrix_shape = seed_od.shape
        zone_labels = sorted(
            set(pair_origins) | set(pair_destinations), key=lambda value: int(value)
        )
        zone_index = {zone: idx for idx, zone in enumerate(zone_labels)}
        symmetric_pairs, directed_to_variable = self.symmetric_pair_index(
            pair_origins, pair_destinations
        )

        targets_all = targets.reset_index(drop=True).copy()
        p_all = self.build_incidence_matrix(od_paths, targets_all)
        p_variables = self.collapse_directed_incidence(
            p_all, directed_to_variable, len(symmetric_pairs)
        )
        targets_all, p_variables, grouping_report = self.group_screenline_targets_by_incidence(
            targets_all,
            p_variables,
        )
        if self.config.write_calibration_audit:
            grouping_path = self.config.output_dir / "screenline_grouping_report.csv"
            grouping_report.to_csv(grouping_path, index=False)
            logger.info("Wrote incidence-similarity screenline grouping audit: %s", grouping_path)
        observed_all = targets_all["observed_count"].to_numpy(dtype=float)
        target_weights_all = pd.to_numeric(
            targets_all.get("target_weight", pd.Series(1.0, index=targets_all.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
        target_weights_all = np.where(target_weights_all > 0, target_weights_all, 0.0)
        count_stddev_all = np.maximum(
            np.sqrt(np.maximum(observed_all, 1.0)),
            float(self.config.aadt_count_cv) * observed_all,
        )
        path_pair_count = (p_variables > self.config.min_model_flow).sum(axis=1)
        path_supported = path_pair_count > 0
        selected_initial = (
            path_supported
            & np.isfinite(observed_all)
            & (observed_all > 0)
            & (target_weights_all > 0)
        )
        if not selected_initial.any():
            raise ValueError(
                "No objectively selected AADT target has an HQ-to-HQ path."
            )

        configured_seed_vector = self.matrix_to_symmetric_vector(
            ODSynthesizer.symmetrize_od_matrix(seed_od),
            symmetric_pairs,
            zone_index,
        )
        configured_seed_modelled_all = p_variables @ configured_seed_vector
        configured_seed_scale = self.weighted_scale(
            configured_seed_modelled_all,
            observed_all,
            count_stddev_all,
            target_weights_all,
            selected_initial,
        )
        configured_seed_scaled_modelled_all = (
            configured_seed_modelled_all * configured_seed_scale
        )
        pre_beta_diagnostic = self.zero_seed_constraint_mask(
            configured_seed_scaled_modelled_all,
            observed_all,
            selected_initial,
        )
        beta_count_stddev_all, beta_weight_audit = self.gls_count_stddev(
            observed=observed_all,
            modelled_prior=configured_seed_scaled_modelled_all,
            path_pair_count=path_pair_count,
            targets=targets_all,
        )
        selected_for_beta = selected_initial
        if not selected_for_beta.any():
            raise ValueError(
                "No AADT target remained after path-support and target-weight checks."
            )

        selected_beta, raw_seed_matrix, raw_seed_modelled_all, beta_audit = self.select_beta(
            p_variables=p_variables,
            targets=targets_all,
            observed=observed_all,
            count_stddev=beta_count_stddev_all,
            target_weights=target_weights_all,
            valid_targets=selected_for_beta,
            symmetric_pairs=symmetric_pairs,
            zone_index=zone_index,
            production_weights=production_weights,
            attraction_weights=attraction_weights,
            impedance=impedance,
            configured_beta=configured_beta,
        )
        raw_seed_vector = self.matrix_to_symmetric_vector(
            raw_seed_matrix, symmetric_pairs, zone_index
        )
        selected_beta_audit = next(
            item for item in beta_audit if item["beta"] == selected_beta
        )
        prior_scale = self.weighted_scale(
            raw_seed_modelled_all,
            observed_all,
            beta_count_stddev_all,
            target_weights_all,
            selected_for_beta,
        )
        prior_vector = raw_seed_vector * prior_scale
        prior_modelled_all = p_variables @ prior_vector
        count_stddev_all, weight_audit = self.gls_count_stddev(
            observed=observed_all,
            modelled_prior=prior_modelled_all,
            path_pair_count=path_pair_count,
            targets=targets_all,
        )
        prior_scale = self.weighted_scale(
            raw_seed_modelled_all,
            observed_all,
            count_stddev_all,
            target_weights_all,
            selected_for_beta,
        )
        prior_vector = raw_seed_vector * prior_scale
        prior_modelled_all = p_variables @ prior_vector
        count_stddev_all, weight_audit = self.gls_count_stddev(
            observed=observed_all,
            modelled_prior=prior_modelled_all,
            path_pair_count=path_pair_count,
            targets=targets_all,
        )
        post_beta_diagnostic = self.zero_seed_constraint_mask(
            prior_modelled_all,
            observed_all,
            selected_for_beta,
        )
        diagnostic_downweighted = pre_beta_diagnostic | post_beta_diagnostic
        selected = selected_initial
        if not selected.any():
            raise ValueError(
                "No AADT target remained after path-support and target-weight checks."
            )

        prior_matrix = self.symmetric_vector_to_matrix(
            prior_vector, symmetric_pairs, zone_index, matrix_shape
        )

        observed = observed_all[selected]
        target_weights = target_weights_all[selected]
        count_stddev = count_stddev_all[selected]
        p_selected = p_variables[selected, :]
        prior_log_std = float(self.config.district_prior_log_std)
        if prior_log_std <= 0:
            raise ValueError("district_prior_log_std must be positive.")
        if not (0.0 < float(self.config.gls_od_multiplier_min) < 1.0):
            raise ValueError("gls_od_multiplier_min must be in the interval (0, 1).")
        if float(self.config.gls_od_multiplier_max) <= 1.0:
            raise ValueError("gls_od_multiplier_max must be greater than 1.")

        final_demand, final_modelled_all, update_audit = self.weighted_gls_odme_update(
            prior_vector=prior_vector,
            p_all=p_variables,
            p_selected=p_selected,
            observed=observed,
            count_stddev=count_stddev,
            target_weights=target_weights,
            max_iter=max_iter,
            tol=tol,
        )

        def log_row(
            iteration: int,
            demand: np.ndarray,
            modelled: np.ndarray,
            status: str,
            reason: str,
            selected_final: bool,
        ) -> Dict[str, object]:
            count_residual = (modelled[selected] - observed) / count_stddev
            log_multiplier = np.log(demand / prior_vector)
            prior_term = float(
                np.sum(demand * (log_multiplier - 1.0) + prior_vector)
            )
            count_term = 0.5 * float(np.sum(target_weights * count_residual ** 2))
            row = self.metrics_row(
                iteration=iteration,
                observed=observed,
                modelled=modelled[selected],
                target_weights=target_weights,
                total_pcu_per_day=float(2.0 * demand.sum()),
                converged=bool(update_audit["converged"]) if selected_final else False,
                target_mode=self.target_mode_label(targets_all.loc[selected]),
                note=(
                    "Symmetric district-level weighted GLS ODME calibrated to "
                    "two-way AADT with entropy regularization around the gravity prior."
                ),
            )
            row.update(
                {
                    "district_objective_total": count_term + prior_term,
                    "aadt_value_field": self.config.aadt_value_field,
                    "aadt_source_period": self.config.aadt_source_period,
                    "aadt_units": self.config.aadt_units,
                    "demand_units": self.config.aadt_units,
                    "district_count_term": count_term,
                    "district_entropy_prior_term": prior_term,
                    "district_log_prior_term": prior_term,
                    "prior_scale_factor": prior_scale,
                    "prior_scale_method": "weighted_least_squares_objective_screenlines",
                    "screenline_candidate_count": int(len(targets_all)),
                    "screenline_input_target_count": int(len(grouping_report)),
                    "screenline_constraint_count": int(len(targets_all)),
                    "screenline_grouping_mode": self.config.screenline_grouping_mode,
                    "screenline_group_min_jaccard": self.config.screenline_group_min_jaccard,
                    "screenline_group_min_cosine": self.config.screenline_group_min_cosine,
                    "screenline_group_observed_method": self.config.screenline_group_observed_method,
                    "screenline_multi_station_group_count": int(
                        grouping_report.loc[
                            grouping_report["screenline_group_size"] > 1,
                            "screenline_group_id",
                        ].nunique()
                    ),
                    "screenline_grouped_target_row_count": int(
                        (grouping_report["screenline_group_size"] > 1).sum()
                    ),
                    "objective_target_count": int(selected.sum()),
                    "zero_seed_excluded_target_count": 0,
                    "pre_beta_zero_seed_excluded_target_count": 0,
                    "post_beta_zero_seed_excluded_target_count": 0,
                    "seed_implausibility_diagnostic_target_count": int(
                        diagnostic_downweighted.sum()
                    ),
                    "path_unsupported_target_count": int((~path_supported).sum()),
                    "district_count": matrix_shape[0],
                    "district_pair_variable_count": len(demand),
                    "district_od_cell_count": matrix_shape[0] * (matrix_shape[0] - 1),
                    "district_incidence_rank": int(np.linalg.matrix_rank(p_selected)),
                    "aadt_count_cv": self.config.aadt_count_cv,
                    "district_prior_log_std": prior_log_std,
                    "gls_category_variance_primary": self.config.gls_category_variance_primary,
                    "gls_category_variance_low_volume": self.config.gls_category_variance_low_volume,
                    "gls_category_variance_hq_adjacent": self.config.gls_category_variance_hq_adjacent,
                    "gls_path_dilution_power": self.config.gls_path_dilution_power,
                    "gls_seed_implausibility_floor": self.config.gls_seed_implausibility_floor,
                    "gls_od_multiplier_min": self.config.gls_od_multiplier_min,
                    "gls_od_multiplier_max": self.config.gls_od_multiplier_max,
                    "gls_sigma_min": float(np.min(count_stddev)),
                    "gls_sigma_max": float(np.max(count_stddev)),
                    "gls_weight_min": float(np.min(target_weights / np.maximum(count_stddev ** 2, 1e-12))),
                    "gls_weight_max": float(np.max(target_weights / np.maximum(count_stddev ** 2, 1e-12))),
                    "zero_seed_ratio_threshold": self.config.zero_seed_ratio_threshold,
                    "zero_seed_abs_flow_threshold": self.config.zero_seed_abs_flow_threshold,
                    "beta_selection_mode": self.config.beta_selection_mode,
                    "selected_beta": selected_beta,
                    "beta_candidate_count": len(beta_audit),
                    "beta_validation_rmse": selected_beta_audit["validation_rmse"],
                    "beta_validation_wape": selected_beta_audit["validation_wape"],
                    "beta_validation_audit": json.dumps(beta_audit),
                    "route_choice_mode": self.config.route_choice_mode,
                    "route_choice_k": self.config.route_choice_k,
                    "fit_status": status,
                    "stop_reason": reason,
                    "selected_final": selected_final,
                }
            )
            return row

        initial_row = log_row(
            0,
            prior_vector,
            prior_modelled_all,
            "count_scaled_gravity_prior",
            "weighted_least_squares_objective_screenlines",
            False,
        )
        final_status = (
            "weighted_gls_converged"
            if update_audit["converged"]
            else "weighted_gls_stopped"
        )
        final_row = log_row(
            int(update_audit["iterations"]),
            final_demand,
            final_modelled_all,
            final_status,
            str(update_audit["stop_reason"]),
            True,
        )
        final_row["solver_status"] = str(update_audit["solver_status"])
        final_row["solver_optimality"] = float(update_audit["max_abs_gradient"])
        final_row["solver_nit"] = int(update_audit["iterations"])
        final_row["solver_cost"] = float(update_audit["objective"])
        final_row["max_relative_gap"] = float(update_audit["max_relative_gap"])
        final_row["weighted_nrmse_gap"] = float(update_audit["weighted_nrmse_gap"])
        final_row["od_log_multiplier_min"] = float(update_audit["od_log_multiplier_min"])
        final_row["od_log_multiplier_max"] = float(update_audit["od_log_multiplier_max"])
        final_row["od_multiplier_min"] = float(update_audit["od_multiplier_min"])
        final_row["od_multiplier_max"] = float(update_audit["od_multiplier_max"])

        final_matrix = self.symmetric_vector_to_matrix(
            final_demand, symmetric_pairs, zone_index, matrix_shape
        )
        np.fill_diagonal(final_matrix, 0.0)

        target_fit = targets_all.copy()
        target_fit["path_pair_count"] = path_pair_count
        target_fit["used_for_calibration"] = selected
        target_fit["calibration_status"] = np.select(
            [
                selected & diagnostic_downweighted,
                selected,
                ~path_supported,
            ],
            [
                "weighted_gls_seed_implausibility_downweighted",
                "weighted_gls_objective_target",
                "no_hq_od_path",
            ],
            default="excluded_from_objective",
        )
        target_fit["constraint_drop_reason"] = ""
        target_fit.loc[diagnostic_downweighted, "constraint_drop_reason"] = (
            "Retained as a soft GLS target: seed/model AADT ratio is implausible, "
            "so the count variance is inflated rather than forcing an exact fit."
        )
        target_fit.loc[~path_supported, "constraint_drop_reason"] = (
            "No routed HQ-to-HQ path crosses this target."
        )
        target_fit["modelled_configured_seed_scaled"] = configured_seed_scaled_modelled_all
        target_fit["configured_seed_scale_factor"] = configured_seed_scale
        target_fit["modelled_raw_gravity_seed"] = raw_seed_modelled_all
        target_fit["modelled_count_scaled_prior"] = prior_modelled_all
        target_fit["modelled_final"] = final_modelled_all
        target_fit["gls_base_count_stddev"] = weight_audit["base_count_stddev"]
        target_fit["gls_path_dilution_factor"] = weight_audit["path_dilution_factor"]
        target_fit["gls_category_variance_factor"] = weight_audit["category_variance_factor"]
        target_fit["gls_seed_implausibility_factor"] = weight_audit["seed_implausibility_factor"]
        target_fit["gls_count_stddev"] = count_stddev_all
        target_fit["gls_weight"] = np.divide(
            target_weights_all,
            np.maximum(count_stddev_all ** 2, 1e-12),
        )
        target_fit["seed_to_observed_ratio"] = np.divide(
            prior_modelled_all,
            np.maximum(observed_all, self.config.min_model_flow),
        )
        target_fit["count_standard_deviation"] = count_stddev_all
        target_fit["standardized_residual_final"] = (
            final_modelled_all - observed_all
        ) / count_stddev_all
        target_fit["absolute_error_raw_seed"] = raw_seed_modelled_all - observed_all
        target_fit["absolute_error_count_scaled_prior"] = prior_modelled_all - observed_all
        target_fit["absolute_error_final"] = final_modelled_all - observed_all
        target_fit["pct_error_final"] = np.divide(
            target_fit["absolute_error_final"],
            np.maximum(observed_all, self.config.min_model_flow),
        )
        target_fit["geh_final"] = np.sqrt(
            2.0 * target_fit["absolute_error_final"] ** 2
            / np.maximum(final_modelled_all + observed_all, self.config.min_model_flow)
        )
        target_fit["prior_scale_factor"] = prior_scale
        target_fit["prior_scale_method"] = "weighted_least_squares_objective_screenlines"
        target_fit["demand_units"] = self.config.aadt_units
        target_fit["selected_beta"] = selected_beta
        target_fit["aadt_count_cv"] = self.config.aadt_count_cv
        target_fit["district_prior_log_std"] = prior_log_std
        target_fit["district_pair_variable_count"] = len(final_demand)
        target_fit["district_od_cell_count"] = matrix_shape[0] * (matrix_shape[0] - 1)
        target_fit["odme_update_damping"] = self.config.odme_update_damping
        target_fit["odme_multiplier_min"] = self.config.odme_multiplier_min
        target_fit["odme_multiplier_max"] = self.config.odme_multiplier_max
        target_fit["zero_seed_ratio_threshold"] = self.config.zero_seed_ratio_threshold
        target_fit["zero_seed_abs_flow_threshold"] = self.config.zero_seed_abs_flow_threshold
        target_fit["route_choice_mode"] = self.config.route_choice_mode
        target_fit["route_choice_k"] = self.config.route_choice_k

        logger.info(
            "Weighted GLS ODME across %d soft objective targets (%d seed-implausibility diagnostics downweighted): prior RMSE %.2f -> final RMSE %.2f; converged=%s.",
            int(selected.sum()),
            int(diagnostic_downweighted.sum()),
            float(initial_row["rmse_counts"]),
            float(final_row["rmse_counts"]),
            bool(update_audit["converged"]),
        )
        return final_matrix, prior_matrix, pd.DataFrame([initial_row, final_row]), target_fit

    @staticmethod
    def symmetric_pair_index(
        origins: Sequence[str],
        destinations: Sequence[str],
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        pair_to_index: Dict[Tuple[str, str], int] = {}
        pairs: List[Tuple[str, str]] = []
        directed_to_variable: List[int] = []
        for origin, destination in zip(origins, destinations):
            if str(origin) == str(destination):
                continue
            key = tuple(sorted((str(origin), str(destination)), key=lambda value: int(value)))
            if key not in pair_to_index:
                pair_to_index[key] = len(pairs)
                pairs.append(key)
            directed_to_variable.append(pair_to_index[key])
        return pairs, directed_to_variable

    @staticmethod
    def collapse_directed_incidence(
        p_directed: np.ndarray,
        directed_to_variable: Sequence[int],
        variable_count: int,
    ) -> np.ndarray:
        p_variables = np.zeros((p_directed.shape[0], variable_count), dtype=float)
        if p_directed.shape[1] != len(directed_to_variable):
            raise ValueError(
                "Directed incidence column count does not match symmetric pair index."
            )
        for directed_col, variable_col in enumerate(directed_to_variable):
            p_variables[:, int(variable_col)] += p_directed[:, directed_col]
        return p_variables

    def zero_seed_constraint_mask(
        self,
        modelled: np.ndarray,
        observed: np.ndarray,
        candidate: np.ndarray,
    ) -> np.ndarray:
        ratio = np.divide(
            modelled,
            np.maximum(observed, self.config.min_model_flow),
        )
        near_zero = (
            (modelled <= float(self.config.zero_seed_abs_flow_threshold))
            | (ratio <= float(self.config.zero_seed_ratio_threshold))
        )
        large_observed = observed >= float(self.config.min_calibration_aadt)
        return candidate & large_observed & near_zero

    def gls_count_stddev(
        self,
        observed: np.ndarray,
        modelled_prior: np.ndarray,
        path_pair_count: np.ndarray,
        targets: pd.DataFrame,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Build GLS count standard deviations from count, category, path, and seed diagnostics."""
        observed = np.asarray(observed, dtype=float)
        modelled_prior = np.asarray(modelled_prior, dtype=float)
        path_pair_count = np.asarray(path_pair_count, dtype=float)

        count_floor = max(float(self.config.min_calibration_aadt), 1.0)
        base_count_stddev = np.maximum(
            np.sqrt(np.maximum(observed, 1.0)),
            float(self.config.aadt_count_cv)
            * np.maximum(observed, count_floor),
        )
        positive_counts = path_pair_count[path_pair_count > 0]
        mean_path_count = (
            float(np.mean(positive_counts)) if len(positive_counts) else 1.0
        )
        path_ratio = np.divide(
            path_pair_count,
            max(mean_path_count, 1.0),
        )
        path_dilution_factor = np.maximum(
            1.0,
            path_ratio ** float(self.config.gls_path_dilution_power),
        )

        category_variance_factor = pd.to_numeric(
            targets.get(
                "screenline_category_variance_factor",
                pd.Series(
                    self.config.gls_category_variance_all_links,
                    index=targets.index,
                ),
            ),
            errors="coerce",
        ).fillna(float(self.config.gls_category_variance_all_links)).to_numpy(dtype=float)
        category_variance_factor = np.maximum(category_variance_factor, 1.0)

        ratio = np.divide(
            np.maximum(observed, self.config.min_model_flow),
            np.maximum(modelled_prior, self.config.min_model_flow),
        )
        seed_implausibility_factor = np.maximum(
            float(self.config.gls_seed_implausibility_floor),
            np.abs(np.log(np.maximum(ratio, self.config.min_model_flow))),
        )
        seed_implausibility_factor = np.maximum(seed_implausibility_factor, 1.0)

        variance = (
            base_count_stddev ** 2
            * path_dilution_factor
            * category_variance_factor
            * seed_implausibility_factor
        )
        count_stddev = np.sqrt(np.maximum(variance, 1e-12))
        return count_stddev, {
            "base_count_stddev": base_count_stddev,
            "path_dilution_factor": path_dilution_factor,
            "category_variance_factor": category_variance_factor,
            "seed_implausibility_factor": seed_implausibility_factor,
        }

    def weighted_gls_odme_update(
        self,
        prior_vector: np.ndarray,
        p_all: np.ndarray,
        p_selected: np.ndarray,
        observed: np.ndarray,
        count_stddev: np.ndarray,
        target_weights: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Entropy-prior weighted GLS ODME solved over OD log multipliers."""
        prior = prior_vector.astype(float, copy=True)
        if np.any(prior <= 0) or not np.isfinite(prior).all():
            raise ValueError("The count-scaled gravity prior must be positive and finite.")

        observed = np.asarray(observed, dtype=float)
        count_stddev = np.asarray(count_stddev, dtype=float)
        target_weights = np.asarray(target_weights, dtype=float)
        gls_weights = np.divide(
            np.maximum(target_weights, 0.0),
            np.maximum(count_stddev ** 2, 1e-12),
        )
        log_min = math.log(float(self.config.gls_od_multiplier_min))
        log_max = math.log(float(self.config.gls_od_multiplier_max))
        bounds = [(log_min, log_max)] * len(prior)

        def objective_and_gradient(z: np.ndarray) -> Tuple[float, np.ndarray]:
            z = np.asarray(z, dtype=float)
            demand = prior * np.exp(z)
            residual = p_selected @ demand - observed
            entropy = float(np.sum(demand * (z - 1.0) + prior))
            count_term = 0.5 * float(np.sum(gls_weights * residual ** 2))
            gradient_x = z + p_selected.T @ (gls_weights * residual)
            gradient_z = demand * gradient_x
            return entropy + count_term, gradient_z

        initial = np.zeros(len(prior), dtype=float)
        result = minimize(
            fun=lambda z: objective_and_gradient(z),
            x0=initial,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(max_iter),
                "ftol": float(tol),
                "gtol": float(tol),
                "maxls": 50,
            },
        )
        z_final = np.asarray(result.x, dtype=float)
        final_demand = prior * np.exp(z_final)
        if not np.isfinite(final_demand).all():
            raise FloatingPointError("Weighted GLS ODME produced non-finite demand values.")

        final_modelled_all = p_all @ final_demand
        selected_modelled = p_selected @ final_demand
        relative_gap = np.abs(selected_modelled - observed) / np.maximum(observed, 1.0)
        weighted_nrmse_gap = float(
            np.sqrt(
                np.sum(np.maximum(target_weights, 0.0) * relative_gap ** 2)
                / max(float(np.sum(np.maximum(target_weights, 0.0))), 1e-12)
            )
        ) if len(relative_gap) else 0.0
        max_relative_gap = float(np.max(relative_gap)) if len(relative_gap) else 0.0
        objective, gradient = objective_and_gradient(z_final)
        max_abs_gradient = float(np.max(np.abs(gradient))) if len(gradient) else 0.0
        converged = bool(result.success)
        stop_reason = str(result.message)

        return final_demand, final_modelled_all, {
            "iterations": int(getattr(result, "nit", 0)),
            "converged": converged,
            "stop_reason": stop_reason,
            "solver_status": "weighted_gls_lbfgsb",
            "objective": float(objective),
            "max_abs_gradient": max_abs_gradient,
            "max_relative_gap": max_relative_gap,
            "weighted_nrmse_gap": weighted_nrmse_gap,
            "od_log_multiplier_min": float(np.min(z_final)) if len(z_final) else 0.0,
            "od_log_multiplier_max": float(np.max(z_final)) if len(z_final) else 0.0,
            "od_multiplier_min": float(np.exp(np.min(z_final))) if len(z_final) else 1.0,
            "od_multiplier_max": float(np.exp(np.max(z_final))) if len(z_final) else 1.0,
        }

    @staticmethod
    def build_incidence_matrix(od_paths: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
        p = np.zeros((len(targets), len(od_paths)), dtype=float)

        route_sets: List[List[Tuple[Set[str], float]]] = []
        for row in od_paths.itertuples(index=False):
            alternatives: List[Tuple[Set[str], float]] = []
            raw_alternatives = getattr(row, "route_alternatives", "")
            if raw_alternatives:
                try:
                    decoded = json.loads(str(raw_alternatives))
                    for route in decoded:
                        edge_set = set(map(str, route.get("edge_keys", [])))
                        share = float(route.get("share", 0.0))
                        if edge_set and math.isfinite(share) and share > 0:
                            alternatives.append((edge_set, share))
                except (TypeError, ValueError, json.JSONDecodeError):
                    alternatives = []
            if not alternatives:
                edge_set = set(
                    str(getattr(row, "path_edge_keys", "")).split("|")
                ) - {""}
                if edge_set:
                    alternatives = [(edge_set, 1.0)]
            total_share = sum(share for _, share in alternatives)
            if total_share > 0:
                alternatives = [
                    (edge_set, share / total_share) for edge_set, share in alternatives
                ]
            route_sets.append(alternatives)

        for t_idx, target in enumerate(targets.itertuples(index=False)):
            selector = str(target.selector)
            mode = str(target.target_mode)
            if mode == "nepal_nh_link":
                selector_edges = set(selector.split("|")) if selector.strip() else set()
                p[t_idx, :] = [
                    sum(
                        share
                        for edge_set, share in alternatives
                        if selector_edges.intersection(edge_set)
                    )
                    for alternatives in route_sets
                ]
            else:
                raise ValueError(
                    f"Unsupported calibration target mode {mode!r}. "
                    "This OD workflow expects exact Nepal.gpkg NH link-code targets."
                )
        return p

    @staticmethod
    def vector_to_matrix(
        vector: np.ndarray,
        origins: Sequence[str],
        destinations: Sequence[str],
        zone_index: Dict[str, int],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        matrix = np.zeros(shape, dtype=float)
        for value, origin, destination in zip(vector, origins, destinations):
            matrix[zone_index[str(origin)],
                   zone_index[str(destination)]] = value
        return matrix

    @staticmethod
    def symmetric_vector_to_matrix(
        vector: np.ndarray,
        pairs: Sequence[Tuple[str, str]],
        zone_index: Dict[str, int],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        matrix = np.zeros(shape, dtype=float)
        for value, (zone_a, zone_b) in zip(vector, pairs):
            i = zone_index[str(zone_a)]
            j = zone_index[str(zone_b)]
            matrix[i, j] = float(value)
            matrix[j, i] = float(value)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    @staticmethod
    def matrix_to_vector(
        matrix: np.ndarray,
        origins: Sequence[str],
        destinations: Sequence[str],
        zone_index: Dict[str, int],
    ) -> np.ndarray:
        return np.array([matrix[zone_index[str(o)], zone_index[str(d)]] for o, d in zip(origins, destinations)])

    @staticmethod
    def matrix_to_symmetric_vector(
        matrix: np.ndarray,
        pairs: Sequence[Tuple[str, str]],
        zone_index: Dict[str, int],
    ) -> np.ndarray:
        values: List[float] = []
        for zone_a, zone_b in pairs:
            i = zone_index[str(zone_a)]
            j = zone_index[str(zone_b)]
            values.append(float(0.5 * (matrix[i, j] + matrix[j, i])))
        return np.asarray(values, dtype=float)

    @staticmethod
    def target_mode_label(targets: pd.DataFrame) -> str:
        modes = sorted(targets["target_mode"].astype(str).unique())
        return "+".join(modes)

    @staticmethod
    def metrics_row(
        iteration: int,
        observed: np.ndarray,
        modelled: np.ndarray,
        target_weights: Optional[np.ndarray],
        total_pcu_per_day: float,
        converged: bool,
        target_mode: str,
        note: str,
    ) -> Dict[str, object]:
        weights = np.ones_like(observed, dtype=float) if target_weights is None else np.asarray(target_weights, dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
        residual = modelled - observed
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        weighted_rmse = float(np.sqrt(np.sum(weights * residual ** 2) / max(np.sum(weights), 1e-9)))
        weighted_mae = float(np.sum(weights * np.abs(residual)) / max(np.sum(weights), 1e-9))
        mean_obs = float(np.mean(observed)) if len(observed) else np.nan
        relative_rmse = rmse / max(mean_obs, 1e-9)
        relative_residual = residual / np.maximum(observed, 1e-9)
        nrmse_by_target = float(np.sqrt(np.mean(relative_residual ** 2)))
        weighted_nrmse_by_target = float(
            np.sqrt(np.sum(weights * relative_residual ** 2) / max(np.sum(weights), 1e-9))
        )
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        geh = np.sqrt(2.0 * (modelled - observed) ** 2 /
                      np.maximum(modelled + observed, 1e-9))
        return {
            "iteration": iteration,
            "rmse_counts": rmse,
            "mae_counts": mae,
            "weighted_rmse_counts": weighted_rmse,
            "weighted_mae_counts": weighted_mae,
            "r2_counts": r2,
            "relative_rmse": relative_rmse,
            "nrmse_by_target": nrmse_by_target,
            "weighted_nrmse_by_target": weighted_nrmse_by_target,
            "mean_geh": float(np.mean(geh)),
            "share_geh_under_5": float(np.mean(geh < 5.0)),
            "total_pcu_per_day": total_pcu_per_day,
            "converged": converged,
            "target_mode": target_mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": note,
        }


class GravityModel:
    @staticmethod
    def furness_balance(
        productions: np.ndarray,
        attractions: np.ndarray,
        impedance: np.ndarray,
        beta: float,
        max_iter: int = 200,
        tol: float = 1e-8,
    ) -> np.ndarray:
        if impedance.shape != (len(productions), len(attractions)):
            raise ValueError(
                f"Impedance shape {impedance.shape} does not match "
                f"{len(productions)} productions and {len(attractions)} attractions."
            )

        friction = np.exp(-beta * impedance)
        np.fill_diagonal(friction, 0.0)
        trips = np.outer(productions, attractions) * friction
        return GravityModel.ipf_balance(trips, productions, attractions, max_iter=max_iter, tol=tol)

    @staticmethod
    def ipf_balance(
        trips: np.ndarray,
        productions: np.ndarray,
        attractions: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-8,
    ) -> np.ndarray:
        matrix = trips.astype(float, copy=True)
        np.fill_diagonal(matrix, 0.0)

        for iteration in range(1, max_iter + 1):
            row_totals = matrix.sum(axis=1, keepdims=True)
            row_factors = productions[:, None] / np.maximum(row_totals, 1e-12)
            matrix *= row_factors

            col_totals = matrix.sum(axis=0, keepdims=True)
            col_factors = attractions[None, :] / np.maximum(col_totals, 1e-12)
            matrix *= col_factors
            np.fill_diagonal(matrix, 0.0)

            row_error = np.max(
                np.abs(matrix.sum(axis=1) - productions) /
                np.maximum(productions, 1.0)
            )
            col_error = np.max(
                np.abs(matrix.sum(axis=0) - attractions) /
                np.maximum(attractions, 1.0)
            )
            if max(row_error, col_error) <= tol:
                logger.info(
                    "Iterative proportional fitting (IPF) / Furness balancing converged in %d iterations.", iteration)
                break
        else:
            logger.warning(
                "Iterative proportional fitting (IPF) / Furness balancing did not reach tolerance after %d iterations.", max_iter)

        return matrix


if __name__ == "__main__":
    ODSynthesizer(Config()).run()

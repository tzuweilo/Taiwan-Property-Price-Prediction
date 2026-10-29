"""V9: two disjoint Train-only candidates, fixed historical gates, optional export.

Importing this module never fits a model or reads a dataset. Real fits occur only
in the notebook's explicit run_validation/export_if_accepted calls. Test labels
are never accepted. Reviewed Dev/Check windows are retrospective, not blind.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import itertools
import json
import re
import zipfile

import numpy as np
import pandas as pd

import study_pipeline as study
from try_pipeline import single_floor
from v9_reference import PriceReferenceHierarchy, ParkingHelper

ID, TARGET, MONTH = study.ID, study.TARGET, "交易時間序號"
BUILDING, PARKING = study.BUILDING, study.PARKING
IMPLEMENTATION = "v9_prior_anchor_parking_decomposition_1"
SCALE = 1_000_000.0


@dataclass(frozen=True)
class V9Config:
    threads: int = 8
    min_overall_gain: float = 100_000.0
    max_accurate_increase_pct: float = 2.0
    max_new_damaged: int = 0
    weights: tuple = (0.0, 0.25, 0.5, 1.0)
    ordinary_cap: float = 500_000.0
    commercial_relative_cap: float = 0.25
    max_extrapolation_ratio: float = 1.5
    iterations: int = 2400
    seed: int = 42
    minimum_fit_rows: int = 100
    ordinary_reference_rows: int = 20
    parking_reference_rows: int = 5
    stress_projects: int = 5
    minimum_stress_rows: int = 5
    maximum_forecast_months: int = 8
    fit_end_month: int = 194
    dev_end_month: int = 197
    check_end_month: int = 199

    def __post_init__(self):
        if self.threads < 1 or self.iterations < 1 or self.minimum_fit_rows < 2:
            raise ValueError("Positive thread/tree counts and at least two fit rows required")
        if not self.weights or 0.0 not in self.weights or any(w < 0 or w > 1 for w in self.weights):
            raise ValueError("Fixed weights must include zero and lie in [0,1]")
        if self.ordinary_cap <= 0 or not 0 < self.commercial_relative_cap <= 1:
            raise ValueError("Invalid correction caps")
        if self.max_extrapolation_ratio < 1 or self.maximum_forecast_months < 1:
            raise ValueError("Invalid support bounds")


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _hash(path):
    return study.hash_file(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _num(frame, column):
    return study.num(frame[column])


def _unlabelled(frame):
    forbidden = {TARGET, "總價", "actual", "target", "label", "y_true", "answer",
        "ground_truth", "matched_price", "true_price", "test_price"}
    bad = [str(c) for c in frame if str(c).strip().casefold() in forbidden
           or re.search(r"答案|真值|實際總價|匹配總價|補價", str(c))]
    if bad:
        raise ValueError("Test answer columns prohibited: " + ", ".join(bad))


def _text(frame, column):
    return frame[column].astype("string").fillna("Missing").astype(str)


def _clean(value):
    value = str(value)
    return value not in {"", "Missing", "缺失", "nan", "<NA>", "Unknown"} and "缺失" not in value


def deterministic_delay(ids, config):
    """Stable by original ID and fixed seed, independent of row ordering."""
    return np.array([1 + int.from_bytes(hashlib.blake2b(
        (str(config.seed) + "|" + str(i)).encode(), digest_size=8).digest(), "big")
        % config.maximum_forecast_months for i in ids], dtype=int)


def populations(frame):
    kind, use = _text(frame, "建物型態"), _text(frame, "主要用途")
    housing_target = _text(frame, "交易標的").str.contains("建物", regex=False).to_numpy(bool)
    residential = kind.str.contains("住宅|華廈|公寓|透天|套房", regex=True).to_numpy(bool)
    industrial = (kind.str.contains("工廠|廠辦|工業", regex=True)
                  | use.str.contains("工業|工廠|廠房", regex=True)).to_numpy(bool)
    office = (kind.str.contains("辦公|商業|店面|商辦", regex=True)
              | use.str.contains("商業|辦公|商辦|店舖|店鋪", regex=True)).to_numpy(bool)
    family = np.where(industrial, "Industrial", np.where(office, "Office_Retail", "Other"))
    gross = _num(frame, BUILDING)
    ordinary = housing_target & residential & ~industrial & ~office & (gross > 0) & (gross <= 300)
    commercial = housing_target & np.isin(family, ["Industrial", "Office_Retail"]) & (gross > 0)
    return ordinary, commercial, family


def commercial_strata(frame):
    _, _, family = populations(frame)
    return np.array([json.dumps([str(f), str(k), str(u)], ensure_ascii=False)
        for f, k, u in zip(family, _text(frame, "建物型態"), _text(frame, "主要用途"))], object)


def price_keys(frame):
    _, _, family = populations(frame)
    # Explicit residential use is retained; raw project names never pool missing projects.
    uses = _text(frame, "主要用途").tolist()
    kinds = _text(frame, "建物型態").tolist()
    area = _num(frame, BUILDING)
    bands = np.select([area <= 75, area <= 150, area <= 225, area <= 300],
                      ["0_75", "75_150", "150_225", "225_300"], default="Other")
    district, road, project = (_text(frame, c).tolist() for c in ["行政區", "道路地段", "建案行政區"])
    keys = {"Project": [], "Road": [], "District": []}
    for d, r, p, k, u, b in zip(district, road, project, kinds, uses, bands):
        tail = (k, u, str(b))
        keys["Project"].append((p,) + tail if _clean(p) else None)
        keys["Road"].append((d, r) + tail if _clean(d) and _clean(r) else None)
        keys["District"].append((d,) + tail if _clean(d) else None)
    return keys


def _floor_values(value):
    value = str(value).replace("～", "~").replace("至", "~").replace("到", "~")
    pieces = re.split(r"~|、|，|,|;", value)
    result = []
    for piece in pieces:
        try:
            number = float(single_floor(piece.strip()))
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number != 0:
            result.append(number)
    if not result:
        return (np.nan,) * 4
    low, high = min(result), max(result)
    if "~" in value:
        count = high - low + 1 - int(low < 0 < high)
    else:
        count = len(set(result))
    return low, high, count, float(low < 0)


def physical_features(frame):
    x = study.feature_frame(frame, 6)
    _, _, family = populations(frame)
    x["Use_Family_V9"] = family
    floor = np.array([_floor_values(v) for v in frame["移轉層次"]], float).reshape(-1, 4)
    for i, name in enumerate(["Floor_Range_Min", "Floor_Range_Max", "Transferred_Floor_Count", "Range_Has_Basement"]):
        x[name] = floor[:, i]
    gross, park = _num(frame, BUILDING), _num(frame, PARKING)
    x["Net_Area_V9"] = np.where((park >= 0) & (gross > park), gross - park, np.nan)
    x["Log_Gross_Area_V9"] = np.log1p(np.maximum(gross, 0))
    x["Log_Net_Area_V9"] = np.log1p(x["Net_Area_V9"])
    return x


def ordinary_block(frame, helper, config, delay=None, cutoff=None):
    refs = helper.query(price_keys(frame), _num(frame, MONTH), delay=delay, cutoff=cutoff)
    x = physical_features(frame)
    for key, values in refs.items():
        if key == "Reference_Source":
            x["Prior_" + key] = values
        elif key.startswith("Reference_") or key in {"Forecast_Gap"}:
            x["Prior_" + key] = np.asarray(values, float)
    base = refs["Reference_Value"]
    for window in (3, 6):
        value = refs[f"Reference_{window}M"]
        x[f"Prior_Trend_{window}_12"] = np.divide(value, base,
            out=np.full(len(frame), np.nan), where=np.isfinite(base) & (base > 0)) - 1
    ordinary, _, _ = populations(frame)
    mask = (ordinary & np.isfinite(base) & (base > 0)
        & (refs["Reference_Count"] >= config.ordinary_reference_rows)
        & (refs["Reference_Age"] <= 24) & (refs["Forecast_Gap"] >= 1)
        & (refs["Forecast_Gap"] <= config.maximum_forecast_months))
    return x, mask, refs


def commercial_block(frame, helper, config, delay=None, cutoff=None):
    park = helper.query(frame, delay=delay, cutoff=cutoff)
    x = physical_features(frame)
    for name, values in park.items():
        if name != "Parking_Effective_Cutoff":
            x["Prior_" + name] = values
    gross, area, count = _num(frame, BUILDING), _num(frame, PARKING), park["Parking_Count"]
    net = gross - area
    _, commercial, _ = populations(frame)
    zero = (count == 0) & (area == 0)
    positive = (count > 0) & (area > 0) & park["Parking_Available"]
    # Unknown/contradictory parking area/count remains unsupported.
    gap = _num(frame, MONTH) - park["Parking_Effective_Cutoff"]
    x["Forecast_Gap_V9"] = gap
    mask = commercial & (net > 0) & np.isfinite(net) & (zero | positive) & np.isfinite(park["Parking_Price"])
    mask &= (gap >= 1) & (gap <= config.maximum_forecast_months)
    return x, mask, net, park


def _normalise_x(x):
    x = x.copy()
    cats = []
    for col in x:
        if pd.api.types.is_numeric_dtype(x[col]):
            values = pd.to_numeric(x[col], errors="coerce").to_numpy(float, na_value=np.nan, copy=True)
            values[~np.isfinite(values)] = np.nan
            x[col] = values
        else:
            x[col] = x[col].astype("string").fillna("Missing").astype(str)
            cats.append(col)
    if {ID, TARGET, "Actual", "Prediction"}.intersection(x):
        raise ValueError("Identifier/outcome detected in the model schema")
    return x, cats


def _package_versions():
    out = {}
    for name in ("catboost", "pandas", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "Not_Installed"
    return out


def fit_cached(x, y, weight, name, root, config, provenance):
    """The only production ML fit primitive; native cache, exact schema and hashes."""
    from catboost import CatBoostRegressor
    x, cats = _normalise_x(x)
    y, weight = np.asarray(y, float), np.asarray(weight, float)
    if len(x) < config.minimum_fit_rows or not np.isfinite(y).all() or not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Invalid or insufficient branch fitting population")
    parameters = dict(loss_function="MAE", iterations=config.iterations, depth=8,
        learning_rate=0.04, l2_leaf_reg=8, grow_policy="Depthwise", min_data_in_leaf=64,
        random_seed=config.seed, thread_count=config.threads, allow_writing_files=False, verbose=300)
    signature = _digest({"implementation": IMPLEMENTATION, "provenance": provenance,
        "x": study.frame_hash(x), "y": hashlib.sha256(y.tobytes()).hexdigest(),
        "weight": hashlib.sha256(weight.tobytes()).hexdigest(), "parameters": parameters,
        "versions": _package_versions()})
    folder = Path(root) / "models" / name / signature[:16]; folder.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = folder / "model.cbm", folder / "metadata.json"
    if model_path.exists() != metadata_path.exists():
        raise RuntimeError(f"Incomplete model cache: {folder}; remove only this incomplete directory and retry")
    model = CatBoostRegressor(**parameters)
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved["signature"] != signature or saved["sha256"] != _hash(model_path):
            raise ValueError("Model cache provenance/hash mismatch")
        model.load_model(str(model_path))
        print(f"Loaded: {name} | Rows: {len(x)} | Trees: {model.tree_count_}", flush=True)
    else:
        print(f"Training: {name} | Rows: {len(x)} | Features: {x.shape[1]}", flush=True)
        model.fit(x, y, sample_weight=weight, cat_features=cats)
        tmp = folder / "model.tmp.cbm"; model.save_model(str(tmp)); tmp.replace(model_path)
        _json(metadata_path, {"signature": signature, "sha256": _hash(model_path),
            "columns": list(x), "categorical": cats, "parameters": parameters,
            "rows": len(x), "provenance": provenance})
    if list(model.feature_names_) != list(x) or int(model.tree_count_) != config.iterations:
        raise ValueError("Exact native-model schema/tree-count check failed")
    if list(model.get_cat_feature_indices()) != [list(x).index(c) for c in cats]:
        raise ValueError("Exact native categorical-feature positions differ")
    return {"model": model, "columns": list(x), "cats": cats, "folder": folder, "signature": signature}


def predict_cached(bundle, x):
    if list(x) != bundle["columns"]:
        raise ValueError("Prediction feature names/order differ from fitted schema")
    x, cats = _normalise_x(x)
    if cats != bundle["cats"]:
        raise ValueError("Prediction categorical schema differs")
    prediction = np.asarray(bundle["model"].predict(x), float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite branch prediction")
    return prediction


def measure(actual, baseline, candidate, mask, config):
    mask = np.asarray(mask, bool)
    old, new = np.abs(actual - baseline), np.abs(actual - candidate)
    accurate = mask & (old <= 500_000)
    changed = np.abs(candidate - baseline) > 1e-6
    changed_accurate = accurate & changed
    result = {"Rows": int(mask.sum()), "Changed_Rows": int((mask & changed).sum()),
        "Current_MAE": float(old[mask].mean()) if mask.any() else np.nan,
        "Candidate_MAE": float(new[mask].mean()) if mask.any() else np.nan,
        "MAE_Reduction": float((old[mask] - new[mask]).mean()) if mask.any() else np.nan,
        "Median_Residual": float(np.median((actual - candidate)[mask])) if mask.any() else np.nan,
        "Mean_Abs_Change": float(np.abs(candidate - baseline)[mask].mean()) if mask.any() else 0,
        "New_Damaged": int((accurate & (new > 1_000_000 + 1e-6)).sum())}
    for prefix, subset in [("Accurate", accurate), ("Changed_Accurate", changed_accurate)]:
        before = float(old[subset].mean()) if subset.any() else np.nan
        after = float(new[subset].mean()) if subset.any() else np.nan
        increase = 100 * (after - before) / before if before > 0 else (0.0 if not subset.any() or after == 0 else np.inf)
        result.update({prefix + "_Rows": int(subset.sum()), prefix + "_Before": before,
                       prefix + "_After": after, prefix + "_Increase_Pct": increase})
    return result


def protection_passes(metric, config):
    return (metric["New_Damaged"] <= config.max_new_damaged
        and metric["Accurate_Increase_Pct"] <= config.max_accurate_increase_pct + 1e-9
        and metric["Changed_Accurate_Increase_Pct"] <= config.max_accurate_increase_pct + 1e-9)


class V9Experiment:
    def __init__(self, project_root, config=None):
        self.root, self.config = Path(project_root), config or V9Config()
        self.folder = self.root / "outputs" / "v9_candidate"
        self.ready = self.validated = False
        self.result = None

    def _sources(self):
        return {name: _hash(self.root / "src" / name) for name in
            ["study_pipeline.py", "try_pipeline.py", "airport_policy.py", "v9_reference.py", "v9_candidate.py"]}

    def _load_baseline(self, label, start, end):
        path = self.root / "outputs" / "submission_v8" / "predictions" / (label.lower() + ".csv")
        table = pd.read_csv(path, dtype={ID: "string"})
        if not {ID, "Actual", "Prediction"}.issubset(table) or table[ID].isna().any() or not table[ID].is_unique:
            raise ValueError("V8 historical baseline needs unique ID, Actual and Prediction: " + str(path))
        ids = self.train[ID].astype("string")
        if not set(table[ID]).issubset(set(ids)):
            raise ValueError("V8 historical IDs missing from original processed Train")
        query = self.train.set_index(ID, drop=False).loc[table[ID]].reset_index(drop=True).copy()
        month = _num(query, MONTH)
        if not np.all((month >= start) & (month <= end)):
            raise ValueError("Historical baseline includes rows outside the fixed period")
        # Require the full, predeclared unique-ID evaluation window, not a cherry-picked subset.
        expected = self.train.loc[(_num(self.train, MONTH) >= start) & (_num(self.train, MONTH) <= end), ID].astype(str)
        if set(table[ID].astype(str)) != set(expected):
            raise ValueError("V8 evaluation population differs from the fixed unique-ID window")
        np.testing.assert_allclose(_num(query, TARGET), _num(table, "Actual"), rtol=0, atol=.01)
        baseline = _num(table, "Prediction")
        if not np.isfinite(baseline).all() or (baseline < 0).any():
            raise ValueError("Invalid V8 historical predictions")
        raw = self.raw_train.set_index(ID, drop=False).loc[table[ID]].reset_index(drop=True).drop(columns=[TARGET])
        from airport_policy import apply_airport_allocation
        replay, audit = apply_airport_allocation(raw, baseline)
        np.testing.assert_allclose(replay, baseline, rtol=0, atol=.02)
        locked = np.zeros(len(query), bool)
        if len(audit) and "Active" in audit:
            locked[audit.loc[audit.Active, "Position"].astype(int).to_numpy()] = True
        return {"frame": query, "raw": raw, "baseline": baseline, "locked": locked,
                "actual": _num(query, TARGET), "path": path, "hash": _hash(path)}

    def prepare(self):
        if self.ready:
            return self.preflight.copy()
        required = [self.root / "configs" / "input_manifest.json"] + [
            self.root / "outputs" / "submission_v8" / "predictions" / f"{label}.csv"
            for label in ("dev", "check")]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Complete the main notebook through V8, or correct PROJECT_ROOT. Missing: " + "; ".join(missing))
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.root / "configs" / "input_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        train_info = manifest["Train"]
        source = Path(train_info["file"]).expanduser()
        if not source.is_file() or _hash(source) != train_info["sha256"]:
            raise ValueError("Original Train source is missing or differs from the V8 input manifest")
        raw = pd.read_excel(source, dtype={ID: "string"})
        if TARGET not in raw or raw[ID].isna().all():
            raise ValueError("Original labelled Train required")
        self.preprocessor = study.TryPreprocessor(raw)
        # Determine ambiguity after the exact V8 target eligibility rules. A raw
        # invalid-target duplicate must not remove a uniquely valid V8 eval row.
        raw["_V9_Source_Position"] = np.arange(len(raw))
        processed = self.preprocessor.transform(raw, training=True).reset_index(drop=True)
        ids = processed[ID].astype("string")
        safe = ids.notna() & ids.str.strip().ne("").fillna(False) & ~ids.duplicated(keep=False)
        positions = pd.to_numeric(processed.loc[safe, "_V9_Source_Position"], errors="raise").to_numpy(int)
        self.raw_train = raw.iloc[positions].drop(columns=["_V9_Source_Position"]).reset_index(drop=True).copy()
        self.train = processed.loc[safe].drop(columns=["_V9_Source_Position"]).reset_index(drop=True).copy()
        if not self.train[ID].is_unique:
            raise ValueError("Global duplicate purge failed")
        self.manifest = manifest
        c = self.config
        self.sets = {"Dev": self._load_baseline("Dev", c.fit_end_month + 1, c.dev_end_month),
                     "Check": self._load_baseline("Check", c.dev_end_month + 1, c.check_end_month)}
        self.fit = self.train.loc[_num(self.train, MONTH) <= c.fit_end_month].reset_index(drop=True).copy()
        if not len(self.fit):
            raise ValueError("Historical Train is empty")
        for item in self.sets.values():
            if not set(self.fit[ID]).isdisjoint(item["frame"][ID]):
                raise ValueError("Historical fit/evaluation IDs overlap")
        sources = self._sources()
        self.provenance = {"implementation": IMPLEMENTATION, "sources": sources,
            "train_sha256": train_info["sha256"], "config": asdict(c),
            "baseline_hashes": {k: v["hash"] for k, v in self.sets.items()}}
        self.run_id = _digest(self.provenance)[:16]
        self.folder = self.folder / self.run_id; self.folder.mkdir(parents=True, exist_ok=True)
        self.a_helper, self.b_helper = self._helpers(self.fit)
        self.blocks = {}
        rows = [{"Set": "Historical_Fit", "Rows": len(self.fit), "Ordinary_Eligible": None,
                 "Commercial_Eligible": None, "Parking_Sales": int(_text(self.fit, "交易標的").eq("車位").sum()),
                 "Optimistic_Maximum_MAE_Reduction": None}]
        self.feasible = True
        for label, item in self.sets.items():
            frame = item["frame"]
            a = ordinary_block(frame, self.a_helper, c, cutoff=c.fit_end_month)
            b = commercial_block(frame, self.b_helper, c, cutoff=c.fit_end_month)
            a_mask = a[1] & ~item["locked"]
            b_mask = b[1] & (_num(frame, BUILDING) > 600) & ~item["locked"]
            if np.any(a_mask & b_mask):
                raise ValueError("V9 branches must be disjoint")
            self.blocks[label] = {"a_mask": a_mask, "b_mask": b_mask}
            cap = np.where(a_mask, c.ordinary_cap, 0) + np.where(b_mask, c.commercial_relative_cap * item["baseline"], 0)
            upper = float(np.minimum(np.abs(item["actual"] - item["baseline"]), cap).mean())
            self.feasible &= upper + 1e-6 >= c.min_overall_gain
            rows.append({"Set": label, "Rows": len(frame), "Ordinary_Eligible": int(a_mask.sum()),
                "Commercial_Eligible": int(b_mask.sum()), "Parking_Sales": None,
                "Optimistic_Maximum_MAE_Reduction": upper})
        self.preflight = pd.DataFrame(rows)
        del self.raw_train
        gc.collect()
        self.preflight.to_csv(self.folder / "preflight.csv", index=False)
        _json(self.folder / "protocol.json", {**self.provenance,
            "historical_fit_cutoff": "2024-03", "Dev": "2024-04 to 2024-06", "Check": "2024-07 to 2024-08",
            "evaluation_status": "Previously reviewed retrospective windows; not independent confirmation",
            "test_answers_used": False, "maximum_new_model_states": 5,
            "full_train_preprocessing_vocabulary": "Target-free compatibility vocabulary, as in V8",
            "forecast_simulation": "One stable ID-hash delay from 1 to 8 per training row",
            "optimistic_bound": "Sum(min(baseline absolute error, maximum allowed delta)) / all rows; not an attainable score estimate"})
        self.ready = True
        print({"Train_Only_Preflight": True, "Excluded_Ambiguous_ID_Rows": int((~safe).sum()),
            "Historical_Fit_Rows": len(self.fit), "Optimistic_Gain_Bound_Passed": bool(self.feasible),
            "Test_Valid_Read": False, "New_Models_Fitted": 0}, flush=True)
        return self.preflight.copy()

    def _helpers(self, fit):
        ordinary, _, _ = populations(fit)
        unit = np.divide(_num(fit, TARGET), _num(fit, BUILDING), out=np.full(len(fit), np.nan), where=ordinary)
        a = PriceReferenceHierarchy(min_rows=self.config.ordinary_reference_rows).fit(fit, unit, price_keys(fit))
        b = ParkingHelper(min_rows=self.config.parking_reference_rows).fit(fit)
        return a, b

    def _fit_branch(self, name, fit, helper, branch, context):
        if len(fit) < self.config.minimum_fit_rows:
            print(f"Skipped unsupported branch: {name} | Available_Rows: {len(fit)}", flush=True)
            return None
        delay = deterministic_delay(fit[ID], self.config)
        if branch == "a":
            x, mask, refs = ordinary_block(fit, helper, self.config, delay=delay)
            area = _num(fit, BUILDING)
            unit = np.divide(_num(fit, TARGET), area, out=np.full(len(fit), np.nan), where=area > 0)
            target = (unit - refs["Reference_Value"]) / SCALE
        else:
            x, mask, area, park = commercial_block(fit, helper, self.config, delay=delay)
            target = np.divide(_num(fit, TARGET) - park["Parking_Price"], area,
                out=np.full(len(fit), np.nan), where=area > 0) / SCALE
        mask &= np.isfinite(target) & np.isfinite(area) & (area > 0)
        if mask.sum() < self.config.minimum_fit_rows:
            print(f"Skipped unsupported branch: {name} | Fit_Rows: {mask.sum()}", flush=True)
            return None
        mean_area = float(area[mask].mean())
        weight = area[mask] / mean_area
        # For arbitrary predictions, weighted unit AE differs from total AE by one constant.
        probe = target[mask] + np.linspace(-.02, .02, int(mask.sum()))
        left = np.average(np.abs(probe - target[mask]), weights=weight)
        right = np.abs((probe - target[mask]) * area[mask] * SCALE).mean() / (mean_area * SCALE)
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
        bundle = fit_cached(x.loc[mask].reset_index(drop=True), target[mask], weight,
            name, self.folder, self.config, {**self.provenance, "context": context,
                "fit_ids": _digest(fit[ID].astype(str).tolist()), "branch": branch})
        _json(bundle["folder"] / "reference_state.json", helper.to_state())
        bundle.update(helper=helper, mean_area=mean_area)
        if branch == "b":
            strata = commercial_strata(fit)
            bundle["support"] = {str(k): {"rows": int((mask & (strata == k)).sum()),
                "maximum_gross_area": float(_num(fit, BUILDING)[mask & (strata == k)].max()),
                "maximum_net_area": float(area[mask & (strata == k)].max()),
                "median_net_unit": float(np.median(target[mask & (strata == k)]))}
                for k in sorted(set(strata[mask]))}
            _json(bundle["folder"] / "support.json", bundle["support"])
        del x; gc.collect()
        return bundle

    def _b_supported(self, frame, mask, bundle, allowed=None):
        strata, area = commercial_strata(frame), _num(frame, BUILDING)
        net = area - _num(frame, PARKING)
        ratio = np.full(len(frame), np.inf)
        for key, support in bundle["support"].items():
            matched = strata == key
            if support["rows"] >= 20:
                ratio[matched] = np.maximum(area[matched] / support["maximum_gross_area"],
                    net[matched] / support["maximum_net_area"])
        supported = mask & (ratio <= self.config.max_extrapolation_ratio)
        if allowed is not None:
            supported &= np.isin(strata, list(allowed))
        return supported, ratio

    def _stress(self):
        """Train-only project exclusion; no held-out price used to choose projects."""
        fit, c = self.fit, self.config
        _, commercial, _ = populations(fit)
        named = _text(fit, "建案行政區").map(_clean).to_numpy(bool)
        large = commercial & (_num(fit, BUILDING) > 600) & named
        groups = fit.loc[large, ["建案行政區", BUILDING]].copy()
        if not len(groups):
            return set(), pd.DataFrame()
        maximum = groups.groupby("建案行政區", sort=True)[BUILDING].max().sort_values(ascending=False, kind="stable")
        held = set(maximum.head(c.stress_projects).index.astype(str))
        heldmask = _text(fit, "建案行政區").isin(held).to_numpy(bool)
        remain, query = fit.loc[~heldmask].reset_index(drop=True), fit.loc[heldmask & commercial].reset_index(drop=True)
        if not len(remain) or not len(query):
            return set(), pd.DataFrame()
        helper = ParkingHelper(min_rows=c.parking_reference_rows).fit(remain)
        bundle = self._fit_branch("commercial_stress", remain, helper, "b", "Train_Project_Stress")
        if bundle is None:
            return set(), pd.DataFrame()
        x, valid, net, park = commercial_block(query, helper, c)
        valid &= _num(query, BUILDING) > 600
        supported, ratio = self._b_supported(query, valid, bundle)
        raw = net * SCALE * predict_cached(bundle, x) + park["Parking_Price"]
        strata = commercial_strata(query)
        rows, allowed = [], set()
        for key in sorted(set(strata[valid])):
            mask = valid & (strata == key) & supported
            fallback_unit = bundle["support"].get(str(key), {}).get("median_net_unit", np.nan)
            reference = np.maximum(net * SCALE * fallback_unit + park["Parking_Price"], 0)
            candidate = reference + np.clip(np.maximum(raw, 0) - reference,
                -c.commercial_relative_cap * reference, c.commercial_relative_cap * reference)
            actual = _num(query, TARGET)
            baseline_mae = float(np.abs(actual[mask] - reference[mask]).mean()) if mask.any() else np.nan
            candidate_mae = float(np.abs(actual[mask] - candidate[mask]).mean()) if mask.any() else np.nan
            improvement = np.abs(actual[mask] - reference[mask]) - np.abs(actual[mask] - candidate[mask])
            concentration = (float(np.max(np.abs(improvement)) / np.sum(np.abs(improvement)))
                if len(improvement) and np.sum(np.abs(improvement)) > 0 else 0)
            largest_positive = np.sort(np.maximum(improvement, 0))[::-1]
            net_after_top_two = float(improvement.sum() - largest_positive[:2].sum())
            passed = bool(mask.sum() >= c.minimum_stress_rows and candidate_mae <= baseline_mae
                and concentration <= .5)
            if passed:
                allowed.add(str(key))
            rows.append({"Stratum": key, "Held_Out_Rows": int((valid & (strata == key)).sum()),
                "Supported_Rows": int(mask.sum()), "Reference_MAE": baseline_mae,
                "Candidate_MAE": candidate_mae, "Largest_Error_Change_Share": concentration,
                "Net_Error_Saved_Excluding_Top_Two_Gains": net_after_top_two,
                "Maximum_Area_Ratio": float(ratio[mask].max()) if mask.any() else np.nan,
                "Passed": passed})
        report = pd.DataFrame(rows)
        report.to_csv(self.folder / "commercial_project_stress.csv", index=False)
        _json(self.folder / "stress_protocol.json", {"excluded_projects": sorted(held),
            "selection": "Top project maximum building area in historical Train, never price/residual",
            "reference": "Remaining-Train stratum median net unit price + same strictly-past parking estimate",
            "timing": "Same-period internal project exclusion, not out-of-time confirmation; remaining Train can include later dates",
            "not_claimed": "An independent comparison against V8",
            "allowed_strata": sorted(allowed)})
        return allowed, report

    def _predict_branches(self, frame, a, b, cutoff, allowed, locked):
        c = self.config
        a_values, b_values = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
        a_mask, b_mask = np.zeros(len(frame), bool), np.zeros(len(frame), bool)
        ratio = np.full(len(frame), np.inf)
        if a is not None:
            x, a_mask, refs = ordinary_block(frame, a["helper"], c, cutoff=cutoff)
            a_values = np.maximum(_num(frame, BUILDING) * (refs["Reference_Value"] + SCALE * predict_cached(a, x)), 0)
            a_mask &= np.isfinite(a_values) & ~locked
        if b is not None:
            x, b_mask, net, park = commercial_block(frame, b["helper"], c, cutoff=cutoff)
            b_mask &= _num(frame, BUILDING) > 600
            b_mask, ratio = self._b_supported(frame, b_mask, b, allowed)
            b_values = np.maximum(net * SCALE * predict_cached(b, x) + park["Parking_Price"], 0)
            b_mask &= np.isfinite(b_values) & ~locked
        if np.any(a_mask & b_mask):
            raise ValueError("Candidate branch overlap")
        return {"a": a_values, "b": b_values, "a_mask": a_mask, "b_mask": b_mask, "area_ratio": ratio}

    def _mix(self, item, pred, wa, wb):
        baseline, c = item["baseline"], self.config
        candidate = baseline.copy()
        for key, weight in [("a", wa), ("b", wb)]:
            mask = pred[key + "_mask"]
            cap = c.ordinary_cap if key == "a" else c.commercial_relative_cap * baseline[mask]
            candidate[mask] += weight * np.clip(pred[key][mask] - baseline[mask], -cap, cap)
        from airport_policy import apply_airport_allocation
        candidate, _ = apply_airport_allocation(item["raw"], np.maximum(candidate, 0))
        np.testing.assert_allclose(candidate[item["locked"]], baseline[item["locked"]], rtol=0, atol=.02)
        candidate[item["locked"]] = baseline[item["locked"]]
        return candidate

    def _metrics(self, item, pred, candidate):
        masks = {"All": np.ones(len(candidate), bool), "Ordinary_Branch": pred["a_mask"],
                 "Commercial_Branch": pred["b_mask"], "Unchanged": ~(pred["a_mask"] | pred["b_mask"])}
        return {name: measure(item["actual"], item["baseline"], candidate, mask, self.config)
                for name, mask in masks.items()}

    def _pass(self, metrics, wa, wb):
        checks = [protection_passes(metrics["All"], self.config)]
        for name, weight in [("Ordinary_Branch", wa), ("Commercial_Branch", wb)]:
            if weight > 0:
                m = metrics[name]
                checks.append(m["Rows"] > 0 and m["MAE_Reduction"] >= -1e-6 and protection_passes(m, self.config))
        return bool(all(checks))

    def run_validation(self):
        if self.validated:
            return self.result
        self.prepare()
        if not self.feasible:
            self.result = {"selection": pd.DataFrame(), "scores": pd.DataFrame(), "decision": {
                "Accepted": False, "Reason": "Fixed eligibility/caps cannot achieve the requested 100000 overall historical gain even with oracle corrections",
                "Test_Valid_Read": False, "Models_Fitted": 0}}
            return self._finish_validation()
        c = self.config
        eligible_b = any(self.blocks[k]["b_mask"].any() for k in self.sets)
        self.allowed, self.stress_scores = self._stress() if eligible_b else (set(), pd.DataFrame())
        a = self._fit_branch("ordinary_historical", self.fit, self.a_helper, "a", "Historical_Through_2024_03")
        b = (self._fit_branch("commercial_historical", self.fit, self.b_helper, "b", "Historical_Through_2024_03")
             if self.allowed else None)
        self.historical_models = {"a": a, "b": b}
        self.branch_predictions = {label: self._predict_branches(item["frame"], a, b, c.fit_end_month,
            self.allowed, item["locked"]) for label, item in self.sets.items()}
        rows, choices = [], []
        item, pred = self.sets["Dev"], self.branch_predictions["Dev"]
        a_weights, b_weights = c.weights if a else (0.0,), c.weights if b else (0.0,)
        for wa, wb in itertools.product(a_weights, b_weights):
            candidate = self._mix(item, pred, wa, wb)
            metrics = self._metrics(item, pred, candidate)
            passed = self._pass(metrics, wa, wb)
            rows.append({"Ordinary_Weight": wa, "Commercial_Weight": wb,
                "Dev_MAE": metrics["All"]["Candidate_MAE"], "MAE_Reduction": metrics["All"]["MAE_Reduction"],
                "Protection_Passed": passed, "New_Damaged": metrics["All"]["New_Damaged"],
                "Changed_Accurate_Increase_Pct": metrics["All"]["Changed_Accurate_Increase_Pct"]})
            if passed:
                choices.append((metrics["All"]["Candidate_MAE"], wa + wb, wa, wb))
        if not choices:
            raise RuntimeError("Even the zero-change control failed; inspect baseline replay")
        _, _, wa, wb = min(choices)
        self.weights = (float(wa), float(wb))
        scores, period_passes = [], {}
        self.candidates = {}
        for label, item in self.sets.items():
            pred = self.branch_predictions[label]
            candidate = self._mix(item, pred, wa, wb)
            self.candidates[label] = candidate
            metrics = self._metrics(item, pred, candidate)
            period_passes[label] = (self._pass(metrics, wa, wb)
                and metrics["All"]["MAE_Reduction"] >= c.min_overall_gain)
            scores.extend({"Set": label, "Segment": name, **metric} for name, metric in metrics.items())
            # Compact evidence: largest changes only, never full duplicate matrices.
            top = np.argsort(np.abs(candidate - item["baseline"]))[-200:][::-1]
            detail = item["frame"].iloc[top][[ID, "行政區", "建案名稱", "建物型態", BUILDING]].reset_index(drop=True)
            detail["Actual"], detail["Baseline"], detail["Candidate"] = item["actual"][top], item["baseline"][top], candidate[top]
            detail.to_csv(self.folder / f"largest_changes_{label.lower()}.csv", index=False)
        self.result = {"selection": pd.DataFrame(rows).sort_values("Dev_MAE", kind="stable"),
            "scores": pd.DataFrame(scores), "decision": {"Accepted": bool(all(period_passes.values())),
                "Selected_Ordinary_Weight": wa, "Selected_Commercial_Weight": wb,
                "Weight_Selection": "Dev only; Check is accept/reject without fallback retuning",
                "Period_Passed": period_passes, "Minimum_Overall_MAE_Reduction": c.min_overall_gain,
                "Commercial_Allowed_Strata": sorted(self.allowed),
                "Evaluation_Status": "Previously reviewed retrospective windows; no blind confirmation claimed",
                "Test_Valid_Read": False, "Reason": "All declared gates passed" if all(period_passes.values())
                    else "Required overall gain or accuracy protection failed; preserve V8"}}
        return self._finish_validation()

    def _finish_validation(self):
        self.result["selection"].to_csv(self.folder / "dev_weight_selection.csv", index=False)
        self.result["scores"].to_csv(self.folder / "paired_scores.csv", index=False)
        _json(self.folder / "decision.json", self.result["decision"])
        self.validation_lock = _digest({"decision": self.result["decision"],
            "weights": getattr(self, "weights", None), "allowed": sorted(getattr(self, "allowed", set())),
            "config": asdict(self.config), "provenance": self.provenance})
        _json(self.folder / "validation_lock.json", {"digest": self.validation_lock})
        self.validated = True
        print({"Validation_Completed": True, "Accepted": self.result["decision"]["Accepted"],
               "Test_Valid_Read": False, "Submission_Exported": False}, flush=True)
        return self.result

    def plot_validation(self):
        if not self.validated:
            raise RuntimeError("Run validation first")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        scores = self.result["scores"]
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
            "axes.spines.right": False, "axes.titleweight": "bold", "font.size": 10})
        fig = plt.figure(figsize=(16, 9), facecolor="#f6f8fb")
        grid = fig.add_gridspec(2, 2, left=.17, right=.96, top=.82, bottom=.11, hspace=.48, wspace=.34)
        fig.text(.17, .94, "V9 | Evidence before deployment", fontsize=23, weight="bold", color="#122b42")
        fig.text(.17, .895, "Train-only candidates  /  fixed Dev selection  /  Check accept-or-reject", color="#506477", fontsize=12)
        if scores.empty:
            ax = fig.add_subplot(grid[:, :]); ax.axis("off")
            ax.text(.05, .65, "Preflight stopped: requested gain is infeasible under fixed caps", size=17, weight="bold", wrap=True)
            ax.text(.05, .42, "No models fitted. Existing V8 preserved.\nSee preflight.csv and decision.json.", size=14)
        else:
            all_rows = scores.loc[scores.Segment.eq("All")].set_index("Set").loc[["Dev", "Check"]]
            ax = fig.add_subplot(grid[0, 0]); positions = np.arange(2)
            ax.barh(positions + .17, all_rows.Current_MAE, .32, color="#b5c2d2", label="V8")
            ax.barh(positions - .17, all_rows.Candidate_MAE, .32, color="#1b8e9b", label="V9 candidate")
            ax.set_yticks(positions, ["Dev", "Check"]); ax.invert_yaxis(); ax.set_title("Overall MAE"); ax.legend(frameon=False, fontsize=9)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M")); ax.set_xlabel("TWD; lower is better")
            ax = fig.add_subplot(grid[0, 1]); gains = all_rows.MAE_Reduction.to_numpy()
            bars = ax.bar(["Dev", "Check"], gains / 1000, color=["#1b8e9b" if g >= self.config.min_overall_gain else "#d89042" for g in gains])
            ax.axhline(self.config.min_overall_gain / 1000, color="#b44545", linestyle="--", label="Required: 100k")
            ax.bar_label(bars, fmt="%.1f", padding=4); ax.set_title("Overall error reduction"); ax.set_ylabel("Thousand TWD"); ax.legend(frameon=False)
            ax = fig.add_subplot(grid[1, 0]); branch = scores.loc[scores.Segment.isin(["Ordinary_Branch", "Commercial_Branch"])].copy()
            labels = [r.Set + " / " + ("Residential" if r.Segment == "Ordinary_Branch" else "Commercial") for r in branch.itertuples()]
            values = branch.Changed_Accurate_Increase_Pct.fillna(0).to_numpy()
            ax.barh(labels, values, color="#677bb1"); ax.axvline(self.config.max_accurate_increase_pct, linestyle="--", color="#b44545")
            ax.set_title("Changed accurate cohort: MAE increase"); ax.set_xlabel("Percent; required <= 2%")
            ax = fig.add_subplot(grid[1, 1]); ax.axis("off")
            accepted = self.result["decision"]["Accepted"]
            ax.text(0, .90, "ACCEPTED" if accepted else "V8 PRESERVED", fontsize=23, weight="bold", color="#157b79" if accepted else "#a76824")
            wa, wb = self.weights
            ax.text(0, .64, f"Locked weights: residential {wa:g} / commercial {wb:g}\nNew damaged: Dev {int(all_rows.loc['Dev','New_Damaged'])}, Check {int(all_rows.loc['Check','New_Damaged'])}", fontsize=12, linespacing=1.7)
            ax.text(0, .24, "Retrospective windows; Kaggle gain is not guaranteed.\nUnsupported products retain V8. Airport rows stay locked.", fontsize=10, color="#506477", linespacing=1.8)
        fig.text(.17, .035, "No Test answers or Valid labels used for fitting, weight selection, or acceptance.", color="#506477", fontsize=10)
        fig.savefig(self.folder / "validation_overview.png", dpi=300, facecolor=fig.get_facecolor())
        fig.savefig(self.folder / "validation_overview.svg", facecolor=fig.get_facecolor())
        return fig

    def export_if_accepted(self):
        if not self.validated:
            raise RuntimeError("Run validation first")
        if not self.result["decision"]["Accepted"]:
            return {"Exported": False, "Reason": self.result["decision"]["Reason"],
                    "Test_Read": False, "Production_Models_Fitted": 0, "Existing_V8_Preserved": True}
        locked = _digest({"decision": self.result["decision"], "weights": self.weights,
            "allowed": sorted(self.allowed), "config": asdict(self.config), "provenance": self.provenance})
        saved_decision = json.loads((self.folder / "decision.json").read_text())
        saved_lock = json.loads((self.folder / "validation_lock.json").read_text())["digest"]
        if locked != self.validation_lock or locked != saved_lock or saved_decision != self.result["decision"]:
            raise ValueError("Validation decision, weights, support strata or config changed; rerun validation")
        # Guard the in-memory decision against edits, source drift and baseline replacements.
        if self._sources() != self.provenance["sources"]:
            raise ValueError("Source implementation changed after validation; rerun validation")
        if _hash(Path(self.manifest["Train"]["file"])) != self.provenance["train_sha256"]:
            raise ValueError("Train changed after validation")
        for item in self.sets.values():
            if _hash(item["path"]) != item["hash"]:
                raise ValueError("Historical baseline changed after validation")
        wa, wb = self.weights
        source = Path(self.manifest["Test"]["file"])
        if _hash(source) != self.manifest["Test"]["sha256"]:
            raise ValueError("Original unlabelled Test differs from the V8 manifest")
        # Header check rejects answer columns before any Test price can be read.
        _unlabelled(pd.read_excel(source, nrows=0))
        raw = pd.read_excel(source, dtype={ID: "string"}); _unlabelled(raw)
        if raw[ID].isna().any() or not raw[ID].is_unique:
            raise ValueError("Unlabelled Test IDs must be unique/nonmissing")
        baseline_path = self.root / "outputs" / "submission_v8" / "submission_v8.csv"
        index_path = self.root / "presentation" / "tables" / "submission_index.csv"
        if index_path.is_file():
            index = pd.read_csv(index_path)
            expected = index.loc[index.Version.eq("submission_v8"), "SHA256"]
            if len(expected) != 1 or expected.iloc[0] != _hash(baseline_path):
                raise ValueError("V8 production submission differs from the saved submission index")
        saved = pd.read_csv(baseline_path, dtype={ID: "string"})
        if len(saved) != len(raw) or not saved[ID].is_unique or set(saved[ID]) != set(raw[ID]):
            raise ValueError("V8 submission/Test IDs differ")
        baseline = pd.to_numeric(saved.set_index(ID).loc[raw[ID], TARGET], errors="raise").to_numpy(float, copy=True)
        from airport_policy import apply_airport_allocation
        replay, audit = apply_airport_allocation(raw, baseline)
        np.testing.assert_allclose(replay, baseline, rtol=0, atol=.02)
        locked = np.zeros(len(raw), bool)
        if len(audit) and "Active" in audit:
            locked[audit.loc[audit.Active, "Position"].astype(int).to_numpy()] = True
        frame = self.preprocessor.transform(raw, training=False).reset_index(drop=True)
        np.testing.assert_array_equal(frame[ID].astype(str), raw[ID].astype(str))
        a_helper, b_helper = self._helpers(self.train)
        a = self._fit_branch("ordinary_full_train", self.train, a_helper, "a", "Full_Train") if wa > 0 else None
        b = self._fit_branch("commercial_full_train", self.train, b_helper, "b", "Full_Train") if wb > 0 else None
        if (wa > 0 and a is None) or (wb > 0 and b is None):
            raise ValueError("Accepted branch unavailable in full Train; no submission exported")
        pred = self._predict_branches(frame, a, b, float(np.nanmax(_num(self.train, MONTH))), self.allowed, locked)
        if (wa > 0 and not pred["a_mask"].any()) or (wb > 0 and not pred["b_mask"].any()):
            raise ValueError("An accepted branch has no production support; no unchanged V8 copy will be exported as V9")
        item = {"baseline": baseline, "raw": raw, "locked": locked}
        candidate = self._mix(item, pred, wa, wb)
        np.testing.assert_array_equal(candidate[locked], baseline[locked])
        from project import write_submission
        destination = self.root / "outputs" / "submission_v9" / self.run_id
        submission = write_submission(destination / "submission_v9.csv", raw, candidate)
        # Save only top changes and abstention counts, not a second full Test matrix.
        top = np.argsort(np.abs(candidate - baseline))[-100:][::-1]
        detail = frame.iloc[top][[ID, "建案名稱", "行政區", "建物型態", BUILDING]].reset_index(drop=True)
        detail["Baseline"], detail["Candidate"] = baseline[top], candidate[top]
        detail["Difference"] = candidate[top] - baseline[top]
        detail.to_csv(destination / "largest_test_changes.csv", index=False)
        _, commercial, _ = populations(frame)
        self._commercial_audit(frame, baseline, pred, b).to_csv(
            destination / "commercial_support_audit.csv", index=False)
        summary = {"Exported": True, "Submission": str(submission), "sha256": _hash(submission),
            "Existing_V8_SHA256": _hash(baseline_path), "Existing_V8_Preserved": True,
            "Ordinary_Active_Rows": int(pred["a_mask"].sum()), "Commercial_Active_Rows": int(pred["b_mask"].sum()),
            "Large_Commercial_Abstained": int((commercial & (_num(frame, BUILDING) > 600) & ~pred["b_mask"]).sum()),
            "Airport_Locked_Rows": int(locked.sum()), "Test_Answers_Used": False,
            "Kaggle_Uploaded": False, "Weights": self.weights}
        _json(destination / "recipe.json", {**self.provenance, **summary, "historical_decision": self.result["decision"]})
        _json(self.folder / "export.json", summary)
        return summary

    def _commercial_audit(self, frame, baseline, pred, bundle):
        _, commercial, family = populations(frame)
        gross, net = _num(frame, BUILDING), _num(frame, BUILDING) - _num(frame, PARKING)
        strata = commercial_strata(frame)
        positions = np.flatnonzero(commercial & (gross > 600))
        rows = []
        for i in positions:
            support = bundle["support"].get(str(strata[i]), {}) if bundle else {}
            gross_ratio = gross[i] / support.get("maximum_gross_area", np.nan)
            net_ratio = net[i] / support.get("maximum_net_area", np.nan)
            if pred["b_mask"][i]:
                reason = "Eligible"
            elif bundle is None:
                reason = "Commercial_Branch_Not_Selected"
            elif support.get("rows", 0) < 20:
                reason = "Insufficient_Same_Type_Use_Training_Rows"
            elif max(gross_ratio, net_ratio) > self.config.max_extrapolation_ratio:
                reason = "Gross_Or_Net_Area_Outside_Training_Support"
            elif str(strata[i]) not in self.allowed:
                reason = "Stratum_Not_Approved_By_Train_Project_Stress"
            else:
                reason = "Parking_Or_Forecast_Eligibility_Failed_Or_Airport_Locked"
            rows.append({ID: frame.iloc[i][ID], "Project": frame.iloc[i]["建案名稱"],
                "Use_Family": family[i], "Stratum": strata[i], "Gross_Area": gross[i],
                "Net_Area": net[i], "Training_Rows": support.get("rows", 0),
                "Gross_Area_Ratio": gross_ratio, "Net_Area_Ratio": net_ratio,
                "Baseline": baseline[i], "Raw_Commercial_Candidate": pred["b"][i],
                "Applied": bool(pred["b_mask"][i]), "Reason": reason})
        return pd.DataFrame(rows).sort_values("Gross_Area", ascending=False) if rows else pd.DataFrame()

    def export_review(self):
        if not self.validated:
            raise RuntimeError("Run validation first")
        destination = self.folder / "v9_review.zip"
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.folder.iterdir()):
                if path.is_file() and path.suffix.lower() in {".csv", ".json", ".png", ".svg"}:
                    archive.write(path, path.name)
        return destination

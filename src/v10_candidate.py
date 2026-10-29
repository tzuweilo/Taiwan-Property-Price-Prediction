"""One new joint-quantile calibrator of a load-only, uncalibrated V8 ensemble.

Historical base models end in Aug 2023. Their later regional/age calibrators
are deliberately absent from training inputs. Importing this file never fits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import copy
import hashlib
import importlib.metadata
import json
import zipfile

import numpy as np
import pandas as pd

import study_pipeline as study
from v10_base import FrozenBase

ID, TARGET, MONTH = study.ID, study.TARGET, "交易時間序號"
AREA, UNIT_SCALE = study.BUILDING, 1000.0
IMPLEMENTATION = "raw_ensemble_joint_quantile_calibration_v10_1"


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    default=str), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def numbers(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float, na_value=np.nan, copy=True)


def check_unlabelled(frame):
    study.assert_unlabelled(frame)
    extra = [str(c) for c in frame if any(s in str(c).casefold() for s in
             ("answer", "actual", "ground_truth", "true_price", "補價", "答案", "實際總價"))]
    if extra:
        raise ValueError("Original unlabelled Test only: " + ", ".join(extra))


def unique_ids(frame):
    ids = frame[ID].astype("string")
    return (ids.notna() & ids.str.strip().ne("").fillna(False) &
            ~ids.duplicated(keep=False)).to_numpy(bool, copy=True)


def housing_mask(frame):
    area = numbers(frame, AREA)
    kind = frame["建物型態"].astype("string").fillna("")
    use = frame["主要用途"].astype("string").fillna("")
    target = frame["交易標的"].astype("string").fillna("")
    return (np.isfinite(area) & (area >= 10) & (area <= 300)
        & kind.str.contains("住宅|華廈|公寓|透天|套房", regex=True).to_numpy(bool)
        & ~use.str.contains("工業|工廠|商業|辦公|店舖|店鋪", regex=True).to_numpy(bool)
        & target.str.contains("房地|建物", regex=True).to_numpy(bool)
        & ~(numbers(frame, "交易建物筆數") > 1))


@dataclass(frozen=True)
class V10Config:
    threads: int = 8
    iterations: int = 2200
    depth: int = 6
    learning_rate: float = .035
    l2_leaf_reg: float = 25.0
    minimum_leaf_rows: int = 100
    seed: int = 42
    old_cutoff: int = 187
    train_end: int = 196
    tune_month: int = 197
    check_end: int = 199
    weights: tuple = (0.0, .25, .5, .75, 1.0)
    maximum_change_twd: float = 3_000_000.0
    maximum_relative_change: float = .20
    minimum_stratum_rows: int = 40
    tune_accurate_increase_pct: float = 10.0
    check_accurate_increase_pct: float = 15.0
    maximum_damaged_fraction: float = .01
    minimum_tune_gain: float = 10_000.0
    minimum_check_gain: float = 25_000.0
    early_stopping_rounds: int = 200

    def __post_init__(self):
        if self.threads < 1 or self.iterations < 1:
            raise ValueError("Positive thread and tree counts required")
        if not (self.old_cutoff < self.train_end < self.tune_month < self.check_end):
            raise ValueError("Invalid fixed chronology")
        if not self.weights or 0 not in self.weights or any(not 0 <= x <= 1 for x in self.weights):
            raise ValueError("Weights must include zero and lie in [0,1]")
        if self.maximum_change_twd <= 0 or not 0 < self.maximum_relative_change <= 1:
            raise ValueError("Invalid prospective caps")


def market_table(train, cutoff):
    """Simple historical aggregations, not an additional predictive model."""
    mask = housing_mask(train) & (numbers(train, MONTH) <= cutoff)
    data = train.loc[mask, ["行政區", "縣市", "建物型態", "主要用途", MONTH, AREA, TARGET]].copy()
    data["Unit"] = numbers(data, TARGET) / numbers(data, AREA)
    result = {}
    for name, columns in (("District_Type", ["行政區", "建物型態"]),
                          ("City_Type", ["縣市", "建物型態"])):
        group = data.assign(Key=data[columns].astype(str).agg("|".join, axis=1))
        tables = []
        for window in (6, 12, 24):
            recent = group.loc[numbers(group, MONTH) > cutoff-window]
            summary = recent.groupby("Key", sort=True).agg(
                Unit=("Unit", "median"), Rows=("Unit", "size"), Months=(MONTH, "nunique"))
            summary.columns = [f"{name}_{window}M_{c}" for c in summary]
            tables.append(summary)
        result[name] = pd.concat(tables, axis=1)
    return result


def model_features(frame, detail, tables):
    """Every input is available without the query's price or error status."""
    source = frame.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)
    x = study.feature_frame(source, 6)
    # Avoid memorising one project's residual after a different base origin.
    excluded = [c for c in x if c.startswith("Project") or c in
                {"Block", "Transaction_Year", "Month_Index", "Transaction_Month", "Completion_Year"}]
    x = x.drop(columns=excluded)
    area = numbers(source, AREA)
    detail = detail.drop(columns=[ID, TARGET, "Actual", "Prediction", "Residual", "AE"], errors="ignore").reset_index(drop=True)
    if "Raw_Base" not in detail or len(detail) != len(source):
        raise ValueError("Raw component predictions are unavailable or misaligned")
    raw_base = numbers(detail, "Raw_Base")
    for col in detail:
        if pd.api.types.is_numeric_dtype(detail[col]):
            values = numbers(detail, col)
            x["Base_" + col] = values
            is_total_price = (col in {"Raw_Base", "Legacy_Raw", "Try_History", "Try_History_Unit",
                "Try_Matched", "Try_LGBM", "Try_Final", "Try_Unit_Total",
                "Legacy_Project_Reference", "Legacy_District_Reference"}
                or col.endswith("_total") or "_Reference_Price" in col)
            if is_total_price:
                x["Per_Area_" + col] = study.ratio(values, area)
                x["Relative_To_Base_" + col] = study.ratio(values, np.maximum(raw_base, 1)) - 1
        else:
            x["Base_" + col] = detail[col].astype("string").fillna("Missing").astype(str)
    for name, columns in (("District_Type", ["行政區", "建物型態"]),
                          ("City_Type", ["縣市", "建物型態"])):
        key = source[columns].astype(str).agg("|".join, axis=1)
        table = tables[name]
        for col in table:
            x[col] = key.map(table[col]).to_numpy(float, na_value=np.nan, copy=True)
        six, twelve = x[f"{name}_6M_Unit"].to_numpy(float), x[f"{name}_12M_Unit"].to_numpy(float)
        count = x[f"{name}_6M_Rows"].fillna(0).to_numpy(float)
        trend = np.clip(study.ratio(six, twelve)-1, -.25, .25) * count/(count+100)
        x[name + "_Shrunk_Trend"] = trend
        x[name + "_Trend_Horizon"] = trend * numbers(detail, "Forecast_Gap")
        x[name + "_Reference_vs_Base"] = study.ratio(six * area, np.maximum(raw_base, 1))-1
    price_cols = [col for col in detail if col in {"Legacy_Raw", "Try_History", "Try_LGBM", "Try_Final"}]
    if price_cols:
        values = detail[price_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        x["Component_Spread_Relative"] = study.ratio(np.nanmax(values, axis=1)-np.nanmin(values, axis=1), np.maximum(raw_base, 1))
    for col in x:
        if pd.api.types.is_numeric_dtype(x[col]):
            v = numbers(x, col).astype("float32"); v[~np.isfinite(v)] = np.nan; x[col] = v
        else:
            x[col] = x[col].astype("string").fillna("Missing").astype(str)
    if not x.columns.is_unique or {ID, TARGET, "Actual", "Residual", "Prediction"}.intersection(x):
        raise ValueError("Forbidden input column or duplicate feature")
    return x


def quantile_prices(raw_base, area, predicted):
    predicted = np.asarray(predicted, float)
    if predicted.shape != (len(raw_base), 3) or not np.isfinite(predicted).all():
        raise ValueError("Expected three finite joint-quantile outputs per transaction")
    crossed = predicted[:, 0] > predicted[:, 1]
    crossed |= predicted[:, 1] > predicted[:, 2]
    # Preserve the model's central q50. Expand crossed bounds deterministically.
    ordered = np.column_stack([np.min(predicted, axis=1), predicted[:, 1], np.max(predicted, axis=1)])
    prices = np.maximum(np.asarray(raw_base)[:, None] + np.asarray(area)[:, None]*UNIT_SCALE*ordered, 0)
    return prices, crossed


def correction(baseline, prices, eligible, weight, config):
    baseline = np.array(baseline, dtype=float, copy=True)
    eligible = np.asarray(eligible, bool)
    outside = (baseline < prices[:, 0]) | (baseline > prices[:, 2])
    active = eligible & outside & np.isfinite(prices).all(axis=1)
    maximum = np.minimum(config.maximum_change_twd, config.maximum_relative_change * baseline)
    delta = np.clip(prices[:, 1]-baseline, -maximum, maximum)
    result = baseline.copy()
    result[active] += float(weight)*delta[active]
    result = np.maximum(result, 0)
    np.testing.assert_array_equal(result[~active], baseline[~active])
    return result, active & (result != baseline)


def score(actual, baseline, candidate, mask, label, segment):
    mask = np.asarray(mask, bool)
    changed = mask & (np.abs(candidate-baseline) > 1e-6)
    old, new = np.abs(actual-baseline), np.abs(actual-candidate)
    accurate = mask & (old <= 500_000)
    changed_accurate = accurate & changed
    def mean(a, m):
        return float(np.mean(a[m])) if m.any() else np.nan
    before, after = mean(old, mask), mean(new, mask)
    ab, aa = mean(old, accurate), mean(new, accurate)
    cb, ca = mean(old, changed_accurate), mean(new, changed_accurate)
    return {"Set": label, "Segment": segment, "Rows": int(mask.sum()), "Changed_Rows": int(changed.sum()),
        "V8_MAE": before, "Candidate_MAE": after, "MAE_Reduction": before-after,
        "Improvement_Pct": 100*(1-after/before) if before > 0 else 0,
        "Median_Residual": float(np.median((actual-candidate)[mask])) if mask.any() else np.nan,
        "Accurate_Rows": int(accurate.sum()), "Accurate_Before": ab, "Accurate_After": aa,
        "Accurate_Increase_Pct": 100*(aa/ab-1) if ab > 0 else 0,
        "Changed_Accurate_Rows": int(changed_accurate.sum()),
        "Changed_Accurate_Increase_Pct": 100*(ca/cb-1) if cb > 0 else 0,
        "New_Damaged": int((accurate & (new > 1_000_000)).sum()),
        "Mean_Abs_Change": mean(np.abs(candidate-baseline), mask)}


class V10Experiment:
    def __init__(self, root, config=None):
        self.root, self.config = Path(root).expanduser().resolve(), config or V10Config()
        self.ready = self.trained = self.validated = False
        self.folder = None

    def prepare(self):
        if self.ready:
            return self.preflight.copy()
        c = self.config
        self.manifest = json.loads((self.root / "configs/input_manifest.json").read_text())
        source = Path(self.manifest["Train"]["file"])
        if study.hash_file(source) != self.manifest["Train"]["sha256"]:
            raise ValueError("Train does not match the completed V8 run")
        # Original dtypes are essential to the native cached-model signatures.
        raw = pd.read_excel(source)
        self.base = FrozenBase(self.root, raw, c.threads)
        tagged = raw.assign(_V10_Source_Position=np.arange(len(raw)))
        processed = self.base.processor.transform(tagged, training=True)
        safe = unique_ids(processed)
        positions = numbers(processed.loc[safe], "_V10_Source_Position").astype(int)
        self.train = processed.loc[safe].drop(columns=["_V10_Source_Position"]).reset_index(drop=True)
        self.raw_train = raw.iloc[positions].reset_index(drop=True).copy()
        self.train[ID] = self.train[ID].astype(str)
        self.raw_train[ID] = self.raw_train[ID].astype(str)
        np.testing.assert_array_equal(self.train[ID], self.raw_train[ID])
        self.provenance = {"implementation": IMPLEMENTATION, "train_sha256": self.manifest["Train"]["sha256"],
            "configuration": asdict(c), "sources": {p.name: study.hash_file(p) for p in
                [Path(__file__), Path(__file__).with_name("v10_base.py"), Path(study.__file__)]},
            "environment": {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "catboost", "lightgbm")}}
        self.run_id = digest(self.provenance)[:16]
        self.folder = self.root / "outputs/submission_v10_onefit" / self.run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        month = numbers(self.train, MONTH)
        history_mask = (month > c.old_cutoff) & (month <= c.check_end)
        self.query = self.train.loc[history_mask].reset_index(drop=True)
        self.raw_query = self.raw_train.loc[history_mask].drop(columns=[TARGET]).reset_index(drop=True)
        self.base.prepare_historical()
        # Fail before using the single fit budget if either old context is absent.
        self.base.prepare_production()
        result = self.base.predict(self.raw_query, "historical")
        self.detail = result["detail"].reset_index(drop=True)
        np.testing.assert_array_equal(self.detail[ID].astype(str), self.query[ID].astype(str))
        # Validate exact Try replay before the only authorised training call.
        self.baseline = np.full(len(self.query), np.nan)
        query_month = numbers(self.query, MONTH)
        for label, low, high in (("dev", 195, 197), ("check", 198, 199)):
            path = self.root / f"outputs/submission_v8/predictions/{label}.csv"
            saved = pd.read_csv(path, dtype={ID: "string"})
            saved[ID] = saved[ID].astype(str)
            mask = (query_month >= low) & (query_month <= high)
            self._verify_saved(saved, self.query.loc[mask])
            aligned = saved.set_index(ID).loc[self.query.loc[mask, ID]]
            self.baseline[mask] = numbers(aligned, "Prediction")
            if not np.isfinite(self.baseline[mask]).all() or (self.baseline[mask] < 0).any():
                raise ValueError("V8 historical predictions must be finite and nonnegative")
            v7 = pd.read_csv(self.root / f"outputs/submission_v7/predictions/{label}.csv", dtype={ID: "string"})
            v7[ID] = v7[ID].astype(str); v7 = v7.set_index(ID).loc[self.query.loc[mask, ID]]
            legacy, fallback = numbers(v7, "Legacy"), v7["Legacy_Fallback"].astype(bool).to_numpy()
            expected_try = 2*numbers(v7, "Try_V03")-legacy
            actual_try = numbers(self.detail.loc[mask], "Try_Final")
            np.testing.assert_allclose(actual_try, expected_try, rtol=1e-10, atol=.05)
            np.testing.assert_array_equal(fallback, self.detail.loc[mask, "Legacy_Fallback"].to_numpy(bool))
        self.locked = self._airport_locked(self.raw_query)
        base_fit = self.train.loc[numbers(self.train, MONTH) <= c.old_cutoff]
        self.market = market_table(base_fit, c.old_cutoff)
        self.x = model_features(self.query, self.detail, self.market)
        self.actual, self.area = numbers(self.query, TARGET), numbers(self.query, AREA)
        self.eligible = housing_mask(self.query) & ~self.locked
        self.fit_mask = self.eligible & (query_month <= c.train_end)
        self.tune_mask = self.eligible & (query_month == c.tune_month)
        self.check_mask = query_month > c.tune_month
        if self.fit_mask.sum() < 1000 or self.tune_mask.sum() < 100:
            raise ValueError("Insufficient fixed training/June rows; no new model fitted")
        self.stratum_columns = ["縣市", "建物型態", "主要用途"]
        key = self.query[self.stratum_columns].astype(str).agg("|".join, axis=1)
        fit_keys = key[self.fit_mask]
        counts = fit_keys.value_counts()
        self.support = {str(k): int(v) for k, v in counts.items()}
        self.support_mask = key.map(counts).fillna(0).to_numpy(float) >= c.minimum_stratum_rows
        # The outcome is never concatenated into x. Only this vector uses y.
        self.target = np.divide(self.actual-numbers(self.detail, "Raw_Base"), self.area*UNIT_SCALE,
            out=np.full(len(self.area), np.nan), where=np.isfinite(self.area) & (self.area > 0))
        if not np.isfinite(self.target[self.fit_mask | self.tune_mask]).all():
            raise ValueError("Invalid signed residual training target")
        if set(self.query.loc[self.fit_mask, ID]) & set(self.query.loc[~(query_month <= c.train_end), ID]):
            raise ValueError("Temporal ID leakage")
        self.provenance["old_model_provenance"] = copy.deepcopy(self.base.provenance)
        save_json(self.folder / "protocol.json", {**self.provenance,
            "maximum_new_model_fits": 1, "new_model": "Joint MultiQuantile residual of uncalibrated ensemble",
            "training": "2023-09 to 2024-05", "tuning": "2024-06", "check": "2024-07 to 2024-08",
            "check_is_blind": False, "test_answers_used": False, "production_refit": False,
            "scope": "Housing, single building or unspecified count, 10 to 300 sqm; known support; future months only",
            "confidence_note": "Conditional quantiles are an abstention heuristic; not a calibrated coverage guarantee"})
        self.preflight = pd.DataFrame([
            {"Set": "Residual_Train_2023_09_2024_05", "Rows": int((query_month <= c.train_end).sum()), "Eligible_Rows": int(self.fit_mask.sum())},
            {"Set": "Tune_2024_06", "Rows": int((query_month == c.tune_month).sum()), "Eligible_Rows": int(self.tune_mask.sum())},
            {"Set": "Check_2024_07_08", "Rows": int(self.check_mask.sum()), "Eligible_Rows": int((self.check_mask & self.eligible).sum())}])
        self.preflight.to_csv(self.folder / "preflight.csv", index=False)
        self.ready = True
        print({"Maximum_New_Models": 1, "Old_Models_Load_Only": True, "Features": self.x.shape[1],
               "Test_Valid_Read": False, "Production_Refit": False}, flush=True)
        return self.preflight.copy()

    @staticmethod
    def _verify_saved(saved, frame):
        if not saved[ID].is_unique or set(saved[ID].astype(str)) != set(frame[ID].astype(str)):
            raise ValueError("Saved V8 evaluation population changed")
        aligned = saved.set_index(ID).loc[frame[ID].astype(str)]
        np.testing.assert_allclose(numbers(aligned, "Actual"), numbers(frame, TARGET), rtol=0, atol=.01)

    @staticmethod
    def _airport_locked(raw):
        from airport_policy import apply_airport_allocation
        _, audit = apply_airport_allocation(raw.drop(columns=[TARGET], errors="ignore"), np.ones(len(raw)))
        locked = np.zeros(len(raw), bool)
        if len(audit) and "Active" in audit:
            locked[audit.loc[audit.Active, "Position"].astype(int).to_numpy()] = True
        return locked

    def train_once(self):
        self.prepare()
        if self.trained:
            return self.training_summary
        from catboost import CatBoostRegressor, Pool
        c = self.config
        columns = list(self.x.columns)
        categorical = self.x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        params = dict(loss_function="MultiQuantile:alpha=0.2,0.5,0.8", iterations=c.iterations,
            learning_rate=c.learning_rate, depth=c.depth, l2_leaf_reg=c.l2_leaf_reg,
            grow_policy="Depthwise", min_data_in_leaf=c.minimum_leaf_rows, random_seed=c.seed,
            thread_count=c.threads, task_type="CPU", verbose=200,
            allow_writing_files=True, train_dir=str(self.folder / "training_log"))
        key = digest({"provenance": self.provenance, "x": study.frame_hash(self.x.loc[self.fit_mask]),
            "y": self.target[self.fit_mask].tolist(), "params": params, "columns": columns})
        path, metadata_path = self.folder / "model.cbm", self.folder / "model.json"
        self.model = CatBoostRegressor(**params)
        if path.is_file() and metadata_path.is_file():
            meta = json.loads(metadata_path.read_text())
            if meta["signature"] != key or study.hash_file(path) != meta["sha256"]:
                raise ValueError("Saved one-fit model changed; automatic refitting is prohibited")
            self.model.load_model(str(path)); fits = 0
            print("Loaded completed V10 model. Additional training calls: 0", flush=True)
        else:
            if path.exists() or metadata_path.exists():
                raise ValueError("Incomplete final model files; keep the snapshot and inspect before retrying")
            ledger_path = self.root / "outputs/submission_v10_onefit/fit_budget.json"
            if ledger_path.is_file():
                ledger = json.loads(ledger_path.read_text())
                if ledger["signature"] != key:
                    raise RuntimeError("The one-model training budget is already assigned to another configuration; no second model will be fitted")
                if ledger.get("state") == "Completed":
                    raise RuntimeError("The one authorised fit was already completed, but its model cache is missing. Restore that cache; automatic retraining is prohibited")
            save_json(ledger_path, {"signature": key, "run_id": self.run_id, "maximum_distinct_models": 1,
                                   "state": "Started", "folder": str(self.folder)})
            average_area = float(self.area[self.fit_mask].mean())
            fit_pool = Pool(self.x.loc[self.fit_mask], self.target[self.fit_mask],
                cat_features=categorical, weight=self.area[self.fit_mask]/average_area)
            tune_pool = Pool(self.x.loc[self.tune_mask], self.target[self.tune_mask],
                cat_features=categorical, weight=self.area[self.tune_mask]/average_area)
            print({"Training": "ONE joint-quantile residual model", "Rows": int(self.fit_mask.sum()),
                "June_Eval_Rows": int(self.tune_mask.sum()), "Maximum_Trees": c.iterations,
                "Snapshot_Interval_Seconds": 60, "Production_Refit": False}, flush=True)
            # The only authorised new predictive-model fit in this module.
            self.model.fit(fit_pool, eval_set=tune_pool, use_best_model=True,
                early_stopping_rounds=c.early_stopping_rounds, save_snapshot=True,
                snapshot_file=str(self.folder / "training.snapshot"), snapshot_interval=60)
            temporary = path.with_name("model.pending.cbm")
            self.model.save_model(str(temporary)); temporary.replace(path)
            meta = {"signature": key, "sha256": study.hash_file(path), "columns": columns,
                    "categorical": categorical, "trees": int(self.model.tree_count_), "params": params}
            save_json(metadata_path, meta)
            save_json(ledger_path, {"signature": key, "run_id": self.run_id, "maximum_distinct_models": 1,
                "state": "Completed", "folder": str(self.folder), "model_sha256": meta["sha256"]})
            del fit_pool, tune_pool; gc.collect(); fits = 1
        if list(self.model.feature_names_) != columns:
            raise ValueError("Named feature schema changed")
        expected_categorical = [columns.index(col) for col in categorical]
        if list(self.model.get_cat_feature_indices()) != expected_categorical:
            raise ValueError("Categorical schema changed")
        self.training_summary = {"New_Model_Fits_This_Run": fits, "Trees": int(self.model.tree_count_),
            "Features": len(columns), "Model_Path": str(path), "Production_Refit": False}
        self.trained = True
        return self.training_summary

    def _historical_prediction(self, weight):
        raw = numbers(self.detail, "Raw_Base")
        prices, crossed = quantile_prices(raw, self.area, self.model.predict(self.x, thread_count=self.config.threads))
        eligible = self.eligible & self.support_mask & (numbers(self.detail, "Forecast_Gap") > 0)
        values, active = correction(self.baseline, prices, eligible, weight, self.config)
        return values, active, prices, crossed

    def validate(self):
        if self.validated:
            return self.result
        self.train_once()
        c = self.config
        month = numbers(self.query, MONTH)
        tune = month == c.tune_month
        rows, choices = [], []
        # This finite selection reads June only; Check cannot change the choice.
        for w in c.weights:
            pred, active, prices, crossed = self._historical_prediction(w)
            metric = score(self.actual, self.baseline, pred, tune, "Tune_2024_06", "All")
            changed = score(self.actual, self.baseline, pred, tune & active, "Tune_2024_06", "Changed")
            allowed = (metric["Accurate_Increase_Pct"] <= c.tune_accurate_increase_pct and
                changed["Accurate_Increase_Pct"] <= c.tune_accurate_increase_pct and
                metric["New_Damaged"] <= max(1, c.maximum_damaged_fraction * metric["Accurate_Rows"]))
            rows.append({"Weight": w, **metric, "Protection_Passed": bool(allowed)})
            if allowed and (w == 0 or metric["MAE_Reduction"] >= c.minimum_tune_gain):
                choices.append((metric["Candidate_MAE"], w))
        self.weight = float(min(choices)[1])
        self.selection = pd.DataFrame(rows)
        save_json(self.folder / "locked_selection.json", {"weight": self.weight, "selection_month": c.tune_month,
            "check_used_for_selection": False, "config": asdict(c),
            "model_sha256": study.hash_file(self.folder / "model.cbm")})
        self.selection.to_csv(self.folder / "june_selection.csv", index=False)
        prediction, active, prices, crossed = self._historical_prediction(self.weight)
        self.historical_prediction, self.historical_active = prediction, active
        uncertainty = []
        for label, mask in (("Tune_2024_06", tune), ("Check_2024_07_08", month > c.tune_month)):
            m = mask & self.eligible & self.support_mask
            uncertainty.append({"Set": label, "Eligible_Rows": int(m.sum()),
                "Empirical_Central_Interval_Coverage_Pct": float(100*np.mean(
                    (self.actual[m] >= prices[m,0]) & (self.actual[m] <= prices[m,2]))) if m.any() else np.nan,
                "Median_Interval_Width": float(np.median(prices[m,2]-prices[m,0])) if m.any() else np.nan,
                "Quantile_Crossing_Rows": int((crossed & m).sum()),
                "Changed_Rows": int((active & mask).sum())})
        self.uncertainty = pd.DataFrame(uncertainty)
        self.uncertainty.to_csv(self.folder / "quantile_diagnostics.csv", index=False)
        scores = []
        for label, mask in (("Tune_2024_06", tune), ("Check_2024_07_08", month > c.tune_month)):
            for segment, sub in (("All", np.ones(len(month), bool)), ("Eligible_Housing", self.eligible),
                ("Changed", active), ("Unchanged", ~active), ("Originally_Accurate", np.abs(self.actual-self.baseline) <= 500_000)):
                scores.append(score(self.actual, self.baseline, prediction, mask & sub, label, segment))
        self.scores = pd.DataFrame(scores)
        check = self.scores.loc[(self.scores.Set == "Check_2024_07_08") & (self.scores.Segment == "All")].iloc[0]
        changed_check = self.scores.loc[(self.scores.Set == "Check_2024_07_08") & (self.scores.Segment == "Changed")].iloc[0]
        protection = (check.Accurate_Increase_Pct <= c.check_accurate_increase_pct and
            changed_check.Accurate_Increase_Pct <= c.check_accurate_increase_pct and
            check.New_Damaged <= max(1, c.maximum_damaged_fraction * check.Accurate_Rows))
        self.decision = {"Selected_Weight": self.weight, "New_Predictive_Models": 1,
            "Production_Refit": False, "June_Selection_Only": True,
            "Recommended": bool(self.weight > 0 and check.MAE_Reduction >= c.minimum_check_gain and protection),
            "Check_MAE_Reduction": float(check.MAE_Reduction), "Check_Protection_Passed": bool(protection),
            "Quantile_Crossing_Rows": int(crossed.sum()),
            "Check_Is_Retrospective": True, "Test_Valid_Answers_Read": False,
            "Kaggle_Target": "1.5M to 1.7M is a goal, not a score prediction or guarantee",
            "Export_Policy": "Export the June-locked nonzero candidate; report Check recommendation separately without retuning"}
        self.scores.to_csv(self.folder / "historical_scores.csv", index=False)
        save_json(self.folder / "decision.json", self.decision)
        save_json(self.folder / "support.json", self.support)
        self.result = {"selection": self.selection, "scores": self.scores,
                       "decision": self.decision, "uncertainty": self.uncertainty}
        self.validation_lock = digest({"decision": self.decision, "weight": self.weight,
            "model_sha256": study.hash_file(self.folder / "model.cbm"), "provenance": self.provenance})
        self.validated = True
        return self.result

    def plot(self):
        self.validate()
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor="#f6f8fb", layout="constrained")
        fig.suptitle("V10 | One new model, evidence before submission", fontsize=20, fontweight="bold")
        all_scores = self.scores.loc[self.scores.Segment.eq("All")]
        labels = ["June selection", "July-August check"]
        positions = np.arange(2)
        axes[0,0].barh(positions+.16, all_scores.V8_MAE/1e6, .3, color="#aab9c9", label="V8")
        axes[0,0].barh(positions-.16, all_scores.Candidate_MAE/1e6, .3, color="#128b8c", label="V10 candidate")
        axes[0,0].set_yticks(positions, labels); axes[0,0].set_xlabel("MAE / million TWD"); axes[0,0].legend(frameon=False)
        axes[0,0].set_title("Overall error; lower is better")
        axes[0,1].plot(self.selection.Weight, self.selection.Candidate_MAE/1e6, "o-", color="#128b8c")
        axes[0,1].axvline(self.weight, color="#bd6a40", linestyle="--", label="June-locked weight")
        axes[0,1].set(xlabel="Correction weight", ylabel="June MAE / million TWD", title="Finite selection on June only")
        axes[0,1].legend(frameon=False)
        original = self.scores.loc[self.scores.Segment.eq("Originally_Accurate")]
        axes[1,0].bar(labels, original.Accurate_Increase_Pct, color="#697fb3")
        axes[1,0].axhline(0, color="#7c8795", linewidth=.8)
        axes[1,0].set(ylabel="MAE change (%)", title="Originally accurate transactions")
        month = numbers(self.query, MONTH)
        rows = []
        for value in (197, 198, 199):
            m = month == value
            row = score(self.actual, self.baseline, self.historical_prediction, m, str(value), "All")
            rows.append(row)
        axes[1,1].plot(["Jun", "Jul", "Aug"], [r["MAE_Reduction"]/1000 for r in rows], "o-", color="#128b8c")
        axes[1,1].axhline(0, color="#7c8795", linewidth=.8)
        axes[1,1].set(ylabel="MAE reduction / thousand TWD", title="June tuning; July-August retrospective check")
        for ax in axes.flat:
            ax.set_facecolor("#ffffff"); ax.grid(axis="y", alpha=.15)
        fig.savefig(self.folder / "validation_overview.png", dpi=300, bbox_inches="tight")
        fig.savefig(self.folder / "validation_overview.svg", bbox_inches="tight")
        return fig

    def export(self):
        self.validate()
        if self.weight == 0:
            return {"Exported": False, "Reason": "June selected no correction; preserve V8", "Production_Refit": False}
        lock = digest({"decision": self.decision, "weight": self.weight,
            "model_sha256": study.hash_file(self.folder / "model.cbm"), "provenance": self.provenance})
        if lock != self.validation_lock:
            raise ValueError("The frozen June/Check decision changed")
        for name, expected in self.provenance["sources"].items():
            if study.hash_file(Path(__file__).with_name(name)) != expected:
                raise ValueError("V10 source changed after fitting")
        info = self.manifest["Test"]
        path = Path(info["file"])
        if study.hash_file(path) != info["sha256"]:
            raise ValueError("Test workbook differs from the original V8 input")
        check_unlabelled(pd.read_excel(path, nrows=0))
        raw = pd.read_excel(path); check_unlabelled(raw)
        if not unique_ids(raw).all():
            raise ValueError("Unlabelled Test IDs must be unique")
        baseline_path = self.root / "outputs/submission_v8/submission_v8.csv"
        frozen = pd.read_csv(baseline_path, dtype={ID: "string"})
        frozen[ID] = frozen[ID].astype(str)
        if set(frozen[ID]) != set(raw[ID].astype(str)) or not frozen[ID].is_unique:
            raise ValueError("V8/Test row alignment changed")
        baseline = numbers(frozen.set_index(ID).loc[raw[ID].astype(str)], TARGET)
        index = pd.read_csv(self.root / "presentation/tables/submission_index.csv")
        expected = index.loc[index.Version.eq("submission_v8"), "SHA256"]
        if len(expected) != 1 or expected.iloc[0] != study.hash_file(baseline_path):
            raise ValueError("The frozen V8 submission changed")
        self.base.prepare_production()
        result = self.base.predict(raw, "production")
        frame, detail = result["frame"], result["detail"]
        latest = int(np.nanmax(numbers(self.train, MONTH)))
        market = market_table(self.train, latest)
        x = model_features(frame, detail, market)
        if list(x.columns) != list(self.model.feature_names_):
            raise ValueError("Production feature schema differs")
        area = numbers(frame, AREA)
        prices, crossed = quantile_prices(numbers(detail, "Raw_Base"), area,
            self.model.predict(x, thread_count=self.config.threads))
        key = frame[self.stratum_columns].astype(str).agg("|".join, axis=1)
        supported = key.map(self.support).fillna(0).to_numpy(float) >= self.config.minimum_stratum_rows
        locked = self._airport_locked(raw)
        gap = numbers(detail, "Forecast_Gap")
        eligible = housing_mask(frame) & supported & ~locked & (gap >= 1) & (gap <= 9)
        prediction, active = correction(baseline, prices, eligible, self.weight, self.config)
        np.testing.assert_array_equal(prediction[locked | ~eligible], baseline[locked | ~eligible])
        from project import write_submission
        submission = write_submission(self.folder / "submission_v10_candidate.csv", raw, prediction)
        audit = pd.DataFrame({ID: raw[ID], "Year": frame["交易年"], "Forecast_Gap": gap,
            "Eligible": eligible, "Changed": active, "V8": baseline, "Candidate": prediction,
            "Difference": prediction-baseline, "Interval_Lower": prices[:,0], "Interval_Upper": prices[:,2]})
        summary = audit.groupby("Year", dropna=False).agg(Rows=(ID,"size"),
            Active_Pct=("Changed",lambda v: 100*v.mean()), Mean_Abs_Change=("Difference",lambda v:v.abs().mean()),
            Median_Change=("Difference","median"))
        summary.to_csv(self.folder / "test_change_summary.csv")
        top = np.argsort(np.abs(prediction-baseline))[-100:][::-1]
        audit.iloc[top].to_csv(self.folder / "largest_test_changes.csv", index=False)
        exported = {"Exported": True, "Recommended": self.decision["Recommended"], "Submission": str(submission),
            "SHA256": study.hash_file(submission), "Selected_Weight": self.weight,
            "New_Production_Models": 0, "Old_V8_Preserved": True, "Airport_Locked_Rows": int(locked.sum()),
            "Within_Training_Period_Unchanged": int((gap <= 0).sum()), "Changed_Rows": int(active.sum()),
            "Quantile_Crossing_Rows": int(crossed.sum()), "Test_Answers_Read": False,
            "Kaggle_Uploaded": False, "Score_Guarantee": None}
        save_json(self.folder / "export.json", exported)
        self.exported = exported
        return exported

    def review(self):
        self.validate()
        destination = self.folder / "v10_review.zip"
        extensions = {".csv", ".json", ".svg", ".png"}
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(self.folder.iterdir()):
                if p.is_file() and p.suffix in extensions and not p.name.startswith("submission_"):
                    archive.write(p, p.name)
        return destination

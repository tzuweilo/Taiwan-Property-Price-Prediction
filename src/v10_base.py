"""Load-only replay of the frozen V7 component models for one-fit residual work.

No ML fitting is permitted here.  The old TrySystem.fit API is used only to
reconstruct deterministic Train reference aggregates and verify/load its exact
native caches.  Both backend fit methods are replaced by raising guards for
that entire call.  Legacy global, regional and age calibrators are deliberately
not applied: Raw_Base uses unit-scale component assembly in both contexts,
before all label-fitted calibration corrections.

Call setup/prediction serially, never while another thread is fitting models.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import gc
import hashlib
import importlib
import json
import sys

import numpy as np
import pandas as pd

ID, TARGET, MONTH = "編號", "總價元", "交易時間序號"
IMPLEMENTATION = "frozen_base_load_only_v2_unit_scale"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


def _frame_hash(frame):
    h = hashlib.sha256()
    h.update(json.dumps([(str(c), str(t)) for c, t in frame.dtypes.items()],
                        ensure_ascii=False).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    default=str), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def forbid_ml_fit():
    """Hard guard: a missing old cache must raise rather than start any ML fit."""
    import catboost
    import lightgbm

    def forbidden(*args, **kwargs):
        raise RuntimeError("Frozen old-model cache is missing or incompatible. "
                           "Old ML fitting is forbidden; complete V8 first in "
                           "the original environment and retain its caches.")

    originals = [(catboost.CatBoostRegressor, "fit", catboost.CatBoostRegressor.fit),
                 (lightgbm.LGBMRegressor, "fit", lightgbm.LGBMRegressor.fit)]
    try:
        for cls, name, _ in originals:
            setattr(cls, name, forbidden)
        yield
    finally:
        for cls, name, method in originals:
            setattr(cls, name, method)


class FrozenBase:
    """Exact old component caches; no global/regional/age calibration labels.

    Parameters
    ----------
    root : Path
        The completed ML_HW1_Price_Study folder containing src and outputs.
    raw_train : DataFrame
        Original labelled Train read by pd.read_excel with the same default
        dtypes as the completed main notebook. Do not pre-clean or cast it.
    threads : int
        Prediction thread budget. Old training cache identities retain 8.

    Public API
    ----------
    prepare_historical(), prepare_production(): setup/load old caches only.
    transform(raw): target-free, exact compatibility Try preprocessing.
    predict(raw, context): dict(frame=processed, detail=component DataFrame).
    provenance: original cache hashes and fixed raw-baseline definition.
    """

    def __init__(self, root, raw_train, threads=8):
        self.root = Path(root).expanduser().resolve()
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError("Positive prediction thread count required")
        if TARGET not in raw_train or ID not in raw_train:
            raise ValueError("Original labelled Train is required")
        source_dir = self.root / "src"
        if not source_dir.is_dir():
            raise FileNotFoundError("Missing completed project src folder: " + str(source_dir))
        sys.path.insert(0, str(source_dir)) if str(source_dir) not in sys.path else None
        self.legacy = importlib.import_module("legacy_pipeline")
        self.trypipe = importlib.import_module("try_pipeline")
        for module in (self.legacy, self.trypipe):
            if Path(module.__file__).resolve().parent != source_dir:
                raise RuntimeError("A different project module is already imported: " + module.__name__
                                   + ". Restart this notebook kernel, then run from the top.")
        self.reference_dir = self.root / "outputs/submission_v7/models/reference_ensemble"
        manifest_path = self.reference_dir / "system_manifest.json"
        recipe_path = self.root / "outputs/submission_v7/recipe.json"
        if not manifest_path.is_file() or not recipe_path.is_file():
            raise FileNotFoundError("Complete V7/V8 and retain system_manifest.json and recipe.json")
        self.system_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.v7_recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        self.recipe = self.system_manifest["recipe"]
        if self.recipe != self.legacy.DEFAULT_RECIPE:
            raise ValueError("Saved old recipe differs from the frozen V8 source recipe")
        if self.system_manifest.get("code_signature") != self.legacy.legacy_implementation_signature():
            raise ValueError("Legacy predictive source differs from the frozen V8 model source")
        self.processor = self.trypipe.TryPreprocessor(raw_train)
        self.processed_train = self.processor.transform(raw_train, training=True)
        self.legacy_processor = self.legacy.DatasetPreprocessor().fit(raw_train)
        legacy_train = self.legacy_processor.transform(raw_train, training=True)
        if self.legacy.frame_digest(raw_train) != self.system_manifest["raw_train_signature"]:
            raise ValueError("Original Train values/dtypes differ from the completed V8 input. "
                             "Read the untouched workbook with pd.read_excel default dtypes.")
        if self.legacy.frame_digest(legacy_train) != self.system_manifest["processed_train_signature"]:
            raise ValueError("Legacy processed Train digest differs from the saved V8 model population")
        np.testing.assert_array_equal(legacy_train[ID].astype(str), self.processed_train[ID].astype(str))
        self._legacy_training = legacy_train[["建案名稱", "建案行政區"]].drop_duplicates().reset_index(drop=True)
        self._historical_legacy_training, _ = self.legacy.historical_split(
            legacy_train, cutoff=self.recipe["historical_cutoff"],
            horizon=self.recipe["historical_horizon"])
        self._historical_legacy_training = self._historical_legacy_training[
            ["建案名稱", "建案行政區"]].drop_duplicates().reset_index(drop=True)
        self._try = {}
        self._legacy_manifests = {}
        self.provenance = {
            "implementation": IMPLEMENTATION,
            "loader_source_sha256": file_hash(Path(__file__)),
            "ml_fits_permitted": 0,
            "calibrators_applied": False,
            "global_scale_applied": False,
            "legacy_assembly_scale": 1.0,
            "raw_base_definition": "50% unit-scale uncalibrated legacy assembly + 50% Try_V02; "
                "ordinary all-reference-missing fallback to uncalibrated legacy assembly",
            "raw_train_signature": self.system_manifest["raw_train_signature"],
            "processed_train_signature": self.system_manifest["processed_train_signature"],
            "legacy_system_manifest_sha256": file_hash(manifest_path),
            "v7_recipe_sha256": file_hash(recipe_path),
            "source_sha256": {name: file_hash(source_dir / name)
                for name in ("legacy_pipeline.py", "try_pipeline.py")},
            "contexts": {},
        }
        del legacy_train
        gc.collect()

    @staticmethod
    def _features(raw):
        features = raw.drop(columns=[TARGET], errors="ignore").copy()
        bad = [str(c) for c in features if str(c).strip().casefold() in
               {"actual", "target", "label", "answer", "y_true", "ground_truth",
                "matched_price", "true_price", "test_price", "總價"}
               or any(s in str(c) for s in ("答案", "真值", "補價", "實際總價", "匹配總價"))]
        if bad:
            raise ValueError("Answer-like query columns prohibited: " + ", ".join(bad))
        if ID not in features or features[ID].isna().any():
            raise ValueError("Query transaction IDs are required")
        return features

    def transform(self, raw):
        features = self._features(raw)
        frame = self.processor.transform(features, training=False)
        if len(frame) != len(raw):
            raise ValueError("Query preprocessing changed the row population")
        np.testing.assert_array_equal(frame[ID].astype(str), raw[ID].astype(str))
        if TARGET in frame:
            raise AssertionError("Target reached the query feature frame")
        return frame

    def _manifests(self, context):
        key = "historical_models" if context == "historical" else "final_models"
        saved = self.system_manifest[key]
        result, hashes = {}, {}
        for name in self.recipe["components"]:
            folder = self.reference_dir / "models" / context / name
            path = folder / "manifest.json"
            if not path.is_file():
                raise FileNotFoundError("Missing frozen legacy component: " + str(path))
            item = json.loads(path.read_text(encoding="utf-8"))
            for k in ("signature", "model_sha256", "bundle_sha256"):
                if item[k] != saved[name][k]:
                    raise ValueError("Frozen legacy manifest mismatch: " + name + "/" + k)
            model, bundle = folder / "model.cbm", folder / "features.pkl"
            for p, expected in ((model, item["model_sha256"]), (bundle, item["bundle_sha256"])):
                if not p.is_file() or file_hash(p) != expected:
                    raise ValueError("Frozen legacy cache checksum failed: " + str(p))
            item["model_path"], item["bundle_path"] = str(model), str(bundle)
            result[name] = item
            hashes[name] = {"manifest_sha256": file_hash(path),
                            "model_sha256": item["model_sha256"],
                            "features_sha256": item["bundle_sha256"]}
        return result, hashes

    def _prepare(self, context):
        if context not in {"historical", "production"}:
            raise ValueError("Context must be historical or production")
        if context in self._try:
            return self
        manifests, hashes = self._manifests(context)
        if context == "historical":
            population, _ = self.legacy.historical_split(self.processed_train, cutoff=187, horizon=12)
            schema = self.processed_train.loc[
                self.processed_train[MONTH].le(194)].reset_index(drop=True)
            directory = self.root / "outputs/submission_v7/models/comparison_ensemble/validation"
        else:
            population = self.processed_train
            schema = self.trypipe.original_schema_fit(population)
            directory = self.root / "outputs/submission_v7/models/comparison_ensemble/full"
        if not directory.is_dir():
            raise FileNotFoundError("Missing frozen Try cache folder: " + str(directory))
        before = {}
        for stem, extension in (("history", ".cbm"), ("unit", ".cbm"),
                                ("matched", ".cbm"), ("lgbm", ".txt")):
            matches = list(directory.glob(stem + "_*" + extension))
            if not matches:
                raise FileNotFoundError("Missing frozen Try model: " + stem + " in " + str(directory))
            for model_path in matches:
                metadata_path = model_path.with_suffix(".json")
                if not metadata_path.is_file():
                    raise FileNotFoundError("Missing frozen Try metadata: " + str(metadata_path))
                before[model_path.name] = file_hash(model_path)
                before[metadata_path.name] = file_hash(metadata_path)
        # The called method computes exact matrix/target/weight identity before
        # each load. A missing matching cache reaches the raising backend guard.
        # HistoricalComparables.fit is a deterministic aggregation, not an ML fit.
        with forbid_ml_fit():
            system = self.trypipe.TrySystem(8).fit(population, directory, schema_fit=schema)
        expected_columns = self.v7_recipe["original_schema_columns"]
        if list(system.base_columns) != list(expected_columns):
            raise ValueError("Frozen Try base feature schema differs from V7 recipe")
        for name, expected in before.items():
            if file_hash(directory / name) != expected:
                raise ValueError("A frozen Try cache was modified during load-only setup")
        self._try[context], self._legacy_manifests[context] = system, manifests
        self.provenance["contexts"][context] = {
            "fit_rows": len(population), "cutoff": float(system.cutoff),
            "legacy_components": hashes, "try_cache_files": before,
            "exact_try_cache_identity_checked": True,
            "old_model_fits": 0,
        }
        print("Loaded frozen context:", context, "| Old ML fits: 0", flush=True)
        return self

    def prepare_historical(self):
        return self._prepare("historical")

    def prepare_production(self):
        return self._prepare("production")

    def _predict_uncached(self, raw, context):
        self._prepare(context)
        features = self._features(raw)
        frame = self.transform(features)
        old_frame = self.legacy_processor.transform(features, training=False)
        np.testing.assert_array_equal(frame[ID].astype(str), old_frame[ID].astype(str))
        config = self.legacy.LegacyConfig(threads=self.threads, batch_size=20000)
        components = self.legacy.predict_suite(old_frame, self._legacy_manifests[context],
                                               self.recipe, config)
        membership = self._historical_legacy_training if context == "historical" else self._legacy_training
        # Frozen historical_scale/production_scale were calibrated with labels.
        # Training a new residual model on earlier months must not reuse either.
        # The same unit-scale assembly is used during historical and Test replay.
        audit = self.legacy.assemble_predictions(old_frame, components, membership, 1.0, self.recipe)
        if "Actual" in audit or "Residual" in audit:
            raise AssertionError("Query target reached frozen legacy prediction")
        system = self._try[context]
        try_result = system.predict(frame)
        fused = self.trypipe.combine_legacy(audit["Baseline"].to_numpy(float), try_result)
        detail = pd.DataFrame({ID: frame[ID].to_numpy(copy=True),
            "Raw_Base": fused["Try_V04"].to_numpy(float),
            "Legacy_Raw": audit["Baseline"].to_numpy(float),
            "Try_History": try_result["History"].to_numpy(float),
            "Try_History_Unit": try_result["History_Unit"].to_numpy(float),
            "Try_Matched": try_result["Matched_Raw"].to_numpy(float),
            "Try_LGBM": try_result["LightGBM_Raw"].to_numpy(float),
            "Try_Final": try_result["Try_V02"].to_numpy(float),
            "Forecast_Gap": try_result["Horizon"].to_numpy(float),
            "Project_Seen": frame["建案行政區"].isin(system.known_projects).to_numpy(bool),
            "Matched_Active": try_result["Matched_Active"].to_numpy(bool),
            "Unit_Weight": try_result["Unit_Weight"].to_numpy(float),
            "LGBM_Weight": try_result["LightGBM_Weight"].to_numpy(float),
            "Legacy_Fallback": fused["Legacy_Fallback"].to_numpy(bool),
            "Area": try_result["Area"].to_numpy(float)})
        for name in self.recipe["components"]:
            detail["Legacy_" + name] = components[name].to_numpy(float)
        matrix = system.feature_frame(frame)
        area, eligible = self.trypipe.unit_area_mask(frame)
        unit_total = np.full(len(frame), np.nan)
        if eligible.any():
            unit_total[eligible] = np.maximum(np.asarray(system.models["unit"].predict(
                matrix.loc[eligible], thread_count=self.threads), float), 0) * area[eligible]
        detail["Try_Unit_Total"] = unit_total
        for level in ("Project", "Road", "District"):
            for suffix in ("Unit", "Price", "Rows", "Age", "Missing"):
                name = level + "_Reference_" + suffix
                detail[name] = pd.to_numeric(matrix[name], errors="coerce").to_numpy(
                    float, na_value=np.nan, copy=True)
        # These names identify raw reference-branch quantities, not its final
        # label-fitted regional/age calibration. Preserve that distinction.
        detail["Legacy_Project_Reference"] = audit["Project_Reference"].to_numpy(float)
        detail["Legacy_District_Reference"] = audit["District_Reference"].to_numpy(float)
        if not np.isfinite(detail[["Raw_Base", "Legacy_Raw", "Try_Final"]]).all().all():
            raise ValueError("Frozen raw baseline contains non-finite predictions")
        detail = detail.reset_index(drop=True)
        del matrix, old_frame, components, audit, try_result, fused
        gc.collect()
        return {"frame": frame.reset_index(drop=True), "detail": detail}

    def predict(self, raw_query, context="historical", cache=True):
        """Return target-free processed rows and compact frozen component table.

        Target columns are dropped BEFORE hashing, preprocessing, references or
        models. Historical mode permits only future months188..199. Production
        mode accepts unlabelled raw features; callers must reject labelled Test
        files before reading their values. Identical feature queries reuse a
        compact prediction cache, never a saved design matrix.
        """
        self._prepare(context)
        features = self._features(raw_query)
        frame = self.transform(features)
        months = pd.to_numeric(frame[MONTH], errors="coerce").to_numpy(float, na_value=np.nan)
        if context == "historical" and not np.all(np.isfinite(months) & (months >= 188) & (months <= 199)):
            raise ValueError("Historical raw replay is restricted to months188..199")
        identity = {"implementation": IMPLEMENTATION, "context": context,
                    "query_features": _frame_hash(features),
                    "frozen_provenance": self.provenance}
        signature = _digest(identity)
        folder = self.root / "outputs/v10_base_cache" / context / signature[:16]
        data_path, metadata_path = folder / "components.pkl", folder / "metadata.json"
        if cache and data_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["identity"] != identity or file_hash(data_path) != metadata["sha256"]:
                raise ValueError("Frozen component prediction cache checksum/identity failed")
            detail = pd.read_pickle(data_path)
            np.testing.assert_array_equal(detail[ID].astype(str), frame[ID].astype(str))
            print("Loaded frozen component predictions:", context, "| Rows:", len(detail), flush=True)
            return {"frame": frame, "detail": detail}
        result = self._predict_uncached(features, context)
        if cache:
            folder.mkdir(parents=True, exist_ok=True)
            temporary = data_path.with_name("components.pending.pkl")
            result["detail"].to_pickle(temporary)
            temporary.replace(data_path)
            _write_json(metadata_path, {"identity": identity, "sha256": file_hash(data_path),
                                        "rows": len(frame), "ml_fits": 0})
        return result

    def replay_try_validation(self, result, label):
        """Exact Try replay against V7 csv, independent of Legacy calibration.

        Saved Try_V03 = .5*saved calibrated Legacy + .5*Try_Final. This is an
        audit only and its Actual column never becomes an input feature.
        """
        label = str(label).lower()
        if label not in {"dev", "check"}:
            raise ValueError("Replay label must be Dev or Check")
        path = self.root / "outputs/submission_v7/predictions" / (label + ".csv")
        saved = pd.read_csv(path, dtype={ID: "string"})
        detail = result["detail"]
        ids = detail[ID].astype(str)
        if not ids.is_unique or not saved[ID].is_unique or set(ids) != set(saved[ID].astype(str)):
            raise ValueError("Try replay requires the complete unique saved evaluation cohort")
        aligned = saved.assign(**{ID: saved[ID].astype(str)}).set_index(ID).loc[ids]
        expected = 2 * aligned["Try_V03"].to_numpy(float) - aligned["Legacy"].to_numpy(float)
        np.testing.assert_allclose(detail["Try_Final"], expected, rtol=1e-12, atol=.02)
        np.testing.assert_array_equal(detail["Legacy_Fallback"].to_numpy(bool),
                                      aligned["Legacy_Fallback"].to_numpy(bool))
        return {"Set": label.title(), "Rows": len(ids), "Try_Replay_Passed": True,
                "Maximum_Absolute_Difference": float(np.max(abs(detail["Try_Final"].to_numpy(float) - expected))),
                "Legacy_Raw_Is_Intentionally_Uncalibrated": True,
                "Old_ML_Fits": 0}

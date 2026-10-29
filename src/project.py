"""Project I/O and explicit reproduction of the fixed price ensemble.

The module imports definitions only. All model training is initiated by the
notebook's named training cells, on the user's machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import re
import shutil
import zipfile

import numpy as np
import pandas as pd

TARGET, ID = "總價元", "編號"
KNOWN_SCORE = 1_918_506
KNOWN_PREDICTION_SHA256 = "0fe22eafd43db8f78a8f8ec54696afe53d7fe7aadebb41a4f67d24f626a5cebb"
INPUT_NAMES = {"Train": "train-v2.xlsx", "Valid": "valid-v2.xlsx", "Test": "test-reindex-test-v2.2.xlsx"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def create_layout(root):
    root = Path(root).expanduser().resolve()
    for relative in ["data/raw", "data/processed", "src", "configs", "references",
                     "presentation/figures", "presentation/tables"]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for version in range(1, 9):
        for relative in ["models", "predictions", "tables", "figures"]:
            (root / "outputs" / f"submission_v{version}" / relative).mkdir(parents=True, exist_ok=True)
    return root


def load_data(root, source_dir, copy_raw=False, cache_raw=False):
    """Read exactly three original workbooks; copying and raw pickle caches are opt-in."""
    root, source_dir = Path(root), Path(source_dir).expanduser()
    manifest, frames = {}, {}
    pandas_version = importlib.metadata.version("pandas")
    for label, name in INPUT_NAMES.items():
        local, source = root / "data/raw" / name, source_dir / name
        if copy_raw:
            if not local.is_file():
                if not source.is_file():
                    raise FileNotFoundError(f"Place {name} in data/raw or set SOURCE_DIR to its folder")
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, local)
            elif source.is_file() and source.resolve() != local.resolve() and file_hash(source) != file_hash(local):
                raise ValueError(f"Different raw files found for {name}; choose one source explicitly")
            chosen = local
        else:
            chosen = source if source.is_file() else local
            if not chosen.is_file():
                raise FileNotFoundError(f"Place {name} in data/raw or set SOURCE_DIR to its folder")
        digest = file_hash(chosen)
        cache = root / "data/processed" / f"raw_{label.lower()}_{digest[:12]}_{pandas_version}.pkl"
        # Check Test headers before either full workbook values or a cached frame.
        if label == "Test":
            header = pd.read_excel(chosen, nrows=0)
            forbidden = {"總價元", "總價", "actual", "target", "label", "answer", "y_true", "ground_truth", "matched_price"}
            suspect = [str(c) for c in header if str(c).casefold().strip() in forbidden
                       or any(word in str(c) for word in ["答案", "真值", "補價", "實際總價", "匹配總價"])]
            if suspect:
                raise ValueError("Use the original unlabelled Test file; answer columns found: " + ", ".join(suspect))
        if cache_raw and cache.is_file():
            frame = pd.read_pickle(cache)
        else:
            frame = pd.read_excel(chosen)
            if cache_raw:
                cache.parent.mkdir(parents=True, exist_ok=True)
                frame.to_pickle(cache)
        if label == "Test" and TARGET in frame:
            raise ValueError("Test labels are not accepted")
        if label != "Test" and TARGET not in frame:
            raise ValueError(f"{label} must contain the original total-price target")
        if ID not in frame:
            raise ValueError(f"{label} has no transaction ID")
        frames[label] = frame
        manifest[label] = {"file": str(chosen.resolve()), "sha256": digest, "rows": len(frame),
            "columns": frame.columns.tolist(), "target_available": TARGET in frame}
        manifest[label].update(raw_copy_enabled=bool(copy_raw), raw_cache_enabled=bool(cache_raw))
    write_json(root / "configs/input_manifest.json", manifest)
    return frames


def preprocessing_audit(raw, prepared):
    """Report frozen input definitions as decisions, not claims of truth."""
    rows = []
    for label in ["Train", "Valid", "Test"]:
        before, after = raw[label], prepared[label.lower()]
        rows.append({"Dataset": label, "Raw_Rows": len(before), "Processed_Rows": len(after),
            "Removed_Rows": len(before) - len(after), "Processed_Columns": len(after.columns),
            "Missing_Transaction_Month": int(after["交易時間序號"].isna().sum()),
            "Missing_Project_Name": int(after["建案名稱"].eq("缺失").sum())})
    decisions = pd.DataFrame([
        {"Rule": "Numeric parsing", "Treatment": "Coerce invalid numeric text to missing; keep raw files"},
        {"Rule": "Target eligibility", "Treatment": "Train: positive numeric target, excluding fixed source anomalies 520 and 1760"},
        {"Rule": "Area parsing", "Treatment": "Negative areas to missing; two fixed source land-area anomalies to missing in Train"},
        {"Rule": "Layout compatibility", "Treatment": "Counts below zero or at least 20 become unknown for the frozen recipe, not proof of invalidity"},
        {"Rule": "Dates", "Treatment": "Parse ROC dates; invalid dates remain missing; preserve original date text"},
        {"Rule": "Address", "Treatment": "Normalize city names; derive district, road, parcel and project keys"},
        {"Rule": "Known site-area conflict", "Treatment": "Report Hanlin 7029.51 sqm cohort; do not silently change the evaluated recipe"},
        {"Rule": "Derived area formulas", "Treatment": "V1-V6 use explicit correct area fields; V7 has documented frozen compatibility expressions"},
        {"Rule": "Cancellation and ID", "Treatment": "Never treat row ID or post-transaction cancellation text as predictive evidence"},
    ])
    return pd.DataFrame(rows), decisions


def align_predictions(frame, table, prediction_column):
    if frame[ID].isna().any() or table[ID].isna().any():
        raise ValueError("Prediction alignment requires non-missing transaction IDs")
    ids = frame[ID].astype(str)
    table = table.copy()
    table[ID] = table[ID].astype(str)
    subset = table.loc[table[ID].isin(ids)]
    if not ids.is_unique or not subset[ID].is_unique or set(ids) != set(subset[ID]):
        raise ValueError("Prediction rows do not match the fixed unique evaluation IDs")
    aligned = subset.set_index(ID).loc[ids]
    if TARGET in frame and "Actual" in aligned:
        np.testing.assert_allclose(pd.to_numeric(frame[TARGET]), aligned.Actual, rtol=0, atol=.01)
    values = pd.to_numeric(aligned[prediction_column], errors="raise").to_numpy(float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite aligned predictions")
    return values


def write_submission(path, raw_test, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids, predictions = raw_test[ID].astype("string"), np.asarray(predictions, float)
    if TARGET in raw_test or not ids.is_unique or ids.isna().any():
        raise ValueError("Original unlabelled unique Test rows required")
    if len(predictions) != len(ids) or not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Invalid submission vector")
    output = pd.DataFrame({ID: raw_test[ID].to_numpy(copy=True), TARGET: predictions})
    temporary = path.with_suffix(".tmp.csv")
    output.to_csv(temporary, index=False, float_format="%.10f")
    roundtrip = pd.read_csv(temporary, dtype={ID: "string"})
    np.testing.assert_array_equal(roundtrip[ID], ids)
    np.testing.assert_allclose(roundtrip[TARGET], predictions, rtol=1e-12, atol=1e-6)
    temporary.replace(path)
    return path


class BestRecipeRunner:
    """Two explicit branches plus the locked public-price rule.

    Validation replays earlier predictions whose labels were not used by the
    component models or the calibrators. Production fits use Train only.
    """

    def __init__(self, root, threads=8):
        self.root, self.threads = Path(root), int(threads)
        if self.threads < 1:
            raise ValueError("threads must be a positive integer")

    def fit(self, raw_train):
        if __package__:
            from .legacy_pipeline import LegacyConfig, LegacySystem, historical_split
            from .try_pipeline import TryPreprocessor, TrySystem, main_feature_frame, original_schema_fit
        else:
            from legacy_pipeline import LegacyConfig, LegacySystem, historical_split
            from try_pipeline import TryPreprocessor, TrySystem, main_feature_frame, original_schema_fit
        if TARGET not in raw_train:
            raise ValueError("Train must contain the original total-price target")
        folder = self.root / "outputs/submission_v7/models"
        self.processor = TryPreprocessor(raw_train)
        processed = self.processor.transform(raw_train, training=True)
        schema = original_schema_fit(processed)
        march = processed.loc[processed["交易時間序號"].le(194)].reset_index(drop=True)
        def active_columns(frame):
            matrix = main_feature_frame(frame)
            return [column for column in matrix if matrix[column].nunique(dropna=False) > 1]
        if active_columns(march) != active_columns(schema):
            raise ValueError("The pre-evaluation schema differs from the frozen production schema")
        fit, _ = historical_split(processed, cutoff=187, horizon=12)
        # Complete schema and chronological preflight before any expensive fit.
        self.legacy = LegacySystem(LegacyConfig(threads=self.threads))
        self.legacy.fit(raw_train, folder / "reference_ensemble")
        self.try_validation = TrySystem(self.threads).fit(fit, folder / "comparison_ensemble/validation", schema_fit=march)
        self.try_production = TrySystem(self.threads).fit(processed, folder / "comparison_ensemble/full", schema_fit=schema)
        self.training_signature = hashlib.sha256(pd.util.hash_pandas_object(raw_train, index=True).to_numpy().tobytes()).hexdigest()
        write_json(self.root / "outputs/submission_v7/recipe.json", {
            "system": "Fixed two-branch ensemble", "reference_branch": self.legacy.recipe,
            "comparison_branch": self.try_production.recipe, "branch_weight": .5,
            "missing_reference_fallback": "100% reference branch for ordinary rows missing all three comparison references",
            "training_signature": self.training_signature, "test_valid_used_for_fit": False,
            "validation_model_cutoff_month": 187,
            "calibration_last_label_month": 194,
            "development_months": [195, 197], "check_months": [198, 199],
            "original_schema_columns": self.try_production.base_columns,
            "reproduction_note": "The comparison branch preserves its originally evaluated area-variable compatibility mapping"})
        return self

    def predict_validation(self, frame):
        if __package__:
            from .try_pipeline import combine_legacy
        else:
            from try_pipeline import combine_legacy
        if not hasattr(self, "try_validation") or not hasattr(self, "legacy"):
            raise RuntimeError("Call fit(raw_train) before predicting")
        if frame.empty or "交易時間序號" not in frame:
            raise ValueError("Validation requires nonempty processed April-August 2024 rows")
        months = pd.to_numeric(frame["交易時間序號"], errors="coerce")
        if months.isna().any() or not months.between(195, 199).all() or not months.eq(months.round()).all():
            raise ValueError("Validation is restricted to April-August 2024 (months 195-199); calibrators use labels through month 194")
        legacy = align_predictions(frame, self.legacy.calibration_audit, "Prediction")
        result = self.try_validation.predict(frame.drop(columns=[TARGET], errors="ignore"))
        fused = combine_legacy(legacy, result)
        fused["Prediction"] = fused["Try_V04"]
        return fused

    def predict_raw(self, raw_frame):
        if __package__:
            from .try_pipeline import combine_legacy
        else:
            from try_pipeline import combine_legacy
        if not hasattr(self, "try_production") or not hasattr(self, "legacy"):
            raise RuntimeError("Call fit(raw_train) before predicting")
        features = raw_frame.drop(columns=[TARGET], errors="ignore").copy()
        legacy = self.legacy.predict_raw(features)["Prediction"].to_numpy(float)
        processed = self.processor.transform(features, training=False)
        result = self.try_production.predict(processed)
        fused = combine_legacy(legacy, result)
        np.testing.assert_array_equal(fused[ID].astype(str), raw_frame[ID].astype(str))
        fused["Prediction"] = fused["Try_V04"]
        return fused


def best_parity(raw_test, predictions, reference_path, output_dir):
    reference_path, output_dir = Path(reference_path), Path(output_dir)
    if file_hash(reference_path) != KNOWN_PREDICTION_SHA256:
        raise ValueError("Frozen prediction reference SHA256 changed")
    reference = pd.read_csv(reference_path, dtype={ID: "string"})
    ids = raw_test[ID].astype("string")
    if len(reference) != len(ids) or not reference[ID].is_unique or set(ids) != set(reference[ID]):
        raise ValueError("Frozen reference IDs differ from original Test")
    expected = reference.set_index(ID).loc[ids, TARGET].to_numpy(float)
    delta = np.asarray(predictions, float) - expected
    equivalent = bool(np.allclose(predictions, expected, rtol=1e-12, atol=1e-5))
    result = {"Frozen_Kaggle_MAE": KNOWN_SCORE, "Frozen_SHA256": KNOWN_PREDICTION_SHA256,
        "Prediction_Parity_Passed": equivalent, "Maximum_Absolute_Difference": float(abs(delta).max()),
        "Mean_Absolute_Difference": float(abs(delta).mean()), "Rows": len(delta),
        "New_Kaggle_MAE": None, "Training_Used_Reference_Predictions": False,
        "Interpretation": "Numerically equivalent replay" if equivalent else "Newly trained candidate; submit for its own score"}
    write_json(output_dir / "reproduction_check.json", result)
    return pd.DataFrame([result])


def transition_evidence(scores, registry):
    """Describe observed transitions; never manufacture monotonic progress."""
    required = {"Version", "Segment", "MAE"}
    if not required.issubset(scores.columns) or not {"Set", "Period"}.intersection(scores.columns):
        raise ValueError("Scores require Version, Segment, MAE and Set or Period")
    all_scores = scores.loc[scores["Segment"].eq("All")].copy()

    def version_name(value):
        text = str(value).strip()
        if isinstance(value, (int, np.integer)):
            number = int(value)
        else:
            match = re.fullmatch(r"(?:submission_)?v?([1-8])", text, re.I)
            if match is None:
                raise ValueError(f"Unknown study version: {value!r}")
            number = int(match.group(1))
        if not 1 <= number <= 8:
            raise ValueError(f"Unknown study version: {value!r}")
        return f"submission_v{number}"

    all_scores["Version"] = all_scores["Version"].map(version_name)
    split = pd.Series(pd.NA, index=all_scores.index, dtype="string")
    for column in ["Set", "Period"]:
        if column not in all_scores:
            continue
        current = all_scores[column].astype("string").str.strip()
        current = current.mask(current.eq(""))
        conflict = split.notna() & current.notna() & split.str.casefold().ne(current.str.casefold())
        if conflict.any():
            raise ValueError("Set and Period disagree for the same score row")
        split = split.fillna(current)
    all_scores["_Split"] = split.str.casefold().map({"dev": "Dev", "check": "Check"})
    all_scores = all_scores.loc[all_scores["_Split"].notna()].copy()
    if all_scores.duplicated(["Version", "_Split"]).any():
        raise ValueError("One All-segment MAE is required per version and evaluation set")
    all_scores["MAE"] = pd.to_numeric(all_scores["MAE"], errors="raise")
    if not np.isfinite(all_scores["MAE"]).all() or all_scores["MAE"].lt(0).any():
        raise ValueError("Evaluation MAE must be finite and nonnegative")
    pivot = all_scores.pivot(index="Version", columns="_Split", values="MAE").reindex(columns=["Dev", "Check"])
    rows = []
    for version in range(1, 9):
        name, previous = f"submission_v{version}", f"submission_v{version-1}"
        if name not in pivot.index:
            continue
        entry = next((registry[key] for key in [version, str(version), f"V{version}", name] if key in registry), None)
        if entry is None or "added" not in entry:
            raise ValueError(f"Missing added-information description for version {version}")
        row = {"Version": name, "Added_Information": entry["added"],
               "Comparison_Scope": "Fixed model; feature-block change" if version <= 6 else
                   "Ensemble and calibration change" if version == 7 else "Published allocation rule"}
        for period in ["Dev", "Check"]:
            row[f"{period}_MAE"] = pivot.loc[name, period]
            row[f"{period}_MAE_Reduction"] = (pivot.loc[previous, period] - pivot.loc[name, period]
                                              if previous in pivot.index else np.nan)
        row["Dev_Supports_Addition"] = (bool(row["Dev_MAE_Reduction"] > 0)
            if version > 1 and pd.notna(row["Dev_MAE_Reduction"]) else None)
        row["Interpretation_Status"] = "Pending written interpretation after execution"
        rows.append(row)
    return pd.DataFrame(rows)


def export_review_bundle(root):
    root = Path(root)
    path = root / "presentation/study_review.zip"
    members = []
    members.extend((root / "presentation").rglob("*.png"))
    members.extend((root / "presentation").rglob("*.svg"))
    members.extend((root / "presentation").rglob("*.csv"))
    members.extend((root / "configs").rglob("*.json"))
    for version in range(1, 9):
        folder = root / "outputs" / f"submission_v{version}"
        members.extend((folder / "tables").glob("*.csv"))
        members.extend(folder.glob("recipe.json"))
        members.extend(folder.glob("reproduction_check.json"))
    members.extend((root / "presentation").glob("*.md"))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(set(members)):
            if member.is_file():
                archive.write(member, member.relative_to(root))
    return path

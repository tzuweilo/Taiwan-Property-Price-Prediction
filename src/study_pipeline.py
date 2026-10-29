"""Explicit V1--V6 study engine. Only fit_version() fits an ML model.

The shared 54-column compatibility preprocessing is kept separate from the
correctly named structural feature arithmetic below. No target reference,
validation label, raw transaction ID, or full-address identifier is a feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import hashlib
import json
import re
import platform

import numpy as np
import pandas as pd

try:
    from .try_pipeline import TryPreprocessor, roc_dates, single_floor
except ImportError:
    from try_pipeline import TryPreprocessor, roc_dates, single_floor

TARGET, ID = "總價元", "編號"
LAND, BUILDING, PARKING = "土地移轉總面積平方公尺", "建物移轉總面積平方公尺", "車位移轉總面積平方公尺"
SCALE, MISSING = 1_000_000., "Missing"
FEATURE_IMPLEMENTATION = "study_correct_structural_v1"
STAGE_TITLES = {1: "Area and time", 2: "Location", 3: "Property characteristics",
                4: "Floor, layout and parking", 5: "Project identity", 6: "Remarks and other information"}
DIRECT = {
    1: {LAND: "Land_Area", BUILDING: "Building_Area", PARKING: "Parking_Area",
        "交易年": "Transaction_Year", "交易月": "Transaction_Month", "交易時間序號": "Month_Index"},
    2: {"縣市": "City", "鄉鎮市區": "Town", "行政區": "District", "道路地段": "Road_Section"},
    3: {"交易標的": "Transaction_Target", "建物型態": "Building_Type", "主要用途": "Primary_Use",
        "都市土地使用分區": "Urban_Zoning", "非都市土地使用分區": "Rural_Zoning",
        "非都市土地使用編定": "Rural_Land_Use", "土地使用類型": "Land_Use_Family", "主要建材": "Material"},
    4: {"移轉層次": "Transferred_Floor_Text", "總樓層數": "Total_Floors", "移轉樓層": "Transferred_Floor",
        "整棟交易": "Whole_Building", "多樓層交易": "Multiple_Floors", "相對樓層": "Relative_Floor",
        "建物現況格局-房": "Rooms", "建物現況格局-廳": "Living_Rooms", "建物現況格局-衛": "Bathrooms",
        "車位類別": "Parking_Type", "交易土地筆數": "Land_Count", "交易建物筆數": "Building_Count",
        "交易車位筆數": "Parking_Count"},
    5: {"建案名稱": "Project", "建案行政區": "Project_District", "棟別": "Block"},
    6: {"建物現況格局-隔間": "Partition", "有無管理組織": "Management", "有備註": "Has_Remarks",
        "特殊關係交易": "Special_Relationship", "含裝潢家具家電資訊": "Furnishing_Mention",
        "毛胚屋": "Unfinished_Interior", "雨遮不計價": "Canopy_Not_Priced", "持分交易": "Broad_Share_Mention",
        "含增建": "Extension_Mention", "預售資訊": "Presale_Mention"},
}
NUMERIC = {LAND, BUILDING, PARKING, "交易年", "交易月", "交易時間序號", "總樓層數", "移轉樓層",
    "整棟交易", "多樓層交易", "相對樓層", "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛",
    "交易土地筆數", "交易建物筆數", "交易車位筆數", "有備註", "特殊關係交易", "含裝潢家具家電資訊",
    "毛胚屋", "雨遮不計價", "持分交易", "含增建", "預售資訊"}
REMARK_PATTERNS = {"特殊關係交易": "親友|員工|共有人|特殊關係", "含裝潢家具家電資訊": "裝潢|傢俱|家具|家電|電器",
    "毛胚屋": "毛胚|毛坯", "雨遮不計價": "雨遮.*不計價|雨遮不計價", "持分交易": "持分|部分移轉|部分權利",
    "含增建": "增建|未登記|違建", "預售資訊": "預售|尚未完工|興建中"}
SOURCE_GROUPS = {
    "Area": [LAND, BUILDING, PARKING],
    "Time": ["交易年月日", "交易年", "交易月", "交易時間序號"],
    "Location": ["土地位置建物門牌", "縣市", "鄉鎮市區", "行政區", "道路地段", "地址基地"],
    "Property": list(DIRECT[3]),
    "Structure": ["移轉層次", "總樓層數", "移轉樓層", "整棟交易", "多樓層交易", "相對樓層",
                  "建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛", "建築完成年月"],
    "Parking_Counts": ["交易筆棟數", "交易土地筆數", "交易建物筆數", "交易車位筆數", "車位類別"],
    "Project": ["建案名稱", "建案行政區", "棟及號", "棟別"],
    "Remarks": ["備註"] + list(DIRECT[6]),
}


@dataclass(frozen=True)
class StudyConfig:
    cache_dir: str | Path
    threads: int = 8
    fit_end_month: int = 194
    dev_end_month: int = 197
    check_end_month: int = 199
    iterations: int = 2400
    seed: int = 42
    permutation_rows: int = 5000
    permutation_repeats: int = 3
    persist_predictions: bool = False


def num(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)


def text(values):
    return values.astype("string").fillna(MISSING).astype(str)


def ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.full(len(a), np.nan), where=np.isfinite(b) & (b > 0))


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(frame):
    h = hashlib.sha256("|".join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def assert_unlabelled(frame):
    forbidden = {TARGET, "總價", "actual", "target", "label", "y_true", "answer", "ground_truth", "matched_price"}
    bad = [c for c in frame if str(c).casefold() in forbidden or re.search("答案|真值|匹配總價|實際總價", str(c))]
    if bad:
        raise ValueError("Prospective Test features must not contain answer columns: " + ", ".join(map(str, bad)))


def prepare_study(raw_train, raw_valid, raw_test):
    """Fit text-recovery vocabulary on Train only; do not alter input objects."""
    assert_unlabelled(raw_test)
    preprocessor = TryPreprocessor(raw_train)
    frames = {"train": preprocessor.transform(raw_train, training=True),
              "valid": preprocessor.transform(raw_valid, training=False),
              "test": preprocessor.transform(raw_test, training=False)}
    if len(frames["test"]) != len(raw_test):
        raise ValueError("Unlabelled Test preprocessing changed its row population")
    np.testing.assert_array_equal(frames["test"][ID].astype(str), raw_test[ID].astype(str))
    frames.update(preprocessor=preprocessor, coverage=column_roles(frames["train"]))
    return frames


def split_study(train, config):
    """One fixed chronological cohort with unique, globally unambiguous evaluation IDs."""
    month = num(train["交易時間序號"])
    masks = {"fit": month <= config.fit_end_month,
             "dev": (month > config.fit_end_month) & (month <= config.dev_end_month),
             "check": (month > config.dev_end_month) & (month <= config.check_end_month)}
    ids = train[ID].astype("string")
    duplicate = ids.duplicated(keep=False).to_numpy(bool)
    missing = (ids.isna() | ids.str.strip().eq("").fillna(False)).to_numpy(bool)
    evaluation = masks["dev"] | masks["check"]
    excluded_duplicate = evaluation & duplicate
    excluded_missing = evaluation & missing
    masks["dev"] &= ~(duplicate | missing)
    masks["check"] &= ~(duplicate | missing)
    future_ids = set(train.loc[masks["dev"] | masks["check"], ID].astype(str))
    overlap = masks["fit"] & train[ID].astype(str).isin(future_ids).to_numpy(bool)
    masks["fit"] &= ~overlap
    result = {name: train.loc[mask].reset_index(drop=True).copy() for name, mask in masks.items()}
    if not all(len(result[name]) for name in masks):
        raise ValueError("Fixed chronological split has an empty fit/development/check population")
    for name in ["dev", "check"]:
        if not result[name][ID].is_unique:
            raise ValueError("Duplicate transaction IDs remain in evaluation")
        if set(result["fit"][ID].astype(str)).intersection(result[name][ID].astype(str)):
            raise ValueError("A transaction ID crosses fit and evaluation")
    if set(result["dev"][ID].astype(str)).intersection(result["check"][ID].astype(str)):
        raise ValueError("A transaction ID crosses development and check")
    result["counts"] = pd.DataFrame([{"Set": k.title(), "Rows": len(result[k])} for k in masks] + [
        {"Set": "Missing month excluded", "Rows": int((~np.isfinite(month)).sum())},
        {"Set": "Duplicate evaluation ID rows excluded", "Rows": int(excluded_duplicate.sum())},
        {"Set": "Missing evaluation ID rows excluded", "Rows": int(excluded_missing.sum())},
        {"Set": "Fit ID overlap excluded", "Rows": int(overlap.sum())}])
    exclusions = train.loc[excluded_duplicate | excluded_missing, [ID, "交易時間序號"]].copy()
    exclusions["Reason"] = np.where(missing[excluded_duplicate | excluded_missing], "Missing ID", "Globally duplicated ID")
    result["excluded_evaluation_ids"] = exclusions.reset_index(drop=True)
    return result


def _dates(frame, column):
    return roc_dates(frame[column], minimum=0)


def _coarse_parcel(series):
    s = text(series).str.replace("台", "臺", regex=False)
    s = s.str.replace(r"^.{2,3}[縣市]", "", regex=True)
    return s.str.extract(r"([^\d,，;；]{1,30}?段)(?:\d|之|小段)", expand=False).fillna(MISSING)


def feature_frame(frame, version=6):
    """Cumulative English-named features; this is NOT the old swapped-area Try frame."""
    version = int(str(version).lstrip("Vv"))
    if not 1 <= version <= 6:
        raise ValueError("Study feature versions are V1 through V6")
    x = pd.DataFrame(index=np.arange(len(frame)))
    for stage in range(1, version + 1):
        for original, alias in DIRECT[stage].items():
            if original not in frame:
                raise KeyError("Missing canonical processed feature: " + original)
            x[alias] = num(frame[original]) if original in NUMERIC else text(frame[original]).to_numpy(copy=True)
    month = num(frame["交易月"])
    x["Month_Sin"], x["Month_Cos"] = np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)
    if version >= 2:
        x["Parcel_Section"] = _coarse_parcel(frame["土地位置建物門牌"]).to_numpy(copy=True)
    if version >= 3:
        for alias, a, b in [("District_Type", "行政區", "建物型態"), ("District_Use", "行政區", "主要用途"),
                             ("Road_Type", "道路地段", "建物型態"), ("Type_Use", "建物型態", "主要用途")]:
            x[alias] = (text(frame[a]) + "|" + text(frame[b])).to_numpy(copy=True)
    if version >= 4:
        building, land, parking = (num(frame[c]) for c in [BUILDING, LAND, PARKING])
        x["Building_Net_Parking"] = np.where((building > 0) & (parking >= 0) & (building >= parking), building - parking, np.nan)
        x["Parking_Building_Ratio"], x["Land_Building_Ratio"] = ratio(parking, building), ratio(land, building)
        x["Building_Per_Count"] = ratio(building, num(frame["交易建物筆數"]))
        x["Land_Per_Count"] = ratio(land, num(frame["交易土地筆數"]))
        x["Parking_Per_Count"] = ratio(parking, num(frame["交易車位筆數"]))
        x["Building_Per_Room"] = ratio(building, num(frame["建物現況格局-房"]))
        x["Ground_Floor"] = (num(frame["移轉樓層"]) == 1).astype("int8")
        x["Basement"] = (num(frame["移轉樓層"]) < 0).astype("int8")
        transaction, completed = _dates(frame, "交易年月日"), _dates(frame, "建築完成年月")
        age = (transaction - completed).dt.days / 365.25
        x["Completion_Year"], x["Building_Age"] = completed.dt.year.to_numpy(float), age.where(age >= 0).to_numpy(float)
    if version >= 5:
        for alias, a, b in [("Project_Type", "建案行政區", "建物型態"), ("Project_Block", "建案行政區", "棟別")]:
            x[alias] = (text(frame[a]) + "|" + text(frame[b])).to_numpy(copy=True)
    if version >= 6:
        remarks = text(frame["備註"])
        x["Surface_Rights_Mention"] = remarks.str.contains("地上權", na=False).to_numpy("int8")
        x["Joint_Buyers_Mention"] = remarks.str.contains("共同買受|共同購買|共同承購|買受人.*[二兩多]人", na=False).to_numpy("int8")
        transaction, completed = _dates(frame, "交易年月日"), _dates(frame, "建築完成年月")
        x["Transaction_Day"] = transaction.dt.day.to_numpy(float)
        x["Transaction_Date_Missing"], x["Completion_Date_Missing"] = transaction.isna().to_numpy("int8"), completed.isna().to_numpy("int8")
        x["Completion_After_Transaction"] = (completed > transaction).to_numpy("int8")
        for original, alias in [(LAND, "Land"), (BUILDING, "Building"), (PARKING, "Parking")]:
            x[alias + "_Area_Missing"] = (~np.isfinite(num(frame[original]))).astype("int8")
        x["Unparsed_Transaction_Counts"] = pd.to_numeric(frame["交易建物筆數"], errors="coerce").isna().to_numpy("int8")
    for column in x:
        if pd.api.types.is_numeric_dtype(x[column]):
            values = num(x[column]).astype("float32"); values[~np.isfinite(values)] = np.nan; x[column] = values
        else:
            x[column] = text(x[column])
    if not x.columns.is_unique or {ID, TARGET, "Actual", "_Source_Row"}.intersection(x.columns):
        raise ValueError("Feature schema contains an identifier/outcome or duplicate name")
    return x


def column_roles(frame):
    direct = {column: (stage, alias) for stage, mapping in DIRECT.items() for column, alias in mapping.items()}
    derived = {"交易年月日": (1, "Year/month/index, month cycle; day/validity in V6"),
        "土地位置建物門牌": (2, "Coarse city/district/road/parcel only; full address excluded"),
        "地址基地": (2, "Coarse location already represented; exact base-address category excluded"),
        "建築完成年月": (4, "Completion year, age, validity"), "交易筆棟數": (4, "Parsed land/building/parking counts"),
        "棟及號": (5, "Block prefix only; unit identifier excluded"), "備註": (6, "Semantic flags only; raw text excluded")}
    records = []
    for column in frame:
        if column in {ID, "_Source_Row"}:
            role, stage, representation = "Excluded identifier", None, "Row alignment only"
        elif column == TARGET:
            role, stage, representation = "Outcome", None, "Training/evaluation target only"
        elif column == "解約情形":
            role, stage, representation = "Excluded post-outcome", None, "Cancellation status excluded"
        elif column in direct:
            stage, representation = direct[column]; role = "Model predictor"
        elif column in derived:
            stage, representation = derived[column]; role = "Derived/coarse representation"
        else:
            role, stage, representation = "Excluded unconfigured field", None, "Not in the frozen V1--V6 design"
        group = next((g for g, cols in SOURCE_GROUPS.items() if column in cols), "Outcome/Excluded")
        records.append({"Feature": column, "Role": role, "First_Version": stage,
                        "Group": group, "Representation": representation})
    return pd.DataFrame(records)


def _parameters(config):
    return {"loss_function": "MAE", "eval_metric": "MAE", "iterations": config.iterations,
        "learning_rate": .04, "depth": 8, "l2_leaf_reg": 8, "random_seed": config.seed,
        "thread_count": config.threads, "allow_writing_files": False, "verbose": 200}


def _check_schema(model, columns, categorical):
    if not model.is_fitted() or list(model.feature_names_ or []) != columns:
        raise ValueError("Saved CatBoost named schema differs")
    if list(model.get_cat_feature_indices()) != categorical:
        raise ValueError("Saved CatBoost categorical schema differs")


def predict_version(bundle, frame):
    x = feature_frame(frame, bundle["version"])
    if x.columns.tolist() != bundle["columns"]:
        raise ValueError("Prediction schema differs from the version's fitted schema")
    prediction = np.maximum(np.asarray(bundle["model"].predict(x, thread_count=bundle["threads"]), float) * bundle["scale"], 0)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite study predictions")
    return prediction


def metrics(actual_frame, prediction, known_projects, version, set_name, baseline_median):
    y, area = num(actual_frame[TARGET]), num(actual_frame[BUILDING])
    valid = actual_frame["交易標的"].astype("string").str.contains("房地|建物", na=False).to_numpy(bool) & np.isfinite(area) & (area > 0)
    ordinary = valid & (area <= 300)
    named = actual_frame["建案名稱"].astype("string").fillna("缺失").ne("缺失").to_numpy(bool)
    seen = actual_frame["建案行政區"].isin(known_projects).to_numpy(bool)
    masks = {"All": np.ones(len(y), bool), "Ordinary": ordinary,
        "Seen_Ordinary": ordinary & named & seen, "Unseen_Ordinary": ordinary & named & ~seen,
        "Large_Area": valid & (area > 300)}
    records = []
    for segment, mask in masks.items():
        if mask.any():
            error = abs(y[mask] - prediction[mask]); residual = y[mask] - prediction[mask]
            records.append({"Version": f"V{version}", "Set": set_name, "Segment": segment, "Rows": int(mask.sum()),
                "MAE": error.mean(), "Median_Residual": np.median(residual),
                "Underprediction_Pct": 100 * np.mean(residual > 0),
                "Median_Baseline_MAE": abs(y[mask] - baseline_median).mean()})
    return pd.DataFrame(records)


def fit_version(train_processed, evaluations: Mapping[str, pd.DataFrame] | None, version, config):
    """Train or load one total-price model; evaluate every row without filtering."""
    from catboost import CatBoostRegressor
    import catboost
    version = int(str(version).lstrip("Vv")); evaluations = dict(evaluations or {})
    if TARGET not in train_processed or not len(train_processed):
        raise ValueError("A nonempty labelled fitting frame is required")
    y = num(train_processed[TARGET]) / SCALE
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Training labels must be finite and positive")
    for label, frame in evaluations.items():
        if label.casefold() == "test":
            assert_unlabelled(frame)
        if set(train_processed[ID].astype(str)).intersection(frame[ID].astype(str)):
            raise ValueError("A transaction ID crosses fit and evaluation: " + label)
    x = feature_frame(train_processed.drop(columns=[TARGET]), version)
    params = _parameters(config)
    categorical_names = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    categorical = [x.columns.get_loc(c) for c in categorical_names]
    identity = {"stage": version, "feature_implementation": FEATURE_IMPLEMENTATION,
        "x_sha256": frame_hash(x), "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "ids_sha256": frame_hash(train_processed[[ID]]), "parameters": params,
        "scale": SCALE, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                     "pandas": pd.__version__, "catboost": catboost.__version__}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(config.cache_dir) / f"V{version}" / signature[:12]; folder.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = folder / "model.cbm", folder / "metadata.json"
    if model_path.is_file() != metadata_path.is_file():
        raise ValueError("Incomplete study model cache; model and metadata must both exist")
    if model_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity or metadata["sha256"] != hash_file(model_path):
            raise ValueError("Study model cache failed its identity/checksum validation")
        model = CatBoostRegressor(); model.load_model(str(model_path))
        print(f"Loaded V{version}: {len(train_processed):,} rows, {len(x.columns)} features")
    else:
        print(f"Training V{version}: {len(train_processed):,} rows, {len(x.columns)} features", flush=True)
        model = CatBoostRegressor(**params)
        model.fit(x, y, cat_features=categorical_names)
        temporary = model_path.with_name("model.pending.cbm"); model.save_model(str(temporary)); temporary.replace(model_path)
        save_json(metadata_path, {"identity": identity, "signature": signature, "sha256": hash_file(model_path),
            "columns": x.columns.tolist(), "categorical_indices": categorical, "rows": len(y)})
    _check_schema(model, x.columns.tolist(), categorical)
    known = set(train_processed.loc[train_processed["建案名稱"].astype(str).ne("缺失"), "建案行政區"].astype(str))
    bundle = {"model": model, "version": version, "columns": x.columns.tolist(), "categorical_indices": categorical,
        "threads": config.threads, "scale": SCALE, "known_projects": known, "folder": folder, "parameters": params}
    scores, paired = [], {}
    baseline = float(np.median(num(train_processed[TARGET])))
    for label, frame in evaluations.items():
        prediction = predict_version(bundle, frame.drop(columns=[TARGET], errors="ignore"))
        detail = frame[[ID]].copy(); detail["Prediction"] = prediction
        if TARGET in frame:
            detail["Actual"] = num(frame[TARGET])
            scores.append(metrics(frame, prediction, known, version, label, baseline))
        paired[label] = detail
        if config.persist_predictions:
            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(label))
            detail.to_csv(folder / f"predictions_{safe_name}.csv", index=False)
    score_frame = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    score_frame.to_csv(folder / "scores.csv", index=False)
    return {"bundle": bundle, "scores": score_frame, "predictions": paired, "folder": folder}


def export_version(bundle, test_processed, path):
    assert_unlabelled(test_processed)
    if test_processed[ID].isna().any() or not test_processed[ID].is_unique:
        raise ValueError("Test IDs must be unique and nonmissing")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    prediction = predict_version(bundle, test_processed)
    result = pd.DataFrame({ID: test_processed[ID].to_numpy(copy=True), TARGET: prediction})
    if path.exists():
        old = pd.read_csv(path, dtype={ID: "string"})
        np.testing.assert_array_equal(old[ID].astype(str), result[ID].astype(str))
        np.testing.assert_allclose(num(old[TARGET]), prediction, rtol=1e-12, atol=1e-5)
    else:
        result.to_csv(path, index=False, float_format="%.10f")
    return path


def _attribute_values(frame, column):
    """One source attribute, represented without a transaction-unique text ID."""
    if column in {"土地位置建物門牌", "地址基地"}:
        return text(frame["道路地段"]), "Coarse location from address", False
    if column == "棟及號":
        return text(frame[column]).str.extract(r"^([A-Za-z]+\d{0,2})", expand=False).fillna(MISSING), "Block prefix", False
    if column == "備註":
        s = text(frame[column]); flags = [s.str.contains(pattern, na=False).astype(str) for pattern in REMARK_PATTERNS.values()]
        value = flags[0]
        for flag in flags[1:]:
            value = value + "|" + flag
        return value, "Semantic remark pattern", False
    if column in {"交易年月日", "建築完成年月"}:
        date = _dates(frame, column)
        values = pd.Series(np.where(date.notna(), date.astype("int64") / 86_400_000_000_000, np.nan), index=frame.index)
        return values, "Parsed date ordinal", True
    numeric = pd.api.types.is_numeric_dtype(frame[column]) or column in NUMERIC
    return frame[column], "Numeric quantile bins" if numeric else "Supported categorical median", numeric


def _univariate_keys(training, other, is_numeric, bins):
    if is_numeric:
        a = num(training); b = num(other); finite = a[np.isfinite(a)]
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))[1:-1] if len(finite) else np.array([])
        def key(values):
            return pd.Series(np.where(np.isfinite(values), np.searchsorted(edges, values).astype(str), MISSING))
        return key(a), key(b)
    return text(training).reset_index(drop=True), text(other).reset_index(drop=True)


def univariate_attribute_scores(fit, dev, check, bins=20, minimum_rows=10, smoothing=50.):
    """All processed columns; Train-only lookup regression, no ML training."""
    y = num(fit[TARGET])
    if not len(y) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Univariate fitting labels must be finite and positive")
    if bins < 2 or minimum_rows < 1 or smoothing < 0:
        raise ValueError("Invalid fixed univariate configuration")
    global_median = float(np.median(y))
    roles = column_roles(fit).set_index("Feature"); rows = []
    for column in fit.columns:
        role = roles.loc[column, "Role"]
        excluded = role.startswith("Excluded") or role == "Outcome"
        for label, evaluation in [("Dev", dev), ("Check", check)]:
            baseline = abs(num(evaluation[TARGET]) - global_median).mean()
            record = {"Feature": column, "Set": label, "Role": role, "Group": roles.loc[column, "Group"],
                "Rows": len(evaluation), "Baseline_MAE": baseline, "Univariate_MAE": np.nan,
                "Improvement_Pct": np.nan, "Unseen_Pct": np.nan, "Train_Distinct": np.nan,
                "Status": "Excluded" if excluded else "Evaluated", "Representation": roles.loc[column, "Representation"]}
            if not excluded:
                train_value, representation, numeric = _attribute_values(fit, column)
                query_value, _, _ = _attribute_values(evaluation, column)
                train_key, query_key = _univariate_keys(train_value, query_value, numeric, bins)
                grouped = pd.DataFrame({"Key": train_key, "Y": y}).groupby("Key", dropna=False).Y.agg(["size", "median"])
                weight = grouped["size"] / (grouped["size"] + smoothing)
                lookup = (weight * grouped["median"] + (1 - weight) * global_median).where(grouped["size"] >= minimum_rows, global_median)
                mapped = query_key.map(lookup)
                prediction = mapped.fillna(global_median).to_numpy(float)
                mae = abs(num(evaluation[TARGET]) - prediction).mean()
                record.update(Representation=representation, Univariate_MAE=mae,
                    Improvement_Pct=100 * (1 - mae / baseline) if baseline else np.nan,
                    Unseen_Pct=100 * mapped.isna().mean(), Train_Distinct=int(train_key.nunique()),
                    Status="Constant/Unavailable" if train_key.nunique() <= 1 else "Evaluated")
            rows.append(record)
    return pd.DataFrame(rows)


def _refresh(frame, changed, original=None):
    """Refresh deterministic descendants after permuting a processed source."""
    changed = set(changed)
    if "交易年月日" in changed:
        date = _dates(frame, "交易年月日")
        frame["交易年"], frame["交易月"] = date.dt.year, date.dt.month
        frame["交易時間序號"] = (frame["交易年"] - 2008) * 12 + frame["交易月"] - 1
    elif changed.intersection({"交易年", "交易月", "交易時間序號"}):
        if "交易時間序號" in changed:
            m = pd.to_numeric(frame["交易時間序號"], errors="coerce")
            frame["交易年"], frame["交易月"] = 2008 + np.floor(m / 12), m % 12 + 1
        frame["交易時間序號"] = (pd.to_numeric(frame["交易年"], errors="coerce") - 2008) * 12 + pd.to_numeric(frame["交易月"], errors="coerce") - 1
        old_date = _dates(frame, "交易年月日")
        first = pd.to_datetime({"year": frame["交易年"], "month": frame["交易月"], "day": np.ones(len(frame))}, errors="coerce")
        day = np.minimum(old_date.dt.day.fillna(1).to_numpy(), first.dt.days_in_month.to_numpy())
        date = first + pd.to_timedelta(day - 1, unit="D")
        frame["交易年月日"] = ((date.dt.year - 1911).astype("Int64").astype("string").str.zfill(3)
            + date.dt.month.astype("Int64").astype("string").str.zfill(2)
            + date.dt.day.astype("Int64").astype("string").str.zfill(2))
    if "交易筆棟數" in changed:
        counts = frame["交易筆棟數"].astype("string").str.extract(r"^土地(\d+)建物(\d+)車位(\d+)$")
        for i, column in enumerate(["交易土地筆數", "交易建物筆數", "交易車位筆數"]):
            frame[column] = pd.to_numeric(counts[i], errors="coerce")
    if "移轉層次" in changed:
        floor = frame["移轉層次"].astype("string")
        frame["移轉樓層"] = floor.map(single_floor)
        frame["整棟交易"] = (floor.isin(["全", "全部", "全棟", "整棟", "全.透天厝"]) | floor.str.contains("整棟", na=False)).astype("int8")
        frame["多樓層交易"] = (frame["移轉樓層"].isna() & frame["整棟交易"].eq(0) & floor.str.contains(r"[+＋\-~～、，,＆&及與至到]|樓中樓", na=False)).astype("int8")
        frame.loc[floor.str.contains("申報事項|加購", na=False), "多樓層交易"] = 0
    if changed.intersection({"移轉層次", "移轉樓層", "總樓層數"}):
        floor, total = num(frame["移轉樓層"]), num(frame["總樓層數"])
        frame["相對樓層"] = np.where((floor > 0) & (floor <= total), ratio(floor, total), np.nan)
    if "備註" in changed:
        s = frame["備註"].astype("string")
        frame["有備註"] = (s.notna() & s.ne("缺失")).astype("int8")
        for column, pattern in REMARK_PATTERNS.items():
            frame[column] = s.str.contains(pattern, na=False).astype("int8")
    if "棟及號" in changed:
        frame["棟別"] = frame["棟及號"].astype("string").str.extract(r"^([A-Za-z]+\d{0,2})", expand=False).fillna("缺失")
    if changed.intersection({"縣市", "鄉鎮市區", "行政區"}) and "土地位置建物門牌" not in changed:
        old = (original if original is not None else frame)["行政區"].astype(str).copy()
        if "行政區" in changed:
            pieces = frame["行政區"].astype("string").str.split("_", n=1, expand=True)
            frame["縣市"], frame["鄉鎮市區"] = pieces[0].fillna("缺失"), pieces[1].fillna("缺失")
        else:
            frame["行政區"] = text(frame["縣市"]) + "_" + text(frame["鄉鎮市區"])
        if "道路地段" not in changed:
            frame["道路地段"] = [new + road[len(previous):] if road.startswith(previous) else road
                for road, previous, new in zip(frame["道路地段"].astype(str), old, frame["行政區"].astype(str))]
    if changed.intersection({"都市土地使用分區", "非都市土地使用分區"}):
        present = lambda column: frame[column].notna() & ~text(frame[column]).isin(["缺失", MISSING, ""])
        urban, rural = present("都市土地使用分區"), present("非都市土地使用分區")
        frame["土地使用類型"] = np.select([urban & ~rural, ~urban & rural, urban & rural],
            ["都市", "非都市", "混合"], default="其他或未標示")
    if changed.intersection({"建案名稱", "行政區", "縣市", "鄉鎮市區", "土地位置建物門牌"}) and "建案行政區" not in changed:
        frame["建案行政區"] = text(frame["行政區"]) + "_" + text(frame["建案名稱"])
    return frame


def grouped_source_permutation(bundle, processed_dev, version=6, repeats=3, max_rows=5000,
                               seed=42, include_columns=True, include_groups=True):
    """Return one integrated table. Group/column rows explicitly identify perturbed sources.

    Uses a uniform fixed sample, so sampled MAE is descriptive rather than a
    precision estimate for rare giant transactions. No true-price sampling.
    """
    if int(version) != bundle["version"] or TARGET not in processed_dev:
        raise ValueError("Permutation needs the matching fitted version and labelled holdout")
    if repeats < 1 or max_rows < 1 or not len(processed_dev):
        raise ValueError("Permutation requires a nonempty sample and at least one repeat")
    sample = processed_dev.sample(n=min(max_rows, len(processed_dev)), random_state=seed).reset_index(drop=True).copy()
    actual = num(sample[TARGET]); baseline = predict_version(bundle, sample.drop(columns=[TARGET])); base_error = abs(actual - baseline).mean()
    roles = column_roles(sample).set_index("Feature"); tasks = []
    if include_groups:
        tasks += [(group, "Group", [c for c in columns if c in sample]) for group, columns in SOURCE_GROUPS.items()]
    if include_columns:
        tasks += [(c, "Column", [c]) for c in sample if c not in {TARGET, "_Source_Row"}]
    rng = np.random.default_rng(seed); records = []
    for name, scope, columns in tasks:
        if scope == "Column" and (roles.loc[name, "Role"].startswith("Excluded") or roles.loc[name, "Role"] == "Outcome"):
            records.append({"Feature": name, "Group": roles.loc[name, "Group"], "Scope": scope, "Status": "Excluded",
                            "MAE_Increase": np.nan, "MAE_Increase_Pct": np.nan, "Std": np.nan, "Rows": len(sample)})
            continue
        if not columns:
            continue
        if name == "土地位置建物門牌" and scope == "Column":
            columns = [c for c in SOURCE_GROUPS["Location"] if c in sample]
        increases = []
        for _ in range(repeats):
            permuted = sample.drop(columns=[TARGET]).copy(deep=True)
            order = rng.permutation(len(sample))
            for column in columns:
                permuted[column] = sample[column].iloc[order].to_numpy(copy=True)
            permuted = _refresh(permuted, columns, original=sample)
            increases.append(abs(actual - predict_version(bundle, permuted)).mean() - base_error)
        records.append({"Feature": name, "Group": name if scope == "Group" else roles.loc[name, "Group"],
            "Scope": scope, "Status": "Evaluated", "Rows": len(sample), "Repeats": repeats,
            "Source_Columns": " | ".join(columns), "Baseline_MAE": base_error,
            "MAE_Increase": float(np.mean(increases)), "MAE_Increase_Pct": 100 * np.mean(increases) / base_error if base_error else np.nan,
            "Std": float(np.std(increases, ddof=1)) if repeats > 1 else 0.,
            "Sampling": "Uniform fixed holdout sample; no rare-event guarantee"})
    return pd.DataFrame(records)


def prediction_values_change(bundle):
    values = np.asarray(bundle["model"].get_feature_importance(type="PredictionValuesChange"), float)
    if len(values) != len(bundle["columns"]):
        raise ValueError("Native feature importance schema differs")
    return pd.DataFrame({"Feature": bundle["columns"], "Importance": values,
                         "Method": "PredictionValuesChange; predictive, not causal"}).sort_values("Importance", ascending=False)


def shap_values(bundle, frame, max_rows=256, seed=42):
    """Optional CatBoost-native SHAP; no external shap dependency."""
    from catboost import Pool
    sample = frame.sample(n=min(max_rows, len(frame)), random_state=seed)
    x = feature_frame(sample.drop(columns=[TARGET], errors="ignore"), bundle["version"])
    pool = Pool(x, cat_features=bundle["categorical_indices"])
    values = np.asarray(bundle["model"].get_feature_importance(data=pool, type="ShapValues")) * bundle["scale"]
    return pd.DataFrame(values, columns=bundle["columns"] + ["Expected_Value"]), sample[[ID]].reset_index(drop=True)

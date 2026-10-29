"""Independent reproduction of the locked Try model branch.

All inputs, feature schemas and reference objects are explicit. Only fit() trains.
Raw validation/test labels are never accepted as fitting inputs. The original
preprocessing is retained for compatibility with the recorded best submission;
its historical >=20 layout rule is documented rather than called optimal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
import json
import pickle
import re
import unicodedata
import numpy as np
import pandas as pd

TARGET, ID_COLUMN = "總價元", "編號"

AREA_COLUMNS = [
    "土地移轉總面積平方公尺", "建物移轉總面積平方公尺", "車位移轉總面積平方公尺"
]

LAYOUT_COLUMNS = ["建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛"]

BASE_NUMERIC = AREA_COLUMNS + LAYOUT_COLUMNS + [
    "總樓層數", "交易年", "交易月", "交易土地筆數",
    "交易建物筆數", "交易車位筆數", "移轉樓層", "相對樓層"
]

BASE_CATEGORICAL = [
    "鄉鎮市區", "交易標的", "都市土地使用分區", "非都市土地使用分區",
    "非都市土地使用編定", "建物型態", "主要用途", "主要建材",
    "建物現況格局-隔間", "車位類別", "土地使用類型", "縣市", "行政區"
]

CITY_REPLACE = {
    "台北市": "臺北市", "台中市": "臺中市",
    "台南市": "臺南市", "桃園縣": "桃園市"
}

CITY_FILL = {
    "清水區": "臺中市", "員山鄉": "宜蘭縣",
    "左營區": "高雄市", "梅山鄉": "嘉義縣"
}

def normalize_text(value):
    if pd.isna(value):
        return pd.NA
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return value if value and value.lower() not in {"nan", "none", "null", "<na>"} else pd.NA

def roc_dates(series, minimum=90):
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    text = text.where(text.str.fullmatch(r"\d{6,7}", na=False)).str.zfill(7)
    year = pd.to_numeric(text.str[:3], errors="coerce")
    gregorian = (year + 1911).astype("Int64").astype("string") + text.str[3:]
    dates = pd.to_datetime(gregorian, format="%Y%m%d", errors="coerce")
    return dates.where(year.between(minimum, 115))

def chinese_integer(text):
    digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text.isdigit():
        return int(text)
    if text in digits:
        return digits[text]
    if text.count("十") == 1:
        left, right = text.split("十")
        if (not left or left in digits) and (not right or right in digits):
            return digits.get(left, 1) * 10 + digits.get(right, 0)
    return np.nan

def single_floor(value):
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(
        r"(地下|地上)?([零一二三四五六七八九十\d]+)(?:層|樓)?", str(value).strip())
    if not match:
        return np.nan
    result = chinese_integer(match.group(2))
    return -result if match.group(1) == "地下" else result

def safe_ratio(numerator, denominator):
    return numerator.div(denominator.where(denominator > 0)).replace(
        [np.inf, -np.inf], np.nan)

def recover_district(address, districts):
    if pd.isna(address):
        return pd.NA
    rest = re.sub(r"^.{2,3}[縣市]", "", str(address))
    return next((district for district in districts if rest.startswith(district)), pd.NA)

def _preprocess_exact(raw, districts, training=False):
    df = raw.copy()
    if TARGET in df:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.loc[df[TARGET].notna() & df[TARGET].gt(0)].copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col not in [TARGET, ID_COLUMN]:
            df[col] = df[col].map(normalize_text).astype("string")
    for col in BASE_NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in AREA_COLUMNS:
        df.loc[df[col] < 0, col] = np.nan
    for col in LAYOUT_COLUMNS:
        df.loc[(df[col] >= 20) | (df[col] < 0), col] = np.nan

    dates = roc_dates(df["交易年月日"])
    df["交易年"], df["交易月"] = dates.dt.year, dates.dt.month
    df["交易時間序號"] = (df["交易年"] - 2008) * 12 + df["交易月"] - 1
    parsed = df["交易筆棟數"].str.extract(r"^土地(\d+)建物(\d+)車位(\d+)$")
    for index, col in enumerate(["交易土地筆數", "交易建物筆數", "交易車位筆數"]):
        df[col] = pd.to_numeric(parsed[index], errors="coerce")

    floor = df["移轉層次"]
    df["移轉樓層"] = floor.map(single_floor)
    df["整棟交易"] = (
        floor.isin(["全", "全部", "全棟", "整棟", "全.透天厝"])
        | floor.str.contains("整棟", na=False)).astype("int8")
    df["多樓層交易"] = (
        df["移轉樓層"].isna() & df["整棟交易"].eq(0)
        & floor.str.contains(r"[+＋\-~～、，,＆&及與至到]|樓中樓", na=False)
    ).astype("int8")
    df.loc[floor.str.contains("申報事項|加購", na=False), "多樓層交易"] = 0
    df.loc[df["總樓層數"] <= 0, "總樓層數"] = np.nan
    valid_floor = df["移轉樓層"].gt(0) & df["移轉樓層"].le(df["總樓層數"])
    df["相對樓層"] = safe_ratio(df["移轉樓層"], df["總樓層數"]).where(valid_floor)

    conflict = (
        df[AREA_COLUMNS[1]].eq(0) & df["交易建物筆數"].gt(0)
        & df["交易標的"].ne("車位")).fillna(False)
    df.loc[conflict, AREA_COLUMNS[1]] = np.nan
    if training:
        df.loc[df[AREA_COLUMNS[0]].isin([1044217.0, 27884.90]), AREA_COLUMNS[0]] = np.nan
        df = df.loc[~df[TARGET].isin([520, 1760])].copy()

    for col in ["主要用途", "主要建材"]:
        df[col] = df[col].replace({"見其它登記事項": "見其他登記事項"})
    urban, rural = df["都市土地使用分區"].notna(), df["非都市土地使用分區"].notna()
    df["土地使用類型"] = np.select(
        [urban & ~rural, ~urban & rural, urban & rural],
        ["都市", "非都市", "混合"], default="其他或未標示")

    address = df["土地位置建物門牌"].str.replace("台", "臺", regex=False)
    df["縣市"] = address.str.extract(r"^(.{2,3}[縣市])", expand=False).replace(CITY_REPLACE)
    missing = df["鄉鎮市區"].isna()
    df.loc[missing, "鄉鎮市區"] = address.loc[missing].map(
        lambda value: recover_district(value, districts))
    df["縣市"] = df["縣市"].fillna(df["鄉鎮市區"].map(CITY_FILL))
    df["行政區"] = df["縣市"].fillna("缺失") + "_" + df["鄉鎮市區"].fillna("缺失")

    remark = df["備註"]
    df["有備註"] = remark.notna().astype("int8")
    patterns = {
        "特殊關係交易": "親友|員工|共有人|特殊關係",
        "含裝潢家具家電資訊": "裝潢|傢俱|家具|家電|電器",
        "毛胚屋": "毛胚|毛坯", "雨遮不計價": "雨遮.*不計價|雨遮不計價",
        "持分交易": "持分|部分移轉|部分權利",
        "含增建": "增建|未登記|違建", "預售資訊": "預售|尚未完工|興建中"
    }
    for col, pattern in patterns.items():
        df[col] = remark.str.contains(pattern, regex=True, na=False).astype("int8")

    rest = address.str.replace(r"^.{2,3}[縣市]", "", regex=True)
    road = rest.str.extract(
        r"((?:[^\d,，;；]{1,30}?)(?:大道|路|街)(?:[一二三四五六七八九十\d]+段)?)",
        expand=False)
    parcel = rest.str.extract(r"([^\d,，;；]{1,30}?段)(?:\d|之|小段)", expand=False)
    df["道路地段"] = df["行政區"] + "_" + road.fillna(parcel).fillna("缺失")
    df["地址基地"] = address.str.replace(
        r"(?:地下|地上)?[零一二三四五六七八九十\d]+[樓層].*$", "", regex=True).fillna("缺失")
    df["建案行政區"] = df["行政區"] + "_" + df["建案名稱"].fillna("缺失")
    df["棟別"] = df["棟及號"].str.extract(
        r"^([A-Za-z]+\d{0,2})", expand=False).fillna("缺失")

    for col in BASE_CATEGORICAL + [
        "建案名稱", "道路地段", "地址基地", "建案行政區", "棟別"
    ]:
        df[col] = df[col].fillna("缺失").astype(str)
    return df.reset_index(drop=True)

def main_feature_frame(frame):
    """Frozen feature arithmetic used by the recorded best Try branch.

    Compatibility note: the original notebook unpacked AREA_COLUMNS (land,
    building, parking) into variables named (building, land, parking). Seven
    engineered labels therefore describe the wrong source area. This function
    intentionally reproduces that arithmetic; new corrected feature experiments
    must be labelled separately and cannot inherit the old Kaggle score.
    """
    excluded = {TARGET, ID_COLUMN, "交易年月日", "建築完成年月"}
    matrix = frame.drop(columns=list(excluded), errors="ignore").copy()
    building, land, parking = [
        pd.to_numeric(frame[col], errors="coerce") for col in AREA_COLUMNS
    ]
    count = pd.to_numeric(frame["交易建物筆數"], errors="coerce")
    housing = frame["交易標的"].astype("string").str.contains("房地|建物", na=False)
    matrix["建物筆數面積衝突"] = (
        housing & building.gt(0) & (count.le(0) | count.isna())
    ).fillna(False).astype("int8")
    matrix["扣車位建物面積"] = (building - parking).where(
        building.gt(0) & parking.ge(0) & building.ge(parking))
    matrix["車位面積占比"] = safe_ratio(parking, building)
    matrix["土地建物面積比"] = safe_ratio(land, building)
    matrix["每筆建物面積"] = safe_ratio(building, count)
    matrix["每車位面積"] = safe_ratio(
        parking, pd.to_numeric(frame["交易車位筆數"], errors="coerce"))
    matrix["每房建物面積"] = safe_ratio(
        building, pd.to_numeric(frame[LAYOUT_COLUMNS[0]], errors="coerce"))
    matrix["行政區_型態"] = frame["行政區"].astype(str) + "_" + frame["建物型態"].astype(str)
    matrix["道路_型態"] = frame["道路地段"].astype(str) + "_" + frame["建物型態"].astype(str)
    matrix["建案_棟別"] = frame["建案行政區"].astype(str) + "_" + frame["棟別"].astype(str)

    for col in matrix:
        if pd.api.types.is_numeric_dtype(matrix[col]):
            values = pd.to_numeric(matrix[col], errors="coerce").to_numpy(
                dtype=np.float32, na_value=np.nan, copy=True)
            values[~np.isfinite(values)] = np.nan
            matrix[col] = values
        else:
            matrix[col] = matrix[col].astype("string").fillna("缺失").astype(str)
    return matrix

class HistoricalComparables:
    def __init__(self, window=24, minimum_rows=5):
        self.window, self.minimum_rows = window, minimum_rows

    def keys(self, frame, level):
        parking = pd.to_numeric(frame["交易車位筆數"], errors="coerce")
        parking = parking.clip(upper=3).astype("Int64").astype("string").fillna("缺失")
        location = frame[{
            "Project": "建案行政區", "Road": "道路地段", "District": "行政區"
        }[level]].astype("string").fillna("缺失")
        key = (
            location + "｜" + frame["建物型態"].astype("string").fillna("缺失")
            + "｜" + frame["交易標的"].astype("string").fillna("缺失")
            + "｜" + parking)
        if level == "Project":
            key = key.where(frame["建案名稱"].ne("缺失"), "Unavailable")
        if level == "Road":
            key = key.where(
                ~frame["道路地段"].astype("string").str.endswith("_缺失", na=True),
                "Unavailable")
        return key.astype(str)

    def fit(self, frame):
        area = frame[AREA_COLUMNS[1]].to_numpy(dtype=float, na_value=np.nan)
        month = frame["交易時間序號"].to_numpy(dtype=float, na_value=np.nan)
        price = frame[TARGET].to_numpy(dtype=float)
        unit = np.divide(price, area, out=np.full(len(frame), np.nan), where=area > 0)
        housing = frame["交易標的"].astype("string").str.contains(
            "房地|建物", na=False).to_numpy(dtype=bool)
        keep = (
            housing & np.isfinite(month) & np.isfinite(unit) & (unit > 0)
            & (area > 0) & (area <= 600)
            & frame["特殊關係交易"].eq(0).to_numpy(dtype=bool)
            & ~frame[ID_COLUMN].duplicated(keep=False).to_numpy())
        self.cutoff = float(np.nanmax(month))
        self.tables = {}

        for level in ["Project", "Road", "District"]:
            source = pd.DataFrame({
                "Key": self.keys(frame, level).to_numpy(),
                "Month": month, "Log_Unit": np.log(np.where(unit > 0, unit, np.nan))
            }).loc[keep]
            source = source.loc[source["Key"].ne("Unavailable")]
            monthly = source.groupby(["Key", "Month"]).agg(
                Rows=("Log_Unit", "size"), Log_Sum=("Log_Unit", "sum")).reset_index()
            tables = {}
            for key, group in monthly.groupby("Key", sort=False):
                group = group.sort_values("Month")
                tables[key] = (
                    group["Month"].to_numpy(dtype=float),
                    np.r_[0, group["Rows"].to_numpy(dtype=float).cumsum()],
                    np.r_[0, group["Log_Sum"].to_numpy(dtype=float).cumsum()]
                )
            self.tables[level] = tables
        return self

    def transform(self, frame):
        frame = frame.reset_index(drop=True)
        area = frame[AREA_COLUMNS[1]].to_numpy(dtype=float, na_value=np.nan)
        month = frame["交易時間序號"].to_numpy(dtype=float, na_value=np.nan)
        cutoff = np.minimum(month - 1, self.cutoff)
        out = pd.DataFrame(index=frame.index)

        for level in ["Project", "Road", "District"]:
            unit = np.full(len(frame), np.nan)
            rows = np.zeros(len(frame))
            age = np.full(len(frame), np.nan)
            keys = self.keys(frame, level)

            for key, positions in keys.groupby(keys, sort=False).indices.items():
                table = self.tables[level].get(key)
                if table is None:
                    continue
                positions = np.asarray(positions)
                positions = positions[np.isfinite(cutoff[positions])]
                if not len(positions):
                    continue
                times, counts, log_sums = table
                upper = np.searchsorted(times, cutoff[positions], side="right")
                lower = np.searchsorted(
                    times, cutoff[positions] - self.window + 1, side="left")
                support = counts[upper] - counts[lower]
                available = (upper > lower) & (support >= self.minimum_rows)
                selected = positions[available]
                if not len(selected):
                    continue
                unit[selected] = np.exp(
                    (log_sums[upper[available]] - log_sums[lower[available]])
                    / support[available])
                rows[selected] = support[available]
                age[selected] = month[selected] - times[upper[available] - 1]

            usable = np.isfinite(area) & (area > 0)
            out[f"{level}_Reference_Unit"] = unit
            out[f"{level}_Reference_Price"] = np.where(usable, unit * area, np.nan)
            out[f"{level}_Reference_Rows"] = np.log1p(rows)
            out[f"{level}_Reference_Age"] = age
            out[f"{level}_Reference_Missing"] = np.isnan(unit).astype("int8")
        return out

def unit_area_mask(frame):
    area = pd.to_numeric(frame[AREA_COLUMNS[1]], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    housing = frame["交易標的"].astype("string").str.contains(
        "房地|建物", na=False).to_numpy(dtype=bool)
    mask = housing & np.isfinite(area) & (area > 0)
    return area, mask

class AreaMatchedHistory(HistoricalComparables):
    def __init__(self, minimum_rows=5):
        super().__init__(window=24, minimum_rows=minimum_rows)
        self.windows = [3, 6, 12, 24]
        self.area_edges = [0, 75, 100, 150, 200, 300, 600, np.inf]

    def keys(self, frame, level):
        key = super().keys(frame, level)
        if level == "Project":
            return key
        area = pd.to_numeric(frame[AREA_COLUMNS[1]], errors="coerce")
        band = pd.cut(area, self.area_edges, right=False).astype("string")
        available = key.ne("Unavailable") & band.notna() & area.gt(0)
        return (key + "｜" + band.fillna("缺失")).where(
            available, "Unavailable").astype(str)

    def transform(self, frame):
        frame = frame.reset_index(drop=True)
        area = frame[AREA_COLUMNS[1]].to_numpy(dtype=float, na_value=np.nan)
        month = frame["交易時間序號"].to_numpy(dtype=float, na_value=np.nan)
        cutoff = np.minimum(month - 1, self.cutoff)
        columns = {}

        for level in ["Road", "District"]:
            groups = self.keys(frame, level)
            positions_by_key = groups.groupby(groups, sort=False).indices
            values_by_window = {
                window: {
                    "Unit": np.full(len(frame), np.nan),
                    "Rows": np.zeros(len(frame)),
                    "Latest_Age": np.full(len(frame), np.nan),
                    "Mean_Age": np.full(len(frame), np.nan)
                }
                for window in self.windows
            }

            for key, positions in positions_by_key.items():
                table = self.tables[level].get(key)
                if table is None or key == "Unavailable":
                    continue
                positions = np.asarray(positions)
                positions = positions[np.isfinite(month[positions])]
                if not len(positions):
                    continue

                times, counts, log_sums = table
                month_sums = np.r_[0, np.cumsum(times * np.diff(counts))]
                upper = np.searchsorted(times, cutoff[positions], side="right")

                for window in self.windows:
                    lower = np.searchsorted(
                        times, month[positions] - window, side="left")
                    lower = np.minimum(lower, upper)
                    support = counts[upper] - counts[lower]
                    available = (upper > lower) & (support >= self.minimum_rows)
                    selected = positions[available]
                    if not len(selected):
                        continue

                    hi, lo = upper[available], lower[available]
                    n = support[available]
                    values = values_by_window[window]
                    values["Unit"][selected] = np.exp(
                        (log_sums[hi] - log_sums[lo]) / n)
                    values["Rows"][selected] = n
                    values["Latest_Age"][selected] = month[selected] - times[hi - 1]
                    values["Mean_Age"][selected] = (
                        month[selected] - (month_sums[hi] - month_sums[lo]) / n)
                    assert (values["Latest_Age"][selected] >= 1).all()
                    assert (values["Mean_Age"][selected] <= window + 1e-6).all()

            for window, values in values_by_window.items():
                prefix = f"Matched_{level}_{window}M"
                unit = values["Unit"]
                columns[f"{prefix}_Unit"] = unit
                columns[f"{prefix}_Price"] = np.where(
                    np.isfinite(area) & (area > 0), unit * area, np.nan)
                columns[f"{prefix}_Log_Rows"] = np.log1p(values["Rows"])
                columns[f"{prefix}_Latest_Age"] = values["Latest_Age"]
                columns[f"{prefix}_Mean_Age"] = values["Mean_Age"]
                columns[f"{prefix}_Missing"] = np.isnan(unit).astype(float)

            long_unit = values_by_window[24]["Unit"]
            for window in [3, 6, 12]:
                short_unit = values_by_window[window]["Unit"]
                ratio = np.divide(
                    short_unit, long_unit, out=np.full(len(frame), np.nan),
                    where=np.isfinite(long_unit) & (long_unit > 0))
                columns[f"Matched_{level}_{window}M_vs_24M"] = np.log(ratio)

        return pd.DataFrame(columns, index=frame.index).astype("float32")

def prepare_lgb_frames(training_matrix, other_matrices):
    original_columns = training_matrix.columns.tolist()
    feature_names = [f"F{i:03d}" for i in range(len(original_columns))]
    categorical = training_matrix.select_dtypes(
        include=["object", "string", "category"]).columns.tolist()
    categories = {
        col: training_matrix[col].astype("string").fillna("缺失").unique().tolist()
        for col in categorical
    }

    def convert(matrix):
        assert matrix.columns.tolist() == original_columns
        out = pd.DataFrame(index=np.arange(len(matrix)))
        for col, name in zip(original_columns, feature_names):
            if col in categories:
                out[name] = pd.Categorical(
                    matrix[col].astype("string").fillna("缺失").to_numpy(),
                    categories=categories[col])
            else:
                out[name] = pd.to_numeric(matrix[col], errors="coerce").to_numpy(
                    dtype=np.float32, na_value=np.nan, copy=True)
        return out

    return (
        convert(training_matrix), [convert(matrix) for matrix in other_matrices],
        {"columns": original_columns, "names": feature_names, "categories": categories}
    )

params_lgb = {
    "objective": "regression_l1", "metric": "l1",
    "n_estimators": 8000, "learning_rate": 0.03,
    "num_leaves": 63, "min_child_samples": 50,
    "reg_lambda": 10, "colsample_bytree": 0.9,
    "subsample": 0.8, "subsample_freq": 1,
    "cat_smooth": 30, "min_data_per_group": 50,
    "random_state": 42, "n_jobs": 8, "verbosity": -1
}


class TryPreprocessor:
    """Fit address-recovery vocabulary on Train, then transform each dataset."""

    def __init__(self, train_raw: pd.DataFrame):
        if TARGET not in train_raw:
            raise ValueError("Training data must include the total-price target")
        self.districts = sorted({
            str(normalize_text(value))
            for value in train_raw["鄉鎮市區"].dropna()
            if str(value) != "The villages and towns urban district"
        }, key=len, reverse=True)

    def transform(self, raw: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        return _preprocess_exact(raw, self.districts, training=training)


CATBOOST_PARAMETERS = {
    "loss_function": "MAE", "eval_metric": "MAE", "iterations": 2500,
    "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 8,
    "random_seed": 42, "thread_count": 8,
    "allow_writing_files": False, "verbose": 200,
}
LOCKED_RECIPE = {
    "history_trees": 2499, "unit_trees": 2469, "matched_trees": 2500,
    "lgbm_trees": 7949, "maximum_unit_weight": 1.0,
    "matched_weight": 0.1, "lgbm_maximum_weight": 0.5,
    "lgbm_half_life_months": 6.0,
}


def original_schema_fit(train: pd.DataFrame) -> pd.DataFrame:
    """Select the original last-four-month development split's fitting rows.

    This selects feature columns, not model hyperparameters. Full-data models
    used this frozen 59-column base schema in the recorded best submission.
    """
    months = np.sort(train["交易時間序號"].dropna().unique())
    if len(months) < 6:
        return train.reset_index(drop=True)
    dev_start = months[-4]
    excluded = train.loc[train["交易時間序號"].ge(dev_start), ID_COLUMN]
    return train.loc[
        train["交易時間序號"].lt(dev_start)
        & ~train[ID_COLUMN].isin(excluded)
    ].reset_index(drop=True)


def _matrix_digest(matrix: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(matrix.columns), ensure_ascii=False).encode())
    digest.update(json.dumps([str(dtype) for dtype in matrix.dtypes]).encode())
    digest.update(pd.util.hash_pandas_object(matrix, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _numeric_digest(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _verify_cat_schema(model, columns):
    # Some loaded CatBoost versions report n_features_in_ == 0; feature_names_
    # remains the meaningful ordered schema and is verified directly.
    if list(model.feature_names_) != list(columns):
        raise ValueError("Saved CatBoost feature names or order differ")


def _convert_lgb_matrix(matrix: pd.DataFrame, encoding: dict) -> pd.DataFrame:
    if list(matrix.columns) != encoding["columns"]:
        raise ValueError("LightGBM input feature names or order differ")
    result = pd.DataFrame(index=np.arange(len(matrix)))
    for original, encoded in zip(encoding["columns"], encoding["names"]):
        if original in encoding["categories"]:
            result[encoded] = pd.Categorical(
                matrix[original].astype("string").fillna("缺失").to_numpy(),
                categories=encoding["categories"][original],
            )
        else:
            result[encoded] = pd.to_numeric(matrix[original], errors="coerce").to_numpy(
                dtype=np.float32, na_value=np.nan, copy=True)
    return result


class TrySystem:
    """Locked four-model Try branch, independent of notebook global variables.

    fit() accepts preprocessed Train only. Cache identities include actual
    matrix, target, weight and parameter digests; a reporting-only code change
    does not trigger another fit. No validation labels select the recipe.
    """

    def __init__(self, thread_count: int = 8):
        self.thread_count = int(thread_count)
        self.recipe = dict(LOCKED_RECIPE)
        self.models = {}
        self.encodings = None

    def fit(self, train: pd.DataFrame, cache_dir=None,
            schema_fit: pd.DataFrame | None = None):
        from catboost import CatBoostRegressor
        import lightgbm as lgb
        import catboost

        frame = train.reset_index(drop=True).copy()
        if TARGET not in frame or not frame[TARGET].gt(0).all():
            raise ValueError("Fit requires positive Train targets")
        schema_frame = frame if schema_fit is None else schema_fit.reset_index(drop=True)
        schema_matrix = main_feature_frame(schema_frame)
        self.base_columns = [
            column for column in schema_matrix
            if schema_matrix[column].nunique(dropna=False) > 1
        ]
        self.history = HistoricalComparables(window=24, minimum_rows=5).fit(frame)
        self.matched_history = AreaMatchedHistory(minimum_rows=5).fit(frame)
        base = main_feature_frame(frame)[self.base_columns]
        references = self.history.transform(frame)
        matrix = pd.concat([base.reset_index(drop=True), references], axis=1)
        matched = pd.concat([matrix, self.matched_history.transform(frame)], axis=1)
        self.history_columns = list(matrix.columns)
        self.matched_columns = list(matched.columns)
        self.categorical = matrix.select_dtypes(
            include=["object", "string", "category"]).columns.tolist()
        self.known_projects = set(frame.loc[
            frame["建案名稱"].ne("缺失"), "建案行政區"])
        self.cutoff = float(self.history.cutoff)
        self.fit_rows = len(frame)
        self.fit_signature = _matrix_digest(frame)
        actual = frame[TARGET].to_numpy(dtype=float)
        area, unit_valid = unit_area_mask(frame)
        if not unit_valid.any():
            raise ValueError("No positive housing building areas are available")
        self.unit_training_mean_area = float(area[unit_valid].mean())
        root = Path(cache_dir) if cache_dir is not None else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

        def cat_fit(name, values, target, trees, weights=None):
            parameters = dict(CATBOOST_PARAMETERS)
            parameters.update(iterations=int(trees), thread_count=self.thread_count)
            identity = {
                "backend": "CatBoost", "version": catboost.__version__,
                "matrix": _matrix_digest(values), "target": _numeric_digest(target),
                "weights": None if weights is None else _numeric_digest(weights),
                "parameters": parameters, "columns": list(values.columns),
            }
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            path = None if root is None else root / f"{name}_{key[:16]}.cbm"
            model = CatBoostRegressor(**parameters)
            if path is not None and path.is_file():
                metadata = json.loads(path.with_suffix(".json").read_text())
                if metadata["identity"] != identity:
                    raise ValueError(f"Cache identity mismatch: {name}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                    raise ValueError(f"Cache checksum mismatch: {name}")
                model.load_model(str(path))
                _verify_cat_schema(model, values.columns)
                print(f"Loaded: Try {name} | Trees: {trees}", flush=True)
            else:
                print(f"Training: Try {name} | Rows: {len(target)} | Trees: {trees}", flush=True)
                categorical = values.select_dtypes(
                    include=["object", "string", "category"]).columns.tolist()
                model.fit(values, target, cat_features=categorical, sample_weight=weights)
                if path is not None:
                    model.save_model(str(path))
                    path.with_suffix(".json").write_text(json.dumps({
                        "identity": identity,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
            return model

        self.models["history"] = cat_fit(
            "history", matrix, actual, self.recipe["history_trees"])
        self.models["unit"] = cat_fit(
            "unit", matrix.loc[unit_valid], actual[unit_valid] / area[unit_valid],
            self.recipe["unit_trees"], area[unit_valid] / self.unit_training_mean_area)
        self.models["matched"] = cat_fit(
            "matched", matched, actual, self.recipe["matched_trees"])

        lgb_matrix, _, self.encodings = prepare_lgb_frames(matrix, [])
        parameters = dict(params_lgb)
        parameters.update(n_estimators=self.recipe["lgbm_trees"], n_jobs=self.thread_count)
        identity = {
            "backend": "LightGBM", "version": lgb.__version__,
            "matrix": _matrix_digest(matrix), "target": _numeric_digest(actual),
            "parameters": parameters, "encoding": self.encodings,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = None if root is None else root / f"lgbm_{key[:16]}.txt"
        if path is not None and path.is_file():
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata["identity"] != identity:
                raise ValueError("LightGBM cache identity mismatch")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("LightGBM cache checksum mismatch")
            self.models["lgbm"] = lgb.Booster(model_file=str(path))
            print("Loaded: Try LightGBM | Trees: 7949", flush=True)
        else:
            print(f"Training: Try LightGBM | Rows: {len(frame)} | Trees: 7949", flush=True)
            model = lgb.LGBMRegressor(**parameters)
            model.fit(lgb_matrix, actual)
            self.models["lgbm"] = model.booster_
            if path is not None:
                self.models["lgbm"].save_model(str(path))
                path.with_suffix(".json").write_text(json.dumps({
                    "identity": identity,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)
        matrix = pd.concat([
            main_feature_frame(features)[self.base_columns],
            self.history.transform(features),
        ], axis=1)
        if list(matrix.columns) != self.history_columns:
            raise ValueError("Try feature names or order differ")
        return matrix

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.drop(columns=[TARGET], errors="ignore").reset_index(drop=True)
        matrix = self.feature_frame(features)
        history = np.maximum(self.models["history"].predict(matrix), 0)
        area, valid_area = unit_area_mask(features)
        large = valid_area & (area > 300)
        ramp = np.zeros(len(features))
        ramp[large] = self.recipe["maximum_unit_weight"] * np.clip(
            np.log(area[large] / 300) / np.log(2), 0, 1)
        history_unit = history.copy()
        unit_total = history.copy()
        if large.any():
            unit_total[large] = np.maximum(
                self.models["unit"].predict(matrix.loc[large]), 0) * area[large]
            history_unit[large] += ramp[large] * (unit_total[large] - history[large])
        ordinary = valid_area & (area <= 300)
        matched_active = (
            ordinary & features["建案名稱"].ne("缺失").to_numpy()
            & ~features["建案行政區"].isin(self.known_projects).to_numpy())
        try_v01 = history_unit.copy()
        raw_matched = np.full(len(features), np.nan)
        if matched_active.any():
            selected = features.loc[matched_active].reset_index(drop=True)
            matched_matrix = pd.concat([
                matrix.loc[matched_active].reset_index(drop=True),
                self.matched_history.transform(selected),
            ], axis=1)
            if list(matched_matrix.columns) != self.matched_columns:
                raise ValueError("Matched feature names or order differ")
            raw_matched[matched_active] = np.maximum(
                self.models["matched"].predict(matched_matrix), 0)
            try_v01[matched_active] += self.recipe["matched_weight"] * (
                raw_matched[matched_active] - try_v01[matched_active])
        raw_lgbm = np.maximum(self.models["lgbm"].predict(
            _convert_lgb_matrix(matrix, self.encodings)), 0)
        horizon = pd.to_numeric(features["交易時間序號"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan) - self.cutoff
        lgb_active = ordinary & np.isfinite(horizon)
        lgb_weight = np.zeros(len(features))
        lgb_weight[lgb_active] = self.recipe["lgbm_maximum_weight"] * np.exp2(
            -np.maximum(horizon[lgb_active] - 1, 0) / self.recipe["lgbm_half_life_months"])
        try_v02 = try_v01.copy()
        try_v02[lgb_active] += lgb_weight[lgb_active] * (
            raw_lgbm[lgb_active] - try_v01[lgb_active])
        reference_cols = [f"{level}_Reference_Price" for level in ["Project", "Road", "District"]]
        missing_reference = matrix[reference_cols].isna().all(axis=1).to_numpy()
        result = features[[ID_COLUMN]].copy()
        result["Area"] = area
        result["Ordinary"] = ordinary
        result["History"] = history
        result["History_Unit"] = history_unit
        result["Unit_Weight"] = ramp
        result["Matched_Raw"] = raw_matched
        result["Matched_Active"] = matched_active
        result["Try_V01"] = try_v01
        result["LightGBM_Raw"] = raw_lgbm
        result["LightGBM_Weight"] = lgb_weight
        result["Horizon"] = horizon
        result["Try_V02"] = try_v02
        result["Reference_Missing"] = missing_reference
        for column in reference_cols:
            result[column] = matrix[column].to_numpy()
        if not np.isfinite(result[["History", "History_Unit", "Try_V01", "Try_V02"]]).all().all():
            raise ValueError("Non-finite Try predictions")
        np.testing.assert_array_equal(try_v01[large], history_unit[large])
        np.testing.assert_array_equal(try_v02[~lgb_active], try_v01[~lgb_active])
        return result


def combine_legacy(legacy_prediction, try_result: pd.DataFrame) -> pd.DataFrame:
    """Exact Try V03 50/50 blend, then locked missing-reference fallback."""
    legacy = np.asarray(legacy_prediction, dtype=float)
    if legacy.shape != (len(try_result),) or not np.isfinite(legacy).all():
        raise ValueError("Legacy predictions are missing or misaligned")
    fallback = (
        try_result["Ordinary"].to_numpy(dtype=bool)
        & try_result["Reference_Missing"].to_numpy(dtype=bool))
    blended = 0.5 * legacy + 0.5 * try_result["Try_V02"].to_numpy(dtype=float)
    prediction = blended.copy()
    prediction[fallback] = legacy[fallback]
    result = try_result[[ID_COLUMN]].copy()
    result["Legacy"] = legacy
    result["Try_V03"] = blended
    result["Legacy_Fallback"] = fallback
    result["Try_V04"] = prediction
    np.testing.assert_array_equal(prediction[~fallback], blended[~fallback])
    return result

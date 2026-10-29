"""Strictly earlier-month, dictionary-serializable references for V9.

This module performs reference aggregation only.  It contains no ML models,
file reads, Test labels, or correction fitting.  Monthly medians are averaged
with the number of eligible original rows in each month as weights; this is
not the pooled transaction median.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd


MONTH = "交易時間序號"
TARGET = "總價元"


def _numbers(values, length=None):
    if np.isscalar(values) or values is None:
        if length is None:
            raise ValueError("A length is required for scalar values")
        return np.full(length, np.nan if values is None else float(values), dtype=float)
    result = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype=float, na_value=np.nan, copy=True)
    if length is not None and len(result) != length:
        raise ValueError("Reference input lengths differ")
    return result


def _months(values, length=None):
    result = _numbers(values, length)
    finite = np.isfinite(result)
    if np.any(result[finite] != np.floor(result[finite])):
        raise ValueError("Month indices must be integer-valued")
    return result


def _key(value):
    if isinstance(value, (tuple, list)):
        parts = [_key(part) for part in value]
        return None if any(part is None for part in parts) else "tuple:" + json.dumps(parts, ensure_ascii=False)
    if value is None or bool(pd.isna(value)):
        return None
    result = str(value)
    return result if result else None


def _keys(values, length=None):
    result = [_key(value) for value in values]
    if length is not None and len(result) != length:
        raise ValueError("Reference key lengths differ")
    return result


def _empty(length):
    return {"value": np.full(length, np.nan), "count": np.zeros(length, dtype=np.int64),
            "age": np.full(length, np.nan), "last_month": np.full(length, np.nan)}


def _effective_cutoff(query_month, max_fit_month, delay=None, cutoff=None):
    month = _months(query_month)
    lag = _numbers(1.0 if delay is None else delay, len(month))
    if np.any(np.isfinite(lag) & ((lag < 1) | (lag != np.floor(lag)))):
        raise ValueError("Forecast delays must be integer months >= 1")
    result = month - lag
    if cutoff is not None:
        result = np.minimum(result, _months(cutoff, len(month)))
    if max_fit_month is None or not np.isfinite(max_fit_month):
        result[:] = np.nan
    else:
        result = np.minimum(result, float(max_fit_month))
    return month, result


class TemporalHistory:
    """Monthly median references with O(log months) interval queries per key.

    ``fit(frame, values, levels)`` accepts an insertion-ordered mapping of
    level names to positional key sequences. ``query`` accepts the same level
    names with new positional keys, cutoff months, a trailing window length,
    and optionally the actual query months used to calculate reference age.
    Only source months <= cutoff and >= cutoff-window+1 are used.
    """

    def __init__(self, month_column=MONTH):
        self.month_column = month_column
        self.max_fit_month = None
        self.level_names = []
        self.levels = {}

    @staticmethod
    def _group(months, medians, counts):
        months = np.asarray(months, dtype=np.int64)
        medians = np.asarray(medians, dtype=float)
        counts = np.asarray(counts, dtype=np.int64)
        if (len(months) != len(medians) or len(months) != len(counts)
                or np.any(np.diff(months) <= 0) or np.any(counts <= 0)
                or not np.isfinite(medians).all()):
            raise ValueError("Invalid serialized monthly reference group")
        return {"months": months, "medians": medians, "counts": counts,
                "sum_value": np.r_[0.0, np.cumsum(medians * counts, dtype=float)],
                "sum_count": np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]}

    def fit(self, frame, values, levels):
        if not isinstance(levels, Mapping) or not levels:
            raise ValueError("An ordered nonempty level-to-keys mapping is required")
        n = len(frame)
        months = _months(frame[self.month_column], n)
        numeric = _numbers(values, n)
        finite_month = np.isfinite(months)
        self.max_fit_month = float(months[finite_month].max()) if finite_month.any() else None
        self.level_names = list(levels)
        self.levels = {}
        valid_value = finite_month & np.isfinite(numeric)
        for name, values_for_keys in levels.items():
            keys = _keys(values_for_keys, n)
            eligible = valid_value & np.fromiter((key is not None for key in keys), bool, count=n)
            data = pd.DataFrame({"key": pd.Series(keys, dtype=object),
                                 "month": months, "value": numeric}).loc[eligible]
            monthly = data.groupby(["key", "month"], sort=True, observed=True)["value"].agg(["median", "size"]).reset_index()
            groups = {}
            for key, group in monthly.groupby("key", sort=False, observed=True):
                groups[key] = self._group(group["month"], group["median"], group["size"])
            self.levels[name] = groups
        return self

    def query_level(self, level, keys, datecutoff, window, query_month=None, min_rows=5):
        if level not in self.levels:
            raise ValueError(f"Unknown reference level: {level}")
        if int(window) != window or window < 1 or int(min_rows) != min_rows or min_rows < 1:
            raise ValueError("Window and min_rows must be positive integers")
        keys = _keys(keys)
        n = len(keys)
        cutoff = _months(datecutoff, n)
        month = cutoff + 1 if query_month is None else _months(query_month, n)
        finite = np.isfinite(cutoff) & np.isfinite(month)
        if np.any(finite & (cutoff >= month)):
            raise ValueError("Own-month/future labels are prohibited: cutoff must precede query month")
        out = _empty(n)
        positions = {}
        for i, key in enumerate(keys):
            if finite[i] and key is not None:
                positions.setdefault(key, []).append(i)
        for key, indices in positions.items():
            group = self.levels[level].get(key)
            if group is None:
                continue
            indices = np.asarray(indices, dtype=np.int64)
            hi = np.searchsorted(group["months"], cutoff[indices], side="right")
            lo = np.searchsorted(group["months"], cutoff[indices] - int(window) + 1, side="left")
            count = group["sum_count"][hi] - group["sum_count"][lo]
            out["count"][indices] = count
            present = count > 0
            if present.any():
                last = group["months"][hi[present] - 1]
                out["last_month"][indices[present]] = last
                out["age"][indices[present]] = month[indices[present]] - last
            supported = count >= int(min_rows)
            if supported.any():
                numerator = group["sum_value"][hi[supported]] - group["sum_value"][lo[supported]]
                out["value"][indices[supported]] = numerator / count[supported]
        return out

    def query(self, keys, datecutoff, window, query_month=None, min_rows=5):
        if not isinstance(keys, Mapping):
            raise ValueError("Query keys must map level names to key sequences")
        return {name: self.query_level(name, values, datecutoff, window, query_month, min_rows)
                for name, values in keys.items()}

    def to_state(self):
        return {"schema": "TemporalHistory.v1", "month_column": self.month_column,
                "max_fit_month": self.max_fit_month, "level_names": self.level_names.copy(),
                "levels": {level: {key: {"months": value["months"].tolist(),
                    "medians": value["medians"].tolist(), "counts": value["counts"].tolist()}
                    for key, value in groups.items()} for level, groups in self.levels.items()}}

    @classmethod
    def from_state(cls, state):
        if state.get("schema") != "TemporalHistory.v1":
            raise ValueError("Unknown temporal reference schema")
        result = cls(state["month_column"])
        result.max_fit_month = state["max_fit_month"]
        result.level_names = list(state["level_names"])
        if set(result.level_names) != set(state["levels"]):
            raise ValueError("Serialized reference levels disagree")
        result.levels = {level: {key: cls._group(value["months"], value["medians"], value["counts"])
            for key, value in state["levels"][level].items()} for level in result.level_names}
        return result


class PriceReferenceHierarchy:
    """Project/road/district references; choose the first supported 12M level.

    The caller supplies keys, including property type/use/area bands where
    needed. Missing project identities should be passed as None, not pooled
    into one synthetic project. Selected 3M/6M references retain the SAME level
    chosen at 12M, so price changes do not switch geographic populations.
    """

    def __init__(self, min_rows=5, windows=(3, 6, 12), month_column=MONTH):
        self.min_rows = int(min_rows)
        self.windows = tuple(int(window) for window in windows)
        if self.min_rows < 1 or 12 not in self.windows or any(window < 1 for window in self.windows):
            raise ValueError("A positive min_rows and a 12M selection window are required")
        self.history = TemporalHistory(month_column)

    def fit(self, frame, values, keys):
        numeric = _numbers(values, len(frame))
        self.history.fit(frame, np.where(numeric > 0, numeric, np.nan), keys)
        return self

    def query(self, keys, query_month, delay=None, cutoff=None):
        month, effective = _effective_cutoff(query_month, self.history.max_fit_month, delay, cutoff)
        n = len(month)
        if list(keys) != self.history.level_names:
            raise ValueError("Query hierarchy levels/order differ from fitting")
        by_window = {window: self.history.query(keys, effective, window, month, self.min_rows)
                     for window in self.windows}
        result = {"Reference_Value": np.full(n, np.nan), "Reference_Count": np.zeros(n, dtype=np.int64),
                  "Reference_Age": np.full(n, np.nan), "Reference_Source": np.full(n, "Missing", dtype=object),
                  "Effective_Cutoff": effective, "Forecast_Gap": month - effective}
        for window in self.windows:
            result[f"Reference_{window}M"] = np.full(n, np.nan)
            result[f"Reference_{window}M_Count"] = np.zeros(n, dtype=np.int64)
            result[f"Reference_{window}M_Age"] = np.full(n, np.nan)
        for level in self.history.level_names:
            selected = (result["Reference_Source"] == "Missing") & np.isfinite(by_window[12][level]["value"])
            for window in self.windows:
                block = by_window[window][level]
                for field, suffix in [("value", "Value"), ("count", "Count"), ("age", "Age")]:
                    result[f"{level}_{window}M_{suffix}"] = block[field].copy()
                result[f"Reference_{window}M"][selected] = block["value"][selected]
                result[f"Reference_{window}M_Count"][selected] = block["count"][selected]
                result[f"Reference_{window}M_Age"][selected] = block["age"][selected]
            result["Reference_Value"][selected] = by_window[12][level]["value"][selected]
            result["Reference_Count"][selected] = by_window[12][level]["count"][selected]
            result["Reference_Age"][selected] = by_window[12][level]["age"][selected]
            result["Reference_Source"][selected] = level
        return result

    def to_state(self):
        return {"schema": "PriceReferenceHierarchy.v1", "min_rows": self.min_rows,
                "windows": list(self.windows), "history": self.history.to_state()}

    @classmethod
    def from_state(cls, state):
        if state.get("schema") != "PriceReferenceHierarchy.v1":
            raise ValueError("Unknown price hierarchy schema")
        result = cls(state["min_rows"], state["windows"], state["history"]["month_column"])
        result.history = TemporalHistory.from_state(state["history"])
        return result


def _parking_count(frame):
    if "交易車位筆數" in frame:
        return _numbers(frame["交易車位筆數"], len(frame))
    if "交易筆棟數" not in frame:
        raise ValueError("Parking counts or the original transaction-count string are required")
    parsed = frame["交易筆棟數"].astype("string").str.extract(r"^土地\d+建物\d+車位(\d+)$", expand=False)
    return _numbers(parsed, len(frame))


def _parking_keys(frame):
    n = len(frame)
    if "縣市" in frame:
        city = frame["縣市"].astype("string")
    elif "土地位置建物門牌" in frame:
        city = frame["土地位置建物門牌"].astype("string").str.extract(r"^(.{2,3}[縣市])", expand=False)
    else:
        city = pd.Series([None] * n, dtype="string")
    city = city.str.replace("台", "臺", regex=False)
    if "行政區" in frame:
        district = frame["行政區"].astype("string")
    elif "鄉鎮市區" in frame:
        district = city + "_" + frame["鄉鎮市區"].astype("string")
    else:
        district = pd.Series([None] * n, dtype="string")
    def clean(values):
        return [None if pd.isna(value) or str(value) in {"", "缺失", "Unknown", "nan", "<NA>"}
                or "缺失" in str(value) else str(value) for value in values]
    return {"District": clean(district), "City": clean(city), "Global": ["Global"] * n}


class ParkingHelper:
    """Per-bay prices from earlier pure-parking sales, district/city/global.

    Missing or invalid counts remain unavailable. Count==0 needs no estimate:
    total price and per-bay reference are zero and Parking_Available is True.
    Positive counts require a supported earlier reference; no silent zero fill.
    """

    def __init__(self, window=24, min_rows=5, month_column=MONTH):
        self.window, self.min_rows = int(window), int(min_rows)
        if self.window < 1 or self.min_rows < 1:
            raise ValueError("Parking window and min_rows must be positive")
        self.history = TemporalHistory(month_column)

    def fit(self, frame, target=None):
        counts = _parking_count(frame)
        total = _numbers(frame[TARGET] if target is None else target, len(frame))
        pure = frame["交易標的"].astype("string").str.strip().eq("車位").fillna(False).to_numpy(bool)
        eligible = pure & np.isfinite(counts) & (counts > 0) & (counts == np.floor(counts)) & np.isfinite(total) & (total > 0)
        values = np.full(len(frame), np.nan)
        values[eligible] = total[eligible] / counts[eligible]
        self.history.fit(frame, values, _parking_keys(frame))
        return self

    def query(self, frame, delay=None, cutoff=None):
        counts = _parking_count(frame)
        month, effective = _effective_cutoff(frame[self.history.month_column], self.history.max_fit_month, delay, cutoff)
        n = len(frame)
        blocks = self.history.query(_parking_keys(frame), effective, self.window, month, self.min_rows)
        count_valid = np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))
        no_parking = count_valid & (counts == 0)
        required = count_valid & (counts > 0)
        per_bay = np.full(n, np.nan)
        support = np.zeros(n, dtype=np.int64)
        age = np.full(n, np.nan)
        source = np.full(n, "Missing", dtype=object)
        per_bay[no_parking], source[no_parking] = 0.0, "Not_Required"
        for level in self.history.level_names:
            block = blocks[level]
            choose = required & ~np.isfinite(per_bay) & np.isfinite(block["value"]) & (block["value"] > 0)
            per_bay[choose] = block["value"][choose]
            support[choose], age[choose], source[choose] = block["count"][choose], block["age"][choose], level
        available = no_parking | (required & np.isfinite(per_bay))
        total = np.full(n, np.nan)
        total[no_parking] = 0.0
        total[required & available] = counts[required & available] * per_bay[required & available]
        return {"Parking_Count": counts, "Parking_Count_Valid": count_valid,
                "Parking_Price_Per_Bay": per_bay, "Parking_Price": total,
                "Parking_Required": required, "Parking_Available": available,
                "Parking_Estimated": required & available, "Parking_Reference_Count": support,
                "Parking_Reference_Age": age, "Parking_Reference_Source": source,
                "Parking_Effective_Cutoff": effective}

    transform = query

    def to_state(self):
        return {"schema": "ParkingHelper.v1", "window": self.window, "min_rows": self.min_rows,
                "history": self.history.to_state()}

    @classmethod
    def from_state(cls, state):
        if state.get("schema") != "ParkingHelper.v1":
            raise ValueError("Unknown parking helper schema")
        result = cls(state["window"], state["min_rows"], state["history"]["month_column"])
        result.history = TemporalHistory.from_state(state["history"])
        return result


def synthetic_self_test():
    """Synthetic aggregation/leakage/serialization tests; no real model fit."""
    frame = pd.DataFrame({MONTH: [1, 1, 1, 2, 3, 4]})
    keys = {"Project": ["P"] * len(frame), "Road": ["R"] * len(frame), "District": ["D"] * len(frame)}
    values = np.array([1., 1., 1000., 30., 999., 777.])
    history = TemporalHistory().fit(frame, values, keys)
    query_keys = {name: [key[0]] for name, key in keys.items()}
    block = history.query(query_keys, [2], window=2, query_month=[3], min_rows=1)["Project"]
    np.testing.assert_allclose(block["value"], [8.25])
    np.testing.assert_array_equal(block["count"], [4])
    np.testing.assert_allclose(block["age"], [1])
    other = TemporalHistory().fit(frame, np.array([1., 1., 1000., 30., -999999., 999999.]), keys)
    np.testing.assert_array_equal(other.query(query_keys, [2], 2, [3], 1)["Project"]["value"], block["value"])
    restored = TemporalHistory.from_state(json.loads(json.dumps(history.to_state())))
    np.testing.assert_array_equal(restored.query(query_keys, [2], 2, [3], 1)["Project"]["value"], block["value"])
    hierarchy = PriceReferenceHierarchy(min_rows=3).fit(frame, values, keys)
    out = hierarchy.query(query_keys, [3], delay=[2])
    np.testing.assert_allclose(out["Reference_Value"], [1.])
    np.testing.assert_allclose(out["Effective_Cutoff"], [1.])
    assert out["Reference_Source"].tolist() == ["Project"]
    parking_frame = pd.DataFrame({MONTH: [1, 1, 2, 3], "交易標的": ["車位"] * 4,
        "交易車位筆數": [1, 2, 1, 1], TARGET: [100., 220., 120., 999999.],
        "行政區": ["臺北市_中山區"] * 4, "縣市": ["臺北市"] * 4})
    parking = ParkingHelper(window=12, min_rows=2).fit(parking_frame)
    query = parking_frame.iloc[[2, 2, 0]].drop(columns=[TARGET]).reset_index(drop=True)
    query[MONTH], query["交易車位筆數"] = [3, 3, 1], [1, 0, 1]
    out = parking.query(query)
    np.testing.assert_allclose(out["Parking_Price"][:2], [110., 0.])
    np.testing.assert_array_equal(out["Parking_Available"], [True, True, False])
    assert np.isnan(out["Parking_Price"][2])
    restored = ParkingHelper.from_state(json.loads(json.dumps(parking.to_state())))
    np.testing.assert_allclose(restored.query(query)["Parking_Price"], out["Parking_Price"], equal_nan=True)
    try:
        history.query(query_keys, [3], 12, [3], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Own-month cutoff was not rejected")
    return {"monthly_median_row_weighting": True, "future_label_invariance": True,
            "deterministic_delay": True, "zero_parking_eligible": True,
            "unsupported_parking_not_zero_filled": True, "json_state_roundtrip": True,
            "own_month_labels_rejected": True}


if __name__ == "__main__":
    print(json.dumps(synthetic_self_test(), indent=2))

"""Self-contained fixed ensemble used by the accepted current-best blend.

No code executes training at import. All data and runtime configuration are explicit.
Training is started only by LegacySystem.fit(), on user-supplied Train data.
"""
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TARGET = "總價元"
ID_COLUMN = "編號"
MAX_YEAR = 2026
OOF_FOLDS = 5


CITY_REPLACE = {"台北市": "臺北市", "台中市": "臺中市", "台南市": "臺南市", "桃園縣": "桃園市"}

CITY_FILL = {"清水區": "臺中市", "員山鄉": "宜蘭縣", "左營區": "高雄市", "梅山鄉": "嘉義縣"}

BASE_CATEGORICAL = ["鄉鎮市區", "交易標的", "都市土地使用分區", "非都市土地使用分區", "非都市土地使用編定", "建物型態", "主要用途", "主要建材", "建物現況格局-隔間", "車位類別", "土地使用類型", "縣市", "行政區"]

AREA_COLUMNS = ["土地移轉總面積平方公尺", "建物移轉總面積平方公尺", "車位移轉總面積平方公尺"]

LAYOUT_COLUMNS = ["建物現況格局-房", "建物現況格局-廳", "建物現況格局-衛"]

BASE_NUMERIC = AREA_COLUMNS + LAYOUT_COLUMNS + ["總樓層數", "交易年", "交易月", "交易土地筆數", "交易建物筆數", "交易車位筆數", "移轉樓層", "相對樓層"]

BASE_FLAGS = ["整棟交易", "多樓層交易", "有備註", "特殊關係交易", "含裝潢家具家電資訊", "毛胚屋", "雨遮不計價"]

def normalize_text(value):
    if pd.isna(value):
        return pd.NA
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return value if value and value.lower() not in {"nan", "none", "null", "<na>"} else pd.NA

def roc_dates(series, minimum=90):
    text = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    allowed = text.str.fullmatch(r"\d{6,7}", na=False)
    text = text.where(allowed).str.zfill(7)
    year = pd.to_numeric(text.str[:3], errors="coerce")
    gregorian = (year + 1911).astype("Int64").astype("string") + text.str[3:]
    dates = pd.to_datetime(gregorian, format="%Y%m%d", errors="coerce")
    return dates.where(year.between(minimum, MAX_YEAR - 1911))

def chinese_integer(text):
    digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
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
    text = str(value).strip()
    match = re.fullmatch(r"(地下|地上)?([零一二三四五六七八九十\d]+)(?:層|樓)?", text)
    if not match:
        return np.nan
    result = chinese_integer(match.group(2))
    return -result if match.group(1) == "地下" else result

def safe_ratio(numerator, denominator):
    return numerator.div(denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)

def recover_district(address, districts):
    if pd.isna(address):
        return pd.NA
    rest = re.sub(r"^.{2,3}[縣市]", "", str(address))
    return next((district for district in districts if rest.startswith(district)), pd.NA)

def preprocess(raw, training=False, districts=None):
    if districts is None:
        raise ValueError("Fit DatasetPreprocessor on the raw training data first.")
    df = raw.copy()
    if TARGET in df:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.loc[df[TARGET].notna() & (df[TARGET] > 0)].copy()
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
    date = roc_dates(df["交易年月日"])
    df["交易年"] = date.dt.year
    df["交易月"] = date.dt.month
    df["交易時間序號"] = (df["交易年"] - 2008) * 12 + df["交易月"] - 1
    parsed = df["交易筆棟數"].str.extract(r"^土地(\d+)建物(\d+)車位(\d+)$")
    for index, col in enumerate(["交易土地筆數", "交易建物筆數", "交易車位筆數"]):
        df[col] = pd.to_numeric(parsed[index], errors="coerce")
    df["移轉樓層"] = df["移轉層次"].map(single_floor)
    floor = df["移轉層次"]
    df["整棟交易"] = (floor.isin(["全", "全部", "全棟", "整棟", "全.透天厝"]) | floor.str.contains("整棟", na=False)).astype("int8")
    df["多樓層交易"] = (df["移轉樓層"].isna() & (df["整棟交易"] == 0) & floor.str.contains(r"[+＋\-~～、，,＆&及與至到]|樓中樓", na=False)).astype("int8")
    df.loc[floor.str.contains("申報事項|加購", na=False), "多樓層交易"] = 0
    df.loc[df["總樓層數"] <= 0, "總樓層數"] = np.nan
    valid_floor = (df["移轉樓層"] > 0) & (df["移轉樓層"] <= df["總樓層數"])
    df["相對樓層"] = safe_ratio(df["移轉樓層"], df["總樓層數"]).where(valid_floor)
    conflict = (df[AREA_COLUMNS[1]] == 0) & (df["交易建物筆數"] > 0) & (df["交易標的"] != "車位")
    df.loc[conflict, AREA_COLUMNS[1]] = np.nan
    if training:
        df.loc[df[AREA_COLUMNS[0]].isin([1044217.0, 27884.90]), AREA_COLUMNS[0]] = np.nan
        df = df.loc[~df[TARGET].isin([520, 1760])].copy()
    for col in ["主要用途", "主要建材"]:
        df[col] = df[col].replace({"見其它登記事項": "見其他登記事項"})
    urban = df["都市土地使用分區"].notna()
    rural = df["非都市土地使用分區"].notna()
    df["土地使用類型"] = np.select([urban & ~rural, ~urban & rural, urban & rural], ["都市", "非都市", "混合"], default="其他或未標示")
    address = df["土地位置建物門牌"].str.replace("台", "臺", regex=False)
    df["縣市"] = address.str.extract(r"^(.{2,3}[縣市])", expand=False).replace(CITY_REPLACE)
    missing = df["鄉鎮市區"].isna()
    df.loc[missing, "鄉鎮市區"] = address.loc[missing].map(lambda value: recover_district(value, districts))
    df["縣市"] = df["縣市"].fillna(df["鄉鎮市區"].map(CITY_FILL))
    df["行政區"] = df["縣市"].fillna("缺失") + "_" + df["鄉鎮市區"].fillna("缺失")
    remark = df["備註"]
    df["有備註"] = remark.notna().astype("int8")
    patterns = {"特殊關係交易": "親友|員工|共有人|特殊關係", "含裝潢家具家電資訊": "裝潢|傢俱|家具|家電|電器", "毛胚屋": "毛胚|毛坯", "雨遮不計價": "雨遮.*不計價|雨遮不計價", "持分交易": "持分|部分移轉|部分權利", "含增建": "增建|未登記|違建", "預售資訊": "預售|尚未完工|興建中"}
    for col, pattern in patterns.items():
        df[col] = remark.str.contains(pattern, regex=True, na=False).astype("int8")
    rest = address.str.replace(r"^.{2,3}[縣市]", "", regex=True)
    road = rest.str.extract(r"((?:[^\d,，;；]{1,30}?)(?:大道|路|街)(?:[一二三四五六七八九十\d]+段)?)", expand=False)
    parcel = rest.str.extract(r"([^\d,，;；]{1,30}?段)(?:\d|之|小段)", expand=False)
    df["道路地段"] = df["行政區"] + "_" + road.fillna(parcel).fillna("缺失")
    df["地址基地"] = address.str.replace(r"(?:地下|地上)?[零一二三四五六七八九十\d]+[樓層].*$", "", regex=True).fillna("缺失")
    df["建案行政區"] = df["行政區"] + "_" + df["建案名稱"].fillna("缺失")
    df["棟別"] = df["棟及號"].str.extract(r"^([A-Za-z]+\d{0,2})", expand=False).fillna("缺失")
    for col in BASE_CATEGORICAL + ["建案名稱", "道路地段", "地址基地", "建案行政區", "棟別"]:
        df[col] = df[col].fillna("缺失").astype(str)
    return df.reset_index(drop=True)

class DatasetPreprocessor:
    """Keep the district dictionary explicit and reusable across data splits."""
    def fit(self, raw):
        self.districts = sorted({
            str(normalize_text(value)) for value in raw["鄉鎮市區"].dropna()
            if str(value) != "The villages and towns urban district"
        }, key=len, reverse=True)
        return self

    def transform(self, raw, training=False):
        return preprocess(raw, training=training, districts=self.districts)

    def fit_transform(self, raw, training=True):
        return self.fit(raw).transform(raw, training=training)


def feature_frame(df, level):
    out = df[BASE_CATEGORICAL + BASE_NUMERIC + BASE_FLAGS].copy()
    out["建案名稱"] = df["建案名稱"]
    out["移轉樓層_缺失"] = df["移轉樓層"].isna().fillna(False).astype("int8")
    out["相對樓層_缺失"] = df["相對樓層"].isna().fillna(False).astype("int8")
    if level == 2:
        out["縣市_交易年"] = df["縣市"] + "_" + df["交易年"].astype("Int64").astype("string").fillna("缺失")
    if level == 3:
        out["建案名稱"] = df["建案名稱"]
    if level >= 4:
        for name, count in [("有建物", "交易建物筆數"), ("有車位", "交易車位筆數")]:
            out[name] = (df[count] > 0).fillna(False).astype("int8")
        out["多筆土地交易"] = (df["交易土地筆數"] > 1).fillna(False).astype("int8")
        for name, area, count in [("每筆土地面積", AREA_COLUMNS[0], "交易土地筆數"), ("每筆建物面積", AREA_COLUMNS[1], "交易建物筆數"), ("每車位平均面積", AREA_COLUMNS[2], "交易車位筆數")]:
            out[name] = safe_ratio(df[area], df[count]).where(df[count] > 0, 0)
    if level >= 5:
        out["交易時間序號"] = df["交易時間序號"]
    if level >= 6:
        for col in ["道路地段", "建案行政區", "地址基地", "棟別", "持分交易", "含增建", "預售資訊"]:
            out[col] = df[col]
        out["行政區_建物型態"] = df["行政區"] + "_" + df["建物型態"]
        out["行政區_交易標的"] = df["行政區"] + "_" + df["交易標的"]
        out["行政區_土地使用"] = df["行政區"] + "_" + df["土地使用類型"]
        out["行政區_都市分區"] = df["行政區"] + "_" + df["都市土地使用分區"]
        out["行政區_非都市編定"] = df["行政區"] + "_" + df["非都市土地使用編定"]
        out["建物型態_用途"] = df["建物型態"] + "_" + df["主要用途"]
        land, building, parking = [df[col] for col in AREA_COLUMNS]
        out["扣車位建物面積"] = (building - parking).where((building >= parking) & (building > 0))
        out["車位面積占比"] = safe_ratio(parking, building)
        out["土地建物面積比"] = safe_ratio(land, building)
        out["建物土地面積比"] = safe_ratio(building, land)
        out["格局總數"] = df[LAYOUT_COLUMNS].sum(axis=1, min_count=3)
        out["每房建物面積"] = safe_ratio(building, df[LAYOUT_COLUMNS[0]])
        out["每格局建物面積"] = safe_ratio(building, out["格局總數"])
        out["衛房比"] = safe_ratio(df[LAYOUT_COLUMNS[2]], df[LAYOUT_COLUMNS[0]])
        out["車位建物筆數比"] = safe_ratio(df["交易車位筆數"], df["交易建物筆數"])
        out["土地每層面積"] = safe_ratio(land, df["總樓層數"])
        out["一樓"] = (df["移轉樓層"] == 1).fillna(False).astype("int8")
        out["頂樓"] = ((df["移轉樓層"] == df["總樓層數"]) & (df["移轉樓層"] > 0)).fillna(False).astype("int8")
        out["地下樓"] = (df["移轉樓層"] < 0).fillna(False).astype("int8")
        out["純土地"] = ((df["交易建物筆數"] == 0) & (df["交易土地筆數"] > 0)).fillna(False).astype("int8")
        for col in AREA_COLUMNS + ["每筆土地面積", "每筆建物面積", "扣車位建物面積"]:
            values = pd.to_numeric(out[col], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan, copy=True)
            values[~np.isfinite(values)] = np.nan
            out[f"log_{col}"] = np.log1p(np.maximum(values, 0))
        for col in AREA_COLUMNS + LAYOUT_COLUMNS + ["總樓層數", "交易土地筆數", "交易建物筆數", "交易車位筆數", "交易年", "交易月"]:
            out[f"{col}_缺失"] = df[col].isna().fillna(False).astype("int8")
    if level != 3 and level < 6:
        out = out.drop(columns="建案名稱")
    return out.replace([np.inf, -np.inf], np.nan)

class FeatureBuilder:
    def __init__(self, level):
        self.level = level

    def fit(self, df):
        base = feature_frame(df, self.level)
        self.categorical = base.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.frequencies = {"建案名稱": df["建案名稱"].value_counts(normalize=True)}
        if self.level >= 6:
            for col in ["道路地段", "地址基地", "建案行政區", "行政區_建物型態", "行政區_交易標的"]:
                self.frequencies[col] = base[col].value_counts(normalize=True)
        self.medians = base.select_dtypes(include=["number"]).median().fillna(0)
        return self

    def transform(self, df):
        out = feature_frame(df, self.level)
        for col, frequency in self.frequencies.items():
            values = df[col] if col == "建案名稱" else out[col]
            out[f"{col}頻率"] = values.map(frequency).fillna(0).astype("float32")
        if self.level <= 5:
            out[self.medians.index] = out[self.medians.index].fillna(self.medians)
        for col in self.categorical:
            out[col] = out[col].fillna("缺失").astype(str)
        for col in out.columns.difference(self.categorical):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        return out

ANCHOR_GROUPS = {"行政區單價": ["行政區", "交易標的", "建物型態"], "道路單價": ["道路地段", "交易標的", "建物型態"], "建案單價": ["建案行政區", "交易標的"], "基地單價": ["地址基地", "交易標的", "建物型態"]}

def anchor_area(df):
    building = df[AREA_COLUMNS[1]]
    land = df[AREA_COLUMNS[0]]
    parking = df[AREA_COLUMNS[2]]
    area = building.where(df["交易建物筆數"] > 0, land)
    area = area.where(df["交易標的"] != "車位", parking)
    return area.where(area > 0)

def group_keys(df, columns):
    key = df[columns[0]].astype(str)
    for col in columns[1:]:
        key = key + "\x1f" + df[col].astype(str)
    return key

class PriceAnchor:
    def fit(self, df, target):
        area = anchor_area(df)
        unit = pd.Series(np.asarray(target), index=df.index).div(area)
        usable = unit.between(100, 10_000_000) & area.notna() & (df["特殊關係交易"] == 0)
        log_unit = pd.Series(np.log(unit.where(usable).to_numpy(dtype=float, na_value=np.nan)), index=df.index)
        self.global_log = float(log_unit.median()) if log_unit.notna().any() else np.log(100_000)
        self.tables = {}
        for name, columns in ANCHOR_GROUPS.items():
            frame = pd.DataFrame({"key": group_keys(df, columns), "value": log_unit})
            if name == "建案單價":
                frame.loc[df["建案名稱"].eq("缺失"), "value"] = np.nan
            if name == "道路單價":
                frame.loc[df["道路地段"].str.endswith("_缺失"), "value"] = np.nan
            table = frame.groupby("key")["value"].agg(["median", "count"])
            self.tables[name] = table
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        area = anchor_area(df)
        fallback = pd.Series(self.global_log, index=df.index)
        for name, columns in ANCHOR_GROUPS.items():
            table = self.tables[name]
            key = group_keys(df, columns)
            count = key.map(table["count"]).fillna(0)
            median = key.map(table["median"]).fillna(fallback)
            log_estimate = (count * median + 20 * fallback) / (count + 20)
            out[name] = np.exp(log_estimate)
            out[f"{name}_筆數"] = np.log1p(count)
            out[f"{name}_面積估價"] = out[name] * area
            if name == "行政區單價":
                fallback = log_estimate
        return out.astype("float32")

def oof_anchors(df, target):
    out = None
    splitter = GroupKFold(n_splits=OOF_FOLDS)
    for fit_index, hold_index in splitter.split(df, groups=df[ID_COLUMN]):
        fold = PriceAnchor().fit(df.iloc[fit_index].reset_index(drop=True), np.asarray(target)[fit_index])
        transformed = fold.transform(df.iloc[hold_index].reset_index(drop=True))
        if out is None:
            out = pd.DataFrame(np.nan, index=np.arange(len(df)), columns=transformed.columns)
        out.iloc[hold_index] = transformed.to_numpy()
    assert out.notna().any(axis=1).all()
    return out.astype("float32"), PriceAnchor().fit(df, target)

class TimeTrend:
    def fit(self, df, target):
        area = anchor_area(df)
        unit = pd.Series(np.asarray(target), index=df.index).div(area)
        frame = pd.DataFrame({"key": group_keys(df, ["行政區", "交易標的", "建物型態"]), "year": df["交易年"], "log_unit": np.log(unit.where(unit.between(100, 10_000_000)).to_numpy(dtype=float, na_value=np.nan))})
        self.last_year = int(df["交易年"].max())
        frame = frame.loc[frame["year"].between(self.last_year - 3, self.last_year)]
        annual = frame.groupby(["key", "year"])["log_unit"].agg(["median", "count"]).reset_index()
        annual = annual.loc[annual["count"] >= 20]
        self.slopes = {}
        for key, rows in annual.groupby("key"):
            if len(rows) >= 3:
                self.slopes[key] = float(np.clip(np.polyfit(rows["year"], rows["median"], 1)[0], -0.15, 0.20))
        return self

    def transform(self, df):
        slope = group_keys(df, ["行政區", "交易標的", "建物型態"]).map(self.slopes).fillna(0)
        horizon = (df["交易年"] + (df["交易月"] - 1) / 12 - self.last_year).clip(-4, 2).fillna(0)
        return pd.DataFrame({"區域歷史單價年增率": np.expm1(slope), "區域時間調整倍率": np.exp(slope * horizon)}, index=df.index).astype("float32")

def oof_trends(df, target):
    out = pd.DataFrame(np.nan, index=df.index, columns=["區域歷史單價年增率", "區域時間調整倍率"])
    for fit_index, hold_index in GroupKFold(n_splits=OOF_FOLDS).split(df, groups=df[ID_COLUMN]):
        trend = TimeTrend().fit(df.iloc[fit_index].reset_index(drop=True), np.asarray(target)[fit_index])
        out.iloc[hold_index] = trend.transform(df.iloc[hold_index].reset_index(drop=True)).to_numpy()
    return out.astype("float32"), TimeTrend().fit(df, target)

def prepare_features(training, target, others, level):
    builder = FeatureBuilder(level).fit(training)
    x_fit = builder.transform(training)
    x_others = [builder.transform(frame) for frame in others]
    anchor = None
    trend = None
    if level >= 7:
        encoded, anchor = oof_anchors(training, target)
        x_fit = pd.concat([x_fit, encoded], axis=1)
        x_others = [pd.concat([matrix, anchor.transform(frame)], axis=1) for matrix, frame in zip(x_others, others)]
    if level >= 8:
        encoded, trend = oof_trends(training, target)
        x_fit = pd.concat([x_fit, encoded], axis=1)
        x_others = [pd.concat([matrix, trend.transform(frame)], axis=1) for matrix, frame in zip(x_others, others)]
        for matrix in [x_fit, *x_others]:
            matrix["歷史區域時間面積估價"] = matrix["行政區單價_面積估價"] * matrix["區域時間調整倍率"]
    assert all(x_fit.columns.equals(matrix.columns) for matrix in x_others)
    categorical = x_fit.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return x_fit, x_others, categorical, {"builder": builder, "anchor": anchor, "trend": trend}

def recency_weights(df):
    month = df["交易時間序號"]
    gap = (month.max() - month).clip(lower=0).fillna(48)
    weights = np.maximum(0.25, np.exp(-np.log(2) * gap / 36))
    return weights / weights.mean()

class RecentHistory:
    def fit(self, df, y):
        area = anchor_area(df)
        unit = pd.Series(np.asarray(y, dtype=float), index=df.index).div(area)
        usable = unit.between(100, 10_000_000) & df["交易時間序號"].notna()
        usable &= df["特殊關係交易"].eq(0)
        self.tables = []
        groups = [("區域近期", ["行政區", "建物型態", "交易標的"]),
                  ("建案近期", ["建案行政區", "交易標的"])]
        for name, columns in groups:
            keep = usable & (df["建案名稱"].ne("缺失") if name == "建案近期" else True)
            frame = pd.DataFrame({
                "key": group_keys(df, columns),
                "month": df["交易時間序號"].astype(float),
                "log_unit": np.log(unit.where(usable).to_numpy(dtype=float, na_value=np.nan))
            }).loc[keep]
            table = frame.groupby(["key", "month"])["log_unit"].agg(
                ["median", "count"]).reset_index().sort_values(["key", "month"])
            table["unit"] = np.exp(table.groupby("key")["median"].transform(
                lambda s: s.ewm(span=4, adjust=False).mean()))
            table["support"] = table.groupby("key")["count"].transform(
                lambda s: s.rolling(3, min_periods=1).sum())
            table["snapshot"] = table["month"]
            self.tables.append((name, columns, table[
                ["key", "month", "unit", "support", "snapshot"]].sort_values("month")))
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        area = anchor_area(df).to_numpy(dtype=float, na_value=np.nan)
        for name, columns, table in self.tables:
            left = pd.DataFrame({
                "key": group_keys(df, columns).to_numpy(),
                "month": df["交易時間序號"].to_numpy(dtype=float, na_value=np.nan),
                "row": np.arange(len(df))
            }).dropna(subset=["month"]).sort_values("month")
            if table.empty or left.empty:
                matched = pd.DataFrame(index=np.arange(len(df)),
                    columns=["unit", "support", "snapshot"], dtype=float)
            else:
                matched = pd.merge_asof(left, table, on="month", by="key",
                    direction="backward", allow_exact_matches=False)
                matched = matched.set_index("row").reindex(np.arange(len(df)))
            lag = df["交易時間序號"].to_numpy(dtype=float, na_value=np.nan) - matched["snapshot"].to_numpy(dtype=float, na_value=np.nan)
            available = (lag > 0) & (lag <= 12)
            available &= matched["support"].fillna(0).to_numpy() >= 5
            out[name + "單價"] = matched["unit"].where(available).to_numpy(dtype=float, na_value=np.nan)
            out[name + "面積估價"] = out[name + "單價"] * area
            out[name + "支持筆數"] = np.log1p(
                matched["support"].where(available, 0).fillna(0).to_numpy(dtype=float, na_value=np.nan))
            out[name + "間隔月數"] = np.where(available, lag, np.nan)
        return out.astype("float32")

def comparable_keys(df):
    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)

    area = anchor_area(df).to_numpy(dtype=float, na_value=np.nan)
    net = num(AREA_COLUMNS[1]) - num(AREA_COLUMNS[2])
    net = np.where((net > 0) & (net <= area), net, area)
    floor, relative = num("移轉樓層"), num("相對樓層")
    floor_band = np.select(
        [floor < 0, floor == 1, relative <= .35, relative <= .7, relative > .7],
        [0, 1, 2, 3, 4], default=-1)

    frame = df.reset_index(drop=True).copy()
    for name, values in {
        "NetBand": np.floor(net / 30), "Rooms": num(LAYOUT_COLUMNS[0]),
        "Parking": num("交易車位筆數"), "FloorBand": floor_band
    }.items():
        frame[name] = pd.Series(values).round().astype("Int64").astype(str)

    common = ["建物型態", "交易標的", "NetBand", "Rooms", "Parking", "車位類別"]
    keys = {
        name: group_keys(frame, [col, *common, *extra])
        for name, col, extra in [
            ("Project", "建案行政區", ["FloorBand"]),
            ("Road", "道路地段", ["FloorBand"]),
            ("District", "行政區", [])
        ]
    }
    keys["Project"] = keys["Project"].where(frame["建案名稱"].ne("缺失"))
    keys["Road"] = keys["Road"].where(~frame["道路地段"].str.endswith("_缺失"))
    return keys, area

class ComparableHistory:
    def __init__(self, window=24, min_rows=5):
        self.window, self.min_rows = window, min_rows

    def fit(self, df):
        df = df.reset_index(drop=True)
        keys, area = comparable_keys(df)
        month = df["交易時間序號"].to_numpy(dtype=float, na_value=np.nan)
        y = df[TARGET].to_numpy(dtype=float, na_value=np.nan)
        unit = np.divide(y, area, out=np.full(len(df), np.nan), where=area > 0)
        keep = np.isfinite(month) & (area > 0) & (area <= 600)
        keep &= (unit >= 100) & (unit <= 1e7)
        keep &= df["特殊關係交易"].eq(0).to_numpy()
        keep &= ~df[ID_COLUMN].duplicated(keep=False).to_numpy()
        self.cutoff, self.tables = float(np.nanmax(month)), {}

        for name, key in keys.items():
            source = pd.DataFrame({
                "Key": key, "Month": month,
                "Log": np.log(unit, where=unit > 0, out=np.full(len(df), np.nan))
            }).loc[keep & key.notna().to_numpy()]
            source["Log2"] = source["Log"] ** 2
            table = source.groupby(["Key", "Month"]).agg(
                N=("Log", "size"), S=("Log", "sum"), Q=("Log2", "sum")
            ).reset_index()
            table[["N", "S", "Q"]] = table.groupby("Key", sort=False)[["N", "S", "Q"]].cumsum()
            self.tables[name] = table.sort_values("Month")
        return self

    def transform(self, df, gap=1):
        keys, area = comparable_keys(df)
        month = df["交易時間序號"].to_numpy(dtype=float, na_value=np.nan)
        gap = np.broadcast_to(np.asarray(gap, dtype=float), month.shape)
        assert np.isfinite(gap).all() and (gap >= 1).all()
        cutoff = np.minimum(month - gap, self.cutoff)
        out = pd.DataFrame({"Comparable_Gap": month - cutoff}, index=np.arange(len(df)))

        def lookup(key, date, table):
            left = pd.DataFrame({"Key": key, "Cut": date, "Row": np.arange(len(df))})
            left = left.dropna(subset=["Key", "Cut"]).sort_values("Cut")
            if table.empty or left.empty:
                return pd.DataFrame(
                    np.nan, index=np.arange(len(df)), columns=["Month", "N", "S", "Q"])
            return pd.merge_asof(
                left, table, left_on="Cut", right_on="Month", by="Key",
                direction="backward"
            ).set_index("Row").reindex(np.arange(len(df)))

        for name, key in keys.items():
            upper = lookup(key, cutoff, self.tables[name])
            lower = lookup(key, cutoff - self.window, self.tables[name])
            sums = upper[["N", "S", "Q"]].fillna(0) - lower[["N", "S", "Q"]].fillna(0)
            n = sums["N"].clip(lower=0)
            mean = sums["S"] / n.where(n > 0)
            available = n.ge(self.min_rows)
            out[name + "_Price"] = (np.exp(mean) * area).where(available)
            out[name + "_N"] = n
            out[name + "_Age"] = (month - upper["Month"]).where(available)
            out[name + "_Spread"] = np.sqrt(
                (sums["Q"] / n.where(n > 0) - mean ** 2).clip(lower=0)
            ).where(available)
            assert (
                out.loc[available, name + "_Age"] >= out.loc[available, "Comparable_Gap"]
            ).all()

        out["Comparable_Price"] = (
            out["Project_Price"].fillna(out["Road_Price"]).fillna(out["District_Price"]))
        out["Comparable_Level"] = np.select(
            [out["Project_Price"].notna(), out["Road_Price"].notna(),
             out["District_Price"].notna()], [3, 2, 1], default=0)
        out.index = df.index
        return out.astype("float32")

class NamePrefix:
    def keys(self, df):
        name = df["建案名稱"].astype("string").fillna("").str.normalize("NFKC").str.upper()
        name = name.str.strip().str.replace(r"^[^A-Z㐀-䶿一-鿿]+", "", regex=True)
        return name.str[:2].where(name.str.len().gt(2), "缺失")

    def fit(self, df):
        source = pd.DataFrame({"Prefix": self.keys(df), "Project": df["建案行政區"]})
        self.stats = source.loc[source["Prefix"].ne("缺失")].groupby("Prefix").agg(
            Rows=("Project", "size"), Projects=("Project", "nunique"))
        self.known = set(df.loc[df["建案名稱"].ne("缺失"), "建案行政區"])
        return self

    def supported(self, df):
        key = self.keys(df)
        return key.map(self.stats["Rows"]).ge(50) & key.map(self.stats["Projects"]).ge(3)

    def transform(self, df):
        key = self.keys(df)
        keep = self.supported(df)
        out = pd.DataFrame(index=df.index)
        out["名稱前綴"] = key.where(keep, "缺失").astype(str)
        out["前綴_縣市"] = (key + "_" + df["縣市"].astype(str)).where(keep, "缺失").astype(str)
        out["前綴_建物型態"] = (key + "_" + df["建物型態"].astype(str)).where(keep, "缺失").astype(str)
        out["前綴歷史建案數"] = np.log1p(key.map(self.stats["Projects"]).fillna(0)).astype("float32")
        out["前綴歷史筆數"] = np.log1p(key.map(self.stats["Rows"]).fillna(0)).astype("float32")
        return out

def section_number(text):
    if not text:
        return ""
    if text.isdigit():
        return str(int(text))
    digits = dict(zip("零一二三四五六七八九", range(10)))
    if "十" in text:
        left, right = text.split("十", 1)
        return str(digits.get(left, 1) * 10 + digits.get(right, 0))
    return str(digits[text]) if text in digits else text

@lru_cache(maxsize=200000)
def location_parts(address, city, district):
    body = address.removeprefix(city).removeprefix(district)
    body = re.sub(r"^.{1,8}[里村][零一二三四五六七八九十\d]+鄰", "", body)
    number = r"[零一二三四五六七八九十\d]+"
    roads = re.findall(
        rf"([^\d\s,，;；、段號棟巷弄]+?(?:大道|路|街))(?:({number})段)?", body)
    roads = [
        name.lstrip("與及、，").strip() + (section_number(section) + "段" if section else "")
        for name, section in roads
    ]
    road = roads[0] if roads else ""
    lane = re.search(rf"({number})巷(?:({number})弄)?", body)
    lane = (
        section_number(lane.group(1)) + "巷" +
        (section_number(lane.group(2)) + "弄" if lane.group(2) else "")
    ) if lane else ""
    crossing = "|".join(sorted(set(roads[:2]))) if len(set(roads[:2])) == 2 else ""
    parcel = re.search(r"([^\d\s,，;；路街號]+?段)([^\d\s,，;；號]+?小段)?", body)
    parcel = "".join(x or "" for x in parcel.groups()) if parcel and ("地號" in body or not road) else ""
    return road, road + lane if road and lane else "", crossing, parcel

def fine_location(frame):
    clean = lambda s: s.astype("string").fillna("").str.normalize("NFKC").str.replace("台", "臺", regex=False)
    parts = [
        location_parts(*values)
        for values in zip(clean(frame["土地位置建物門牌"]),
                          clean(frame["縣市"]), clean(frame["鄉鎮市區"]))
    ]
    out = pd.DataFrame(parts, index=frame.index, columns=[
        "細地段_道路段號", "細地段_巷弄", "細地段_路口", "細地段_土地小段"])
    for col in out:
        out[col] = (frame["行政區"].astype(str) + "|" + out[col]).where(out[col].ne(""), "缺失")
    out["細地段_道路型態"] = (
        out["細地段_道路段號"] + "|" + frame["建物型態"].astype(str)
    ).where(out["細地段_道路段號"].ne("缺失"), "缺失")
    return out.astype(str)

def area_weights(area, lower, upper, maximum):
    area = np.asarray(area, dtype=float)
    weights = np.zeros(len(area))
    valid = np.isfinite(area) & (area > 0)
    weights[valid] = maximum * np.clip(
        np.log(area[valid] / lower) / np.log(upper / lower), 0, 1)
    return weights

def named_project(frame):
    name = frame["建案名稱"].astype("string").fillna("").str.normalize("NFKC").str.strip()
    return ~name.str.upper().isin(["", "缺失", "NAN", "NONE", "NULL", "<NA>"])

def project_status(frame, training):
    named = named_project(frame).to_numpy(dtype=bool)
    known = set(training.loc[named_project(training), "建案行政區"].astype(str))
    seen = frame["建案行政區"].astype(str).isin(known).to_numpy()
    return pd.Series(
        np.select([~named, seen], ["Missing_Name", "Seen_Named"], default="Unseen_Named"),
        index=frame.index, name="Project_Status"
    )

def ordinary_mask(df):
    return anchor_area(df).between(0, 300, inclusive="right").fillna(False).to_numpy(dtype=bool)


class FineLocationFeatures:
    """Fit location counts on the training split and preserve the original column order."""
    def fit(self, frame):
        extra = fine_location(frame)
        self.categorical = extra.columns.tolist()
        self.stats = {}
        for column in self.categorical:
            table = pd.DataFrame({"Key": extra[column], "Project": frame["建案行政區"]})
            self.stats[column] = table.groupby("Key").agg(
                Rows=("Project", "size"), Projects=("Project", "nunique")
            ).drop(index="缺失", errors="ignore")
        return self

    def transform(self, frame):
        extra = fine_location(frame)
        for column in self.categorical:
            for metric in ["Rows", "Projects"]:
                values = extra[column].map(self.stats[column][metric]).fillna(0).to_numpy(dtype=float)
                extra[f"{column}_{metric}"] = np.log1p(values).astype("float32")
        return extra


def numeric_frame(frame):
    """Use the native floating-point representation used by the legacy fine-location model."""
    out = frame.copy()
    for column in out.select_dtypes(include="number"):
        out[column] = out[column].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    return out



import copy
import ast
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
from catboost import CatBoostRegressor


DEFAULT_RECIPE = {'model_defaults': {'loss_function': 'MAE',
                    'eval_metric': 'MAE',
                    'depth': 9,
                    'learning_rate': 0.04,
                    'l2_leaf_reg': 8,
                    'random_seed': 42,
                    'allow_writing_files': False,
                    'verbose': 200},
 'components': {'structural_total': {'trees': 2994, 'features': 95, 'params': {}},
                'anchored_total': {'trees': 3500,
                                   'features': 110,
                                   'params': {'learning_rate': 0.035, 'l2_leaf_reg': 10}},
                'recent_total': {'trees': 2400, 'features': 103, 'params': {}},
                'unit_price': {'trees': 1948, 'features': 95, 'params': {}},
                'comparable_total': {'trees': 3493, 'features': 110, 'params': {}},
                'prefix_total': {'trees': 3491, 'features': 115, 'params': {}},
                'fine_location_total': {'trees': 2400, 'features': 118, 'params': {}}},
 'historical_cutoff': 187,
 'historical_horizon': 12,
 'historical_scale': 1.0089489408031578,
 'production_scale': 1.0297437329436667,
 'early_tree_budget': 851,
 'uplift_guard': 1.1,
 'unit_ramp': [300, 600, 1],
 'fine_weight': 0.25,
 'comparable_weight': 0.25,
 'calibration': {'shrink': 200, 'min_projects': 5, 'weight': 0.5, 'cap': 250000},
 'calibration_fit_horizon': [1, 7],
 'calibration_dev_horizon': [8, 10],
 'calibration_check_horizon': [11, 12]}

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()

def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

def frame_digest(frame):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(json.dumps([(c, str(t)) for c, t in frame.dtypes.items()], ensure_ascii=False).encode())
    return digest.hexdigest()

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)

def fit_feature_bundle(frame, kind):
    target = frame[TARGET].to_numpy(dtype=float)
    if kind == 'anchored_total':
        matrix, _, categorical, bundle = prepare_features(frame, target, [], 8)
    else:
        builder = FeatureBuilder(6).fit(frame)
        matrix = builder.transform(frame)
        categorical, bundle = builder.categorical.copy(), {'builder': builder}
        if kind in ['recent_total', 'fine_location_total']:
            bundle['history'] = RecentHistory().fit(frame, target)
            matrix = pd.concat([matrix, bundle['history'].transform(frame)], axis=1)
        if kind in ['comparable_total', 'prefix_total']:
            bundle['reference'] = ComparableHistory(window=24, min_rows=5).fit(frame)
            matrix = pd.concat([matrix, bundle['reference'].transform(frame)], axis=1)
        if kind == 'prefix_total':
            bundle['prefix'] = NamePrefix().fit(frame)
            matrix = pd.concat([matrix, bundle['prefix'].transform(frame)], axis=1)
            categorical += ['名稱前綴', '前綴_縣市', '前綴_建物型態']
        if kind == 'fine_location_total':
            bundle['fine'] = FineLocationFeatures().fit(frame)
            matrix = pd.concat([matrix, bundle['fine'].transform(frame)], axis=1)
            categorical += bundle['fine'].categorical
    bundle.update(kind=kind, categorical=categorical, columns=matrix.columns.tolist())
    return matrix, bundle

def transform_feature_bundle(bundle, frame):
    kind = bundle['kind']
    matrix = bundle['builder'].transform(frame)
    if kind == 'anchored_total':
        matrix = pd.concat([matrix, bundle['anchor'].transform(frame), bundle['trend'].transform(frame)], axis=1)
        matrix['歷史區域時間面積估價'] = matrix['行政區單價_面積估價'] * matrix['區域時間調整倍率']
    if 'history' in bundle:
        matrix = pd.concat([matrix, bundle['history'].transform(frame)], axis=1)
    if 'reference' in bundle:
        matrix = pd.concat([matrix, bundle['reference'].transform(frame)], axis=1)
    if 'prefix' in bundle:
        matrix = pd.concat([matrix, bundle['prefix'].transform(frame)], axis=1)
    if 'fine' in bundle:
        matrix = pd.concat([matrix, bundle['fine'].transform(frame)], axis=1)
    assert matrix.columns.tolist() == bundle['columns'], 'Feature order mismatch.'
    return matrix

def train_component(name, frame, context, recipe, context_dir, code_signature, config):
    spec = recipe['components'][name]
    params = {**recipe['model_defaults'], **spec.get('params', {}), 'iterations': spec['trees']}
    params['thread_count'] = config.threads
    if config.smoke:
        params.update(iterations=8, depth=4, verbose=False)
    numeric_cast = context == 'historical' or name == 'fine_location_total'
    signature = json_digest({'fit': frame_digest(frame), 'params': params, 'name': name,
        'context': context, 'numeric_cast': numeric_cast, 'code': code_signature,
        'recipe': recipe, 'environment': config.environment, 'mode': ('smoke' if config.smoke else 'full')})
    folder = context_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    model_path, bundle_path, manifest_path = folder / 'model.cbm', folder / 'features.pkl', folder / 'manifest.json'
    if config.reuse_cache and all(p.exists() for p in [model_path, bundle_path, manifest_path]):
        manifest = json.loads(manifest_path.read_text())
        if (manifest['signature'] == signature and manifest['model_sha256'] == file_sha256(model_path)
                and manifest['bundle_sha256'] == file_sha256(bundle_path)):
            manifest['model_path'] = str(model_path)
            manifest['bundle_path'] = str(bundle_path)
            print(f'Loaded: {context} / {name}')
            return manifest
    started = time.time()
    data = numeric_frame(frame) if numeric_cast else frame
    matrix, bundle = fit_feature_bundle(data, name)
    assert matrix.shape[1] == spec['features'], f'Unexpected feature count: {name}'
    target = data[TARGET].to_numpy(dtype=float)
    weights = None
    mask = np.ones(len(data), dtype=bool)
    if name == 'unit_price':
        area = anchor_area(data).to_numpy(dtype=float, na_value=np.nan)
        mask = np.isfinite(area) & (area > 0)
        target = target[mask] / area[mask]
        weights = area[mask] / area[mask].mean()
    elif name in ['comparable_total', 'prefix_total']:
        mask = ordinary_mask(data)
        target = target[mask]
    elif name == 'anchored_total':
        weights = np.asarray(recency_weights(data), dtype=float)
    assert mask.any() and np.isfinite(target).all(), 'Empty or invalid training target.'
    print(f'Training: {context} / {name} | Rows: {int(mask.sum())} | Trees: {params["iterations"]}')
    model = CatBoostRegressor(**params)
    model.fit(matrix.loc[mask], target, cat_features=bundle['categorical'], sample_weight=weights)
    bundle['numeric_cast'] = numeric_cast
    model.save_model(str(model_path))
    with bundle_path.open('wb') as stream:
        cloudpickle.dump(bundle, stream)
    manifest = {'signature': signature, 'context': context, 'component': name,
        'fit_rows': int(mask.sum()), 'trees': int(model.tree_count_), 'features': len(bundle['columns']),
        'columns': bundle['columns'], 'categorical': bundle['categorical'], 'params': params,
        'minutes': (time.time() - started) / 60, 'model_path': str(model_path),
        'bundle_path': str(bundle_path), 'model_sha256': file_sha256(model_path),
        'bundle_sha256': file_sha256(bundle_path)}
    write_json(manifest_path, manifest)
    del model, matrix, bundle, data
    gc.collect()
    return manifest

def train_suite(frame, context, recipe, run_dir, code_signature, config):
    folder = run_dir / 'models' / context
    return {name: train_component(name, frame, context, recipe, folder, code_signature, config)
            for name in recipe['components']}

def predict_suite(frame, manifests, recipe, config):
    frame = frame.reset_index(drop=True)
    output = pd.DataFrame(index=frame.index)
    for name, manifest in manifests.items():
        assert file_sha256(manifest['model_path']) == manifest['model_sha256']
        assert file_sha256(manifest['bundle_path']) == manifest['bundle_sha256']
        model = CatBoostRegressor()
        model.load_model(manifest['model_path'])
        with open(manifest['bundle_path'], 'rb') as stream:
            bundle = cloudpickle.load(stream)
        chunks = []
        for start in range(0, len(frame), config.batch_size):
            batch = frame.iloc[start:start + config.batch_size].reset_index(drop=True)
            if bundle['numeric_cast']:
                batch = numeric_frame(batch)
            matrix = transform_feature_bundle(bundle, batch)
            assert matrix.columns.tolist() == model.feature_names_
            part = pd.DataFrame({name: np.maximum(model.predict(matrix), 0)})
            if name == 'recent_total':
                budget = min(recipe['early_tree_budget'], model.tree_count_)
                part['early_total'] = np.maximum(model.predict(matrix, ntree_end=budget), 0)
                history = bundle['history'].transform(batch)
                part['Project_Reference'] = history['建案近期面積估價'].to_numpy()
                part['District_Reference'] = history['區域近期面積估價'].to_numpy()
            if name == 'comparable_total':
                refs = bundle['reference'].transform(batch)
                part['Comparable_Reference'] = refs['Comparable_Price'].to_numpy()
            if name == 'prefix_total':
                part['Prefix_Supported'] = bundle['prefix'].supported(batch).fillna(False).to_numpy(dtype=bool)
            chunks.append(part)
        combined = pd.concat(chunks, ignore_index=True)
        for column in combined:
            output[column] = combined[column].to_numpy()
        print(f'Predicted: {name} | Rows: {len(frame)}')
        del model, bundle, chunks, combined, matrix
        gc.collect()
    return output

def assemble_predictions(frame, components, training, scale, recipe):
    assert len(frame) == len(components)
    base = frame.reset_index(drop=True)
    parts = components.reset_index(drop=True)
    area = anchor_area(base).to_numpy(dtype=float, na_value=np.nan)
    ordinary = ordinary_mask(base)
    status = project_status(base, training).to_numpy()
    unseen = status == 'Unseen_Named'
    early = parts['early_total'].to_numpy()
    guard = np.where(early > 0, np.minimum(parts['recent_total'], recipe['uplift_guard'] * early), parts['recent_total'])
    fine_guard = np.where(early > 0, np.minimum(parts['fine_location_total'], recipe['uplift_guard'] * early), parts['fine_location_total'])
    fine_active = ordinary & unseen
    adjusted = guard.copy()
    adjusted[fine_active] = ((1 - recipe['fine_weight']) * guard[fine_active]
                             + recipe['fine_weight'] * fine_guard[fine_active])
    calibrated = scale * (.5 * parts['structural_total'].to_numpy() + .5 * parts['anchored_total'].to_numpy())
    unit_weight = area_weights(area, *recipe['unit_ramp'])
    positive = np.isfinite(area) & (area > 0)
    unit = np.where(positive, parts['unit_price'].to_numpy() * area, parts['structural_total'])
    recent_active = ordinary & parts['Comparable_Reference'].notna().to_numpy()
    prefix_active = recent_active & unseen & parts['Prefix_Supported'].to_numpy(dtype=bool)
    def mix(p13):
        total = .5 * p13 + .5 * calibrated
        result = (1 - unit_weight) * total + unit_weight * unit
        active = recent_active & ~prefix_active
        w = recipe['comparable_weight']
        result[active] = (1 - w) * result[active] + w * parts.loc[active, 'comparable_total'].to_numpy()
        result[prefix_active] = (1 - w) * result[prefix_active] + w * parts.loc[prefix_active, 'prefix_total'].to_numpy()
        assert np.isfinite(result).all() and (result >= 0).all()
        return total, result
    _, without_fine = mix(guard)
    total, prediction = mix(adjusted)
    np.testing.assert_array_equal(prediction[~fine_active], without_fine[~fine_active])
    audit = base[[ID_COLUMN, '行政區', '縣市', '建案名稱', '建案行政區', '交易年', '交易月', '交易時間序號', '建物型態']].copy()
    audit['Area'], audit['Ordinary'], audit['Project_Status'] = area, ordinary, status
    audit['Baseline'], audit['Before_Fine'], audit['Adjusted_Recent'], audit['Total_Component'] = prediction, without_fine, adjusted, total
    audit['Unit_Weight'], audit['Fine_Active'] = unit_weight, fine_active
    for col in parts:
        audit[col] = parts[col].to_numpy()
    if TARGET in base:
        audit['Actual'] = base[TARGET].to_numpy(dtype=float)
        audit['Residual'] = audit['Actual'] - prediction
        audit['AE'] = audit['Residual'].abs()
    return audit

class RegionalCalibration:
    def __init__(self, shrink=200, min_projects=5, weight=.5, cap=250000, fit_horizon=(1, 7)):
        self.shrink, self.min_projects, self.weight, self.cap = shrink, min_projects, weight, cap
        self.fit_horizon = tuple(fit_horizon)

    def keys(self, frame):
        out = frame.copy()
        out['Reference'] = np.select([out['Project_Reference'].notna(), out['District_Reference'].notna()],
                                     ['Project', 'District'], default='Missing')
        out['Parent'] = out['Project_Status'].astype(str) + '|' + out['Reference']
        out['Key'] = out['行政區'].astype(str) + '|' + out['Parent']
        return out

    def fit(self, audit):
        source = self.keys(audit)
        source = source.loc[source['Ordinary'] & source['Horizon'].between(*self.fit_horizon)].copy()
        assert len(source) > 0, 'No calibration rows.'
        source['Residual'] = source['Actual'] - source['Baseline']
        self.stats = source.groupby('Key').agg(Median=('Residual', 'median'), Rows=('Residual', 'size'),
                                               Projects=('建案行政區', 'nunique'))
        self.parents = source.groupby('Parent')['Residual'].median()
        self.fit_rows = len(source)
        return self

    def raw_correction(self, audit):
        frame = self.keys(audit)
        parent = frame['Parent'].map(self.parents).fillna(0).to_numpy(dtype=float)
        count = frame['Key'].map(self.stats['Rows']).fillna(0).to_numpy(dtype=float)
        median = frame['Key'].map(self.stats['Median']).fillna(0).to_numpy(dtype=float)
        supported = frame['Key'].map(self.stats['Projects']).fillna(0).ge(self.min_projects).to_numpy()
        shrunk = (count * median + self.shrink * parent) / (count + self.shrink)
        return np.where(supported, shrunk, parent)

    def predict(self, audit, weight=None):
        weight = self.weight if weight is None else weight
        baseline = audit['Baseline'].to_numpy(dtype=float)
        ordinary = audit['Ordinary'].to_numpy(dtype=bool)
        delta = np.where(ordinary, np.clip(weight * self.raw_correction(audit), -self.cap, self.cap), 0)
        result = np.maximum(baseline + delta, 0)
        np.testing.assert_array_equal(result[~ordinary], baseline[~ordinary])
        assert np.isfinite(result).all() and np.abs(result - baseline).max() <= self.cap + 1e-6
        return result

def evaluate_calibration(audit, calibration, recipe):
    baseline, actual = audit['Baseline'].to_numpy(), audit['Actual'].to_numpy()
    original_error = np.abs(actual - baseline)
    ordinary = audit['Ordinary'].to_numpy(dtype=bool)
    accurate = ordinary & (original_error <= 500000)
    rows = []
    corrected = calibration.predict(audit)
    corrected_error = np.abs(actual - corrected)
    for period, low, high in [('Calibration_Fit', *recipe['calibration_fit_horizon']),
                              ('Calibration_Dev', *recipe['calibration_dev_horizon']),
                              ('Calibration_Check', *recipe['calibration_check_horizon'])]:
        dates = audit['Horizon'].between(low, high).to_numpy()
        for segment, condition in {
            'All': np.ones(len(audit), dtype=bool),
            'Unseen_Ordinary': ordinary & audit['Project_Status'].eq('Unseen_Named').to_numpy(),
            'Originally_Accurate': accurate, 'Other_Areas_Unchanged': ~ordinary
        }.items():
            mask = dates & condition
            if not mask.any():
                continue
            old, new = original_error[mask].mean(), corrected_error[mask].mean()
            rows.append({'Period': period, 'Segment': segment, 'Rows': int(mask.sum()),
                'Baseline_MAE': old, 'Candidate_MAE': new, 'Improvement_Pct': 100 * (1 - new / old),
                'Mean_Abs_Change': np.abs(corrected - baseline)[mask].mean(),
                'Median_Residual': np.median((actual - corrected)[mask])})
    return pd.DataFrame(rows), corrected

class AgeBiasCorrection:
    def __init__(self, cap=500_000, max_slope=200_000, shrink=500):
        self.cap, self.max_slope, self.shrink = cap, max_slope, shrink

    @staticmethod
    def weighted_median(values, weights):
        values, weights = np.asarray(values, float), np.asarray(weights, float)
        order = np.argsort(values)
        return float(values[order][np.searchsorted(
            np.cumsum(weights[order]), weights.sum() / 2)])

    def fit_line(self, points):
        x, y, w = [points[c].to_numpy(float)
                   for c in ["Reference_Age", "Median", "Rows"]]
        slopes = [0.0, self.max_slope]
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[j] != x[i]:
                    slopes.append(np.clip(
                        (y[j] - y[i]) / (x[j] - x[i]), 0, self.max_slope))
        def loss(slope):
            intercept = self.weighted_median(y - slope * x, w)
            return np.sum(w * np.abs(y - intercept - slope * x))
        slope = float(min(slopes, key=loss))
        return self.weighted_median(y - slope * x, w), slope

    @staticmethod
    def scope(frame):
        return (
            frame["Ordinary"].to_numpy(dtype=bool)
            & frame["Project_Status"].eq("Unseen_Named").to_numpy()
            & frame["Reference"].eq("District").to_numpy()
            & frame["Reference_Age"].between(1, 12).to_numpy())

    def fit(self, frame):
        source = frame.loc[
            self.scope(frame) & frame["Horizon"].between(1, 7).to_numpy()
        ].copy()
        source["Residual"] = source["Actual"] - source["Candidate"]

        def points(group):
            return group.groupby("Reference_Age").agg(
                Median=("Residual", "median"), Rows=("Actual", "size")
            ).reset_index()

        self.parent_intercept, self.parent_slope = self.fit_line(points(source))
        rows = []
        for district, group in source.groupby("行政區"):
            projects = group["建案行政區"].nunique()
            if len(group) < 100 or projects < 5 or group["Reference_Age"].nunique() < 4:
                continue
            _, local_slope = self.fit_line(points(group))
            strength = min(
                len(group) / (len(group) + self.shrink),
                projects / (projects + 5))
            slope = (
                strength * local_slope + (1 - strength) * self.parent_slope)
            local_intercept = np.median(
                group["Residual"] - slope * group["Reference_Age"])
            intercept = (
                strength * local_intercept
                + (1 - strength) * self.parent_intercept)
            rows.append({
                "District": district, "Intercept": intercept,
                "Slope": slope, "Rows": len(group), "Projects": projects})
        assert rows, "Insufficient calibration support."
        self.coefficients = pd.DataFrame(rows).set_index("District")
        return self

    def eligible(self, frame):
        return self.scope(frame) & frame["行政區"].isin(
            self.coefficients.index).to_numpy()

    def predict(self, frame, weight):
        intercept = frame["行政區"].map(
            self.coefficients["Intercept"]).fillna(0).to_numpy()
        slope = frame["行政區"].map(
            self.coefficients["Slope"]).fillna(0).to_numpy()
        age = frame["Reference_Age"].fillna(0).clip(1, 12).to_numpy()
        correction = np.maximum(intercept + slope * age, 0)
        active = self.eligible(frame)
        baseline = frame["Candidate"].to_numpy(dtype=float)
        result = baseline + np.where(
            active, np.minimum(weight * correction, self.cap), 0)
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(result[~active], baseline[~active])
        return result


@dataclass
class LegacyConfig:
    """Runtime settings; predictive choices are frozen in DEFAULT_RECIPE."""

    threads: int = 8
    batch_size: int = 40000
    reuse_cache: bool = True
    smoke: bool = False
    save_system_pickle: bool = False

    @property
    def environment(self):
        packages = ["numpy", "pandas", "scikit-learn", "catboost", "cloudpickle"]
        return {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in packages}}


def legacy_implementation_signature():
    """Fingerprint predictive definitions; prose and output formatting are irrelevant."""
    names = {
        'normalize_text', 'roc_dates', 'chinese_integer', 'single_floor', 'safe_ratio',
        'recover_district', 'preprocess', 'DatasetPreprocessor', 'feature_frame',
        'FeatureBuilder', 'anchor_area', 'group_keys', 'PriceAnchor', 'oof_anchors',
        'TimeTrend', 'oof_trends', 'prepare_features', 'recency_weights', 'RecentHistory',
        'comparable_keys', 'ComparableHistory', 'NamePrefix', 'section_number',
        'location_parts', 'fine_location', 'area_weights', 'named_project',
        'project_status', 'ordinary_mask', 'FineLocationFeatures', 'numeric_frame',
        'fit_feature_bundle', 'transform_feature_bundle', 'assemble_predictions',
        'train_component', 'train_suite', 'predict_suite',
        'RegionalCalibration', 'AgeBiasCorrection', 'historical_split', 'attach_reference_age'}

    class WithoutDocstrings(ast.NodeTransformer):
        def _visit(self, node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                    node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            return node
        visit_FunctionDef = _visit
        visit_ClassDef = _visit

    parsed = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    definitions = [WithoutDocstrings().visit(node) for node in parsed.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in definitions} == names
    constants = {key: globals()[key] for key in [
        'TARGET', 'ID_COLUMN', 'MAX_YEAR', 'OOF_FOLDS', 'CITY_REPLACE', 'CITY_FILL',
        'BASE_CATEGORICAL', 'AREA_COLUMNS', 'LAYOUT_COLUMNS', 'BASE_NUMERIC',
        'BASE_FLAGS', 'ANCHOR_GROUPS']}
    return json_digest({'definitions': [ast.dump(node, include_attributes=False) for node in definitions],
                        'constants': constants})


def historical_split(frame, cutoff=187, horizon=12):
    """Keep original row order; remove query IDs from the earlier fit population."""
    month = pd.to_numeric(frame["交易時間序號"], errors="coerce")
    query = frame.loc[month.between(cutoff + 1, cutoff + horizon)].copy()
    fit = frame.loc[month.le(cutoff)].copy()
    fit = fit.loc[~fit[ID_COLUMN].isin(query[ID_COLUMN])]
    fit, query = fit.reset_index(drop=True), query.reset_index(drop=True)
    if fit.empty or query.empty:
        raise ValueError("Training data do not cover the fixed calibration time window.")
    assert fit["交易時間序號"].max() <= cutoff < query["交易時間序號"].min()
    assert set(fit[ID_COLUMN]).isdisjoint(query[ID_COLUMN])
    return fit, query


def attach_reference_age(audit, frame, recent_manifest, calibration, batch_size=40000):
    """Recover the age of exactly the reference already used by the ensemble."""
    assert file_sha256(recent_manifest["bundle_path"]) == recent_manifest["bundle_sha256"]
    with open(recent_manifest["bundle_path"], "rb") as stream:
        bundle = cloudpickle.load(stream)
    chunks = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start:start + batch_size].reset_index(drop=True)
        if bundle["numeric_cast"]:
            batch = numeric_frame(batch)
        chunks.append(bundle["history"].transform(batch))
    reference = pd.concat(chunks, ignore_index=True)
    result = calibration.keys(audit).reset_index(drop=True)
    for generated, original in [
        ("建案近期面積估價", "Project_Reference"),
        ("區域近期面積估價", "District_Reference")
    ]:
        np.testing.assert_allclose(reference[generated].to_numpy(),
                                   result[original].to_numpy(),
                                   rtol=1e-6, atol=1, equal_nan=True)
    result["Reference_Age"] = np.select(
        [result["Reference"].eq("Project"), result["Reference"].eq("District")],
        [reference["建案近期間隔月數"].to_numpy(),
         reference["區域近期間隔月數"].to_numpy()], default=np.nan)
    return result


class LegacySystem:
    """Explicit seven-component ensemble, regional correction and age correction.

    Training uses only raw_train. Valid and Test are accepted by predict_raw()
    after fitting and are never used to construct features or calibration.
    The two sets of seven component states serve different training cutoffs.
    They are components of one fixed recipe, not sequential experiment versions.
    """

    def __init__(self, config=None, recipe=None):
        self.config = config or LegacyConfig()
        self.recipe = copy.deepcopy(DEFAULT_RECIPE if recipe is None else recipe)
        self.age_weight = 0.1

    def fit(self, raw_train, cache_dir):
        if TARGET not in raw_train:
            raise ValueError(f"Train must contain {TARGET}.")
        self.directory = Path(cache_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.preprocessor = DatasetPreprocessor().fit(raw_train)
        frame = self.preprocessor.transform(raw_train, training=True)
        self.fit_rows = len(frame)
        # Prediction only needs project membership, not a second full Train copy.
        self.training = frame[["建案名稱", "建案行政區"]].drop_duplicates().reset_index(drop=True)
        self.raw_train_signature = frame_digest(raw_train)
        self.train_signature = frame_digest(frame)
        self.code_signature = legacy_implementation_signature()
        fit, query = historical_split(frame, self.recipe["historical_cutoff"],
                                      self.recipe["historical_horizon"])
        print({"System": "Fixed Seven-Component Ensemble", "Full_Train_Rows": len(frame),
               "Calibration_Model_Fit_Rows": len(fit), "Calibration_Query_Rows": len(query),
               "Component_States": 14, "Test_Valid_Used_For_Fit": False}, flush=True)
        self.historical_models = train_suite(fit, "historical", self.recipe,
                                             self.directory, self.code_signature, self.config)
        components = predict_suite(query, self.historical_models, self.recipe, self.config)
        audit = assemble_predictions(query, components, fit,
                                      self.recipe["historical_scale"], self.recipe)
        audit["Horizon"] = audit["交易時間序號"] - self.recipe["historical_cutoff"]
        self.calibration = RegionalCalibration(
            **self.recipe["calibration"],
            fit_horizon=self.recipe["calibration_fit_horizon"]).fit(audit)
        audit["Candidate"] = self.calibration.predict(audit)
        audit = attach_reference_age(audit, query, self.historical_models["recent_total"],
                                     self.calibration, self.config.batch_size)
        self.age_model = AgeBiasCorrection().fit(audit)
        audit["Prediction"] = self.age_model.predict(audit, self.age_weight)
        self.calibration_audit = audit[[ID_COLUMN, "Actual", "Prediction", "Horizon"]].copy()
        self.calibration_audit.to_csv(self.directory / "calibration_predictions.csv", index=False)
        self.calibration.stats.to_csv(self.directory / "regional_calibration.csv")
        self.age_model.coefficients.to_csv(self.directory / "age_calibration.csv")
        del fit, query, components, audit
        gc.collect()
        self.final_models = train_suite(frame, "production", self.recipe,
                                        self.directory, self.code_signature, self.config)
        self._write_manifest()
        return self

    def _write_manifest(self):
        write_json(self.directory / "system_manifest.json", {
            "system": "Fixed Seven-Component Ensemble", "recipe": self.recipe,
            "raw_train_signature": self.raw_train_signature,
            "processed_train_signature": self.train_signature,
            "training_rows": self.fit_rows, "age_weight": self.age_weight,
            "environment": self.config.environment,
            "calibration_fit_horizon": self.recipe["calibration_fit_horizon"],
            "valid_used_for_training": False, "test_used_for_training": False,
            "code_signature": self.code_signature,
            "historical_models": self.historical_models,
            "final_models": self.final_models,
            "system_pickle_saved": bool(self.config.save_system_pickle)})
        if self.config.save_system_pickle:
            with (self.directory / "system.pkl").open("wb") as stream:
                cloudpickle.dump(self, stream)

    def predict(self, processed_frame):
        if not hasattr(self, "final_models"):
            raise RuntimeError("Call fit(raw_train, cache_dir) before predicting.")
        frame = processed_frame.reset_index(drop=True)
        components = predict_suite(frame, self.final_models, self.recipe, self.config)
        audit = assemble_predictions(frame, components, self.training,
                                      self.recipe["production_scale"], self.recipe)
        audit["Candidate"] = self.calibration.predict(audit)
        audit = attach_reference_age(audit, frame, self.final_models["recent_total"],
                                     self.calibration, self.config.batch_size)
        audit["Prediction"] = self.age_model.predict(audit, self.age_weight)
        assert np.isfinite(audit["Prediction"]).all()
        return audit

    def predict_raw(self, raw_query):
        # Query labels never enter preprocessing or any feature constructor.
        frame = self.preprocessor.transform(raw_query.drop(columns=[TARGET], errors='ignore'), training=False)
        if len(frame) != len(raw_query):
            raise ValueError("Query preprocessing removed rows; provide valid labels or remove the target column.")
        audit = self.predict(frame)
        np.testing.assert_array_equal(audit[ID_COLUMN].astype(str), raw_query[ID_COLUMN].astype(str))
        return audit


LegacyPreprocessor = DatasetPreprocessor

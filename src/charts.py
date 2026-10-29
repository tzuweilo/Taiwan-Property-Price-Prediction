"""Presentation charts for the property-price notebook.

No model is trained and no Test target is required by this module. Descriptive
relationships use labelled Train rows. Model importance must be supplied from a
held-out evaluation, and is never relabelled as a causal effect.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TARGET = "總價元"
BUILDING = "建物移轉總面積平方公尺"
PALETTE = {"background": "#F7F5F0", "paper": "#FFFFFF", "navy": "#17324D",
           "teal": "#147D86", "coral": "#CE6A56", "gold": "#C49A48",
           "muted": "#697A89", "grid": "#DCE4E8", "light": "#DEEDEF"}
COLORS = [PALETTE[key] for key in ["teal", "navy", "coral", "gold", "muted"]] + ["#7D759A", "#81966F", "#6398AF"]
FEATURE_LABELS = {
    "編號": "Transaction ID", "總價元": "Total price", "鄉鎮市區": "Town / district",
    "交易標的": "Transaction target", "土地位置建物門牌": "Address",
    "土地移轉總面積平方公尺": "Land area", "都市土地使用分區": "Urban zoning",
    "非都市土地使用分區": "Non-urban zoning", "非都市土地使用編定": "Non-urban land use",
    "交易年月日": "Transaction date", "交易筆棟數": "Land / building / parking counts",
    "移轉層次": "Transferred-floor text", "總樓層數": "Total floors", "建物型態": "Building type",
    "主要用途": "Primary use", "主要建材": "Construction material", "建築完成年月": "Completion date",
    "建物移轉總面積平方公尺": "Building area", "建物現況格局-房": "Rooms",
    "建物現況格局-廳": "Living rooms", "建物現況格局-衛": "Bathrooms",
    "建物現況格局-隔間": "Partition", "有無管理組織": "Management organization",
    "車位類別": "Parking type", "車位移轉總面積平方公尺": "Parking area",
    "備註": "Remarks", "建案名稱": "Project name", "棟及號": "Block / unit text",
    "解約情形": "Cancellation status", "交易年": "Transaction year", "交易月": "Transaction month",
    "交易日": "Transaction day", "交易季度": "Transaction quarter", "交易時間序號": "Month index",
    "交易土地筆數": "Land count", "交易建物筆數": "Building count", "交易車位筆數": "Parking count",
    "移轉樓層": "Transferred floor", "最低移轉樓層": "Lowest transferred floor", "最高移轉樓層": "Highest transferred floor", "整棟交易": "Whole-building transaction",
    "多樓層交易": "Multiple-floor transaction", "相對樓層": "Relative floor",
    "土地使用類型": "Land-use family", "縣市": "City / county", "行政區": "City and district",
    "有備註": "Has remarks", "特殊關係交易": "Special relationship", "含裝潢家具家電資訊": "Furnishing mention",
    "毛胚屋": "Unfinished interior", "雨遮不計價": "Canopy excluded from price",
    "持分交易": "Partial ownership", "含增建": "Extension mention", "預售資訊": "Presale mention",
    "道路地段": "Road / land section", "地址基地": "Address base", "建案行政區": "Project and district",
    "棟別": "Block", "屋齡": "Building age", "完成年": "Completion year",
    "交易月_sin": "Month sine", "交易月_cos": "Month cosine",
    "建物面積為零": "Zero building area", "建物筆數面積衝突": "Building-count / area conflict",
    "交易筆棟數未解析": "Unparsed transaction counts", "交易日期無效": "Invalid transaction date",
    "完成日在交易日後": "Completion after transaction", "建商前綴": "Developer prefix",
    "Area": "Building area", "Year": "Transaction year", "Month": "Transaction month",
    "Location": "Location", "Property": "Type / use / zoning", "Structure": "Floor / layout / parking",
    "Project": "Project identity", "Remarks": "Remarks / status", "Time": "Time", "Area_Time": "Area and time",
    "扣車位建物面積": "Building area excluding parking", "車位面積占比": "Parking / building area",
    "土地建物面積比": "Land / building area", "建物土地面積比": "Building / land area",
    "格局總數": "Total layout count", "每房建物面積": "Building area per room",
    "每格局建物面積": "Building area per layout count", "衛房比": "Bathrooms / rooms",
    "車位建物筆數比": "Parking / building count", "土地每層面積": "Land area per floor",
    "一樓": "Ground floor", "頂樓": "Top floor", "地下樓": "Basement floor", "純土地": "Land only",
    "多筆土地交易": "Multiple-land transaction", "有建物": "Has building", "有土地": "Has land",
    "有車位": "Has parking", "每筆土地面積": "Area per land record", "每棟建物面積": "Area per building",
    "每筆建物面積": "Area per building", "每車位平均面積": "Area per parking space",
    "每個車位面積": "Area per parking space", "都市分區": "Urban zoning", "非都市編定": "Non-urban land use",
    "土地使用": "Land-use family", "用途": "Primary use",
}
CATEGORY_LABELS = {
    "缺失": "Missing", "住家用": "Residential", "住商用": "Residential / commercial", "商業用": "Commercial",
    "工業用": "Industrial", "辦公室": "Office", "一般事務所": "General office", "店舖": "Retail",
    "住宅大樓(11層含以上有電梯)": "High-rise apartment", "華廈(10層含以下有電梯)": "Mid-rise apartment",
    "公寓(5樓含以下無電梯)": "Walk-up apartment", "透天厝": "Townhouse", "店面(店鋪)": "Retail unit",
    "辦公商業大樓": "Office building", "廠辦": "Factory / office", "工廠": "Factory", "倉庫": "Warehouse",
    "套房(1房1廳1衛)": "Studio", "房地(土地+建物)+車位": "Land + building + parking",
    "房地(土地+建物)": "Land + building", "建物": "Building only", "土地": "Land only",
    "車位": "Parking only", "都市": "Urban", "非都市": "Non-urban", "混合": "Mixed",
    "其他或未標示": "Other / unspecified", "住宅區": "Residential zone", "商業區": "Commercial zone",
    "工業區": "Industrial zone", "農業區": "Agricultural zone", "有": "Yes", "無": "No",
    "鋼筋混凝土造": "Reinforced concrete", "鋼骨鋼筋混凝土造": "Steel-reinforced concrete",
    "鋼骨造": "Steel frame", "加強磚造": "Reinforced brick", "磚造": "Brick", "壁式預鑄鋼筋混凝土造": "Precast concrete",
    "坡道平面": "Ramp / flat", "坡道機械": "Ramp / mechanical", "升降平面": "Lift / flat",
    "升降機械": "Lift / mechanical", "塔式車位": "Tower parking", "一樓平面": "Ground / flat",
}
CITY_LABELS = {"臺北市": "Taipei", "新北市": "New Taipei", "桃園市": "Taoyuan", "臺中市": "Taichung",
    "臺南市": "Tainan", "高雄市": "Kaohsiung", "新竹縣": "Hsinchu County", "新竹市": "Hsinchu City",
    "基隆市": "Keelung", "苗栗縣": "Miaoli", "彰化縣": "Changhua", "南投縣": "Nantou",
    "雲林縣": "Yunlin", "嘉義市": "Chiayi City", "嘉義縣": "Chiayi County", "屏東縣": "Pingtung",
    "宜蘭縣": "Yilan", "花蓮縣": "Hualien", "臺東縣": "Taitung", "澎湖縣": "Penghu", "金門縣": "Kinmen", "連江縣": "Lienchiang"}
DISTRICT_LABELS = {"中山區": "Zhongshan", "大安區": "Da'an", "信義區": "Xinyi", "松山區": "Songshan",
    "中正區": "Zhongzheng", "萬華區": "Wanhua", "大同區": "Datong", "南港區": "Nangang",
    "內湖區": "Neihu", "士林區": "Shilin", "北投區": "Beitou", "文山區": "Wenshan",
    "板橋區": "Banqiao", "淡水區": "Tamsui", "新店區": "Xindian", "新莊區": "Xinzhuang",
    "林口區": "Linkou", "三重區": "Sanchong", "中和區": "Zhonghe", "永和區": "Yonghe",
    "土城區": "Tucheng", "蘆洲區": "Luzhou", "三峽區": "Sanxia", "鶯歌區": "Yingge",
    "汐止區": "Xizhi", "五股區": "Wugu", "泰山區": "Taishan", "樹林區": "Shulin",
    "桃園區": "Taoyuan", "中壢區": "Zhongli", "八德區": "Bade", "龜山區": "Guishan",
    "蘆竹區": "Luzhu", "大園區": "Dayuan", "楊梅區": "Yangmei", "平鎮區": "Pingzhen",
    "龍潭區": "Longtan", "觀音區": "Guanyin", "新屋區": "Xinwu", "竹北市": "Zhubei",
    "竹東鎮": "Zhudong", "新埔鎮": "Xinpu", "西屯區": "Xitun", "南屯區": "Nantun",
    "北屯區": "Beitun", "太平區": "Taiping", "大里區": "Dali", "烏日區": "Wuri",
    "豐原區": "Fengyuan", "沙鹿區": "Shalu", "梧棲區": "Wuqi", "清水區": "Qingshui",
    "北區": "North", "西區": "West", "南區": "South", "東區": "East", "中區": "Central",
    "安南區": "Annan", "安平區": "Anping", "永康區": "Yongkang", "新市區": "Xinshi",
    "鳳山區": "Fengshan", "楠梓區": "Nanzi", "左營區": "Zuoying", "鼓山區": "Gushan",
    "三民區": "Sanmin", "前鎮區": "Qianzhen", "小港區": "Xiaogang", "仁武區": "Renwu",
    "大寮區": "Daliao", "橋頭區": "Qiaotou", "岡山區": "Gangshan", "安樂區": "Anle",
    "七堵區": "Qidu", "暖暖區": "Nuannuan", "仁愛區": "Ren'ai", "中西區": "West Central"}


def location_label(value, fallback="Other district"):
    text = str(value)
    city, separator, district = text.partition("_")
    if separator and city in CITY_LABELS and district in DISTRICT_LABELS:
        return f"{CITY_LABELS[city]} · {DISTRICT_LABELS[district]}"
    if text in CITY_LABELS: return CITY_LABELS[text]
    if text in DISTRICT_LABELS: return DISTRICT_LABELS[text]
    return text if text.isascii() else fallback


def feature_label(column):
    """English label, with deterministic placeholders for unmapped raw fields."""
    text = str(column)
    if text in FEATURE_LABELS:
        return FEATURE_LABELS[text]
    if text.startswith("log_"):
        return "Log " + feature_label(text[4:]).lower()
    for suffix, label in [("_Negative", "negative flag"), ("_Invalid", "invalid flag"),
                          ("_Capacity_Warning", "capacity warning"), ("_Raw", "source value"),
                          ("_缺失", "missing flag"), ("頻率", "frequency")]:
        if text.endswith(suffix):
            return f"{feature_label(text[:-len(suffix)])}: {label}"
    if "_" in text:
        parts = text.split("_")
        if all(part in FEATURE_LABELS or part.isascii() for part in parts):
            return " × ".join(feature_label(part) for part in parts)
    return text if text.isascii() else "Derived attribute"


def apply_theme():
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [name for name in ["Inter", "Aptos", "DejaVu Sans", "Noto Sans CJK TC", "Noto Sans CJK JP"] if name in available]
    plt.rcParams.update({"font.family": preferred or ["sans-serif"], "font.size": 10.5,
                         "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 11,
                         "axes.labelsize": 10, "axes.labelcolor": PALETTE["navy"],
                         "axes.edgecolor": PALETTE["grid"], "axes.linewidth": .8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": PALETTE["paper"], "figure.facecolor": PALETTE["background"],
                         "text.color": PALETTE["navy"], "xtick.color": PALETTE["muted"],
                         "ytick.color": PALETTE["muted"], "grid.color": PALETTE["grid"],
                         "grid.alpha": .65, "legend.frameon": False, "savefig.facecolor": PALETTE["background"],
                         "axes.prop_cycle": plt.cycler(color=COLORS), "svg.fonttype": "none"})


def _numeric(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan, copy=True)


def _table_path(out_dir, stem):
    if out_dir is None:
        return None
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / stem


def _save_table(table, out_dir, stem):
    destination = Path(out_dir).parent / "tables" if out_dir is not None and Path(out_dir).name == "figures" else out_dir
    path = _table_path(destination, stem + ".csv")
    if path is not None:
        table.to_csv(path, index=False, encoding="utf-8-sig")


def save_chart(fig, out_dir, stem):
    if out_dir is not None:
        path = _table_path(out_dir, stem)
        for kind in ["png", "svg"]:
            temporary = path.with_name(f".{path.name}.{kind}.tmp")
            fig.savefig(temporary, format=kind, dpi=300 if kind == "png" else None,
                        bbox_inches="tight", pad_inches=.22)
            temporary.replace(path.with_suffix("." + kind))
    return fig


def _heading(fig, title, subtitle=None):
    height = fig.get_figheight()
    fig.suptitle(title, x=.06, y=1 - .12 / height, ha="left", fontsize=20, fontweight="bold", color=PALETTE["navy"])
    if subtitle:
        fig.text(.06, 1 - .56 / height, subtitle, ha="left", va="top", color=PALETTE["muted"], fontsize=10.5)


def _footnote(fig, text):
    fig.text(.06, .018, text, ha="left", va="bottom", color=PALETTE["muted"], fontsize=9)


def display_chart(fig):
    """Display one returned Figure exactly once in notebook output.

    Closing the pyplot manager before explicit display suppresses the inline
    backend's automatic second rendering. The Figure itself stays displayable.
    The chart's PNG/SVG have already been written by its chart function.
    """
    if not hasattr(fig, "savefig"):
        raise TypeError("display_chart expects the Figure returned by a chart function")
    try:
        from IPython.display import display
    except ImportError:
        fig.show()
        return None
    plt.close(fig)
    display(fig)
    return None


def display_charts(figures):
    """Display all topic figures in order, once each, without duplicate output."""
    for fig in figures:
        display_chart(fig)
    return None


def _compact(axis, which="y"):
    formatter = ticker.EngFormatter(places=0, sep="")
    (axis.yaxis if which == "y" else axis.xaxis).set_major_formatter(formatter)


def _empty(axis, message="No eligible observations"):
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes, color=PALETTE["muted"])
    axis.set_axis_off()


def _category_display(values):
    """Readable translated categories; unfamiliar CJK values use a stable code."""
    mapping = {}
    for index, value in enumerate(values, 1):
        text = str(value)
        translated = CATEGORY_LABELS.get(text, text if text.isascii() else f"Category {index:02d}")
        mapping[text] = translated
    return mapping










def monthly_summary(train, target=TARGET):
    y = _numeric(train[target]); x = _numeric(train[BUILDING])
    if "交易年" in train and "交易月" in train:
        year, month = _numeric(train["交易年"]), _numeric(train["交易月"])
        dates = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": np.ones(len(train))}), errors="coerce")
    elif "Transaction_Date" in train:
        dates = pd.to_datetime(train["Transaction_Date"], errors="coerce")
    else:
        return pd.DataFrame(columns=["Month", "Rows", "Median_Price", "Median_Area", "Median_Unit_Price"])
    data = pd.DataFrame({"Month": np.asarray(dates), "Price": y, "Area": x})
    data["Unit_Price"] = np.divide(y, x, out=np.full(len(y), np.nan), where=np.isfinite(x) & (x > 0))
    return data.dropna(subset=["Month"]).groupby("Month", observed=True).agg(
        Rows=("Price", "size"), Median_Price=("Price", "median"), Median_Area=("Area", "median"),
        Median_Unit_Price=("Unit_Price", "median")).reset_index()




def attribute_audit(train, features=None, target=TARGET, max_bins=8):
    """One descriptive audit row per requested attribute, plus full grouped data.

    Numeric Spearman rho is a marginal monotonic association, not feature
    importance. Categorical values receive category medians, IQR, and counts;
    no in-sample high-cardinality target-encoding score is ranked as evidence.
    """
    features = list(features) if features is not None else [c for c in train if not str(c).startswith("_")]
    y = pd.Series(_numeric(train[target]), index=train.index)
    records, details = [], []
    for number, column in enumerate(features, 1):
        row = {"No": number, "Feature": column, "Label": feature_label(column), "Rows": len(train),
               "Kind": "", "Nonmissing_Rows": 0, "Missing_Pct": np.nan, "Unique_Values": 0,
               "Spearman_Rho": np.nan, "Pearson_R": np.nan, "Status": "Descriptive only"}
        if column not in train:
            row.update(Kind="Absent", Status="Column absent"); records.append(row); continue
        series = train[column]
        missing = series.isna() | series.astype("string").str.strip().isin(["", "缺失", "Missing", "<NA>"])
        row.update(Nonmissing_Rows=int((~missing).sum()), Missing_Pct=float(missing.mean() * 100),
                   Unique_Values=int(series.nunique(dropna=True)))
        if column == target:
            row.update(Kind="Target", Status="Target; never used as an input"); records.append(row); continue
        if column in {"編號", "ID", "_Source_Row"}:
            row.update(Kind="Identifier", Status="Identity / alignment only; excluded from predictive inputs"); records.append(row); continue
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
        if numeric:
            value = pd.Series(_numeric(series), index=train.index)
            good = value.notna() & y.notna()
            row["Kind"] = "Numeric"
            if good.sum() >= 3 and value[good].nunique() > 1 and y[good].nunique() > 1:
                row["Spearman_Rho"] = value[good].corr(y[good], method="spearman")
                row["Pearson_R"] = value[good].corr(y[good])
            if value[good].nunique() > max_bins:
                groups = pd.qcut(value, max_bins, duplicates="drop")
            else:
                groups = value
            detail = pd.DataFrame({"Value": value, "Price": y, "Group": groups}).dropna(subset=["Group", "Price"])
            grouped = detail.groupby("Group", observed=True).agg(Rows=("Price", "size"), X=("Value", "median"),
                Median=("Price", "median"), Q25=("Price", lambda a: a.quantile(.25)), Q75=("Price", lambda a: a.quantile(.75))).reset_index()
            grouped["Category"] = grouped.pop("Group").astype(str)
            grouped["Category_Label"] = grouped["Category"]
        else:
            row["Kind"] = "Categorical"
            category = series.astype("string").fillna("Missing").replace({"缺失": "Missing", "": "Missing"})
            grouped = pd.DataFrame({"Category": category, "Price": y}).dropna(subset=["Price"]).groupby("Category", observed=True).agg(
                Rows=("Price", "size"), Median=("Price", "median"), Q25=("Price", lambda a: a.quantile(.25)),
                Q75=("Price", lambda a: a.quantile(.75))).reset_index().sort_values("Rows", ascending=False)
            mapping = _category_display(grouped.Category.tolist())
            grouped["Category_Label"] = grouped.Category.map(mapping)
            grouped["X"] = np.nan
            if row["Unique_Values"] > 1000:
                row["Status"] = "High cardinality; category medians are not out-of-sample evidence"
        grouped.insert(0, "Feature", column)
        grouped.insert(1, "Feature_Label", feature_label(column))
        grouped["Kind"] = row["Kind"]
        grouped["Rows_Share_Pct"] = 100 * grouped.Rows / max(1, len(train))
        details.append(grouped)
        records.append(row)
    return pd.DataFrame(records), (pd.concat(details, ignore_index=True) if details else pd.DataFrame())




def chart_version_progress(scores, out_dir=None, version_col="Version", mae_col="MAE", split_col="Set"):
    apply_theme(); table = scores.copy()
    if version_col not in table or mae_col not in table:
        raise ValueError(f"scores requires {version_col!r} and {mae_col!r}")
    if split_col not in table:
        table[split_col] = "Evaluation"
    if "Segment" in table:
        table = table.loc[table.Segment.eq("All")].copy()
    versions = list(dict.fromkeys(table[version_col].astype(str)))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.9), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.22, wspace=.32)
    _heading(fig, "What each feature block adds", "Only paired comparisons on the same frozen evaluation rows support a version-to-version claim.")
    for index, (label, group) in enumerate(table.groupby(split_col, sort=False, observed=True)):
        mapped = group.set_index(version_col).reindex(versions)
        values = _numeric(mapped[mae_col]) / 1e6
        axes[0].plot(range(len(versions)), values, "o-", label=str(label), color=COLORS[index % len(COLORS)], linewidth=2)
        for i, value in enumerate(values):
            if np.isfinite(value):
                axes[0].annotate(f"{value:.3f}", (i, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_xticks(range(len(versions)), versions, rotation=25, ha="right")
    axes[0].set_ylabel("MAE (NT$M) · lower is better"); axes[0].grid(axis="y"); axes[0].legend(fontsize=9)
    latest_split = table[split_col].iloc[0] if len(table) else "Evaluation"
    group = table.loc[table[split_col].eq(latest_split)].drop_duplicates(version_col).set_index(version_col).reindex(versions)
    values = _numeric(group[mae_col]); delta = np.r_[np.nan, values[:-1] - values[1:]] / 1000
    axes[1].barh(versions[1:][::-1], delta[1:][::-1], color=[PALETTE["teal"] if v >= 0 else PALETTE["coral"] for v in delta[1:][::-1]])
    axes[1].axvline(0, color=PALETTE["muted"], linewidth=.9)
    axes[1].set_xlabel("MAE saved vs previous version (NT$K)")
    axes[1].set_title(f"Incremental benefit · {latest_split}", loc="left", fontsize=11)
    _footnote(fig, "Model results are measured after you run the notebook. Public Kaggle scores belong to a different evaluation population.")
    _save_table(table, out_dir, "version_progress")
    return save_chart(fig, out_dir, "version_progress")






def chart_quality_overview(frames, cleaning_summary=None, out_dir=None, features=None):
    """One data-quality topic: dataset counts and all-attribute coverage map."""
    apply_theme()
    columns = list(features) if features is not None else list(dict.fromkeys(
        c for frame in frames.values() for c in frame if not str(c).startswith("_") and c != TARGET))
    table = pd.DataFrame({"Feature": columns, "Label": [feature_label(c) for c in columns]})
    for name, frame in frames.items():
        values = []
        for column in columns:
            if column not in frame:
                values.append(np.nan)
            else:
                series = frame[column]
                values.append(100 * (series.notna() & ~series.astype("string").str.strip().isin(["", "缺失", "Missing", "<NA>"])).mean())
        table[name] = values
    height = max(8.5, .255 * len(columns) + 2.5)
    fig = plt.figure(figsize=(15, height))
    grid = fig.add_gridspec(2, 2, left=.07, right=.95, top=1 - 1.36 / height, bottom=.07,
                            width_ratios=[1.0, 1.5], height_ratios=[1, 1.4], wspace=.8, hspace=.32)
    counts_ax, notes_ax, heat_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    _heading(fig, "Data quality and coverage", "The same preprocessing is applied to every workbook; Test remains a feature-only prediction set.")
    labels, counts = list(frames), [len(frame) for frame in frames.values()]
    bars = counts_ax.barh(labels, counts, color=COLORS[:len(labels)])
    counts_ax.invert_yaxis(); counts_ax.set_xlim(0, max([1] + counts) * 1.28)
    counts_ax.bar_label(bars, labels=[f"{value:,}" for value in counts], padding=6, fontsize=11)
    counts_ax.set_xlabel("Transactions"); counts_ax.set_title("Dataset size", loc="left"); _compact(counts_ax, "x")
    notes_ax.set_axis_off(); notes_ax.text(0, 1, "What the map means", transform=notes_ax.transAxes, fontsize=13, fontweight="bold", va="top")
    target_note = "Target shown for completeness; never an input" if TARGET in columns else "Target is excluded from this coverage map"
    notes = ["100% = observed in every row", "0% = entirely missing", "Grey = absent from the workbook", "Missing values stay explicit", "IDs support alignment, not price prediction", target_note]
    for i, note in enumerate(notes):
        notes_ax.text(0, .88 - .115 * i, note, fontsize=10, transform=notes_ax.transAxes, color=PALETTE["muted"], va="top")
    if cleaning_summary is not None and len(cleaning_summary):
        summary = pd.DataFrame(cleaning_summary)
        _save_table(summary, out_dir, "cleaning_summary")
        notes_ax.text(0, .12, "Detailed cleaning counts are exported\nwith the notebook tables.", fontsize=10,
                      color=PALETTE["teal"], transform=notes_ax.transAxes, va="top")
    vals = table[labels].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("coverage", ["#F0EAE1", "#B8DDDE", PALETTE["teal"], PALETTE["navy"]]); cmap.set_bad("#DFE3E6")
    image = heat_ax.imshow(np.ma.masked_invalid(vals), vmin=0, vmax=100, aspect="auto", cmap=cmap)
    heat_ax.set_xticks(range(len(labels)), labels); heat_ax.set_yticks(range(len(columns)), table.Label, fontsize=9)
    heat_ax.set_title("Observed values by attribute (%)", loc="left", fontsize=12)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            value = vals[i, j]
            heat_ax.text(j, i, "N/A" if not np.isfinite(value) else f"{value:.0f}", ha="center", va="center", fontsize=8,
                         color="white" if np.isfinite(value) and value >= 65 else PALETTE["navy"])
    _footnote(fig, "Feature names are translated for presentation; the exported coverage table retains each original source name.")
    _save_table(table, out_dir, "quality_attribute_coverage")
    return save_chart(fig, out_dir, "01_data_quality_atlas")


def chart_feature_influence_atlas(importance, coverage=None, groups=None, features=None, out_dir=None,
                                  metric_label="Held-out MAE increase after shuffling", stem="02_feature_influence_atlas"):
    """ONE ranked overview for every attribute; no per-attribute chart explosion.

    ``importance``: Feature and MAE_Increase (or Univariate_MAE_Gain,
    Delta_MAE / Importance), optional Std, Group, Status. Positive always means
    greater useful predictive information under the supplied evaluation. This
    function does not fit a model or manufacture importance for excluded fields.
    ``coverage``: optional Feature + Missing_Pct (Train audit) or Train coverage.
    """
    apply_theme(); table = importance.copy()
    feature_col = next((c for c in ["Feature", "Attribute", "Column"] if c in table), None)
    value_col = next((c for c in ["Predictive_Gain_Dev", "MAE_Increase", "Univariate_MAE_Gain", "Delta_MAE", "Importance"] if c in table), None)
    if feature_col is None or value_col is None:
        raise ValueError("importance requires Feature and a signed MAE impact column")
    if feature_col != "Feature": table = table.rename(columns={feature_col: "Feature"})
    if table.Feature.duplicated().any():
        raise ValueError("Provide one importance row per attribute, for one fixed evaluation population")
    if features is not None:
        table = table.set_index("Feature").reindex(list(features)).rename_axis("Feature").reset_index()
    table[value_col] = pd.to_numeric(table[value_col], errors="coerce")
    table["Label"] = table.Feature.map(feature_label)
    if groups:
        table["Group"] = table.Feature.map(groups).fillna(table.get("Group", "Other"))
    elif "Group" not in table:
        table["Group"] = "Attribute"
    table["Group"] = table["Group"].fillna("Other").astype(str)
    table["Coverage_Pct"] = np.nan
    if coverage is not None and len(coverage):
        lookup = pd.DataFrame(coverage).drop_duplicates("Feature").set_index("Feature")
        if "Missing_Pct" in lookup:
            table["Coverage_Pct"] = 100 - table.Feature.map(lookup.Missing_Pct)
        elif "Train" in lookup:
            table["Coverage_Pct"] = table.Feature.map(lookup.Train)
    table = table.sort_values(value_col, ascending=False, na_position="last", kind="stable").reset_index(drop=True)
    categories = list(dict.fromkeys(table.Group.fillna("Other").astype(str)))
    colors = {name: COLORS[i % len(COLORS)] for i, name in enumerate(categories)}
    nrows = math.ceil(max(1, len(table)) / 2)
    height = max(8, .32 * nrows + 3.1)
    fig = plt.figure(figsize=(17, height))
    plot_top, plot_bottom = 1 - 1.32 / height, .23
    # Each half needs a full label gutter; ordinary GridSpec wspace makes the
    # second half's long labels collide with the first half's coverage strip.
    axes = [fig.add_axes([.225, plot_bottom, .215, plot_top - plot_bottom]),
            fig.add_axes([.458, plot_bottom, .035, plot_top - plot_bottom]),
            fig.add_axes([.735, plot_bottom, .215, plot_top - plot_bottom]),
            fig.add_axes([.968, plot_bottom, .035, plot_top - plot_bottom])]
    _heading(fig, "Feature influence atlas", "All attributes in one view · size reflects predictive information on the same held-out rows, not a causal price premium.")
    check_col = next((c for c in ["Predictive_Gain_Check", "Univariate_MAE_Gain_Check", "MAE_Increase_Check"] if c in table), None)
    finite = np.concatenate([_numeric(table[value_col])] + ([_numeric(table[check_col])] if check_col else []))
    finite = finite[np.isfinite(finite)] / 1000
    lo = min(0, finite.min() if len(finite) else 0); hi = max(1, finite.max() if len(finite) else 1)
    span = hi - lo
    for half in range(2):
        sub = table.iloc[half * nrows:(half + 1) * nrows].copy()
        axis, cov_ax = axes[half * 2], axes[half * 2 + 1]
        positions = np.arange(len(sub)); values = _numeric(sub[value_col]) / 1000
        if sub.empty:
            axis.set_axis_off(); cov_ax.set_axis_off(); continue
        axis.axvline(0, color=PALETTE["grid"], linewidth=1)
        for pos, (_, row) in enumerate(sub.iterrows()):
            value = row[value_col]
            if pd.notna(value):
                color = colors[str(row.Group)] if value >= 0 else PALETTE["coral"]
                axis.hlines(pos, 0, value / 1000, color=color, linewidth=1.5, alpha=.45)
                std_col = next((c for c in ["Std", "MAE_Increase_Std", "Importance_Std"] if c in sub), None)
                error = row.get(std_col, np.nan) / 1000 if std_col else np.nan
                axis.errorbar(value / 1000, pos - .10 if check_col else pos, xerr=error if pd.notna(error) else None,
                              fmt="o", color=color, markersize=5, capsize=2, linewidth=.9)
            else:
                axis.plot(0, pos, marker="x", color=PALETTE["muted"], markersize=4)
            if check_col and pd.notna(row[check_col]):
                axis.plot(row[check_col] / 1000, pos + .12, marker="D", markerfacecolor="white",
                          markeredgecolor=colors[str(row.Group)], markersize=4, linestyle="none")
        axis.set_yticks(positions, [f"{int(i) + 1:02d}  {label}" for i, label in zip(sub.index, sub.Label)], fontsize=9)
        axis.set_ylim(len(sub) - .4, -.7); axis.set_xlim(lo - span * .07, hi + span * .08)
        axis.set_xlabel("Predictive impact (NT$K)"); axis.grid(axis="x", alpha=.45)
        axis.set_title("Predictive impact", loc="left", fontsize=11)
        for pos in positions:
            if pos % 2 == 0: axis.axhspan(pos - .48, pos + .48, color=PALETTE["background"], zorder=-1)
        cov = _numeric(sub.Coverage_Pct)
        cov_ax.barh(positions, np.nan_to_num(cov, nan=0), height=.5, color=PALETTE["light"])
        cov_ax.scatter(np.nan_to_num(cov, nan=0), positions, s=13, color=PALETTE["muted"])
        cov_ax.set_xlim(0, 115); cov_ax.set_ylim(len(sub) - .4, -.7)
        cov_ax.set_yticks([]); cov_ax.set_xticks([0, 100], ["0", "100"]); cov_ax.set_xlabel("%", fontsize=9)
        cov_ax.set_title("Observed", loc="left", fontsize=9)
        for pos, value in enumerate(cov):
            if not np.isfinite(value): cov_ax.text(50, pos, "N/A", va="center", ha="center", fontsize=7, color=PALETTE["muted"])
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colors[name], label=feature_label(name)) for name in categories]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .102), ncol=min(4, max(1, len(categories))), fontsize=9)
    if check_col:
        fig.text(.06, .083, "● Dev   ◇ Check · Ranked by Dev only; Check remains a paired robustness report.", fontsize=9, color=PALETTE["muted"])
    fig.text(.06, .055, metric_label + "; negative values are retained. × = no estimate / excluded input.", fontsize=9, color=PALETTE["muted"])
    _footnote(fig, "Correlated attributes can substitute for one another. Statistical association, model contribution, and causal effect are different quantities.")
    _save_table(table, out_dir, "feature_influence_all_attributes")
    return save_chart(fig, out_dir, stem)


def chart_area_time_context(train, target=TARGET, out_dir=None):
    """One integrated topic figure: area, time, size mix, and price scale."""
    apply_theme(); fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    fig.subplots_adjust(left=.08, right=.96, top=.83, bottom=.14, hspace=.43, wspace=.3)
    _heading(fig, "Area and time establish the price context", "Size gives the scale; monthly transaction mix and changing market levels explain why scale alone is insufficient.")
    x, y = _numeric(train[BUILDING]), _numeric(train[target]); good = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
    if good.any():
        lx, ly = np.log10(x[good]), np.log10(y[good] / 1e6)
        image = axes[0, 0].hexbin(lx, ly, gridsize=58, bins="log", mincnt=1,
            cmap=LinearSegmentedColormap.from_list("areatime", ["#DDEEEF", PALETTE["teal"], PALETTE["navy"]]))
        axes[0, 0].set_title("Area × total price", loc="left", fontsize=12)
        axes[0, 0].set_xlabel("Building area (m², log scale)"); axes[0, 0].set_ylabel("Price (NT$M, log scale)")
        for which, values in [("x", lx), ("y", ly)]:
            ticks = np.arange(math.floor(values.min()), math.ceil(values.max()) + 1)
            getattr(axes[0, 0], f"set_{which}ticks")(ticks, [f"{10.**v:g}" for v in ticks])
        fig.colorbar(image, ax=axes[0, 0], fraction=.027, pad=.015, label="Transactions")
    else: _empty(axes[0, 0])
    monthly = monthly_summary(train, target)
    metrics = [(axes[0, 1], "Median_Price", 1e6, "Monthly median total price", "NT$M", PALETTE["teal"]),
               (axes[1, 0], "Median_Area", 1, "Changing building-size mix", "m²", PALETTE["gold"]),
               (axes[1, 1], "Median_Unit_Price", 1000, "Monthly median unit price", "NT$K / m²", PALETTE["navy"])]
    for axis, col, scale, title, unit, color in metrics:
        if monthly.empty:
            _empty(axis); continue
        axis.plot(monthly.Month, monthly[col] / scale, color=color, linewidth=2)
        axis.set_title(title, loc="left", fontsize=12); axis.set_ylabel(unit)
        axis.grid(axis="y"); axis.tick_params(axis="x", rotation=25)
        axis.text(.02, .97, f"Monthly n: {monthly.Rows.min():,}–{monthly.Rows.max():,}",
                  transform=axis.transAxes, va="top", fontsize=8, color=PALETTE["muted"])
    _footnote(fig, "Train only · All valid positive prices and areas are retained in the density plot. Monthly medians are not a quality-adjusted price index.")
    _save_table(monthly, out_dir, "area_time_monthly_summary")
    return save_chart(fig, out_dir, "03_area_time_context")


def chart_interaction_atlas(train, target=TARGET, out_dir=None, min_rows=30):
    """ONE topic figure for area × type and location × type, with cell counts."""
    apply_theme(); y, area = _numeric(train[target]), _numeric(train[BUILDING])
    good = np.isfinite(y) & (y > 0) & np.isfinite(area) & (area > 0)
    location = "行政區" if "行政區" in train else "鄉鎮市區"
    data = pd.DataFrame({"Price": y[good], "Area": area[good],
        "Type": train.loc[good, "建物型態"].astype("string").fillna("Missing").to_numpy(),
        "District": train.loc[good, location].astype("string").fillna("Missing").to_numpy()})
    data["Unit_Price"] = data.Price / data.Area
    top_types = data.Type.value_counts().head(4).index.tolist(); mapping = _category_display(top_types)
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.9), gridspec_kw={"width_ratios": [1, 1.3]})
    fig.subplots_adjust(left=.075, right=.94, top=.79, bottom=.21, wspace=.4)
    _heading(fig, "Interactions reveal different pricing regimes", "The same area can imply a different price for another property type or district; sample support matters.")
    bands = ["≤50", "50–100", "100–150", "150–200", "200–300", "300–600", ">600"]
    data["Area_Band"] = pd.cut(data.Area, [0, 50, 100, 150, 200, 300, 600, np.inf], labels=bands)
    grouped = data.groupby(["Type", "Area_Band"], observed=True).agg(Rows=("Price", "size"),
        Median_Price=("Price", "median"), Median_Unit_Price=("Unit_Price", "median")).reset_index()
    for i, value in enumerate(top_types):
        subset = grouped.loc[grouped.Type.eq(value)].set_index("Area_Band").reindex(bands)
        subset.loc[subset.Rows.lt(min_rows), "Median_Price"] = np.nan
        axes[0].plot(range(len(bands)), subset.Median_Price / 1e6, "o-", color=COLORS[i], label=mapping[value], linewidth=1.8, markersize=4)
    axes[0].set_xticks(range(len(bands)), bands, rotation=30, ha="right")
    axes[0].set_xlabel("Building area band (m²)"); axes[0].set_ylabel("Median total price (NT$M)")
    axes[0].set_title("Area × building type", loc="left", fontsize=12); axes[0].grid(axis="y")
    axes[0].legend(fontsize=8.5, loc="upper left")
    top_districts = data.District.value_counts().head(12).index.tolist()
    summary = data.groupby(["District", "Type"], observed=True).agg(Rows=("Price", "size"),
        Median_Unit_Price=("Unit_Price", "median")).reset_index()
    pivot = summary.loc[summary.Rows.ge(min_rows)].pivot(index="District", columns="Type", values="Median_Unit_Price").reindex(index=top_districts, columns=top_types) / 1000
    sizes = summary.pivot(index="District", columns="Type", values="Rows").reindex(index=top_districts, columns=top_types)
    vals = pivot.to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("interaction_atlas", ["#EEE7DC", "#B9DBDC", PALETTE["teal"], PALETTE["navy"]]); cmap.set_bad("#E6E9EA")
    if vals.size:
        image = axes[1].imshow(np.ma.masked_invalid(vals), cmap=cmap, aspect="auto")
        finite = vals[np.isfinite(vals)]; threshold = np.quantile(finite, .65) if len(finite) else np.inf
        axes[1].set_yticks(range(len(top_districts)), [location_label(value, f"District {i:02d}") for i, value in enumerate(top_districts, 1)], fontsize=8.5)
        axes[1].set_xticks(range(len(top_types)), [mapping[t] for t in top_types], rotation=25, ha="right", fontsize=8.5)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isfinite(vals[i, j]):
                    axes[1].text(j, i, f"{vals[i,j]:.1f}\nn={int(sizes.iloc[i,j]):,}", ha="center", va="center", fontsize=8,
                        color="white" if vals[i,j] > threshold else PALETTE["navy"])
        fig.colorbar(image, ax=axes[1], fraction=.025, pad=.025, label="Median unit price (NT$K / m²)")
    else: _empty(axes[1])
    axes[1].set_title("District × building type", loc="left", fontsize=12)
    _footnote(fig, f"Train only · Most frequent districts / types · n < {min_rows} cells hidden · District keys and all cell values are exported in CSV; no model rows are dropped.")
    _save_table(grouped, out_dir, "interaction_area_type")
    _save_table(summary, out_dir, "interaction_district_type")
    _save_table(pd.DataFrame({"Chart_Label": [location_label(value, f"District {i:02d}") for i, value in enumerate(top_districts, 1)], "District": top_districts}), out_dir, "interaction_district_keys")
    return save_chart(fig, out_dir, "04_interaction_atlas")


def chart_residual_diagnostics(actual, predictions, meta=None, out_dir=None, name="Final model"):
    """One paired error topic: calibration, residuals, concentration, and segments."""
    apply_theme(); y, p = np.asarray(actual, dtype=float), np.asarray(predictions, dtype=float)
    if y.shape != p.shape or y.ndim != 1: raise ValueError("Actual and predictions must be aligned one-dimensional arrays")
    good = np.isfinite(y) & np.isfinite(p); y, p = y[good], p[good]
    residual, error = y - p, np.abs(y - p)
    fig, axes = plt.subplots(2, 2, figsize=(14.3, 8.9))
    fig.subplots_adjust(left=.08, right=.965, top=.83, bottom=.14, hspace=.45, wspace=.3)
    _heading(fig, "Where prediction error remains", f"{name} · held-out evaluation · residual = actual − prediction; positive values indicate underprediction.")
    if not len(y):
        for axis in axes.ravel(): _empty(axis)
        return save_chart(fig, out_dir, "06_residual_diagnostics")
    lower, upper = np.quantile(residual, [.01, .99]); selected = (residual >= lower) & (residual <= upper)
    axes[0, 0].hist(residual[selected] / 1e6, bins=55, color=PALETTE["teal"], edgecolor="white", linewidth=.4)
    axes[0, 0].axvline(0, color=PALETTE["navy"], linewidth=1)
    axes[0, 0].axvline(np.median(residual) / 1e6, color=PALETTE["coral"], linewidth=1.6)
    axes[0, 0].set_title("Residual distribution · central 98% shown", loc="left", fontsize=12)
    axes[0, 0].set_xlabel("Residual (NT$M)"); axes[0, 0].set_ylabel("Transactions")
    bins = (pd.qcut(pd.Series(p), min(10, len(np.unique(p))), duplicates="drop")
            if len(np.unique(p)) > 1 else pd.Series("All predictions", index=range(len(p))))
    calibration = pd.DataFrame({"Prediction": p, "Actual": y, "Bin": bins}).groupby("Bin", observed=True).agg(
        Rows=("Actual", "size"), Predicted=("Prediction", "median"), Actual=("Actual", "median")).reset_index(drop=True)
    axes[0, 1].plot(calibration.Predicted / 1e6, calibration.Actual / 1e6, "o-", color=PALETTE["teal"])
    limits = [min(calibration.Predicted.min(), calibration.Actual.min()) / 1e6, max(calibration.Predicted.max(), calibration.Actual.max()) / 1e6]
    axes[0, 1].plot(limits, limits, "--", color=PALETTE["muted"], linewidth=1)
    axes[0, 1].set_title("Calibration by prediction decile", loc="left", fontsize=12)
    axes[0, 1].set_xlabel("Median prediction (NT$M)"); axes[0, 1].set_ylabel("Median actual price (NT$M)")
    sorted_error = np.sort(error)[::-1]; share = np.cumsum(sorted_error) / max(sorted_error.sum(), np.finfo(float).eps) * 100
    ranks = np.arange(1, len(error) + 1) / len(error) * 100
    axes[1, 0].plot(ranks, share, color=PALETTE["navy"], linewidth=2)
    axes[1, 0].plot([0, 100], [0, 100], "--", color=PALETTE["grid"])
    axes[1, 0].set_xlim(0, 100); axes[1, 0].set_ylim(0, 100)
    axes[1, 0].set_title("Error concentration", loc="left", fontsize=12)
    axes[1, 0].set_xlabel("Transactions sorted by largest error (%)"); axes[1, 0].set_ylabel("Cumulative absolute error (%)")
    index = min(len(error) - 1, max(0, math.ceil(.1 * len(error)) - 1))
    axes[1, 0].annotate(f"Top 10% → {share[index]:.1f}% of error", (ranks[index], share[index]), xytext=(24, -7),
                         textcoords="offset points", color=PALETTE["navy"], fontsize=9)
    axis = axes[1, 1]
    if meta is not None and BUILDING in meta:
        if len(meta) != len(good): raise ValueError("Metadata must have the original evaluation row order")
        area = _numeric(meta[BUILDING])[good]
        categories = pd.cut(area, [-np.inf, 100, 200, 300, 600, np.inf], labels=["≤100", "100–200", "200–300", "300–600", ">600"])
        segments = pd.DataFrame({"Band": categories, "Error": error}).groupby("Band", observed=True).agg(Rows=("Error", "size"), MAE=("Error", "mean")).reset_index()
        axis.barh(segments.Band.astype(str), segments.MAE / 1e6, color=PALETTE["coral"])
        axis.set_xlabel("MAE (NT$M)"); axis.set_ylabel("Building area (m²)")
        axis.set_title("Error by area band", loc="left", fontsize=12)
        for pos, row in enumerate(segments.itertuples()):
            axis.text(row.MAE / 1e6, pos, f"  n={row.Rows:,}", va="center", fontsize=8, color=PALETTE["muted"])
        axis.margins(x=.25); _save_table(segments, out_dir, "residual_area_segments")
    else:
        axis.set_axis_off()
        stats = [("MAE", f"NT${error.mean():,.0f}"), ("Median residual", f"NT${np.median(residual):+,.0f}"),
                 ("Underprediction", f"{100 * (residual > 0).mean():.1f}%"), ("Evaluation rows", f"{len(y):,}")]
        for i, (label, value) in enumerate(stats):
            axis.text(0, .95 - i * .23, label, transform=axis.transAxes, fontsize=10, color=PALETTE["muted"])
            axis.text(0, .85 - i * .23, value, transform=axis.transAxes, fontsize=17, fontweight="bold")
    _footnote(fig, "All error metrics use the full evaluation population. Only the residual histogram display is restricted to P1–P99; no records are removed from MAE.")
    _save_table(calibration, out_dir, "residual_calibration_deciles")
    return save_chart(fig, out_dir, "06_residual_diagnostics")


def chart_shap_summary(values, feature_frame, out_dir=None, scale=1., stem="07_model_contribution_atlas"):
    """Native CatBoost SHAP input only; no external SHAP dependency or fitting.

    ``values`` must contain one SHAP contribution column per feature, optionally
    followed by the expected-value column. ``scale`` converts model output to
    TWD. Feature values color numeric inputs; categories use neutral navy to
    avoid imposing arbitrary numerical order on nominal categories.
    """
    apply_theme(); arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != len(feature_frame) or arr.shape[1] not in {feature_frame.shape[1], feature_frame.shape[1] + 1}:
        raise ValueError("SHAP matrix must align with feature-frame rows and exact feature order")
    arr = arr[:, :feature_frame.shape[1]] * float(scale)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("Native SHAP contributions must be nonempty and finite")
    order = np.argsort(np.mean(np.abs(arr), axis=0))[::-1]
    total = len(order); half = math.ceil(total / 2); height = max(8, .3 * half + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(17.2, height))
    fig.subplots_adjust(left=.22, right=.96, top=1 - 1.15 / height, bottom=.12, wspace=.65)
    _heading(fig, "Model contribution atlas", "All direct model inputs · native CatBoost SHAP · signed contributions explain the model, not causal market premiums.")
    rng = np.random.default_rng(42)
    cmap = LinearSegmentedColormap.from_list("shapvalues", [PALETTE["teal"], "#D6DEE1", PALETTE["coral"]])
    records = []
    for h, axis in enumerate(axes):
        selected = order[h * half:(h + 1) * half]
        for pos, column in enumerate(selected):
            effect = arr[:, column] / 1e6
            # Symmetric, deterministic density jitter in fixed x bins.
            if len(effect) > 1500:
                sample = np.sort(rng.choice(len(effect), 1500, replace=False))
            else: sample = np.arange(len(effect))
            ex = effect[sample]; spread = np.nanmax(ex) - np.nanmin(ex) if len(ex) else 0
            bucket = np.floor((ex - np.nanmin(ex)) / max(spread, 1e-12) * 60).astype(int) if len(ex) else np.array([], dtype=int)
            jitter = np.zeros(len(ex))
            for b in np.unique(bucket):
                ix = np.flatnonzero(bucket == b); r = rng.permutation(len(ix))
                jitter[ix] = ((r // 2 + 1) * np.where(r % 2, 1, -1)) / max(1, len(ix)) * .65
            series = feature_frame.iloc[:, column]
            if pd.api.types.is_numeric_dtype(series.dtype):
                val = _numeric(series)[sample]; finite = val[np.isfinite(val)]
                lower, upper = np.quantile(finite, [.05, .95]) if len(finite) else (0, 1)
                col = np.clip((val - lower) / (upper - lower), 0, 1) if upper > lower else np.full(len(val), .5)
                col = np.where(np.isfinite(val), col, .5)
                axis.scatter(ex, pos + jitter, c=col, cmap=cmap, vmin=0, vmax=1, s=4, alpha=.5, rasterized=True, linewidths=0)
            else:
                axis.scatter(ex, pos + jitter, color=PALETTE["navy"], s=4, alpha=.36, rasterized=True, linewidths=0)
            records.append({"Feature": feature_frame.columns[column], "Mean_Abs_SHAP_TWD": np.mean(np.abs(arr[:, column])), "Rank": h * half + pos + 1})
        axis.axvline(0, color=PALETTE["muted"], linewidth=.8)
        axis.set_yticks(range(len(selected)), [feature_label(feature_frame.columns[i]) for i in selected], fontsize=8.5)
        axis.set_ylim(len(selected) - .4, -.6); axis.set_xlabel("Contribution to prediction (NT$M)"); axis.grid(axis="y", alpha=.3)
    _footnote(fig, "Numeric feature color: teal = low, coral = high. Nominal categories are neutral. Maximum 1,500 points per feature displayed; ranks use all supplied rows.")
    _save_table(pd.DataFrame(records).sort_values("Rank"), out_dir, "native_shap_all_features")
    return save_chart(fig, out_dir, stem)


def build_eda_gallery(train, frames=None, raw_features=None, out_dir=None, show=True):
    """Three integrated topic figures, plus complete attribute statistics in CSV."""
    figures = []
    if frames:
        figures.append(chart_quality_overview(frames, out_dir=out_dir, features=raw_features))
    figures.extend([chart_area_time_context(train, out_dir=out_dir), chart_interaction_atlas(train, out_dir=out_dir)])
    audit, details = attribute_audit(train, features=raw_features)
    _save_table(audit, out_dir, "all_attribute_audit"); _save_table(details, out_dir, "all_attribute_group_details")
    if show: display_charts(figures)
    return {"figures": figures, "attribute_audit": audit, "attribute_details": details}


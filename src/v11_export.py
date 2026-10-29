"""Inference-only adaptive correction. Never fit a predictive model."""
from pathlib import Path
import hashlib
import json
import warnings
import zipfile
import numpy as np
import pandas as pd

POLICIES = ("Current", "Boundary50", "Boundary75", "Boundary100",
            "Confidence50", "Confidence75", "Confidence100")
IMPLEMENTATION = "v10_adaptive_no_fit_1"


def review_zip(directory):
    path=directory/"adaptive_review.zip"
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(directory.iterdir()):
            if file.suffix in {".json",".csv"} and not file.name.startswith("submission_"):
                archive.write(file,file.name)
    return str(path)


def adaptive_prediction(baseline, prices, eligible, policy, config, current_weight=.25):
    baseline = np.array(baseline, dtype=float, copy=True)
    lower, median, upper = np.asarray(prices, float).T
    finite = np.isfinite(prices).all(axis=1) & np.isfinite(baseline)
    below, above = baseline < lower, baseline > upper
    active = np.asarray(eligible, bool) & finite & (below | above)
    boundary = np.where(below, lower, np.where(above, upper, baseline))
    median_delta, boundary_delta = median-baseline, boundary-baseline
    maximum = np.minimum(config.maximum_change_twd, config.maximum_relative_change*baseline)
    strength = np.divide(np.abs(boundary_delta), np.abs(median_delta),
        out=np.zeros(len(baseline)), where=np.isfinite(median_delta) & (np.abs(median_delta) > 1e-8))
    strength = np.clip(strength, 0, 1)
    if policy == "Current":
        delta = current_weight*np.clip(median_delta, -maximum, maximum)
    elif policy.startswith("Boundary"):
        weight = float(policy.removeprefix("Boundary"))/100
        delta = weight*np.clip(boundary_delta, -maximum, maximum)
    elif policy.startswith("Confidence"):
        peak = float(policy.removeprefix("Confidence"))/100
        weight = current_weight + (peak-current_weight)*strength
        delta = weight*np.clip(median_delta, -maximum, maximum)
    else:
        raise ValueError("Unknown fixed policy: " + str(policy))
    result = baseline.copy()
    result[active] += delta[active]
    result = np.maximum(result, 0)
    np.testing.assert_array_equal(result[~active], baseline[~active])
    return result, active & (np.abs(result-baseline) > 1e-6)


def protected(metric, changed, config, maximum_increase):
    return bool(metric["Accurate_Increase_Pct"] <= maximum_increase and
        changed["Accurate_Increase_Pct"] <= maximum_increase and
        metric["New_Damaged"] <= max(1, config.maximum_damaged_fraction*metric["Accurate_Rows"]))


def select_june(e, prices, eligible, current_weight, module):
    month = module.numbers(e.query, module.MONTH)
    tune = month == e.config.tune_month
    rows, choices = [], []
    for policy in POLICIES:
        candidate, changed = adaptive_prediction(e.baseline, prices, eligible, policy, e.config, current_weight)
        metric = module.score(e.actual, e.baseline, candidate, tune, "June", "All")
        subset = module.score(e.actual, e.baseline, candidate, tune & changed, "June", "Changed")
        passed = protected(metric, subset, e.config, e.config.tune_accurate_increase_pct)
        rows.append({"Policy": policy, "Protection_Passed": passed, **metric})
        if passed:
            choices.append((metric["Candidate_MAE"], POLICIES.index(policy), policy))
    if not choices:
        return "Current", pd.DataFrame(rows)
    return min(choices)[2], pd.DataFrame(rows)


def run_adaptive_export(project_root, threads=8):
    import v10_candidate as module
    from v10_base import forbid_ml_fit
    from IPython.display import display
    # Any missing/incompatible cache is fatal: this guard also covers the
    # original method that is capable of fitting when run without the guard.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="DataFrame is highly fragmented.*", category=pd.errors.PerformanceWarning)
        with forbid_ml_fit():
            e = module.V10Experiment(project_root, module.V10Config(threads=threads))
            display(e.prepare())
            if not (e.folder/"model.cbm").is_file() or not (e.folder/"model.json").is_file():
                raise FileNotFoundError("Completed V10 model required. No new training is permitted.")
            result = e.train_once()
            if result["New_Model_Fits_This_Run"] != 0:
                raise AssertionError("Inference-only export unexpectedly fitted a model")
            return _run_loaded(e, module)


def _run_loaded(e, module):
    from IPython.display import display
    c = e.config
    selection_path = e.folder/"locked_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Complete the original V10 validation before running this export")
    old = json.loads(selection_path.read_text())
    model_hash = module.study.hash_file(e.folder/"model.cbm")
    if old["model_sha256"] != model_hash:
        raise ValueError("V10 model differs from the previous locked selection")
    current_weight = float(old["weight"])
    prices, _ = module.quantile_prices(module.numbers(e.detail,"Raw_Base"), e.area,
        e.model.predict(e.x, thread_count=c.threads))
    eligible = e.eligible & e.support_mask & (module.numbers(e.detail,"Forecast_Gap") > 0)
    policy, selection = select_june(e, prices, eligible, current_weight, module)
    identity = {"implementation": IMPLEMENTATION, "policy": policy, "policy_grid": POLICIES,
        "current_weight": current_weight, "model_sha256": model_hash, "configuration": module.asdict(c)}
    directory = e.folder/"adaptive_export"/module.digest(identity)[:12]
    directory.mkdir(parents=True, exist_ok=True)
    selection.to_csv(directory/"june_selection.csv",index=False)
    # Freeze before evaluating Check. Check cannot select another policy.
    module.save_json(directory/"locked_policy.json",identity)
    print("June-only policy selection")
    display(selection[["Policy","Candidate_MAE","MAE_Reduction","Accurate_Increase_Pct",
        "Changed_Accurate_Increase_Pct","New_Damaged","Protection_Passed"]].round(3))
    current, _ = adaptive_prediction(e.baseline,prices,eligible,"Current",c,current_weight)
    candidate, active = adaptive_prediction(e.baseline,prices,eligible,policy,c,current_weight)
    month = module.numbers(e.query,module.MONTH)
    rows=[]
    for label,mask in (("June",month==c.tune_month),("July_August",month>c.tune_month)):
        for name, values in (("Current_V10",current),("Adaptive_Candidate",candidate)):
            item=module.score(e.actual,e.baseline,values,mask,label,name)
            item["Gain_vs_Current_V10"] = float(np.mean(np.abs(e.actual[mask]-current[mask]))-item["Candidate_MAE"])
            rows.append(item)
    report=pd.DataFrame(rows)
    report.to_csv(directory/"validation.csv",index=False)
    print("Frozen policy; retrospective comparison")
    display(report[["Set","Segment","Candidate_MAE","Gain_vs_Current_V10","Accurate_Increase_Pct","New_Damaged"]].round(3))
    june = report.loc[(report.Set=="June") & (report.Segment=="Adaptive_Candidate")].iloc[0]
    check = report.loc[(report.Set=="July_August") & (report.Segment=="Adaptive_Candidate")].iloc[0]
    changed_check=module.score(e.actual,e.baseline,candidate,(month>c.tune_month)&active,"Check","Changed")
    protection=protected(check,changed_check,c,c.check_accurate_increase_pct)
    recommended=bool(policy!="Current" and june.Gain_vs_Current_V10>=10_000 and
                     check.Gain_vs_Current_V10>=10_000 and protection)
    decision={**identity,"New_Model_Fits":0,"Recommended":recommended,
        "June_Gain_vs_Current_V10":float(june.Gain_vs_Current_V10),
        "Check_Gain_vs_Current_V10":float(check.Gain_vs_Current_V10),
        "Check_Protection_Passed":protection,"Retrospective_Check":True,
        "Test_Answers_Read":False,"Kaggle_Uploaded":False}
    module.save_json(directory/"decision.json",decision)
    if not recommended:
        result={"Exported":False,**decision,
            "Reason":"No June-selected adaptive policy passed the additional retrospective improvement/protection checks; preserve current V10"}
        module.save_json(directory/"export.json",result)
        result["Review_Package"]=review_zip(directory)
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
        return result
    # Only inference below. No Test labels are read and no parameter changes.
    info=e.manifest["Test"];path=Path(info["file"])
    if module.study.hash_file(path)!=info["sha256"]:
        raise ValueError("Original unlabelled Test changed")
    module.check_unlabelled(pd.read_excel(path,nrows=0))
    raw=pd.read_excel(path);module.check_unlabelled(raw)
    if not module.unique_ids(raw).all():raise ValueError("Test IDs must be unique")
    original_path=e.root/"outputs/submission_v8/submission_v8.csv"
    original_hash=module.study.hash_file(original_path)
    v8=pd.read_csv(original_path,dtype={module.ID:"string"})
    v8[module.ID]=v8[module.ID].astype(str)
    if not v8[module.ID].is_unique or set(v8[module.ID])!=set(raw[module.ID].astype(str)):
        raise ValueError("V8/Test IDs differ")
    baseline=module.numbers(v8.set_index(module.ID).loc[raw[module.ID].astype(str)],module.TARGET)
    predicted=e.base.predict(raw,"production")
    frame,detail=predicted["frame"],predicted["detail"]
    market=module.market_table(e.train,int(np.nanmax(module.numbers(e.train,module.MONTH))))
    x=module.model_features(frame,detail,market)
    if list(x)!=list(e.model.feature_names_):raise ValueError("Production schema mismatch")
    qp,_=module.quantile_prices(module.numbers(detail,"Raw_Base"),module.numbers(frame,module.AREA),
        e.model.predict(x,thread_count=c.threads))
    key=frame[e.stratum_columns].astype(str).agg("|".join,axis=1)
    support=key.map(e.support).fillna(0).to_numpy(float)>=c.minimum_stratum_rows
    gap=module.numbers(detail,"Forecast_Gap");locked=e._airport_locked(raw)
    usable=module.housing_mask(frame)&support&~locked&(gap>=1)&(gap<=9)
    replay,_=adaptive_prediction(baseline,qp,usable,"Current",c,current_weight)
    previous_path=e.folder/"submission_v10_candidate.csv"
    previous=pd.read_csv(previous_path,dtype={module.ID:"string"})
    previous[module.ID]=previous[module.ID].astype(str)
    if not previous[module.ID].is_unique or set(previous[module.ID])!=set(raw[module.ID].astype(str)):
        raise ValueError("Previous V10/Test IDs differ")
    saved=module.numbers(previous.set_index(module.ID).loc[raw[module.ID].astype(str)],module.TARGET)
    np.testing.assert_allclose(replay,saved,rtol=1e-11,atol=.02)
    final,active=adaptive_prediction(baseline,qp,usable,policy,c,current_weight)
    np.testing.assert_array_equal(final[~usable|locked],baseline[~usable|locked])
    if not np.any(np.abs(final-saved)>.01):raise ValueError("No changed predictions; duplicate submission suppressed")
    from project import write_submission
    submission=write_submission(directory/"submission_v10_adaptive.csv",raw,final)
    if module.study.hash_file(original_path)!=original_hash:
        raise AssertionError("V8 was modified")
    decision.update(Exported=True,Submission=str(submission),Changed_From_Current=int((abs(final-saved)>.01).sum()),
        SHA256=module.study.hash_file(submission),Airport_Locked_Rows=int(locked.sum()))
    module.save_json(directory/"export.json",decision)
    summary=pd.DataFrame({"Year":frame["交易年"],"Changed":abs(final-saved)>.01,"Delta":final-saved}).groupby("Year").agg(
        Rows=("Changed","size"),Changed_Pct=("Changed",lambda s:100*s.mean()),
        Mean_Abs_Change=("Delta",lambda s:s.abs().mean()))
    summary.to_csv(directory/"test_changes.csv")
    decision["Review_Package"]=review_zip(directory)
    display(summary.round(3));print(json.dumps(decision,ensure_ascii=False,indent=2,default=str))
    return decision
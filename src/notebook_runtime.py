"""Readable notebook orchestration. Importing this module never starts a fit."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd

import study_pipeline as study
import project


REGISTRY = {
    1: {"added": "Area and transaction time", "reason": "Establish a reproducible size-and-time baseline"},
    2: {"added": "City, district, road and parcel section", "reason": "The same area can have different prices in different locations"},
    3: {"added": "Type, use, zoning, material and location interactions", "reason": "Residential and industrial prices follow different location relationships"},
    4: {"added": "Floors, layout, parking, counts and explicit area ratios", "reason": "Separate physical scale from transaction composition"},
    5: {"added": "Project, block and project-location/type interactions", "reason": "Project identity can explain variation within a district"},
    6: {"added": "Remarks, management, date detail and missingness", "reason": "Account for transaction semantics and information availability"},
    7: {"added": "Fixed reference and comparison ensemble", "reason": "Reproduce the complete evaluated system rather than a feature-only ablation"},
    8: {"added": "Published Airport resettlement unit schedule", "reason": "Use an explicit public-price rule only when unit identity and area agree"},
}


class StudySession:
    def __init__(self, root, raw, prepared, split, config, full_ablations=True):
        self.root, self.raw, self.prepared, self.split = Path(root), raw, prepared, split
        self.config, self.full_ablations = config, bool(full_ablations)
        self.stages, self.score_parts, self.valid_parts, self.submissions = {}, [], [], {}
        self.predictions, self.best = {}, None
        self._best_complete, self._public_rule_complete = False, False
        self.known_projects = set(split["fit"].loc[
            split["fit"]["建案名稱"].ne("缺失"), "建案行政區"].astype(str))
        self.baseline_median = float(np.median(study.num(split["fit"][study.TARGET])))
        project.write_json(self.root / "configs/version_plan.json", REGISTRY)

    def version_folder(self, version):
        return self.root / "outputs" / f"submission_v{version}"

    def _persist_scores(self):
        if self.score_parts:
            self.scores.to_csv(self.root / "presentation/tables/version_metrics.csv", index=False)
        if self.valid_parts:
            pd.concat(self.valid_parts, ignore_index=True).to_csv(
                self.root / "presentation/tables/provided_valid_metrics.csv", index=False)

    @staticmethod
    def _upsert_scores(parts, new_scores):
        """Cell retries replace the same version/set, never duplicate its metrics."""
        keys = set(zip(new_scores["Version"], new_scores["Set"]))
        retained = []
        for part in parts:
            keep = np.array([(v, label) not in keys for v, label in zip(part["Version"], part["Set"])])
            if keep.any():
                retained.append(part.loc[keep].copy())
        parts[:] = retained + [new_scores]

    @property
    def scores(self):
        return pd.concat(self.score_parts, ignore_index=True) if self.score_parts else pd.DataFrame()

    def load_stage_bundle(self, version):
        """Reload a verified native cache when an interpretation cell is rerun."""
        version = int(version)
        if version not in self.stages:
            self.run_stage(version)
        bundle = self.stages[version].get("bundle")
        if bundle is None:
            config = replace(self.config, cache_dir=self.version_folder(version) / "models/historical")
            result = study.fit_version(self.split["fit"], {}, version, config)
            bundle = result["bundle"]
        return bundle

    def run_stage(self, version):
        """One historical fit and, optionally, one full-Train fit per stage."""
        version = int(version)
        if not 1 <= version <= 6:
            raise ValueError("run_stage accepts versions 1 through 6; use run_best/run_public_rule for 7/8")
        if version in self.stages:
            return self.stages[version]["scores"]
        folder = self.version_folder(version)
        historical_config = replace(self.config, cache_dir=folder / "models/historical")
        result = study.fit_version(self.split["fit"],
            {"Dev": self.split["dev"], "Check": self.split["check"]}, version, historical_config)
        scores = result["scores"].copy()
        scores["Version"] = f"submission_v{version}"
        self._upsert_scores(self.score_parts, scores)
        for label, detail in result["predictions"].items():
            detail.to_csv(folder / f"predictions/{label.lower()}.csv", index=False)
            self.predictions[(version, label)] = detail.Prediction.to_numpy(float, copy=True)
        scores.to_csv(folder / "tables/metrics.csv", index=False)
        bundle = result["bundle"]
        schema = pd.DataFrame({"Feature": bundle["columns"],
            "Categorical": [i in bundle["categorical_indices"] for i in range(len(bundle["columns"]))]})
        schema.to_csv(folder / "tables/model_schema.csv", index=False)
        recipe = {"version": version, **REGISTRY[version], "parameters": bundle["parameters"],
            "feature_columns": bundle["columns"], "target_scale": bundle["scale"],
            "historical_fit_last_month": "2024-03", "development": "2024-04 to 2024-06",
            "check": "2024-07 to 2024-08", "full_train_export": self.full_ablations,
            "historical_cache": str(result["folder"]), "test_labels_used": False}
        if self.full_ablations:
            full_config = replace(self.config, cache_dir=folder / "models/full")
            full_result = study.fit_version(self.prepared["train"], {}, version, full_config)
            full_bundle = full_result["bundle"]
            prediction = study.predict_version(full_bundle, self.prepared["test"])
            path = project.write_submission(folder / f"submission_v{version}.csv", self.raw["Test"], prediction)
            self.submissions[version] = str(path)
            valid_prediction = study.predict_version(full_bundle, self.prepared["valid"].drop(columns=[study.TARGET]))
            self._record_valid(version, valid_prediction, full_bundle["known_projects"])
            recipe["full_cache"] = str(full_result["folder"])
            del full_result, full_bundle
            gc.collect()
        project.write_json(folder / "recipe.json", recipe)
        self.stages[version] = {"bundle": bundle if version == 6 else None, "scores": scores}
        if version != 6:
            del result, bundle
            gc.collect()
        self._persist_scores()
        return scores

    def _record_valid(self, version, prediction, known):
        score = study.metrics(self.prepared["valid"], prediction, known, version, "Provided_Valid",
                              float(np.median(study.num(self.prepared["train"][study.TARGET]))))
        score["Version"] = f"submission_v{version}"
        self._upsert_scores(self.valid_parts, score)
        frame = self.prepared["valid"][[study.ID, study.TARGET]].copy()
        frame["Prediction"] = prediction
        frame.to_csv(self.version_folder(version) / "predictions/provided_valid.csv", index=False)

    def _record_best_evaluation(self, version, label, frame, prediction, detail):
        scores = study.metrics(frame, prediction, self.known_projects, version, label, self.baseline_median)
        scores["Version"] = f"submission_v{version}"
        self._upsert_scores(self.score_parts, scores)
        self.predictions[(version, label)] = np.array(prediction, float, copy=True)
        detail = detail.copy()
        detail["Actual"] = study.num(frame[study.TARGET])
        detail["Prediction"] = prediction
        detail.to_csv(self.version_folder(version) / f"predictions/{label.lower()}.csv", index=False)

    def run_best(self):
        """Fit/load 14 reference states and 8 comparison states; no ablation rerun."""
        if self._best_complete:
            return self.scores.loc[self.scores.Version.eq("submission_v7")]
        if self.best is None:
            self.best = project.BestRecipeRunner(self.root, self.config.threads).fit(self.raw["Train"])
        for label, key in [("Dev", "dev"), ("Check", "check")]:
            frame = self.split[key]
            detail = self.best.predict_validation(frame)
            self._record_best_evaluation(7, label, frame, detail.Prediction.to_numpy(float), detail)
        detail = self.best.predict_raw(self.raw["Test"])
        self.test_v7 = detail.Prediction.to_numpy(float, copy=True)
        component_columns = [c for c in detail if pd.api.types.is_numeric_dtype(detail[c])]
        detail[component_columns].describe(percentiles=[.5, .95, .99]).to_csv(
            self.version_folder(7) / "tables/test_component_summary.csv")
        path = project.write_submission(self.version_folder(7) / "submission_v7.csv", self.raw["Test"], self.test_v7)
        self.submissions[7] = str(path)
        valid_prediction = self.best.predict_raw(self.raw["Valid"].drop(columns=[study.TARGET])).Prediction.to_numpy(float)
        self.valid_v7 = valid_prediction
        self._record_valid(7, valid_prediction, set(self.best.try_production.known_projects))
        scores = self.scores.loc[self.scores.Version.eq("submission_v7")]
        scores.to_csv(self.version_folder(7) / "tables/metrics.csv", index=False)
        self._persist_scores()
        self._best_complete = True
        return scores

    def _raw_train_subset(self, frame):
        raw = self.raw["Train"]
        ids = frame[study.ID].astype(str)
        selected = raw.loc[raw[study.ID].astype(str).isin(ids)].copy()
        selected[study.ID] = selected[study.ID].astype(str)
        if not selected[study.ID].is_unique or set(selected[study.ID]) != set(ids):
            raise ValueError("Historical raw features cannot be aligned uniquely")
        return selected.set_index(study.ID, drop=False).loc[ids].reset_index(drop=True).drop(columns=[study.TARGET])

    def run_public_rule(self):
        from airport_policy import apply_airport_allocation, aa_official_table, AA_AREA_TOLERANCE_SQM
        if self._public_rule_complete:
            return self.scores.loc[self.scores.Version.eq("submission_v8")]
        if not self._best_complete:
            raise RuntimeError("Run submission_v7 first")
        folder = self.version_folder(8)
        for label, key in [("Dev", "dev"), ("Check", "check")]:
            frame = self.split[key]
            prediction, audit = apply_airport_allocation(self._raw_train_subset(frame), self.predictions[(7, label)])
            audit.to_csv(folder / f"tables/public_rule_{label.lower()}.csv", index=False)
            detail = frame[[study.ID]].copy()
            self._record_best_evaluation(8, label, frame, prediction, detail)
        prediction, audit = apply_airport_allocation(self.raw["Test"], self.test_v7)
        self.test_v8, self.public_audit = prediction, audit
        audit.to_csv(folder / "tables/public_rule_audit.csv", index=False)
        aa_official_table().to_csv(folder / "tables/public_unit_schedule.csv", index=False)
        valid_prediction, valid_audit = apply_airport_allocation(
            self.raw["Valid"].drop(columns=[study.TARGET]), self.valid_v7)
        valid_audit.to_csv(folder / "tables/public_rule_provided_valid.csv", index=False)
        self._record_valid(8, valid_prediction, set(self.best.try_production.known_projects))
        path = project.write_submission(folder / "submission_v8.csv", self.raw["Test"], prediction)
        self.submissions[8] = str(path)
        project.write_json(folder / "recipe.json", {
            "version": 8, **REGISTRY[8], "new_model_fits": 0, "input_version": 7,
            "source": "Embedded public CAA and city unit schedules, with source date guards",
            "area_tolerance_sqm": AA_AREA_TOLERANCE_SQM,
            "parking_midpoint_twd": 1050000, "parking_range_twd": [850000, 1250000],
            "test_labels_used": False, "prediction_reference_used_for_training": False})
        self.parity = project.best_parity(self.raw["Test"], prediction,
            self.root / "references/current_best_predictions.csv", folder)
        scores = self.scores.loc[self.scores.Version.eq("submission_v8")]
        scores.to_csv(folder / "tables/metrics.csv", index=False)
        self._persist_scores()
        self._public_rule_complete = True
        return scores

    def finish(self):
        evidence = project.transition_evidence(self.scores, REGISTRY)
        evidence.to_csv(self.root / "presentation/tables/transition_evidence.csv", index=False)
        for version in range(1, 9):
            evidence.loc[evidence.Version.eq(f"submission_v{version}")].to_csv(
                self.version_folder(version) / "tables/transition_evidence.csv", index=False)
        index = pd.DataFrame([{"Version": f"submission_v{v}", "Submission": p,
            "SHA256": project.file_hash(p), "New_Kaggle_MAE": np.nan} for v, p in sorted(self.submissions.items())])
        index.to_csv(self.root / "presentation/tables/submission_index.csv", index=False)
        notes = ["# Analysis Notes", "", "本次尚未填寫主觀結果結論；以下數值為執行後的自動紀錄。", "",
            "V1–V6 的參數與歷史評估列固定；V7 同時改變元件與校準，不能將全部差異解釋成特徵效果。",
            "V8 為公開單位價格規則，受來源日期、戶別、樓層與面積一致性限制。", "",
            "單欄位估計、來源置換與 SHAP 都是預測解釋，不是因果價格效果。", "",
            "## Measured Transitions", ""]
        for row in evidence.to_dict("records"):
            notes.append(f"{row['Version']}: Dev MAE={row.get('Dev_MAE', np.nan):,.3f}; "
                         f"Check MAE={row.get('Check_MAE', np.nan):,.3f}。中文原因與採用結論待檢視完整分群後補入。\n")
        notes += ["## Pending Interpretation", "", "請回傳執行後的 notebook 與 study_review.zip，以補上圖表判讀、"
            "退步原因、版本選擇與簡報結論。已知 1,918,506 僅屬於凍結參考檔；新訓練檔以重現檢查與實際提交為準。"]
        (self.root / "presentation/analysis_notes.md").write_text("\n".join(notes), encoding="utf-8")
        project.write_json(self.root / "configs/completion.json", {
            "submissions": self.submissions, "versions_evaluated": sorted(self.scores.Version.unique()),
            "automatic_kaggle_upload": False, "test_target_loaded": False})
        return evidence, index, project.export_review_bundle(self.root)

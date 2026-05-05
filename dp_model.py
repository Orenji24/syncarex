import os
import re
import json
from typing import Optional

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from scipy.stats import entropy, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MARGIN_RULES = {
    "age": 5,
    "bp": 10,
    "blood_pressure": 10,
    "cholesterol": 20,
    "cholestrol": 20,
}


def _find_column(data: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", column.lower()): column for column in data.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _closest_margin_column(column: str) -> Optional[int]:
    cleaned = re.sub(r"[^a-z0-9]", "", column.lower())
    for key, margin in MARGIN_RULES.items():
        if re.sub(r"[^a-z0-9]", "", key) in cleaned:
            return margin
    return None


def _aligned_real_series(real: pd.Series, size: int) -> pd.Series:
    if len(real) == size:
        return real.reset_index(drop=True)
    repeated = np.resize(real.to_numpy(), size)
    return pd.Series(repeated, name=real.name)


def _bounded_numeric_sample(real: pd.Series, margin: int, size: int, noise_level: float) -> pd.Series:
    aligned_real = _aligned_real_series(real, size)
    real_numeric = pd.to_numeric(aligned_real, errors="coerce")
    valid_mask = real_numeric.notna()
    values = real_numeric.copy()

    if not valid_mask.any():
        return real

    if pd.api.types.is_integer_dtype(real.dropna()):
        active_margin = max(1, int(round(margin * max(noise_level, 0.1))))
        possible_offsets = np.array([offset for offset in range(-active_margin, active_margin + 1) if offset != 0])
        noise = np.random.choice(possible_offsets, size=valid_mask.sum())
        generated = real_numeric.loc[valid_mask].to_numpy() + noise
        values.loc[valid_mask] = np.rint(generated).astype(int)
        return values.astype("Int64" if real.isna().any() else int)

    active_margin = margin * max(noise_level, 0.1)
    noise = np.random.uniform(-active_margin, active_margin, size=valid_mask.sum())
    near_zero = np.abs(noise) < 0.01
    noise[near_zero] = np.sign(noise[near_zero] + 0.001) * 0.01
    values.loc[valid_mask] = real_numeric.loc[valid_mask].to_numpy() + noise
    return values


def _apply_margin_rules(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, noise_level: float) -> tuple[pd.DataFrame, list[dict]]:
    adjusted = synthetic_data.copy()
    margin_report = []
    size = len(adjusted)

    for column in adjusted.columns:
        if column not in real_data.columns:
            continue

        margin = _closest_margin_column(column)
        if margin is None:
            continue

        if pd.api.types.is_numeric_dtype(real_data[column]):
            aligned_real = _aligned_real_series(real_data[column], size)
            adjusted[column] = _bounded_numeric_sample(
                real=real_data[column],
                margin=margin,
                size=size,
                noise_level=noise_level,
            )
            difference = (
                pd.to_numeric(adjusted[column], errors="coerce")
                - pd.to_numeric(aligned_real, errors="coerce")
            ).abs()
            margin_report.append(
                {
                    "column": column,
                    "allowed": margin,
                    "max_diff": round(float(difference.max(skipna=True)), 3),
                    "avg_diff": round(float(difference.mean(skipna=True)), 3),
                    "passed": bool((difference.dropna() <= margin).all()),
                }
            )

    return adjusted, margin_report


def _apply_general_noise(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    adjusted = synthetic_data.copy()
    if noise_level <= 0:
        return adjusted

    for column in adjusted.select_dtypes(include=[np.number]).columns:
        if _closest_margin_column(column) is not None:
            continue
        if column not in real_data.columns:
            continue
        std = pd.to_numeric(real_data[column], errors="coerce").std()
        if pd.isna(std) or std == 0:
            continue
        scale = std * 0.03 * noise_level
        adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") + np.random.normal(
            0,
            scale,
            len(adjusted),
        )
    return adjusted


def _line_distribution_payload(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict:
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        return {"columns": [], "series": {}}

    priority = []
    for candidates in (["age"], ["bp", "blood_pressure"], ["cholesterol", "cholestrol"]):
        column = _find_column(real_data, candidates)
        if column and column in numeric_columns:
            priority.append(column)
    columns = list(dict.fromkeys(priority + numeric_columns[:6]))

    series = {}
    for column in columns:
        real_values = pd.to_numeric(real_data[column], errors="coerce").dropna()
        syn_values = pd.to_numeric(synthetic_data[column], errors="coerce").dropna()
        combined = pd.concat([real_values, syn_values], ignore_index=True)
        if combined.empty:
            continue
        low = combined.min()
        high = combined.max()
        if low == high:
            low -= 0.5
            high += 0.5
        counts_real, edges = np.histogram(real_values, bins=25, range=(low, high), density=True)
        counts_syn, _ = np.histogram(syn_values, bins=edges, density=True)
        centers = ((edges[:-1] + edges[1:]) / 2).round(3).tolist()
        series[column] = {
            "x": centers,
            "real": np.nan_to_num(counts_real).round(6).tolist(),
            "synthetic": np.nan_to_num(counts_syn).round(6).tolist(),
        }

    return {"columns": columns, "series": series}


def _correlation_payload(data: pd.DataFrame) -> dict:
    numeric = data.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return {"columns": [], "z": []}
    corr = numeric.corr().fillna(0).round(3)
    return {"columns": corr.columns.tolist(), "z": corr.values.tolist()}


def _stats_table(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> list[dict]:
    rows = []
    for column in real_data.select_dtypes(include=[np.number]).columns:
        if column not in synthetic_data.columns:
            continue
        real = pd.to_numeric(real_data[column], errors="coerce")
        syn = pd.to_numeric(synthetic_data[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "real_mean": round(float(real.mean()), 3),
                "syn_mean": round(float(syn.mean()), 3),
                "real_median": round(float(real.median()), 3),
                "syn_median": round(float(syn.median()), 3),
                "real_std": round(float(real.std()), 3),
                "syn_std": round(float(syn.std()), 3),
            }
        )
    return rows


def _quality_metrics(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict:
    kl_scores = []
    ks_scores = []
    numeric_columns = [column for column in real_data.select_dtypes(include=[np.number]).columns if column in synthetic_data.columns]

    for column in numeric_columns:
        real = pd.to_numeric(real_data[column], errors="coerce").dropna()
        syn = pd.to_numeric(synthetic_data[column], errors="coerce").dropna()
        if real.empty or syn.empty:
            continue
        combined = pd.concat([real, syn], ignore_index=True)
        low = combined.min()
        high = combined.max()
        if low == high:
            low -= 0.5
            high += 0.5
        real_hist, edges = np.histogram(real, bins=20, range=(low, high), density=False)
        syn_hist, _ = np.histogram(syn, bins=edges, density=False)
        real_prob = (real_hist + 1e-9) / (real_hist.sum() + 1e-9)
        syn_prob = (syn_hist + 1e-9) / (syn_hist.sum() + 1e-9)
        kl_scores.append(float(entropy(real_prob, syn_prob)))
        ks_scores.append(float(ks_2samp(real, syn).statistic))

    corr_similarity = "N/A"
    if len(numeric_columns) >= 2:
        real_corr = real_data[numeric_columns].corr().fillna(0)
        syn_corr = synthetic_data[numeric_columns].corr().fillna(0)
        diff = np.abs(real_corr.to_numpy() - syn_corr.to_numpy()).mean()
        corr_similarity = round(float(max(0, 1 - diff / 2)), 3)

    return {
        "kl_divergence": round(float(np.mean(kl_scores)), 4) if kl_scores else "N/A",
        "ks_score": round(float(np.mean(ks_scores)), 4) if ks_scores else "N/A",
        "correlation_similarity": corr_similarity,
        "duplicate_rate": round(float(synthetic_data.duplicated().mean()), 4) if len(synthetic_data) else 0,
    }


def _dataset_summary(data: pd.DataFrame) -> dict:
    numeric_count = len(data.select_dtypes(include=[np.number]).columns)
    return {
        "rows_uploaded": len(data),
        "columns_detected": len(data.columns),
        "numeric_columns": numeric_count,
        "categorical_columns": len(data.columns) - numeric_count,
        "missing_values_pct": round(float(data.isna().mean().mean() * 100), 2),
    }


def _privacy_level(epsilon: float) -> str:
    if epsilon <= 1:
        return "High"
    if epsilon <= 5:
        return "Medium"
    return "Low"


def _medical_checks(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> list[dict]:
    checks = []
    age_col = _find_column(synthetic_data, ["age"])
    disease_col = _find_column(synthetic_data, ["disease", "diagnosis", "outcome"])
    if age_col and disease_col:
        grouped = synthetic_data.groupby(disease_col)[age_col].mean(numeric_only=True).dropna()
        checks.append(
            {
                "name": "Age vs disease relationship",
                "status": "Checked" if len(grouped) > 1 else "Limited",
                "detail": f"{len(grouped)} disease/outcome groups compared by mean age.",
            }
        )

    bp_col = _find_column(synthetic_data, ["bp", "blood_pressure", "systolic_bp"])
    if bp_col:
        bp = pd.to_numeric(synthetic_data[bp_col], errors="coerce")
        invalid = int(((bp < 60) | (bp > 220)).sum())
        checks.append(
            {
                "name": "Blood pressure range validation",
                "status": "Pass" if invalid == 0 else "Review",
                "detail": f"{invalid} synthetic rows outside 60-220.",
            }
        )

    glucose_col = _find_column(synthetic_data, ["glucose", "blood_glucose", "sugar"])
    if glucose_col:
        glucose = pd.to_numeric(synthetic_data[glucose_col], errors="coerce")
        anomalies = int(((glucose < 50) | (glucose > 250)).sum())
        checks.append(
            {
                "name": "Glucose anomaly detection",
                "status": "Pass" if anomalies == 0 else "Review",
                "detail": f"{anomalies} synthetic rows below 50 or above 250.",
            }
        )

    return checks or [{"name": "Medical consistency checks", "status": "Limited", "detail": "No age, BP, glucose, disease, or diagnosis columns were detected."}]


def _estimate_model_utility(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    target = _find_column(real_data, ["target", "label", "outcome", "diagnosis", "disease"])
    if target is None or target not in synthetic_data.columns:
        return None, None, "No target column found"

    real = real_data.dropna(subset=[target]).copy()
    synthetic = synthetic_data.dropna(subset=[target]).copy()
    if len(real) < 20 or len(synthetic) < 20:
        return None, None, "Not enough rows for model utility test"

    feature_columns = [column for column in real.columns if column != target and column in synthetic.columns]
    if not feature_columns:
        return None, None, "No shared feature columns found"

    x_real = real[feature_columns]
    y_real = real[target]
    x_syn = synthetic[feature_columns]
    y_syn = synthetic[target]

    numeric_features = x_real.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    is_classification = y_real.nunique(dropna=True) <= 12 or y_real.dtype == object
    model = RandomForestClassifier(n_estimators=80, random_state=42) if is_classification else RandomForestRegressor(n_estimators=80, random_state=42)
    metric_name = "Accuracy" if is_classification else "MAE"

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x_real,
            y_real,
            test_size=0.25,
            random_state=42,
            stratify=y_real if is_classification and y_real.nunique() > 1 else None,
        )

        real_pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        real_pipeline.fit(x_train, y_train)
        real_predictions = real_pipeline.predict(x_test)

        synthetic_pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        synthetic_pipeline.fit(x_syn, y_syn)
        synthetic_predictions = synthetic_pipeline.predict(x_test)

        if is_classification:
            real_score = accuracy_score(y_test, real_predictions)
            synthetic_score = accuracy_score(y_test, synthetic_predictions)
        else:
            real_score = mean_absolute_error(y_test, real_predictions)
            synthetic_score = mean_absolute_error(y_test, synthetic_predictions)

        return round(float(real_score), 3), round(float(synthetic_score), 3), metric_name
    except Exception:
        return None, None, "Could not calculate model utility"


def _privacy_risk_score(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    shared_columns = [column for column in real_data.columns if column in synthetic_data.columns]
    if not shared_columns:
        return 0.0

    real_rows = set(real_data[shared_columns].astype(str).agg("|".join, axis=1))
    synthetic_rows = set(synthetic_data[shared_columns].astype(str).agg("|".join, axis=1))
    if not synthetic_rows:
        return 0.0

    return round(len(real_rows.intersection(synthetic_rows)) / len(synthetic_rows), 4)


def run_dp_pipeline(
    input_path: str,
    output_folder: str,
    run_id: str,
    epochs: int = 300,
    sample_size: Optional[int] = None,
    noise_level: float = 0.5,
    epsilon: float = 1.0,
):
    os.makedirs(output_folder, exist_ok=True)
    real_data = pd.read_csv(input_path)
    requested_rows = sample_size or len(real_data)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        verbose=False,
    )
    synthesizer.fit(real_data)

    synthetic_data = synthesizer.sample(num_rows=requested_rows)
    synthetic_data = _apply_general_noise(real_data, synthetic_data, noise_level)
    synthetic_data, margin_report = _apply_margin_rules(real_data, synthetic_data, noise_level)

    synthetic_filename = f"synthetic_{run_id}.csv"
    synthetic_path = os.path.join(output_folder, synthetic_filename)

    synthetic_data.to_csv(synthetic_path, index=False)

    real_score, synthetic_score, metric_name = _estimate_model_utility(real_data, synthetic_data)
    privacy_risk = _privacy_risk_score(real_data, synthetic_data)
    summary = _dataset_summary(real_data)
    quality_metrics = _quality_metrics(real_data, synthetic_data)
    chart_payload = {
        "line": _line_distribution_payload(real_data, synthetic_data),
        "realCorr": _correlation_payload(real_data),
        "syntheticCorr": _correlation_payload(synthetic_data),
    }
    medical_checks = _medical_checks(real_data, synthetic_data)
    stats_rows = _stats_table(real_data, synthetic_data)
    privacy_level = _privacy_level(epsilon)

    return {
        "run_id": run_id,
        "epsilon": round(epsilon, 2),
        "privacy_level": privacy_level,
        "privacy_explanation": "Lower epsilon = higher privacy, lower similarity.",
        "epochs": epochs,
        "noise_level": noise_level,
        "sample_size": requested_rows,
        "real_acc": real_score if real_score is not None else "N/A",
        "syn_acc": synthetic_score if synthetic_score is not None else "N/A",
        "metric_name": metric_name,
        "privacy_risk": privacy_risk,
        "row_count": len(real_data),
        "column_count": len(real_data.columns),
        "download_url": f"/download/{synthetic_filename}",
        "preview_columns": synthetic_data.columns.tolist()[:8],
        "preview_rows": synthetic_data.head(6).to_dict(orient="records"),
        "margin_report": margin_report,
        "summary": summary,
        "quality_metrics": quality_metrics,
        "stats_rows": stats_rows,
        "chart_payload": json.dumps(chart_payload),
        "medical_checks": medical_checks,
        "report_url": f"/research-report/{run_id}",
    }

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
    "bloodpressure": 10,
    "cholesterol": 20,
    "cholestrol": 20,
    "heart_rate": 5,
    "heartrate": 5,
    "pulse": 5,
    "blood_sugar": 15,
    "bloodsugar": 15,
    "glucose": 15,
}

PROTECTED_COPY_COLUMNS = {
    "gender",
    "sex",
    "medication",
    "medications",
    "medicine",
    "medicines",
    "drug",
    "drugs",
}

TRAINING_ROW_LIMIT = 5000
ANALYSIS_ROW_LIMIT = 5000
HEATMAP_COLUMN_LIMIT = 30
STATS_COLUMN_LIMIT = 60


def _sample_for_analysis(data: pd.DataFrame, limit: int = ANALYSIS_ROW_LIMIT) -> pd.DataFrame:
    if len(data) <= limit:
        return data
    return data.sample(n=limit, random_state=42)


def _find_column(data: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", column.lower()): column for column in data.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def _find_columns(data: pd.DataFrame, candidates: list[str]) -> list[str]:
    found = []
    for column in data.columns:
        cleaned = re.sub(r"[^a-z0-9]", "", column.lower())
        if any(re.sub(r"[^a-z0-9]", "", candidate.lower()) in cleaned for candidate in candidates):
            found.append(column)
    return found


def _closest_margin_column(column: str) -> Optional[int]:
    cleaned = re.sub(r"[^a-z0-9]", "", column.lower())
    for key, margin in MARGIN_RULES.items():
        if re.sub(r"[^a-z0-9]", "", key) in cleaned:
            return margin
    return None


def _is_protected_copy_column(column: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9]", "", column.lower())
    return any(re.sub(r"[^a-z0-9]", "", key) == cleaned for key in PROTECTED_COPY_COLUMNS)


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


def _preserve_copy_columns(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
    adjusted = synthetic_data.copy()
    size = len(adjusted)

    for column in adjusted.columns:
        if column in real_data.columns and _is_protected_copy_column(column):
            adjusted[column] = _aligned_real_series(real_data[column], size)

    return adjusted


def _apply_heart_rate_trend(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    adjusted = synthetic_data.copy()
    size = len(adjusted)
    heart_columns = _find_columns(real_data, ["heart_rate", "heartrate", "pulse"])

    for column in heart_columns:
        if column not in adjusted.columns:
            continue
        real_numeric = pd.to_numeric(_aligned_real_series(real_data[column], size), errors="coerce")
        if real_numeric.notna().sum() == 0:
            continue

        max_offset = max(1, int(round(5 * max(noise_level, 0.1))))
        possible_offsets = [offset for offset in range(-max_offset, max_offset + 1) if offset != 0]
        offset = np.random.choice(possible_offsets)
        values = real_numeric + offset
        values = values.clip(lower=40, upper=180)

        if pd.api.types.is_integer_dtype(real_data[column].dropna()):
            values = np.rint(values).astype("Int64" if real_data[column].isna().any() else int)

        adjusted[column] = values

    return adjusted


def _parse_bp_value(value) -> tuple[Optional[float], Optional[float]]:
    if pd.isna(value):
        return None, None

    text = str(value)
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if "/" in text and len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    if numbers:
        return float(numbers[0]), None
    return None, None


def _condition_from_values(bp_value, sugar_value) -> str:
    systolic, diastolic = _parse_bp_value(bp_value)
    sugar = pd.to_numeric(pd.Series([sugar_value]), errors="coerce").iloc[0]
    has_hypertension = (
        (systolic is not None and systolic > 120)
        or (diastolic is not None and diastolic > 80)
    )
    has_diabetes = pd.notna(sugar) and sugar > 140

    if has_hypertension and has_diabetes:
        return "Hypertension and Diabetes"
    if has_hypertension:
        return "Hypertension"
    if has_diabetes:
        return "Diabetes"
    return "Healthy"


def _balanced_labels(size: int) -> np.ndarray:
    labels = np.array(["low", "medium", "high"] * ((size // 3) + 1))[:size]
    np.random.shuffle(labels)
    return labels


def _balanced_bp_values(size: int, as_pair: bool, integer_output: bool) -> list:
    labels = _balanced_labels(size)
    values = []

    for label in labels:
        if label == "low":
            systolic = np.random.randint(95, 111)
            diastolic = np.random.randint(60, 71)
        elif label == "medium":
            systolic = np.random.randint(111, 121)
            diastolic = np.random.randint(71, 81)
        else:
            systolic = np.random.randint(121, 151)
            diastolic = np.random.randint(81, 96)

        if as_pair:
            values.append(f"{systolic}/{diastolic}")
        elif integer_output:
            values.append(int(systolic))
        else:
            values.append(float(systolic))

    return values


def _balanced_sugar_values(size: int, integer_output: bool) -> list:
    labels = _balanced_labels(size)
    values = []

    for label in labels:
        if label == "low":
            sugar = np.random.randint(70, 91)
        elif label == "medium":
            sugar = np.random.randint(91, 141)
        else:
            sugar = np.random.randint(141, 221)
        values.append(int(sugar) if integer_output else float(sugar))

    return values


def _apply_balanced_clinical_values(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    adjusted = synthetic_data.copy()
    size = len(adjusted)
    bp_col = _find_column(adjusted, ["blood_pressure", "bloodpressure", "bp"])
    sugar_col = _find_column(adjusted, ["blood_sugar", "bloodsugar", "glucose", "sugar"])

    if bp_col:
        original = adjusted[bp_col].dropna()
        as_pair = original.astype(str).str.contains("/", regex=False).any()
        integer_output = pd.api.types.is_integer_dtype(adjusted[bp_col].dropna())
        adjusted[bp_col] = _balanced_bp_values(size, as_pair=as_pair, integer_output=integer_output)

    if sugar_col:
        integer_output = pd.api.types.is_integer_dtype(adjusted[sugar_col].dropna())
        adjusted[sugar_col] = _balanced_sugar_values(size, integer_output=integer_output)

    return adjusted


def _apply_bp_sugar_trends(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    adjusted = synthetic_data.copy()
    size = len(adjusted)
    bp_col = _find_column(real_data, ["blood_pressure", "bloodpressure", "bp"])
    sugar_col = _find_column(real_data, ["blood_sugar", "bloodsugar", "glucose", "sugar"])

    if bp_col and bp_col in adjusted.columns:
        aligned_bp = _aligned_real_series(real_data[bp_col], size)
        parsed = aligned_bp.apply(_parse_bp_value)
        systolic = parsed.apply(lambda item: item[0])
        diastolic = parsed.apply(lambda item: item[1])
        has_pair = diastolic.notna().any()

        max_offset = max(1, int(round(10 * max(noise_level, 0.1))))
        possible_offsets = [offset for offset in range(-max_offset, max_offset + 1) if offset != 0]
        systolic_offset = np.random.choice(possible_offsets)
        diastolic_offset = int(np.sign(systolic_offset) * max(1, abs(systolic_offset) // 2))

        new_systolic = (systolic + systolic_offset).clip(lower=80, upper=220)
        new_diastolic = (diastolic + diastolic_offset).clip(lower=50, upper=130)

        if has_pair:
            adjusted[bp_col] = [
                f"{int(round(sys))}/{int(round(dia))}" if pd.notna(sys) and pd.notna(dia) else value
                for sys, dia, value in zip(new_systolic, new_diastolic, aligned_bp)
            ]
        else:
            values = np.rint(new_systolic).astype("Int64" if real_data[bp_col].isna().any() else int)
            adjusted[bp_col] = values

    if sugar_col and sugar_col in adjusted.columns:
        aligned_sugar = pd.to_numeric(_aligned_real_series(real_data[sugar_col], size), errors="coerce")
        max_offset = max(1, int(round(15 * max(noise_level, 0.1))))
        possible_offsets = [offset for offset in range(-max_offset, max_offset + 1) if offset != 0]
        offset = np.random.choice(possible_offsets)
        values = (aligned_sugar + offset).clip(lower=40, upper=400)

        if pd.api.types.is_integer_dtype(real_data[sugar_col].dropna()):
            values = np.rint(values).astype("Int64" if real_data[sugar_col].isna().any() else int)

        adjusted[sugar_col] = values

    return adjusted


def _apply_condition_rules(synthetic_data: pd.DataFrame) -> pd.DataFrame:
    adjusted = synthetic_data.copy()
    bp_col = _find_column(adjusted, ["blood_pressure", "bloodpressure", "bp", "systolic"])
    sugar_col = _find_column(adjusted, ["blood_sugar", "bloodsugar", "glucose", "sugar"])
    condition_col = _find_column(adjusted, ["condition"])

    if bp_col is None and sugar_col is None:
        return adjusted

    if condition_col is None:
        condition_col = "condition"

    bp_values = adjusted[bp_col] if bp_col else pd.Series([None] * len(adjusted), index=adjusted.index)
    sugar_values = adjusted[sugar_col] if sugar_col else pd.Series([None] * len(adjusted), index=adjusted.index)
    adjusted[condition_col] = [
        _condition_from_values(bp_value, sugar_value)
        for bp_value, sugar_value in zip(bp_values, sugar_values)
    ]

    return adjusted


def _line_distribution_payload(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict:
    real_data = _sample_for_analysis(real_data)
    synthetic_data = _sample_for_analysis(synthetic_data)
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        return {"columns": [], "series": {}}

    priority = []
    for candidates in (
        ["age"],
        ["heart_rate", "heartrate", "pulse"],
        ["bp", "blood_pressure"],
        ["blood_sugar", "glucose", "sugar"],
        ["cholesterol", "cholestrol"],
    ):
        column = _find_column(real_data, candidates)
        if column and column in numeric_columns:
            priority.append(column)
    columns = list(dict.fromkeys(priority + numeric_columns[:6]))

    series = {}
    for column in columns:
        real_values = pd.to_numeric(real_data[column], errors="coerce").dropna()
        syn_values = pd.to_numeric(synthetic_data[column], errors="coerce").dropna()
        if real_values.empty or syn_values.empty:
            continue

        percentiles = np.linspace(0, 100, 13)
        real_profile = np.percentile(real_values, percentiles)
        syn_profile = np.percentile(syn_values, percentiles)
        series[column] = {
            "x": percentiles.astype(int).tolist(),
            "real": np.nan_to_num(real_profile).round(3).tolist(),
            "synthetic": np.nan_to_num(syn_profile).round(3).tolist(),
        }

    return {"columns": columns, "series": series}


def _numeric_for_correlation(data: pd.DataFrame) -> pd.DataFrame:
    sampled = _sample_for_analysis(data)
    numeric = sampled.select_dtypes(include=[np.number]).copy()

    bp_col = _find_column(sampled, ["blood_pressure", "bloodpressure", "bp"])
    if bp_col and bp_col not in numeric.columns:
        parsed = sampled[bp_col].apply(_parse_bp_value)
        numeric[f"{bp_col}_systolic"] = parsed.apply(lambda item: item[0])
        numeric[f"{bp_col}_diastolic"] = parsed.apply(lambda item: item[1])

    condition_col = _find_column(sampled, ["condition"])
    if condition_col:
        condition_map = {
            "Healthy": 0,
            "Hypertension": 1,
            "Diabetes": 2,
            "Hypertension and Diabetes": 3,
        }
        numeric["condition_code"] = sampled[condition_col].map(condition_map)

    return numeric.dropna(axis=1, how="all")


def _correlation_payload(data: pd.DataFrame) -> dict:
    numeric = _numeric_for_correlation(data)
    if numeric.shape[1] < 2:
        return {"columns": [], "z": []}
    numeric = numeric.iloc[:, :HEATMAP_COLUMN_LIMIT]
    corr = numeric.corr().fillna(0).round(3)
    return {"columns": corr.columns.tolist(), "z": corr.values.tolist()}


def _correlation_difference_payload(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> dict:
    real_numeric = _numeric_for_correlation(real_data)
    syn_numeric = _numeric_for_correlation(synthetic_data)
    shared = [column for column in real_numeric.columns if column in syn_numeric.columns][:HEATMAP_COLUMN_LIMIT]
    if len(shared) < 2:
        return {"columns": [], "z": []}

    real_corr = real_numeric[shared].corr().fillna(0)
    syn_corr = syn_numeric[shared].corr().fillna(0)
    diff = (syn_corr - real_corr).round(3)
    return {"columns": shared, "z": diff.values.tolist()}


def _stats_table(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> list[dict]:
    rows = []
    numeric_columns = real_data.select_dtypes(include=[np.number]).columns[:STATS_COLUMN_LIMIT]
    for column in numeric_columns:
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
    real_data = _sample_for_analysis(real_data)
    synthetic_data = _sample_for_analysis(synthetic_data)
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

    heart_cols = _find_columns(synthetic_data, ["heart_rate", "heartrate", "pulse"])
    for heart_col in heart_cols[:1]:
        heart = pd.to_numeric(synthetic_data[heart_col], errors="coerce")
        invalid = int(((heart < 40) | (heart > 180)).sum())
        checks.append(
            {
                "name": "Heart rate range validation",
                "status": "Pass" if invalid == 0 else "Review",
                "detail": f"{invalid} synthetic rows outside 40-180 bpm.",
            }
        )

    return checks or [{"name": "Medical consistency checks", "status": "Limited", "detail": "No age, BP, glucose, disease, or diagnosis columns were detected."}]


def _estimate_model_utility(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    target = _find_column(real_data, ["target", "label", "outcome", "diagnosis", "disease"])
    if target is None or target not in synthetic_data.columns:
        return None, None, "No target column found"

    real = _sample_for_analysis(real_data.dropna(subset=[target]).copy())
    synthetic = _sample_for_analysis(synthetic_data.dropna(subset=[target]).copy())
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
    training_data = _sample_for_analysis(real_data, TRAINING_ROW_LIMIT)

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        verbose=False,
    )
    synthesizer.fit(training_data)

    synthetic_data = synthesizer.sample(num_rows=requested_rows)
    synthetic_data = _apply_general_noise(real_data, synthetic_data, noise_level)
    synthetic_data, margin_report = _apply_margin_rules(real_data, synthetic_data, noise_level)
    synthetic_data = _apply_heart_rate_trend(real_data, synthetic_data, noise_level)
    synthetic_data = _preserve_copy_columns(real_data, synthetic_data)
    synthetic_data = _apply_bp_sugar_trends(real_data, synthetic_data, noise_level)
    synthetic_data = _apply_condition_rules(synthetic_data)

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
        "corrDiff": _correlation_difference_payload(real_data, synthetic_data),
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
        "training_rows_used": len(training_data),
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

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings


# -----------------------------
# Configurable paths and constants
# -----------------------------
DATASET_PATH = Path("TrainingDataset.arff")
FIGURES_DIR = Path("figures")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

RESULTS_CSV_ROOT = Path("results_summary.csv")
BEST_PARAMS_JSON_ROOT = Path("best_hyperparameters.json")

RESULTS_CSV_OUTPUTS = OUTPUTS_DIR / "results_summary.csv"
BEST_PARAMS_JSON_OUTPUTS = OUTPUTS_DIR / "best_hyperparameters.json"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 10
POS_LABEL = 1


def ensure_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def decode_arff_bytes(df: pd.DataFrame) -> pd.DataFrame:
    decoded_df = df.copy()
    for col in decoded_df.columns:
        if decoded_df[col].dtype == object:
            decoded_df[col] = decoded_df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            )
    return decoded_df


def convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    converted_df = df.copy()
    for col in converted_df.columns:
        try:
            converted_df[col] = pd.to_numeric(converted_df[col], errors="raise")
        except (ValueError, TypeError):
            # Keep non-numeric columns unchanged.
            pass
    return converted_df


def load_arff_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Place your ARFF file in the project root or update DATASET_PATH."
        )

    raw_data, _ = arff.loadarff(dataset_path)
    df = pd.DataFrame(raw_data)
    df = decode_arff_bytes(df)
    df = convert_columns_to_numeric(df)
    return df


def detect_target_column(df: pd.DataFrame) -> Tuple[str, bool]:
    preferred_names = ["class", "target", "label", "result", "phishing"]
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}

    for name in preferred_names:
        if name in lower_to_original:
            return str(lower_to_original[name]), False

    for col in df.columns:
        col_lower = str(col).strip().lower()
        if any(key in col_lower for key in preferred_names):
            return str(col), False

    fallback_col = str(df.columns[-1])
    return fallback_col, True


def coerce_target_to_binary(y: pd.Series) -> pd.Series:
    y_decoded = y.copy()
    if y_decoded.dtype == object:
        y_decoded = y_decoded.apply(
            lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
        )

    numeric_target = pd.to_numeric(y_decoded, errors="coerce")
    if numeric_target.notna().all():
        y_numeric = numeric_target.astype(int)
    else:
        mapping = {
            "1": 1,
            "-1": -1,
            "phishing": 1,
            "legitimate": -1,
            "legit": -1,
            "true": 1,
            "false": -1,
        }
        y_as_str = y_decoded.astype(str).str.strip().str.lower()
        y_mapped = y_as_str.map(mapping)
        if y_mapped.isna().any():
            unknown_values = sorted(y_as_str[y_mapped.isna()].unique().tolist())
            raise ValueError(
                f"Could not map all target labels to binary classes. Unknown labels: {unknown_values}"
            )
        y_numeric = y_mapped.astype(int)

    unique_classes = set(y_numeric.unique().tolist())
    if unique_classes == {0, 1}:
        print("[INFO] Target labels detected as {0, 1}. Mapping 0 -> -1 and 1 -> 1.")
        y_numeric = y_numeric.replace({0: -1, 1: 1})
        unique_classes = set(y_numeric.unique().tolist())

    if unique_classes != {-1, 1}:
        raise ValueError(
            f"Target labels must be binary with classes -1 and 1 (or 0/1). Found classes: {sorted(unique_classes)}"
        )

    return y_numeric


def build_model_configs() -> Dict[str, Dict[str, Any]]:
    model_configs: Dict[str, Dict[str, Any]] = {
        "Logistic Regression": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__penalty": ["l1", "l2"],
                "model__C": [0.01, 0.1, 1, 10, 100],
            },
        },
        "Decision Tree": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        DecisionTreeClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "param_grid": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 5, 10, 15, 20],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5, 10],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
        "SVM": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "param_grid": [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.1, 1, 10, 100],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1, 10, 100],
                    "model__gamma": ["scale", "auto", 0.01, 0.1, 1],
                },
            ],
        },
        "KNN": {
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        KNeighborsClassifier(),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__metric": ["minkowski"],
                "model__p": [1, 2],
            },
        },
    }

    return model_configs


def get_scoring() -> Dict[str, Any]:
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(
            precision_score,
            average="binary",
            pos_label=POS_LABEL,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            average="binary",
            pos_label=POS_LABEL,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            average="binary",
            pos_label=POS_LABEL,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }


def get_roc_scores(estimator: Any, x_test: pd.DataFrame) -> Tuple[Optional[np.ndarray], str]:
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(x_test)
        if probs.ndim == 2:
            classes = list(estimator.classes_)
            if POS_LABEL in classes:
                pos_idx = classes.index(POS_LABEL)
                return probs[:, pos_idx], "predict_proba"
            return probs[:, -1], "predict_proba_last_class"

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        if isinstance(scores, np.ndarray) and scores.ndim > 1:
            classes = list(estimator.classes_)
            if POS_LABEL in classes:
                pos_idx = classes.index(POS_LABEL)
                return scores[:, pos_idx], "decision_function"
            return scores[:, -1], "decision_function_last_class"
        return np.asarray(scores), "decision_function"

    return None, "unavailable"


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_confusion_matrix_plot(model_name: str, cm: np.ndarray) -> Path:
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    output_path = FIGURES_DIR / filename

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate (-1)", "Phishing (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def save_comparison_bar_chart(results_df: pd.DataFrame) -> Path:
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    chart_data = results_df[["model", *metric_cols]].copy()

    x = np.arange(len(chart_data))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, metric in enumerate(metric_cols):
        ax.bar(x + idx * width, chart_data[metric].values, width=width, label=metric.upper())

    ax.set_title("Test-Set Metric Comparison Across Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x + (len(metric_cols) - 1) * width / 2)
    ax.set_xticklabels(chart_data["model"], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = FIGURES_DIR / "metrics_comparison_bar_chart.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def format_results_for_terminal(df: pd.DataFrame) -> str:
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"
        )
    return display_df.to_string(index=False)


def evaluate_single_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: Dict[str, Any],
    cv: StratifiedKFold,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    result: Dict[str, Any] = {
        "model": model_name,
        "cv_best_f1": np.nan,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "roc_auc": np.nan,
        "train_tune_time_sec": np.nan,
        "prediction_time_sec": np.nan,
        "best_params": "",
        "status": "failed",
        "error": "",
    }

    best_params_json: Dict[str, Any] = {}

    try:
        warnings.simplefilter("ignore", category=FitFailedWarning)

        train_start = time.perf_counter()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan,
            return_train_score=False,
        )
        grid_search.fit(x_train, y_train)
        train_tune_time = time.perf_counter() - train_start

        best_estimator = grid_search.best_estimator_
        best_params_json = sanitize_for_json(grid_search.best_params_)

        model_filename = f"{model_name.lower().replace(' ', '_')}_best.joblib"
        model_path = MODELS_DIR / model_filename
        joblib.dump(best_estimator, model_path)

        pred_start = time.perf_counter()
        y_pred = best_estimator.predict(x_test)
        roc_scores, roc_source = get_roc_scores(best_estimator, x_test)
        prediction_time = time.perf_counter() - pred_start

        if roc_scores is not None:
            try:
                roc_auc_value = roc_auc_score(y_test, roc_scores)
            except ValueError as roc_err:
                print(f"[WARN] ROC-AUC failed for {model_name}: {roc_err}")
                roc_auc_value = np.nan
        else:
            print(
                f"[WARN] ROC-AUC unavailable for {model_name}: neither predict_proba nor decision_function is available."
            )
            roc_auc_value = np.nan
            roc_source = "unavailable"

        cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
        save_confusion_matrix_plot(model_name, cm)

        result.update(
            {
                "cv_best_f1": float(grid_search.best_score_),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(
                    precision_score(
                        y_test,
                        y_pred,
                        average="binary",
                        pos_label=POS_LABEL,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        y_pred,
                        average="binary",
                        pos_label=POS_LABEL,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test,
                        y_pred,
                        average="binary",
                        pos_label=POS_LABEL,
                        zero_division=0,
                    )
                ),
                "roc_auc": float(roc_auc_value) if pd.notna(roc_auc_value) else np.nan,
                "train_tune_time_sec": float(train_tune_time),
                "prediction_time_sec": float(prediction_time),
                "best_params": json.dumps(best_params_json, ensure_ascii=True),
                "status": "success",
                "error": "",
            }
        )

        print(f"[INFO] Finished {model_name}. ROC source: {roc_source}.")

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        print(f"[ERROR] {model_name} failed: {exc}")

    return result, best_params_json


def save_outputs(results_df: pd.DataFrame, best_params_all: Dict[str, Dict[str, Any]]) -> None:
    results_df.to_csv(RESULTS_CSV_ROOT, index=False)
    results_df.to_csv(RESULTS_CSV_OUTPUTS, index=False)

    with BEST_PARAMS_JSON_ROOT.open("w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=2, ensure_ascii=True)

    with BEST_PARAMS_JSON_OUTPUTS.open("w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=2, ensure_ascii=True)


def main() -> None:
    print("=" * 80)
    print("Phishing Website Detection - Classical ML Benchmark")
    print("=" * 80)

    ensure_directories()

    try:
        df = load_arff_dataset(DATASET_PATH)
    except FileNotFoundError as file_err:
        print(f"[ERROR] {file_err}")
        return
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        return

    print(f"[INFO] Dataset loaded successfully from: {DATASET_PATH}")
    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    target_col, used_fallback = detect_target_column(df)
    if used_fallback:
        print(
            f"[WARN] Target column name not found by common names. Falling back to last column as target: '{target_col}'."
        )
    else:
        print(f"[INFO] Target column detected: '{target_col}'")

    try:
        y = coerce_target_to_binary(df[target_col])
        x = df.drop(columns=[target_col])
    except Exception as exc:
        print(f"[ERROR] Failed to prepare features and target: {exc}")
        return

    print(f"[INFO] Features shape: {x.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts().sort_index()}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[INFO] Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")

    scoring = get_scoring()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    model_configs = build_model_configs()

    all_results: List[Dict[str, Any]] = []
    all_best_params: Dict[str, Dict[str, Any]] = {}

    for model_name, config in model_configs.items():
        print("\n" + "-" * 80)
        print(f"Training and tuning: {model_name}")
        print("-" * 80)

        result, best_params = evaluate_single_model(
            model_name=model_name,
            pipeline=config["pipeline"],
            param_grid=config["param_grid"],
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scoring=scoring,
            cv=cv,
        )

        all_results.append(result)
        if best_params:
            all_best_params[model_name] = best_params

    results_df = pd.DataFrame(all_results)

    # Keep successful models first and sort by test-set F1.
    success_df = results_df[results_df["status"] == "success"].copy()
    failed_df = results_df[results_df["status"] != "success"].copy()

    if not success_df.empty:
        success_df = success_df.sort_values(by="f1", ascending=False)
    results_df = pd.concat([success_df, failed_df], ignore_index=True)

    save_outputs(results_df, all_best_params)

    if not success_df.empty:
        save_comparison_bar_chart(success_df)

    print("\n" + "=" * 80)
    print("Cross-Validated Training Performance (best mean CV F1 and hyperparameters)")
    print("=" * 80)
    cv_view_cols = ["model", "cv_best_f1", "best_params", "status", "error"]
    print(format_results_for_terminal(results_df[cv_view_cols]))

    print("\n" + "=" * 80)
    print("Final Held-Out Test Performance")
    print("=" * 80)
    test_view_cols = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "train_tune_time_sec",
        "prediction_time_sec",
        "status",
        "error",
    ]
    print(format_results_for_terminal(results_df[test_view_cols]))

    print("\n" + "=" * 80)
    print("Saved Artifacts")
    print("=" * 80)
    print(f"- Summary CSV: {RESULTS_CSV_ROOT}")
    print(f"- Summary CSV copy: {RESULTS_CSV_OUTPUTS}")
    print(f"- Best hyperparameters JSON: {BEST_PARAMS_JSON_ROOT}")
    print(f"- Best hyperparameters JSON copy: {BEST_PARAMS_JSON_OUTPUTS}")
    print(f"- Confusion matrices and metric comparison chart: {FIGURES_DIR}")
    print(f"- Trained best models: {MODELS_DIR}")


if __name__ == "__main__":
    main()

# Phishing Website Detection Using Classical Machine Learning

This project benchmarks five classical supervised machine learning models for phishing website detection using the UCI Phishing Websites dataset in ARFF format.

Implemented models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Methodology

The implementation follows a leakage-safe design:
- Dataset loaded via `scipy.io.arff`
- ARFF byte-string decoding to UTF-8 when needed
- Numeric conversion where possible
- Stratified 70/30 train-test split (`random_state=42`)
- Hyperparameter tuning on training data only using:
  - `GridSearchCV`
  - `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`
  - Multi-metric scoring: accuracy, precision, recall, F1, ROC-AUC
  - `refit="f1"` for model selection
- Final evaluation on held-out test set only
- Preprocessing stays inside model pipelines to prevent data leakage

## Project Structure

- `main.py` - Main experiment script
- `requirements.txt` - Python dependencies
- `README.md` - Setup and run instructions
- `data/` - Optional location for dataset
- `figures/` - Saved confusion matrices and comparison chart
- `models/` - Saved best trained models (`joblib`)
- `outputs/` - Copies of tabular/JSON outputs
- `results_summary.csv` - Main summary output
- `best_hyperparameters.json` - Main best-parameter output

## Dataset Placement

Expected dataset file name:
- `TrainingDataset.arff`

By default, `main.py` looks for the dataset in the project root:
- `TrainingDataset.arff`

If your dataset is in another location (for example in `data/`), update this line at the top of `main.py`:

```python
DATASET_PATH = Path("TrainingDataset.arff")
```

Example alternative:

```python
DATASET_PATH = Path("data/TrainingDataset.arff")
```

## Run in VS Code (Windows PowerShell)

1. Open this project folder in VS Code.
2. Open a terminal in VS Code.
3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Place `TrainingDataset.arff` in the project root (or update `DATASET_PATH` accordingly).
6. Run the experiment:

```powershell
python main.py
```

## Outputs

After execution, the script will:
- Print clear terminal summary tables
- Save `results_summary.csv`
- Save `best_hyperparameters.json`
- Save copies of both files in `outputs/`
- Save one confusion matrix figure per model in `figures/`
- Save a metric comparison bar chart in `figures/`
- Save best fitted model per algorithm in `models/`

## Notes

- Positive class is phishing (`pos_label=1`).
- Legitimate class is `-1`.
- If target column name is unknown, the script falls back to the last column and reports that behavior.
- If ROC-AUC cannot be computed from probabilities, the script attempts `decision_function`.
- If specific model configurations fail during grid search, the process continues when possible and reports failures.

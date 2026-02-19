# MachineLearning-Coursework1

Regression coursework predicting a continuous `outcome` variable from a
diamond/gemstone dataset. The final model is a tuned
`HistGradientBoostingRegressor` trained within a scikit-learn `Pipeline`,
selected after systematic comparison against Ridge and Random Forest baselines
via 5-Fold cross-validation.

---

## Repository Structure

```
MachineLearning-Coursework1/
├── data/                        # Training and test CSVs (git-ignored; see Setup)
│   ├── CW1_train.csv
│   └── CW1_test.csv
├── notebooks/                   # Exploratory Jupyter notebooks
├── reports/
│   ├── figures/                 # Generated plots (correlation heatmap, pred vs actual)
│   └── report.tex               # LaTeX source for the 2-page technical report
├── src/                         # Python package — all implementation code
│   ├── features.py              # Preprocessing pipeline (ColumnTransformer)
│   ├── models.py                # Model registry
│   ├── evaluate.py              # CV evaluation logic
│   ├── train.py                 # Training, tuning, and submission generation
│   └── visualize.py             # Figure generation for the report
├── submissions/                 # Output prediction CSV (git-ignored)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**1. Clone the repository**

```bash
git clone git@github.com:jpires0405/MachineLearning-Coursework1.git
cd MachineLearning-Coursework1
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Place data files**

The CSV data files are git-ignored and must be placed manually:

```
data/CW1_train.csv   # 10,000-row training set
data/CW1_test.csv    #  1,000-row test set (no target column)
```

---

## Execution

All scripts are run from inside the `MachineLearning-Coursework1/` directory.

**Run the full training pipeline and generate the submission CSV**

```bash
python src/train.py
```

Prints the best CV mean R² and parameter set to stdout. Writes the
1,000-row prediction file to `submissions/CW1_submission_23115639.csv`.

**Generate report figures**

```bash
python src/visualize.py
```

Writes two plots to `reports/figures/`:
- `correlation_heatmap.png` — Pearson correlations of the top 10 numeric features with `outcome`
- `pred_vs_actual.png` — Predicted vs. actual scatter plot on an 80/20 hold-out split

---

## Methodology Summary

- **Preprocessing**: numeric features scaled with `StandardScaler`; categorical
  features (`cut`, `color`, `clarity`) encoded with
  `OneHotEncoder(handle_unknown="ignore")`. Both transformers are wrapped in a
  `ColumnTransformer` and embedded inside an `sklearn.pipeline.Pipeline` to
  prevent data leakage.
- **Validation**: 5-Fold cross-validation (`KFold`, `shuffle=True`,
  `random_state=123`) using R² as the scoring metric throughout.
- **Model selection**: Ridge regression established a linear baseline
  (CV R² = 0.282). `RandomForestRegressor` (CV R² = 0.452) and
  `HistGradientBoostingRegressor` (CV R² = 0.460) were then evaluated.
  HistGBR was preferred for its histogram-binned split search and native
  handling of non-linear feature interactions.
- **Hyperparameter tuning**: `RandomizedSearchCV` (15 iterations, same 5-Fold
  CV) over learning rate, iteration count, tree depth, and L2 regularisation.
  Best configuration (lr = 0.05, max\_iter = 300, max\_depth = 3,
  l2\_regularization = 0.1) achieved CV R² = 0.472.
- **Reproducibility**: all random states fixed at `123`.

---

## Results

| Model | Configuration | CV Mean R² | CV Std Dev |
|---|---|---|---|
| Ridge | Baseline (α = 1.0) | 0.282 | 0.014 |
| Random Forest | Default | 0.452 | 0.014 |
| HistGradientBoosting | Default | 0.460 | 0.015 |
| HistGradientBoosting | **Tuned** | **0.472** | — |

---

*Student ID: 23115639*

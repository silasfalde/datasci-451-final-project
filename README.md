# datasci-451-final-project

Python implementation of a multilabel vaccine uptake prediction project using the NHFS training dataset.

## Project Layout

- **src/**: Modular Python utilities for data processing, modeling, and evaluation
  - `config.py`: Configuration, paths, and hyperparameters
  - `data_utils.py`: Data loading, preprocessing, fold creation
  - `model_utils.py`: Model training (logistic, random forest, Bayesian)
  - `eval_utils.py`: Model evaluation and visualization
- **notebooks/**: Unified Jupyter notebook for end-to-end pipeline execution
- **data/raw**: Raw feature and label CSV files
- **data/processed**: Cached joined data
- **outputs/**: Models, predictions, metrics, and figures

## Quick Start

### Option 1: Run the Unified Notebook

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook notebooks/pipeline.ipynb
   ```

3. Run all cells to execute the full pipeline end-to-end

### Option 2: Import and Use Modules Directly

```python
import sys
sys.path.insert(0, 'src')

from config import CONFIG, ensure_output_dirs
from data_utils import load_joined_data, standardize_missing, make_folds
from model_utils import fit_logistic, fit_random_forest, fit_bayesian
from eval_utils import summarize_metrics, plot_roc_curves, plot_calibration

# Use config to access paths and hyperparameters
ensure_output_dirs()
```

## Features

- **Data Preprocessing**: Handles missing values, categorical encoding, and stratified k-fold creation
- **Multilabel Modeling**: Simultaneous prediction of two vaccine outcomes (H1N1 and seasonal)
- **Model Comparison**: Ridge logistic regression, random forest, and Bayesian hierarchical approaches
- **Comprehensive Evaluation**: Per-outcome and macro ROC AUC, Brier scores, ROC/calibration plots

## Configuration

Edit `src/config.py` to customize:
- Number of CV folds: `CONFIG["folds"]`
- Bayesian MCMC settings: `CONFIG["bayes"]`
- Output paths: `CONFIG["paths"]`
- Outcomes of interest: `CONFIG["outcomes"]`

## Notes

- The notebook uses stratified CV that preserves both outcome distributions across folds
- Bayesian hierarchical models include random intercepts by `hhs_geo_region`
- All outputs (predictions, metrics, plots) are saved to the `outputs/` directory
- The pipeline is reproducible with fixed `CONFIG["seed"] = 451`

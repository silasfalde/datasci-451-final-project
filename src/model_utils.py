import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from config import CONFIG


STAN_HIER_LOGIT = """
data {
    int<lower=1> N;
    int<lower=1> K;
    matrix[N, K] X;
    int<lower=1> J;
    array[N] int<lower=1, upper=J> g;
    array[N] int<lower=0, upper=1> y;

    int<lower=1> N_test;
    matrix[N_test, K] X_test;
    array[N_test] int<lower=1, upper=J> g_test;
    real<lower=0> beta_sd;
}
parameters {
    real alpha;
    vector[K] beta;
    vector[J] z_u;
    real<lower=0> sigma_u;
}
transformed parameters {
    vector[J] u;
    u = z_u * sigma_u;
}
model {
    alpha ~ normal(0, 2.5);
    beta ~ normal(0, beta_sd);
    z_u ~ normal(0, 1);
    sigma_u ~ normal(0, 1);

    y ~ bernoulli_logit(alpha + X * beta + u[g]);
}
generated quantities {
    vector[N_test] p_test;
    for (n in 1:N_test) {
        p_test[n] = inv_logit(alpha + dot_product(row(X_test, n), beta) + u[g_test[n]]);
    }
}
"""


def _safe_summary(fit):
    """Return Stan summary dict when available, else None."""
    try:
        return fit.summary()
    except Exception:
        return None


def _extract_diagnostics(fit):
    """Extract lightweight convergence diagnostics from Stan fit."""
    summary = _safe_summary(fit)
    diagnostics = {
        "max_rhat": np.nan,
        "min_bulk_ess": np.nan,
        "min_tail_ess": np.nan,
    }

    if not summary:
        return diagnostics

    cols = summary.get("summary_colnames", [])
    vals = np.asarray(summary.get("summary", []))
    if vals.size == 0 or len(cols) == 0:
        return diagnostics

    idx = {name: i for i, name in enumerate(cols)}

    if "R_hat" in idx:
        diagnostics["max_rhat"] = float(np.nanmax(vals[:, idx["R_hat"]]))
    if "ESS_bulk" in idx:
        diagnostics["min_bulk_ess"] = float(np.nanmin(vals[:, idx["ESS_bulk"]]))
    if "ESS_tail" in idx:
        diagnostics["min_tail_ess"] = float(np.nanmin(vals[:, idx["ESS_tail"]]))

    return diagnostics


def _extract_param_intervals(fit, param_name, vector_param=False):
    """Return mean and central 95% interval for scalar or vector parameters."""
    try:
        draws = np.asarray(fit[param_name])
    except Exception:
        return None

    if vector_param:
        # Keep last axis as parameter dimension; collapse all sample axes.
        if draws.ndim == 1:
            sample_matrix = draws.reshape(-1, 1)
        else:
            sample_matrix = draws.reshape(-1, draws.shape[-1])

        mean = np.nanmean(sample_matrix, axis=0)
        lo = np.nanpercentile(sample_matrix, 2.5, axis=0)
        hi = np.nanpercentile(sample_matrix, 97.5, axis=0)
        return {"mean": mean, "ci_lower": lo, "ci_upper": hi}

    # Scalar parameter: collapse all sample axes to one vector.
    sample_vector = draws.reshape(-1)
    mean = float(np.nanmean(sample_vector))
    lo = float(np.nanpercentile(sample_vector, 2.5))
    hi = float(np.nanpercentile(sample_vector, 97.5))
    return {"mean": mean, "ci_lower": lo, "ci_upper": hi}


def _posterior_pred_summary(p_test, n_test):
    """Return posterior mean and 95% CI for test probabilities with robust shape handling."""
    arr = np.asarray(p_test)

    if arr.ndim == 1:
        if arr.shape[0] != n_test:
            raise ValueError(
                f"Unexpected p_test length {arr.shape[0]} (expected {n_test})."
            )
        mean = arr
        lo = arr
        hi = arr
        return mean, lo, hi

    # Identify which axis corresponds to test examples.
    candidate_axes = [ax for ax, size in enumerate(arr.shape) if size == n_test]
    if not candidate_axes:
        raise ValueError(
            f"Could not find test axis of size {n_test} in p_test with shape {arr.shape}."
        )

    # Use last matching axis to be robust to (draw, n_test) or (n_test, draw) or higher rank.
    test_axis = candidate_axes[-1]
    arr = np.moveaxis(arr, test_axis, -1)

    # Collapse all sample axes; keep final axis as test index.
    sample_matrix = arr.reshape(-1, n_test)
    mean = np.nanmean(sample_matrix, axis=0)
    lo = np.nanpercentile(sample_matrix, 2.5, axis=0)
    hi = np.nanpercentile(sample_matrix, 97.5, axis=0)
    return mean, lo, hi


def fit_logistic(X_train, y_train, X_test):
    """Fit ridge logistic regression and predict on test set."""
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=CONFIG["seed"],
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {"model": model, "predictions": y_pred_proba}


def fit_random_forest(X_train, y_train, X_test, seed=None):
    """Fit random forest classifier and predict on test set."""
    if seed is None:
        seed = CONFIG["seed"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return {"model": model, "predictions": y_pred_proba}


def fit_bayesian(train_df, test_df, outcome):
    """Fit a Bayesian hierarchical logistic model using PyStan."""
    try:
        import stan
    except ImportError as exc:
        raise ImportError(
            "PyStan is required for fit_bayesian. Install with `pip install pystan`."
        ) from exc

    # PyStan invokes asyncio internally; Jupyter already runs an event loop.
    # Patch the loop when available to avoid `asyncio.run()` RuntimeError.
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        pass

    group_col = CONFIG["group_col"]
    id_col = CONFIG["id_col"]

    if group_col not in train_df.columns:
        raise ValueError(f"Grouping column '{group_col}' is required for hierarchical model.")

    train_local = train_df.copy()
    test_local = test_df.copy()

    # Ensure group labels are available and consistent.
    train_local[group_col] = train_local[group_col].fillna("Unknown").astype(str)
    test_local[group_col] = test_local[group_col].fillna("Unknown").astype(str)

    # Build feature matrices: keep group out of X (modeled separately as random intercept).
    drop_cols = [id_col, outcome, group_col, "_strata_key"]
    feature_cols = [c for c in train_local.columns if c not in drop_cols]

    X_train_raw = train_local[feature_cols]
    X_test_raw = test_local[feature_cols]

    # Numeric median imputation and categorical missing handling.
    for col in X_train_raw.columns:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]):
            med = X_train_raw[col].median()
            X_train_raw[col] = X_train_raw[col].fillna(med)
            X_test_raw[col] = X_test_raw[col].fillna(med)
        else:
            X_train_raw[col] = X_train_raw[col].fillna("Unknown").astype(str)
            X_test_raw[col] = X_test_raw[col].fillna("Unknown").astype(str)

    # One-hot encode non-numeric predictors and align train/test.
    X_train = pd.get_dummies(X_train_raw, drop_first=False)
    X_test = pd.get_dummies(X_test_raw, drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_local[outcome].astype(int).to_numpy()

    # Group index (1-based for Stan).
    unique_groups = sorted(set(train_local[group_col]).union(set(test_local[group_col])))
    group_map = {g: i + 1 for i, g in enumerate(unique_groups)}
    g_train = np.array([group_map[g] for g in train_local[group_col]], dtype=np.int32)
    g_test = np.array([group_map[g] for g in test_local[group_col]], dtype=np.int32)

    stan_data = {
        "N": int(X_train.shape[0]),
        "K": int(X_train.shape[1]),
        "X": X_train.to_numpy(dtype=float),
        "J": int(len(unique_groups)),
        "g": g_train,
        "y": y_train,
        "N_test": int(X_test.shape[0]),
        "X_test": X_test.to_numpy(dtype=float),
        "g_test": g_test,
        "beta_sd": float(CONFIG["bayes"].get("prior_sd", 1.0)),
    }

    posterior = stan.build(
        STAN_HIER_LOGIT,
        data=stan_data,
        random_seed=CONFIG["seed"],
    )

    fit = posterior.sample(
        num_chains=CONFIG["bayes"]["chains"],
        num_samples=CONFIG["bayes"]["draws"],
        num_warmup=CONFIG["bayes"]["tune"],
    )

    p_test = fit["p_test"]
    y_pred_proba, pred_ci_lower, pred_ci_upper = _posterior_pred_summary(
        p_test, stan_data["N_test"]
    )

    diagnostics = _extract_diagnostics(fit)
    beta_summary = _extract_param_intervals(fit, "beta", vector_param=True)
    sigma_u_summary = _extract_param_intervals(fit, "sigma_u", vector_param=False)

    return {
        "model": fit,
        "predictions": np.clip(np.asarray(y_pred_proba, dtype=float), 1e-6, 1 - 1e-6),
        "prediction_ci_lower": np.clip(np.asarray(pred_ci_lower, dtype=float), 1e-6, 1 - 1e-6),
        "prediction_ci_upper": np.clip(np.asarray(pred_ci_upper, dtype=float), 1e-6, 1 - 1e-6),
        "diagnostics": diagnostics,
        "beta_summary": beta_summary,
        "sigma_u_summary": sigma_u_summary,
        "feature_names": list(X_train.columns),
    }

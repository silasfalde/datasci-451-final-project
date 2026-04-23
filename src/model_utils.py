import numpy as np
import pandas as pd
import inspect
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


def _rhat_from_chains(samples_by_chain):
    """Compute split-free Gelman-Rubin R-hat from (draws, chains) samples."""
    x = np.asarray(samples_by_chain, dtype=float)
    if x.ndim != 2:
        return np.nan

    n_draws, n_chains = x.shape
    if n_draws < 2 or n_chains < 2:
        return np.nan

    chain_means = np.nanmean(x, axis=0)
    chain_vars = np.nanvar(x, axis=0, ddof=1)
    w = np.nanmean(chain_vars)
    if not np.isfinite(w) or w <= 0:
        return np.nan

    b = n_draws * np.nanvar(chain_means, ddof=1)
    var_hat = ((n_draws - 1) / n_draws) * w + (b / n_draws)
    if not np.isfinite(var_hat) or var_hat <= 0:
        return np.nan

    return float(np.sqrt(var_hat / w))


def _ess_lag1_from_chains(samples_by_chain):
    """Approximate effective sample size from mean lag-1 autocorrelation."""
    x = np.asarray(samples_by_chain, dtype=float)
    if x.ndim != 2:
        return np.nan

    n_draws, n_chains = x.shape
    if n_draws < 3 or n_chains < 1:
        return np.nan

    centered = x - np.nanmean(x, axis=0, keepdims=True)
    denom = np.nansum(centered * centered, axis=0)
    numer = np.nansum(centered[1:] * centered[:-1], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho1_per_chain = numer / denom

    rho1 = float(np.nanmean(rho1_per_chain))
    if not np.isfinite(rho1):
        return np.nan

    rho1 = float(np.clip(rho1, -0.99, 0.99))
    ess = (n_draws * n_chains) * (1.0 - rho1) / (1.0 + rho1)
    return float(np.clip(ess, 1.0, float(n_draws * n_chains)))


def _extract_diagnostics_from_draws(fit):
    """Fallback diagnostics for PyStan 3 using internal chain draws."""
    diagnostics = {
        "max_rhat": np.nan,
        "min_bulk_ess": np.nan,
        "min_tail_ess": np.nan,
        "divergences": 0,
        "treedepth_saturated": 0,
    }

    draws = getattr(fit, "_draws", None)

    all_names = []
    try:
        all_names = list(fit.to_frame().columns)
    except Exception:
        pass
    if draws is not None and (len(all_names) != int(np.asarray(draws).shape[0])):
        all_names = []

    if not all_names:
        all_names = list(getattr(fit, "sample_and_sampler_param_names", []))

    if draws is None or len(all_names) == 0:
        return diagnostics

    # Parameter diagnostics from core model parameters only (exclude p_test and sampler stats).
    def _is_model_param(sample_name):
        return (
            sample_name == "alpha"
            or sample_name == "sigma_u"
            or sample_name.startswith("beta.")
            or sample_name.startswith("z_u.")
        )

    model_idx = [i for i, name in enumerate(all_names) if _is_model_param(name)]
    rhats = []
    esses = []
    for i in model_idx:
        # _draws layout: (variables, samples, chains)
        var_draws = np.asarray(draws[i], dtype=float)
        rhat_val = _rhat_from_chains(var_draws)
        ess_val = _ess_lag1_from_chains(var_draws)
        if np.isfinite(rhat_val):
            rhats.append(rhat_val)
        if np.isfinite(ess_val):
            esses.append(ess_val)

    if rhats:
        diagnostics["max_rhat"] = float(np.max(rhats))
    if esses:
        min_ess = float(np.min(esses))
        diagnostics["min_bulk_ess"] = min_ess
        diagnostics["min_tail_ess"] = min_ess

    # Sampler diagnostics if present.
    for name in ("divergent__", "diverging__", "divergence__"):
        if name in all_names:
            idx = all_names.index(name)
            diagnostics["divergences"] = int(np.nansum(np.asarray(draws[idx], dtype=float)))
            break

    if "treedepth__" in all_names:
        idx = all_names.index("treedepth__")
        treedepth_vals = np.asarray(draws[idx], dtype=float)
        diagnostics["treedepth_saturated"] = int(np.nansum(treedepth_vals >= 10))

    return diagnostics


def _extract_diagnostics(fit):
    """Extract lightweight convergence diagnostics from Stan fit."""
    summary = _safe_summary(fit)
    diagnostics = {
        "max_rhat": np.nan,
        "min_bulk_ess": np.nan,
        "min_tail_ess": np.nan,
        "divergences": 0,
        "treedepth_saturated": 0,
    }

    if not summary:
        return _extract_diagnostics_from_draws(fit)

    cols = summary.get("summary_colnames", [])
    vals = np.asarray(summary.get("summary", []))
    if vals.size == 0 or len(cols) == 0:
        return _extract_diagnostics_from_draws(fit)

    idx = {name: i for i, name in enumerate(cols)}

    if "R_hat" in idx:
        diagnostics["max_rhat"] = float(np.nanmax(vals[:, idx["R_hat"]]))
    if "ESS_bulk" in idx:
        diagnostics["min_bulk_ess"] = float(np.nanmin(vals[:, idx["ESS_bulk"]]))
    if "ESS_tail" in idx:
        diagnostics["min_tail_ess"] = float(np.nanmin(vals[:, idx["ESS_tail"]]))

    # If summary is unavailable/incomplete under a backend, use draw-based fallback.
    if (
        not np.isfinite(diagnostics["max_rhat"])
        or not np.isfinite(diagnostics["min_bulk_ess"])
        or not np.isfinite(diagnostics["min_tail_ess"])
    ):
        fallback = _extract_diagnostics_from_draws(fit)
        for key in ("max_rhat", "min_bulk_ess", "min_tail_ess"):
            if not np.isfinite(diagnostics[key]) and np.isfinite(fallback[key]):
                diagnostics[key] = fallback[key]
        diagnostics["divergences"] = fallback["divergences"]
        diagnostics["treedepth_saturated"] = fallback["treedepth_saturated"]

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

    bayes_cfg = CONFIG.get("bayes", {})
    chains = int(bayes_cfg.get("chains", 2))
    parallel_chains = int(bayes_cfg.get("parallel_chains", chains))
    num_cores = int(bayes_cfg.get("num_cores", parallel_chains))

    # Pass only kwargs supported by the active stan backend.
    build_kwargs = {
        "data": stan_data,
        "random_seed": CONFIG["seed"],
    }
    build_sig = inspect.signature(stan.build)
    if (
        bayes_cfg.get("use_opencl", False)
        and "opencl_ids" in build_sig.parameters
    ):
        build_kwargs["opencl_ids"] = (
            int(bayes_cfg.get("opencl_platform_id", 0)),
            int(bayes_cfg.get("opencl_device_id", 0)),
        )

    sample_kwargs = {
        "num_chains": chains,
        "num_samples": int(bayes_cfg.get("draws", 700)),
        "num_warmup": int(bayes_cfg.get("tune", 350)),
    }

    def _build_and_sample():
        posterior = stan.build(STAN_HIER_LOGIT, **build_kwargs)
        sample_sig = inspect.signature(posterior.sample)
        local_sample_kwargs = dict(sample_kwargs)
        if "num_cores" in sample_sig.parameters:
            local_sample_kwargs["num_cores"] = num_cores
        if "num_chains" in sample_sig.parameters and "parallel_chains" in sample_sig.parameters:
            local_sample_kwargs["num_chains"] = chains
            local_sample_kwargs["parallel_chains"] = parallel_chains
        return posterior.sample(**local_sample_kwargs)

    # Run Stan in a fresh worker thread to avoid Jupyter event-loop conflicts.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        fit = executor.submit(_build_and_sample).result()

    if fit is None:
        raise RuntimeError(
            "Stan sampling returned no fit object. Check the notebook stderr output for the "
            "underlying compiler/sampler error and retry with fewer cores/chains if needed."
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


def run_training_pipeline(df, folds, config=None, verbose=True):
    """Run CV training for baseline and Bayesian models and collect outputs."""
    from data_utils import preprocess_for_baseline, preprocess_for_bayesian, get_X_y

    cfg = CONFIG if config is None else config

    all_predictions = []
    bayes_diagnostics = []
    bayes_effect_summaries = []

    if verbose:
        print("Bayesian fits run serially per outcome; intra-fit parallelism is controlled by Stan chains/cores.")

    for fold_idx, fold_data in enumerate(folds):
        if verbose:
            print(f"Fold {fold_idx+1}/{len(folds)}")

        train_idx, test_idx = fold_data["train"], fold_data["test"]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        for outcome in cfg["outcomes"]:
            train_base, test_base = preprocess_for_baseline(train_df, test_df, outcome)
            X_train_base, y_train_base = get_X_y(train_base, outcome)
            X_test_base, y_test_base = get_X_y(test_base, outcome)

            log_result = fit_logistic(X_train_base, y_train_base, X_test_base)
            log_preds = np.clip(log_result["predictions"], 1e-6, 1 - 1e-6)
            all_predictions.append(
                pd.DataFrame(
                    {
                        "respondent_id": test_df["respondent_id"].values,
                        "fold": fold_idx + 1,
                        "outcome": outcome,
                        "model": "logistic_ridge",
                        "truth": y_test_base.values,
                        "prediction": log_preds,
                    }
                )
            )

            rf_result = fit_random_forest(
                X_train_base,
                y_train_base,
                X_test_base,
                seed=cfg["seed"] + fold_idx,
            )
            rf_preds = np.clip(rf_result["predictions"], 1e-6, 1 - 1e-6)
            all_predictions.append(
                pd.DataFrame(
                    {
                        "respondent_id": test_df["respondent_id"].values,
                        "fold": fold_idx + 1,
                        "outcome": outcome,
                        "model": "random_forest",
                        "truth": y_test_base.values,
                        "prediction": rf_preds,
                    }
                )
            )

            if fold_idx < cfg["max_bayes_folds"]:
                train_bayes, test_bayes = preprocess_for_bayesian(train_df, test_df, outcome)
                bayes_result = fit_bayesian(train_bayes, test_bayes, outcome)
                bayes_preds = np.clip(bayes_result["predictions"], 1e-6, 1 - 1e-6)

                all_predictions.append(
                    pd.DataFrame(
                        {
                            "respondent_id": test_bayes["respondent_id"].values,
                            "fold": fold_idx + 1,
                            "outcome": outcome,
                            "model": "bayesian_hierarchical",
                            "truth": test_bayes[outcome].values,
                            "prediction": bayes_preds,
                            "pred_ci_lower": bayes_result["prediction_ci_lower"],
                            "pred_ci_upper": bayes_result["prediction_ci_upper"],
                        }
                    )
                )

                bayes_diagnostics.append(
                    {
                        "fold": fold_idx + 1,
                        "outcome": outcome,
                        **bayes_result["diagnostics"],
                    }
                )

                sigma_summary = bayes_result["sigma_u_summary"]
                if sigma_summary is not None:
                    bayes_effect_summaries.append(
                        {
                            "fold": fold_idx + 1,
                            "outcome": outcome,
                            "group_sd_mean": float(np.asarray(sigma_summary["mean"]).squeeze()),
                            "group_sd_ci_lower": float(np.asarray(sigma_summary["ci_lower"]).squeeze()),
                            "group_sd_ci_upper": float(np.asarray(sigma_summary["ci_upper"]).squeeze()),
                        }
                    )

                beta_summary = bayes_result.get("beta_summary")
                if beta_summary is not None:
                    beta_means = np.asarray(beta_summary["mean"]).ravel()
                    feature_names = bayes_result.get("feature_names", [])
                    if len(feature_names) == len(beta_means) and len(feature_names) > 0:
                        top_idx = np.argsort(np.abs(beta_means))[-3:][::-1]
                        for j in top_idx:
                            bayes_effect_summaries.append(
                                {
                                    "fold": fold_idx + 1,
                                    "outcome": outcome,
                                    "feature": feature_names[j],
                                    "beta_mean": float(beta_means[j]),
                                    "beta_ci_lower": float(np.asarray(beta_summary["ci_lower"]).ravel()[j]),
                                    "beta_ci_upper": float(np.asarray(beta_summary["ci_upper"]).ravel()[j]),
                                }
                            )

    pred_df = pd.concat(all_predictions, ignore_index=True)

    if verbose:
        print(f"✓ Training complete, collected {len(pred_df)} predictions")
        print(f"✓ Bayesian fits completed: {len(bayes_diagnostics)}")
        print(
            f"✓ Bayesian prior scale in use (for sensitivity analysis): "
            f"{cfg['bayes'].get('prior_sd', 1.0)}"
        )

    return {
        "pred_df": pred_df,
        "bayes_diagnostics": bayes_diagnostics,
        "bayes_effect_summaries": bayes_effect_summaries,
    }

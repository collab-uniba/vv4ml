#!/usr/bin/env python3
"""BoneStrengthML case study for the VV4ML paper.

Stage ``train``: group-aware 70/30 split, hyperparameter optimisation of every
configured regressor for each strain output, evaluation on the held-out test set.
Stage ``verify``: selects the best model per output (lowest test RMSE), runs the
validation test and the four VV4ML verification tests on a shared LHS evaluation
dataset, and stores all results as JSON/CSV.

Usage:
    python run_experiment.py train  --data data/sampleDataset.csv --out results
    python run_experiment.py verify --data data/sampleDataset.csv --out results
    python run_experiment.py all    --data data/sampleDataset.csv --out results

Every stage is resumable: finished models/tests are skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import vv4ml
from vv4ml.utils.model_factory import KERNEL_REGISTRY, MODEL_REGISTRY, recreate_model
from vv4ml.validation import run_validation_test
from vv4ml.verification import (
    build_evaluation_dataset,
    run_convergence_test,
    run_verification,
)
from vv4ml.verification.smoothness import run_smoothness_test

HERE = Path(__file__).resolve().parent
SEED = 42
MAX_RANDOM_ITER = 50


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------- data


def load_data(cfg, data_path: Path):
    fm = cfg.dataset.format_metadata
    df = pd.read_csv(data_path, sep=fm.delimiter, decimal=fm.decimal_separator)
    inputs = [i.name for i in cfg.dataset.inputs]
    outputs = [o.name for o in cfg.dataset.outputs]
    return df[inputs], df[outputs]


def group_split(cfg, X: pd.DataFrame):
    pcs = [c for c in X.columns if c.startswith("PC_")]
    groups = X.groupby(pcs, sort=False).ngroup().values
    frac = cfg.model_development.train_test_split.train_fraction
    gss = GroupShuffleSplit(n_splits=1, train_size=frac, random_state=SEED)
    tr, te = next(gss.split(X, groups=groups))
    return tr, te, groups


# ------------------------------------------------------------------------- models


def base_estimator(model_type: str):
    if model_type == "GaussianProcessRegressor":
        from sklearn.gaussian_process import GaussianProcessRegressor

        return Pipeline(
            [("scaler", StandardScaler()), ("gpr", GaussianProcessRegressor(random_state=SEED))]
        )
    cls = MODEL_REGISTRY[model_type]
    kw = {}
    if model_type == "XGBoostRegressor":
        kw = {"n_jobs": 1, "tree_method": "hist", "random_state": SEED}
    elif model_type == "CatBoostRegressor":
        kw = {"thread_count": 1, "verbose": 0, "random_seed": SEED, "allow_writing_files": False}
    elif model_type == "RandomForestRegressor":
        kw = {"n_jobs": 1, "random_state": SEED}
    elif model_type == "MLPRegressor":
        kw = {"random_state": SEED}
    return cls(**kw)


def split_grid(model_type: str, hp: dict):
    """Separate fixed hyperparameters from search grids."""
    fixed, grid = {}, {}
    for k, v in hp.items():
        if k == "hidden_layer_sizes":
            v = [tuple(x) for x in v] if isinstance(v, list) else v
            (grid if isinstance(v, list) and len(v) > 1 else fixed)[k] = v
            continue
        if isinstance(v, list):
            grid[k] = v
        else:
            fixed[k] = v
    if model_type == "GaussianProcessRegressor":
        fixed = {f"gpr__{k}": v for k, v in fixed.items()}
        grid = {f"gpr__{k}": v for k, v in grid.items()}
        if "gpr__kernel" in grid:
            grid["gpr__kernel"] = [KERNEL_REGISTRY[k] for k in grid["gpr__kernel"]]
    return fixed, grid


def kernel_name(k) -> str:
    for name, kern in KERNEL_REGISTRY.items():
        if repr(kern) == repr(k):
            return name
    return repr(k)


def to_vv4ml_params(model_type: str, params: dict) -> dict:
    """Convert search params into the form accepted by vv4ml.recreate_model."""
    out = {}
    for k, v in params.items():
        k2 = k.replace("gpr__", "")
        if k2 == "kernel":
            v = kernel_name(v)
        out[k2] = v
    if model_type == "XGBoostRegressor":
        out.setdefault("tree_method", "hist")
    if model_type == "CatBoostRegressor":
        out.setdefault("verbose", 0)
        out.setdefault("allow_writing_files", False)
    return out


def eval_metrics(y, yp) -> dict:
    y, yp = np.asarray(y).ravel(), np.asarray(yp).ravel()
    return {
        "MSE": float(mean_squared_error(y, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yp))),
        "MAE": float(mean_absolute_error(y, yp)),
        "MaxAE": float(np.max(np.abs(y - yp))),
        "R2": float(r2_score(y, yp)),
    }


def stage_train(cfg, X, Y, out: Path, models: list[str] | None, n_jobs: int):
    tr, te, groups = group_split(cfg, X)
    dump(
        {
            "n_samples": len(X),
            "n_train": len(tr),
            "n_test": len(te),
            "n_groups": int(len(np.unique(groups))),
            "n_groups_train": int(len(np.unique(groups[tr]))),
            "n_groups_test": int(len(np.unique(groups[te]))),
            "groups_overlap": int(len(set(groups[tr]) & set(groups[te]))),
        },
        out / "split.json",
    )
    np.save(out / "train_idx.npy", tr)
    np.save(out / "test_idx.npy", te)
    cv_folds = cfg.model_development.optimization.cv
    for mcfg in cfg.model_development.model_list:
        if models and mcfg.type not in models:
            continue
        for output in Y.columns:
            mdir = out / "models" / output / mcfg.type
            if (mdir / "metrics.json").exists():
                log(f"skip {output}/{mcfg.type} (done)")
                continue
            log(f"train {output}/{mcfg.type}")
            t0 = time.perf_counter()
            est = base_estimator(mcfg.type)
            fixed, grid = split_grid(mcfg.type, mcfg.hyperparameters)
            est.set_params(**fixed)
            xtr, ytr = X.iloc[tr], Y[output].iloc[tr]
            gtr = groups[tr]
            search_info = {"method": "none", "n_candidates": 1}
            if grid:
                size = int(np.prod([len(v) for v in grid.values()]))
                cv = GroupKFold(n_splits=cv_folds)
                common = dict(
                    estimator=est,
                    scoring="neg_root_mean_squared_error",
                    cv=cv,
                    n_jobs=n_jobs,
                    refit=True,
                    error_score=np.nan,
                    verbose=1,
                )
                if size > MAX_RANDOM_ITER:
                    search = RandomizedSearchCV(
                        param_distributions=grid, n_iter=MAX_RANDOM_ITER, random_state=SEED, **common
                    )
                    search_info = {"method": "random", "grid_size": size, "n_candidates": MAX_RANDOM_ITER}
                else:
                    search = GridSearchCV(param_grid=grid, **common)
                    search_info = {"method": "grid", "grid_size": size, "n_candidates": size}
                search.fit(xtr, ytr, groups=gtr)
                model = search.best_estimator_
                best = {**fixed, **search.best_params_}
                search_info["best_cv_rmse"] = float(-search.best_score_)
            else:
                model = clone(est).fit(xtr, ytr)
                best = dict(fixed)
            fit_s = time.perf_counter() - t0
            m_test = eval_metrics(Y[output].iloc[te], model.predict(X.iloc[te]))
            m_train = eval_metrics(ytr, model.predict(xtr))
            mdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, mdir / "model.joblib")
            dump(
                {
                    "output": output,
                    "model_type": mcfg.type,
                    "best_params": to_vv4ml_params(mcfg.type, best),
                    "search": search_info,
                    "test": m_test,
                    "train": m_train,
                    "fit_seconds": fit_s,
                },
                mdir / "metrics.json",
            )
            log(f"  {output}/{mcfg.type}: test RMSE={m_test['RMSE']:.3e} MaxAE={m_test['MaxAE']:.3e} R2={m_test['R2']:.4f} ({fit_s:.0f}s)")
    # summary table
    rows = [json.loads(p.read_text()) for p in sorted((out / "models").glob("*/*/metrics.json"))]
    pd.DataFrame(
        [{"output": r["output"], "model": r["model_type"], **r["test"], "fit_s": r["fit_seconds"]} for r in rows]
    ).to_csv(out / "model_selection.csv", index=False)


# ------------------------------------------------------------------- verification


def _use_all_cores(model) -> None:
    """Models are fitted single-threaded inside the parallel search; use all
    cores for the (very large) number of predictions required by verification."""
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    params = est.get_params()
    if "n_jobs" in params:
        est.set_params(n_jobs=-1)
    if "thread_count" in params:
        est.set_params(thread_count=-1)


def stage_verify(cfg, X, Y, out: Path, n_seeds: int):
    tr = np.load(out / "train_idx.npy")
    te = np.load(out / "test_idx.npy")
    sel = pd.read_csv(out / "model_selection.csv")
    vdir = out / "verification"
    vdir.mkdir(exist_ok=True)

    log("generate evaluation dataset D")
    t0 = time.perf_counter()
    D = build_evaluation_dataset(cfg)
    ev = cfg.test_configurations.verification.evaluation_dataset
    dump(
        {
            "samples": len(D),
            "dimensions": D.shape[1],
            "optimization": ev.optimization,
            "random_state": ev.random_state,
            "seconds": time.perf_counter() - t0,
        },
        vdir / "evaluation_dataset.json",
    )

    for output in Y.columns:
        odir = vdir / output
        if (odir / "report.json").exists():
            log(f"skip verification {output} (done)")
            continue
        best = sel[sel.output == output].sort_values("RMSE").iloc[0]
        mdir = out / "models" / output / best.model
        meta = json.loads((mdir / "metrics.json").read_text())
        model = joblib.load(mdir / "model.joblib")
        _use_all_cores(model)
        params = meta["best_params"]
        log(f"verify {output} with {best.model}")

        # sanity: rebuilt model reproduces the selected one
        rebuilt = recreate_model(best.model, params, SEED).fit(X.iloc[tr], Y[output].iloc[tr])
        rebuild_diff = float(
            np.max(np.abs(rebuilt.predict(X.iloc[te]).ravel() - model.predict(X.iloc[te]).ravel()))
        )

        vgate = next(g for g in cfg.gate_thresholds.validation if g.field == output)
        val = run_validation_test(model, X.iloc[te], Y[output].iloc[te], vgate.metric, vgate.value)

        rep = run_verification(
            cfg, output, model, best.model, params, X.iloc[tr], Y[output].iloc[tr], x_eval=D, random_state=SEED
        )
        rd = rep.to_dict()
        rd["validation"] = {**val.__dict__, "outcome": val.outcome}
        rd["model_rebuild_max_abs_diff"] = rebuild_diff
        rd["best_params"] = params

        # convergence curve over several subset seeds (sensitivity of the curve)
        conv_cfg = cfg.test_configurations.verification.convergence
        cgate = next(g for g in cfg.gate_thresholds.verification.convergence if g.field == output)
        ref = np.asarray(model.predict(D)).ravel()
        curves = []
        for s in range(n_seeds):
            from vv4ml.verification import convergence as conv_mod

            errs = conv_mod._error_handle(
                X.iloc[tr], Y[output].iloc[tr], D, ref, best.model, params,
                sorted(conv_cfg.n_rows), metric=vv4ml.METRICS_REGISTRY[cgate.metric], random_state=SEED + s,
            )
            curves += [{"seed": SEED + s, "n_rows": n, "error": e} for n, e in zip(sorted(conv_cfg.n_rows), errs)]
        odir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(curves).to_csv(odir / "convergence_curves.csv", index=False)

        # smoothness with range normalisation (diagnostic, not a gate)
        sm = cfg.test_configurations.verification.smoothness
        sgate = next(g for g in cfg.gate_thresholds.verification.smoothness if g.field == output)
        t0 = time.perf_counter()
        sm_range = run_smoothness_test(
            model, D, [tuple(i.range) for i in cfg.dataset.inputs], sm.sample_fraction,
            sm.perturbation_scaled_magnitude, sgate.value, random_state=SEED, normalization="range",
        )
        rd["smoothness_range_normalised"] = {**sm_range.__dict__, "outcome": sm_range.outcome}
        rd["runtime_s"]["smoothness_range"] = time.perf_counter() - t0

        # per-input smoothness tables
        sv = rd["results"]["smoothness"]["max_ratio_per_input"]
        pd.DataFrame(
            {"input": list(sv), "max_ratio_value_norm": list(sv.values()),
             "max_ratio_range_norm": [sm_range.max_ratio_per_input[k] for k in sv]}
        ).to_csv(odir / "smoothness_per_input.csv", index=False)

        # prediction range on D (existence context)
        rd["prediction_on_D"] = {
            "min": float(ref.min()), "max": float(ref.max()),
            "p01": float(np.quantile(ref, 0.01)), "p99": float(np.quantile(ref, 0.99)),
        }
        dump(rd, odir / "report.json")
        log(f"  {output}: " + ", ".join(f"{k}={v['outcome']}" for k, v in rd["results"].items())
            + f", validation={rd['validation']['outcome']}")


def environment(out: Path):
    def ver(m):
        try:
            return __import__(m).__version__
        except Exception:
            return None

    try:
        commit = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    dump(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "vv4ml": ver("vv4ml") or "1.1.0",
            "vv4ml_commit": commit,
            "numpy": ver("numpy"), "pandas": ver("pandas"), "sklearn": ver("sklearn"),
            "xgboost": ver("xgboost"), "catboost": ver("catboost"),
        },
        out / "environment.json",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "verify", "all"])
    ap.add_argument("--config", default=str(HERE / "bonestrength_vv4ml.yaml"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="*", help="subset of model types to train")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-seeds", type=int, default=5, help="seeds for the convergence curve")
    ap.add_argument("--eval-samples", type=int, help="override N_D (for quick tests)")
    a = ap.parse_args()

    cfg = vv4ml.load_config(a.config)
    if a.eval_samples:
        cfg.test_configurations.verification.evaluation_dataset.samples = a.eval_samples
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    environment(out)
    X, Y = load_data(cfg, Path(a.data))
    log(f"data {X.shape} -> outputs {list(Y.columns)}")
    if a.stage in ("train", "all"):
        stage_train(cfg, X, Y, out, a.models, a.n_jobs)
    if a.stage in ("verify", "all"):
        stage_verify(cfg, X, Y, out, a.n_seeds)
    log("done")


if __name__ == "__main__":
    main()

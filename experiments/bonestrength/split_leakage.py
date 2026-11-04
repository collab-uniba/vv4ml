#!/usr/bin/env python3
"""Effect of the splitting strategy on the apparent accuracy of the surrogate.

Trains an XGBoost regressor with fixed hyperparameters on a group-aware split
(femurs never shared between partitions) and on a random split of the rows, and
reports the test R^2 for both outputs. Used in the Results of the VV4ML paper.

    python split_leakage.py --data data/sampleDataset.csv
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n-jobs", type=int, default=-1)
a = ap.parse_args()

df = pd.read_csv(a.data, sep=";")
pcs = [f"PC_{i}" for i in range(1, 94)]
X = df[pcs + ["PosAnt", "MedLat"]]
groups = df.groupby(pcs, sort=False).ngroup().values
splits = {
    "grouped": next(GroupShuffleSplit(1, train_size=0.7, random_state=42).split(X, groups=groups)),
    "random": train_test_split(np.arange(len(X)), train_size=0.7, random_state=42),
}
params = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.5, random_state=42, n_jobs=a.n_jobs)
for out in ["maxStrain_11", "maxStrain_33"]:

    fm = df.groupby(groups)[out].transform("mean")
    r2_femur = r2_score(df[out], fm)
    print(f"{out}: variance explained by femur identity = {r2_femur:.3f}")
    for name, (tr, te) in splits.items():
        m = xgb.XGBRegressor(**params).fit(X.iloc[tr], df[out].iloc[tr])
        print(f"  {name:8s} split: test R2 = {r2_score(df[out].iloc[te], m.predict(X.iloc[te])):.3f}")

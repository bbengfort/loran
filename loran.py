#!/usr/bin/env python3

import json
import time
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from functools import partial
from itertools import combinations

from sklearn.svm import SVR
from sklearn.linear_model import LarsCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score as cvs

# Ignore warnings in this script
warnings.filterwarnings("ignore")

# TODO: don't hardcode models used but make a configuration
ALPHAS = list(np.logspace(-2, 1, 15))
MODELS = {
    "Linear Regression": {
        'Model': LinearRegression, 'params': {},
    },
    "SGD": {
        'Model': SGDRegressor, 'params': {},
    },
    "Huber": {
        'Model': HuberRegressor, 'params': {},
    },
    # "Theil Sen": {
    #     'Model': TheilSenRegressor, 'params': {},
    # },
    "Support Vector RBF": {
        'Model': SVR, 'params': {},
    },
    "kNN Regression": {
        'Model': KNeighborsRegressor, 'params': {},
    },
    # "kNN Radius Regression": {
    #     'Model': RadiusNeighborsRegressor, 'params': {},
    # },
    "Decision Tree": {
        'Model': DecisionTreeRegressor, 'params': {},
    },
    "Random Forest": {
        'Model': RandomForestRegressor, 'params': {},
    },
    "Gradient Boosting": {
        'Model': GradientBoostingRegressor, 'params': {},
    },
    "Extra Tree": {
        'Model': ExtraTreeRegressor, 'params': {},
    },
    # "ARD": {
    #     'Model': ARDRegression, 'params': {},
    # },
    "Bayesian Ridge": {
        'Model': BayesianRidge, 'params': {},
    },
    "Ridge": {
        'Model': RidgeCV, 'params': {'alphas': ALPHAS},
    },
    "Lasso": {
        'Model': LassoCV, 'params': {'alphas': ALPHAS},
    },
    "ElasticNet": {
        'Model': ElasticNetCV, 'params': {'alphas': ALPHAS},
    },
    "Lars": {
        'Model': LarsCV, 'params': {},
    },
    "Lasso Lars": {
        'Model': LassoLarsCV, 'params': {},
    },
    "Bayesian Lasso Lars": {
        'Model': LassoLarsIC, 'params': {},
    },
    "OMP": {
        'Model': OrthogonalMatchingPursuitCV, 'params': {},
    },
    "ANN": {
        'Model': MLPRegressor, 'params': {},
    },
    "Passive Aggressive": {
        'Model': PassiveAggressiveRegressor, 'params': {},
    },
    "Gaussian Process": {
        'Model': GaussianProcessRegressor, 'params': {},
    },
}


def load_dataset(dataset, target_col, feature_cols=None, exclude_cols=None):
    data = pd.read_csv(dataset)

    # TODO: remove hardcoded exclusion columns and add to args
    exclude_cols = (set(exclude_cols) if exclude_cols else set())
    exclude_cols.add(target_col)
    feature_cols = feature_cols or list(set(data.columns) - exclude_cols)

    # Returns X, y
    return data[feature_cols], data[target_col]


def feature_names(*args, **kwargs):
    X, _ = load_dataset(*args, **kwargs)
    return list(X.columns)


def feature_count(*args, **kwargs):
    return len(feature_names(*args, **kwargs))


def fit(name, dataset, target_col, feature_cols=None, exclude_cols=None):
    try:
        # Load Dataset
        X, y = load_dataset(dataset, target_col, feature_cols, exclude_cols)

        # Fetch model to fit and evaluate
        Model = MODELS[name]['Model']
        params = MODELS[name]['params']

        # Fit and evaluate model
        start = time.time()
        model = Model(**params)
        r2_scores = cvs(model, X, y, scoring='r2', cv=12)
        delta = time.time() - start

        # Construct report
        report = {
            'model': name,
            'hyperparameters': model.get_params(),
            'repr': str(model),
            'r2_scores': list(r2_scores),
            'elapsed': delta,
            'target_col': target_col,
            'feature_cols': feature_cols,
            'X_shape': X.shape,
            'y_shape': y.shape,
        }

        return json.dumps(report)
    except Exception as e:
        return json.dumps({
            'model': name,
            'dataset': dataset,
            'target_col': target_col,
            'feature_cols': feature_cols,
            'error': str(e),
        })


def n_jobs(n_features, max_leave_n):
    """
    Computes the total number of jobs in this run
    """
    def n_combos(n, r):
        return int(math.factorial(n) / (math.factorial(n-r)*math.factorial(r)))

    count = 0
    for n in range(max_leave_n+1):
        count += n_combos(n_features, n_features-n)

    return count * len(MODELS)


def jobs(dataset, target_col, max_leave_n, exclude_cols=None):
    """
    Returns an iterable of the args, kwargs for each job
    """
    # First get the name of all features
    features = feature_names(dataset, target_col, exclude_cols=exclude_cols)
    n_features = len(features)
    for leave_n in range(max_leave_n+1):
        for feature_subset in combinations(features, n_features-leave_n):
            for name in MODELS:
                yield (name,), {'feature_cols': list(feature_subset)}


def main(args):
    """
    Run the multiprocessing script on the specified data set.
    """
    full_kwds = {
        'dataset': args.dataset,
        'target_col': args.target,
        'exclude_cols': args.exclude,
    }

    n_features = feature_count(**full_kwds)
    progbar = tqdm(total=n_jobs(n_features, args.max_leave_n), unit='job')
    fit_job = partial(fit, **full_kwds)

    def callback(result):
        with open(args.outpath, 'a') as f:
            f.write(result+"\n")
        progbar.update(1)

    pool = mp.Pool(args.procs)
    for pargs, kwargs in jobs(max_leave_n=args.max_leave_n, **full_kwds):
        pool.apply_async(
            fit_job, pargs, kwargs, callback=callback
        )

    pool.close()
    pool.join()
    progbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="multiprocessing model selection search",
        epilog="this is currently only a prototype",
    )

    args = {
        ('-p', '--procs'): {
            'type': int, 'metavar': 'N', 'default': mp.cpu_count(),
            'help': 'specify the number of python processes to run'
        },
        ('-o', '--outpath'): {
            'metavar': 'PATH', 'default': 'model_rankings.json',
            'help': 'path to write out JSON rankings report',
        },
        ('-t', '--target'): {
            'metavar': 'COL', 'required': True,
            'help': 'specify the name of the column with the target',
        },
        ('-e', '--exclude'): {
            'metavar': 'COLS', 'default': None, 'type': lambda s: s.split(","),
            'help': 'comma separated list of cols to exclude',
        },
        ('-n', '--max-leave-n'): {
            'metavar': 'N', 'default': 2, 'type': int,
            'help': 'maximum number of features to leave out in a run',
        },
        'dataset': {
            'help': 'specify the path to the dataset to train on',
        }
    }

    for pargs, kwargs in args.items():
        if isinstance(pargs, str):
            pargs = (pargs,)
        parser.add_argument(*pargs, **kwargs)

    args = parser.parse_args()
    main(args)

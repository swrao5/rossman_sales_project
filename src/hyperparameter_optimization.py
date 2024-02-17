# src/hyperparameter_optimization.py
from hyperopt import hp, tpe, fmin
from sklearn.model_selection import cross_val_score
import numpy as np

def optimize(space, model, features, target, max_evals=50):
    objective_fn = lambda params: objective(params, model, features, target)
    best_params = fmin(fn=objective_fn, space=space, algo=tpe.suggest, max_evals=max_evals)
    return best_params

def objective(params, model, features, target):
    model.set_params(**params)
    score = -np.mean(cross_val_score(model, features, target, cv=3, scoring='neg_mean_squared_error'))
    return score

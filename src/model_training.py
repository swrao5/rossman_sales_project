# src/model_training.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

def train_random_forest(features, target):
    return RandomForestRegressor().fit(features, target)

def train_xgboost(features, target):
    return XGBRegressor().fit(features, target)

def train_lightgbm(features, target):
    return lgb.LGBMRegressor().fit(features, target)

def evaluate_model(model, features, target):
    score = -np.mean(cross_val_score(model, features, target, cv=3, scoring='neg_mean_squared_error'))
    return np.sqrt(score)

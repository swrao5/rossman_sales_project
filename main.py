# main.py
from src.data_preprocesing import load_data, preprocess_data
from src.feature_engineering import build_features
from src.model_training import train_random_forest, train_xgboost, train_lightgbm, evaluate_model
from src.hyperparameter_optimization import optimize
from hyperopt import hp, tpe, fmin
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split

raw_path = r"C:\Users\raosw\projects\rossman_sales_project\data\raw"
preprocessed_path = r"C:\Users\raosw\projects\rossman_sales_project\data\processed"


# Load data
df_train, df_test, df_store = load_data(raw_path)

# Preprocess data
df_train_preprocessed, df_test_preprocessed, features_train, features_test = preprocess_data(df_train, df_test, df_store, preprocessed_path)

#splitting the data into trainig and validation
train_features, valid_features , train_target, valid_target = train_test_split(df_train_preprocessed[features_train], df_train_preprocessed['Sales'], test_size=0.2, random_state=42)


# Hyperparameter search space for XGBoost with GPU acceleration
xgb_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.choice('max_depth', range(3, 15)),
    'n_estimators': hp.choice('n_estimators', range(100, 1000, 50)),
    'tree_method': 'hist',  # Enable GPU acceleration
    'device' : 'cuda'
    # Add more hyperparameters as needed
}

# Hyperparameter search space for LightGBM with GPU acceleration
lgb_space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.choice('max_depth', range(3, 15)),
    'n_estimators': hp.choice('n_estimators', range(100, 1000, 50)),
    'tree_method': 'hist',  # Enable GPU acceleration
    'device': 'cuda',  # Enable GPU acceleration
    # Add more hyperparameters as needed
}

rf_space = {
    'n_estimators': hp.choice('n_estimators', range(100, 1000, 50)),
    'max_depth': hp.choice('max_depth', range(3, 15)),
    'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
    # Add more hyperparameters as needed
}

# Hyperparameter optimization for XGBoost
best_params_xgb = optimize(xgb_space, XGBRegressor(), train_features, train_target)

# Hyperparameter optimization for RandomForest
best_params_rf = optimize(rf_space, RandomForestRegressor(), train_features, train_target)

# Hyperparameter optimization for LightGBM
best_params_lgb = optimize(lgb_space, lgb.LGBMRegressor(), train_features, train_target)

# Train the models with the best hyperparameters
xgb_model = train_xgboost(train_features, train_target)
rf_model = train_random_forest(train_features, train_target)
lgb_model = train_lightgbm(train_features, train_target)

# Evaluate the models on the test set
xgb_score = evaluate_model(xgb_model, valid_features, valid_target)
rf_score = evaluate_model(rf_model, valid_features, valid_target)
lgb_score = evaluate_model(lgb_model, valid_features, valid_target)

print(f'XGBoost Test RMSE: {xgb_score}')
print(f'Random Forest Test RMSE: {rf_score}')
print(f'LightGBM Test RMSE: {lgb_score}')

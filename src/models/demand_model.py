"""
xgboost model for demand prediction
we predict how many tickets will be sold at a given price point
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.xgboost
import joblib
import os

class DemandPredictor:
    """predicts ticket demand given price and other features"""
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
    
    def train(self, X, y, use_mlflow=True, hyperparameter_tune=False):
        """
        train the demand prediction model
        we use xgboost which handles non-linear relationships well
        """
        
        # split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # default hyperparameters that work well for tabular data
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'early_stopping_rounds': 20
        }
        
        # optional hyperparameter tuning
        if hyperparameter_tune:
            params = self._tune_hyperparameters(X_train, y_train)
        
        if use_mlflow:
            mlflow.set_experiment("demand_prediction")
            mlflow.start_run(run_name="xgboost_demand_model")
            mlflow.log_params(params)
        
        # train the model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # make predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # calculate performance metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        # log feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if use_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(self.model, "demand_model")
            mlflow.end_run()
        
        print(f"\ndemand model training complete:")
        print(f"  validation mae: {metrics['val_mae']:.4f}")
        print(f"  validation r2: {metrics['val_r2']:.4f}")
        print(f"  top 3 features: {self.feature_importance.head(3)['feature'].tolist()}")
        
        return metrics
    
    def _tune_hyperparameters(self, X_train, y_train):
        """simple grid search for hyperparameter tuning"""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.07]
        }
        
        base_model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='neg_mean_absolute_error'
        )
        grid_search.fit(X_train, y_train)
        
        print(f"best parameters from grid search: {grid_search.best_params_}")
        return grid_search.best_params_
    
    def predict(self, X):
        """predict demand for new data points"""
        if self.model is None:
            raise ValueError("model not trained yet. call train() first")
        return self.model.predict(X)
    
    def save(self, path='models/demand_model.pkl'):
        """save trained model to disk"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, path)
        print(f"model saved to {path}")
    
    def load(self, path='models/demand_model.pkl'):
        """load trained model from disk"""
        self.model = joblib.load(path)
        print(f"model loaded from {path}")

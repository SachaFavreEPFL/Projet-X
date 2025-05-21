import json
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    """Load and prepare training data."""
    train_df = pd.read_csv('dataset/stocks_extracted_filtered.csv')
    
    # Features de base
    features = [
        # Ratios de valorisation
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio', 'enterprise_to_ebitda',
        
        # Ratios de qualité
        'return_on_equity', 'return_on_assets', 'operating_margins', 'profit_margins',
        'debt_to_equity', 'current_ratio', 'quick_ratio', 'interest_coverage',
        
        # Ratios de croissance
        'earnings_growth', 'revenue_growth',
        
        # Ratios de marché
        'beta', 'market_daily_change',
        
        # Ratios financiers
        'bs_total_liabilities', 'bs_total_equity', 'bs_cash', 'bs_short_term_investments'
    ]
    
    # Target : score de surévaluation
    target = 'overvaluation_score'
    
    # Filtrer les colonnes qui existent dans le DataFrame
    features = [f for f in features if f in train_df.columns]
    
    X = train_df[features]
    y = train_df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(config, X_train, X_test, y_train, y_test):
    """Evaluate an XGBoost model with a given configuration."""
    model = XGBRegressor(
        **config,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    return model, train_score, test_score, train_rmse, test_rmse

def optimize_hyperparameters(X_train, X_test, y_train, y_test):
    """Optimize XGBoost model hyperparameters."""
    # Define hyperparameter search space
    param_grid = {
        "n_estimators": [100],               # Fixé à une valeur
        "max_depth": [3, 4],                 # 2 valeurs
        "learning_rate": [0.01, 0.05],       # 2 valeurs
        "subsample": [0.7],                  # Fixé à une valeur
        "colsample_bytree": [0.7],           # Fixé à une valeur
        "min_child_weight": [3],             # Fixé à une valeur
        "gamma": [0.1],                      # Fixé à une valeur
        "reg_alpha": [0.1],                  # Fixé à une valeur
        "reg_lambda": [0.1]                  # Fixé à une valeur
    }
    
    # Generate all possible combinations
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in product(*values)]
    
    # Search for best configuration
    best_score = float('-inf')
    best_config = None
    best_model = None
    
    print("\nTesting configurations:")
    print("Config\t\tTrain R²\tTest R²\tTrain RMSE\tTest RMSE\tOverfitting")
    print("-" * 80)
    
    for config in configs:
        model, train_score, test_score, train_rmse, test_rmse = evaluate_model(
            config, X_train, X_test, y_train, y_test
        )
        overfitting = train_score - test_score
        
        # Afficher les résultats pour chaque configuration
        config_str = f"depth={config['max_depth']}, lr={config['learning_rate']}"
        print(f"{config_str}\t{train_score:.4f}\t\t{test_score:.4f}\t{train_rmse:.4f}\t\t{test_rmse:.4f}\t\t{overfitting:.4f}")
        
        if test_score > best_score and overfitting < 0.2:  # Overfitting < 20%
            best_score = test_score
            best_config = config
            best_model = model
    
    print("\nBest configuration found:")
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {best_score:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Overfitting: {overfitting:.4f}")
    
    return best_model, best_config, best_score

def save_results(model, config, score, features):
    """Save results and create visualizations."""
    # Create output directory
    os.makedirs('XGBoost', exist_ok=True)
    
    # Save configuration
    result = {
        "best_config": config,
        "best_score": float(score),
        "feature_importance": pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).to_dict('records')
    }
    
    with open('XGBoost/xgb_best_config.json', 'w') as f:
        json.dump(result, f, indent=4)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance for Overvaluation Prediction')
    plt.tight_layout()
    plt.savefig('XGBoost/feature_importance.png')

def main():
    """Main function."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print("Optimizing hyperparameters...")
    best_model, best_config, best_score = optimize_hyperparameters(
        X_train, X_test, y_train, y_test
    )
    
    print("\nOptimal configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    
    print("\nSaving results...")
    save_results(
        best_model, 
        best_config, 
        best_score,
        X_train.columns
    )
    
    print("\nDone! Results have been saved in the 'XGBoost' directory")

if __name__ == "__main__":
    main() 
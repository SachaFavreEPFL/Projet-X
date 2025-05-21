import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from itertools import product
import matplotlib.pyplot as plt

def load_data():
    """Load and prepare training data."""
    train_df = pd.read_csv('dataset/train_stocks_valuation.csv')
    
    features = [
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
        'roe', 'roa', 'operating_margins', 'profit_margins',
        'debt_to_equity', 'current_ratio',
        'earnings_growth', 'revenue_growth',
        'beta', 'market_cap',
        'sector_encoded'
    ]
    target = 'valuation_class_encoded'
    
    X = train_df[features]
    y = train_df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(config, X_train, X_test, y_train, y_test):
    """Evaluate an XGBoost model with a given configuration."""
    model = XGBClassifier(
        **config,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)
    
    return model, train_score, test_score

def optimize_hyperparameters(X_train, X_test, y_train, y_test):
    """Optimize XGBoost model hyperparameters."""
    # Define hyperparameter search space
    param_grid = {
        "n_estimators": [100],               # Fixé à une valeur
        "max_depth": [3, 4],                 # 2 valeurs
        "learning_rate": [0.01, 0.02],       # 2 valeurs
        "subsample": [0.7],                  # Fixé à une valeur
        "colsample_bytree": [0.7],           # Fixé à une valeur
        "min_child_weight": [3],             # Fixé à une valeur
        "gamma": [0.2],                      # Fixé à une valeur
        "reg_alpha": [0.4],                  # Fixé à une valeur
        "reg_lambda": [0.4]                  # Fixé à une valeur
    }
    
    # Generate all possible combinations
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in product(*values)]
    
    # Search for best configuration
    best_score = 0
    best_config = None
    best_model = None
    
    print("\nTesting configurations:")
    print("Config\t\tTrain Score\tTest Score\tOverfitting")
    print("-" * 60)
    
    for config in configs:
        model, train_score, test_score = evaluate_model(config, X_train, X_test, y_train, y_test)
        overfitting = train_score - test_score
        
        # Afficher les résultats pour chaque configuration
        config_str = f"depth={config['max_depth']}, lr={config['learning_rate']}"
        print(f"{config_str}\t{train_score:.4f}\t\t{test_score:.4f}\t\t{overfitting:.4f}")
        
        if test_score > best_score and overfitting < 0.1:  # Overfitting < 10%
            best_score = test_score
            best_config = config
            best_model = model
    
    print("\nBest configuration found:")
    print(f"Train Score: {train_score:.4f}")
    print(f"Test Score: {best_score:.4f}")
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
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance')
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
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Vérification de l'existence du fichier de configuration
CONFIG_PATH = os.path.join('XGBoost', 'xgb_best_config.json')
if not os.path.exists(CONFIG_PATH):
    print(f"[ERREUR] Le fichier '{CONFIG_PATH}' est introuvable.")
    print("Veuillez d'abord lancer 'tune_xgboost.py' pour générer la meilleure configuration.")
    sys.exit(1)

# Charger la meilleure configuration
with open(CONFIG_PATH, 'r') as f:
    best_config = json.load(f)["best_config"]

def load_and_prepare_data(file_path='dataset/train_stocks_valuation_light.csv'):
    """
    Charge et prépare les données pour l'entraînement
    """
    # Chargement des données
    df = pd.read_csv(file_path)
    
    # Sélection des features
    features = [
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio',
        'roe', 'roa', 'operating_margins', 'profit_margins',
        'debt_to_equity', 'current_ratio',
        'earnings_growth', 'revenue_growth',
        'beta', 'market_cap',
        'sector_encoded'
    ]
    
    # Séparation des features et de la cible
    X = df[features]
    y = df['valuation_class_encoded']  # Utilisation de valuation_class_encoded
    
    return X, y, None  # Plus besoin de label_encoder car les données sont déjà encodées

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Entraîne le modèle XGBoost
    """
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Entraîner le modèle avec la meilleure configuration
    model = xgb.XGBClassifier(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        learning_rate=best_config["learning_rate"],
        subsample=best_config["subsample"],
        colsample_bytree=best_config["colsample_bytree"],
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle
    """
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vrai Label')
    plt.xlabel('Prédiction')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Importance des Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return accuracy, feature_importance

def cross_validate_model(X, y, cv=5):
    """
    Effectue une validation croisée
    """
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(y.unique()),
        learning_rate=0.1,
        max_depth=6
    )
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\nValidation croisée (CV={cv}):")
    print(f"Scores: {scores}")
    print(f"Moyenne: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

def main():
    # Chargement et préparation des données
    print("Chargement et préparation des données...")
    X, y, _ = load_and_prepare_data()
    
    # Entraînement du modèle
    print("\nEntraînement du modèle...")
    model, X_test, y_test = train_model(X, y)
    
    # Évaluation
    print("\nÉvaluation du modèle...")
    accuracy, feature_importance = evaluate_model(model, X_test, y_test)
    
    # Validation croisée
    print("\nValidation croisée...")
    cross_validate_model(X, y)
    
    # Sauvegarde du modèle
    print("\nSauvegarde du modèle...")
    joblib.dump(model, 'xgboost_model.joblib')
    
    print("\nEntraînement terminé !")
    print("Le modèle a été sauvegardé.")
    print("Les graphiques de la matrice de confusion et de l'importance des features ont été générés.")

if __name__ == "__main__":
    main() 
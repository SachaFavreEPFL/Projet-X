import json
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from itertools import product
import time
import matplotlib.pyplot as plt

# Charger les données d'entraînement
train_df = pd.read_csv('dataset/train_stocks_valuation_light.csv')

# Définir les features et la cible
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

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_config(config, X_train, X_test, y_train, y_test):
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
    return train_score, test_score

# Étape 1: Paramètres principaux
print("Étape 1: Recherche des paramètres principaux...")
param_grid_step1 = {
    "n_estimators": [80, 100, 120],      # Réduit encore plus
    "max_depth": [2, 3, 4],              # Réduit encore plus la profondeur
    "learning_rate": [0.005, 0.01, 0.02], # Réduit encore plus le taux d'apprentissage
    "subsample": [0.5, 0.6, 0.7],        # Augmente encore la régularisation
    "colsample_bytree": [0.5, 0.6, 0.7],  # Augmente encore la régularisation
    "reg_alpha": [0.4, 0.5, 0.6],        # Augmente encore la régularisation L1
    "reg_lambda": [0.4, 0.5, 0.6]        # Augmente encore la régularisation L2
}

keys, values = zip(*param_grid_step1.items())
configs_to_test = [dict(zip(keys, v)) for v in product(*values)]

best_score = 0
best_config = None
best_train_score = 0
best_overfitting = float('inf')

for config in configs_to_test:
    train_score, test_score = evaluate_config(config, X_train, X_test, y_train, y_test)
    overfitting = train_score - test_score
    if test_score > best_score and overfitting < 0.1:  # Sur-apprentissage < 10%
        best_score = test_score
        best_config = config
        best_train_score = train_score
        best_overfitting = overfitting

print(f"\nMeilleure configuration de l'étape 1 : {best_config}")
print(f"Score (test) : {best_score:.4f}")
print(f"Score (train): {best_train_score:.4f}")
print(f"Sur-apprentissage: {best_overfitting:.4f}")

# Étape 2: Paramètres secondaires avec les meilleures valeurs de l'étape 1
print("\nÉtape 2: Optimisation fine des paramètres secondaires...")
param_grid_step2 = {
    "min_child_weight": [3, 4, 5],       # Augmenté encore plus
    "gamma": [0.2, 0.3, 0.4],            # Augmenté encore plus
    "reg_alpha": [0.5, 0.6, 0.7],        # Augmenté encore plus
    "reg_lambda": [0.5, 0.6, 0.7]        # Augmenté encore plus
}

# Ajouter les meilleures valeurs de l'étape 1
for key, value in best_config.items():
    param_grid_step2[key] = [value]

keys, values = zip(*param_grid_step2.items())
configs_to_test = [dict(zip(keys, v)) for v in product(*values)]

for config in configs_to_test:
    train_score, test_score = evaluate_config(config, X_train, X_test, y_train, y_test)
    overfitting = train_score - test_score
    if test_score > best_score and overfitting < 0.1:  # Sur-apprentissage < 10%
        best_score = test_score
        best_config = config
        best_train_score = train_score
        best_overfitting = overfitting

# Après la boucle d'optimisation, ajouter :
print("\nAnalyse de l'importance des features :")
best_model = XGBClassifier(
    **best_config,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
best_model.fit(X_train, y_train)

# Calculer les scores pour la meilleure configuration
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
train_score = accuracy_score(y_train, y_pred_train)
test_score = accuracy_score(y_test, y_pred_test)

print("\nRésultats de la meilleure configuration :")
print(f"Précision sur l'ensemble d'entraînement : {train_score:.4f}")
print(f"Précision sur l'ensemble de test        : {test_score:.4f}")
print(f"Écart train-test (sur-apprentissage)   : {train_score - test_score:.4f}")

print("\nInterprétation du sur-apprentissage :")
print("Écart < 5%  : Très bon, le modèle généralise très bien")
print("Écart 5-10% : Acceptable, le modèle généralise bien")
print("Écart > 10% : Signe de sur-apprentissage, le modèle 'apprend par cœur'")

# Obtenir l'importance des features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nImportance des features :")
print(feature_importance)

# Sauvegarde des résultats
result = {
    "best_config": best_config,
    "best_score": float(best_score),  # Conversion en float pour éviter les problèmes de sérialisation
    "feature_importance": feature_importance.to_dict('records')
}

OUTPUT_DIR = 'XGBoost'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(OUTPUT_DIR, 'xgb_best_config.json')

with open(CONFIG_PATH, 'w') as f:
    json.dump(result, f, indent=4)

print(f"\nMeilleure configuration finale :")
print(f"Score de test : {best_score:.4f}")
print(f"Sur-apprentissage : {train_score - test_score:.4f}")
print(f"\nConfiguration :")
for key, value in best_config.items():
    print(f"{key}: {value}")

print(f"\nConfiguration sauvegardée dans {CONFIG_PATH}")

# Créer un graphique d'importance des features
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Importance des Features')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
print("\nGraphique d'importance des features sauvegardé dans 'XGBoost/feature_importance.png'") 